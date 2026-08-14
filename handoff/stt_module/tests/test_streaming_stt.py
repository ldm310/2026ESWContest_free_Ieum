"""실제 Whisper와 마이크 없이 StreamingSTT 공개 API를 검증한다."""

from __future__ import annotations

import threading
import time
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np

from stt.streaming import StreamingSTT, prepare_audio_array, resample_linear
from stt.types import STTResult


def wait_until(predicate, timeout: float = 2.0) -> None:
    """비동기 worker 조건이 충족될 때까지 짧게 기다린다."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("비동기 테스트 조건이 제한 시간 안에 충족되지 않았습니다.")


class StreamingSTTTest(unittest.TestCase):
    """StreamingSTT 입력 계약, lifecycle, callback과 통계를 검증한다."""

    def setUp(self) -> None:
        self.model_patcher = patch("stt.streaming.get_model", return_value=object())
        self.unload_patcher = patch("stt.streaming.ModelManager.unload_model")
        self.model_patcher.start()
        self.mock_unload = self.unload_patcher.start()

    def tearDown(self) -> None:
        self.unload_patcher.stop()
        self.model_patcher.stop()

    def make_stream(
        self,
        callback,
        transcribe=None,
    ) -> tuple[StreamingSTT, Mock]:
        """빠른 테스트 설정의 started StreamingSTT를 만든다."""

        if transcribe is None:
            transcribe = Mock(
                side_effect=lambda model, audio, rate, beam, partial: (
                    "부분 자막" if partial else "최종 자막"
                )
            )
        self.addCleanup(patch.stopall)
        patcher = patch("stt.streaming._transcribe_audio", transcribe)
        patcher.start()
        stream = StreamingSTT(
            on_result=callback,
            sample_rate=1_000,
            partial_interval=0.02,
            silence_threshold=0.01,
            silence_duration=0.02,
            preview_seconds=0.2,
            pre_roll_seconds=0.01,
            min_speech_duration=0.01,
        )
        stream.start()
        return stream, transcribe

    def test_start_stop_and_duplicate_calls(self) -> None:
        stream, _ = self.make_stream(lambda result: None)
        stream.start()
        stream.stop()
        stream.stop()
        self.mock_unload.assert_called_once()

    def test_push_before_start_raises(self) -> None:
        stream = StreamingSTT(lambda result: None)
        with self.assertRaisesRegex(RuntimeError, "start"):
            stream.push_audio(np.ones(10, dtype=np.float32))

    def test_push_after_stop_raises(self) -> None:
        stream, _ = self.make_stream(lambda result: None)
        stream.stop()
        with self.assertRaises(RuntimeError):
            stream.push_audio(np.ones(10, dtype=np.float32), sample_rate=1_000)

    def test_start_failure_emits_error_and_allows_retry(self) -> None:
        results: list[STTResult] = []
        stream = StreamingSTT(results.append, sample_rate=1_000)
        with patch(
            "stt.streaming._transcribe_audio",
            side_effect=RuntimeError("warm-up failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "시작에 실패"):
                stream.start()
        self.assertEqual(results[0].type, "error")
        self.assertIn("warm-up failure", results[0].error or "")

        with patch("stt.streaming._transcribe_audio", return_value=""):
            stream.start()
            stream.stop()

    def test_float_int16_column_and_invalid_multichannel_inputs(self) -> None:
        float_audio = prepare_audio_array(
            np.array([[0.25], [-0.5]], dtype=np.float64), 16_000, 16_000
        )
        self.assertEqual(float_audio.dtype, np.float32)
        np.testing.assert_allclose(float_audio, [0.25, -0.5])

        int_audio = prepare_audio_array(
            np.array([32_767, -32_768], dtype=np.int16), 16_000, 16_000
        )
        np.testing.assert_allclose(int_audio, [32_767 / 32_768, -1.0])

        with self.assertRaises(ValueError):
            prepare_audio_array(np.zeros((10, 2), dtype=np.float32), 16_000, 16_000)
        with self.assertRaises(ValueError):
            prepare_audio_array(np.array([np.nan], dtype=np.float32), 16_000, 16_000)

    def test_empty_array_is_ignored(self) -> None:
        stream, _ = self.make_stream(lambda result: None)
        stream.push_audio(np.array([], dtype=np.float32), sample_rate=1_000)
        self.assertEqual(stream.get_stats()["audio_samples_received"], 0)
        stream.stop()

    def test_pcm16_mono_and_multichannel_conversion(self) -> None:
        stream = StreamingSTT(lambda result: None)
        stream.push_audio = Mock()

        mono = np.array([32_767, -32_768], dtype="<i2")
        stream.push_pcm16(mono.tobytes(), sample_rate=16_000)
        mono_argument = stream.push_audio.call_args.args[0]
        np.testing.assert_allclose(mono_argument, [32_767 / 32_768, -1.0])

        stereo = np.array([[32_767, -32_767], [16_000, 16_000]], dtype="<i2")
        stream.push_pcm16(stereo.tobytes(), sample_rate=16_000, channels=2)
        stereo_argument = stream.push_audio.call_args.args[0]
        np.testing.assert_allclose(stereo_argument, [0.0, 16_000 / 32_768])

    def test_pcm16_rejects_incomplete_frame(self) -> None:
        stream = StreamingSTT(lambda result: None)
        with self.assertRaises(ValueError):
            stream.push_pcm16(b"\x00\x01", channels=2)

    def test_linear_resampling(self) -> None:
        source = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        resampled = resample_linear(source, 4, 8)
        self.assertEqual(resampled.dtype, np.float32)
        self.assertEqual(resampled.size, 8)

    def test_push_audio_resamples_before_queueing(self) -> None:
        stream, _ = self.make_stream(lambda result: None)
        stream.push_audio(np.ones(100, dtype=np.float32), sample_rate=2_000)
        wait_until(lambda: stream.get_stats()["current_audio_queue_size"] == 0)
        self.assertEqual(stream.get_stats()["audio_samples_received"], 50)
        stream.stop()

    def test_partial_and_final_callbacks_and_ids(self) -> None:
        results: list[STTResult] = []
        stream, _ = self.make_stream(results.append)
        voice = np.full(30, 0.2, dtype=np.float32)

        stream.push_audio(voice, sample_rate=1_000)
        stream.push_audio(voice, sample_rate=1_000)
        wait_until(lambda: any(result.type == "partial" for result in results))
        stream.flush()
        self.assertTrue(any(result.type == "final" for result in results))

        stream.push_audio(voice, sample_rate=1_000)
        stream.flush()
        finals = [result for result in results if result.type == "final"]
        self.assertEqual([result.utterance_id for result in finals], [1, 2])
        self.assertEqual(
            [result.sequence_id for result in results],
            list(range(1, len(results) + 1)),
        )
        self.assertTrue(
            all(datetime.fromisoformat(result.timestamp).tzinfo is not None for result in results)
        )
        stream.stop()

    def test_silence_creates_final(self) -> None:
        results: list[STTResult] = []
        stream, _ = self.make_stream(results.append)
        stream.push_audio(np.full(30, 0.2, dtype=np.float32), sample_rate=1_000)
        stream.push_audio(np.zeros(30, dtype=np.float32), sample_rate=1_000)
        wait_until(lambda: any(result.type == "final" for result in results))
        stream.stop()

    def test_duplicate_partial_is_not_emitted(self) -> None:
        results: list[STTResult] = []
        stream, _ = self.make_stream(results.append)
        voice = np.full(25, 0.2, dtype=np.float32)
        for _ in range(5):
            stream.push_audio(voice, sample_rate=1_000)
            time.sleep(0.01)
        wait_until(lambda: stream.get_stats()["partial_count"] >= 2)
        partials = [result for result in results if result.type == "partial"]
        self.assertEqual(len(partials), 1)
        stream.stop()

    def test_callback_exception_does_not_kill_worker(self) -> None:
        callback_calls = []

        def failing_callback(result: STTResult) -> None:
            callback_calls.append(result)
            raise RuntimeError("callback failure")

        stream, _ = self.make_stream(failing_callback)
        stream.push_audio(np.full(30, 0.2, dtype=np.float32), sample_rate=1_000)
        stream.flush()
        stream.push_audio(np.full(30, 0.2, dtype=np.float32), sample_rate=1_000)
        stream.flush()
        self.assertGreaterEqual(len(callback_calls), 2)
        stream.stop()

    def test_inference_error_emits_error_result(self) -> None:
        results: list[STTResult] = []

        def failing_transcribe(model, audio, rate, beam, partial):
            if audio.size == 250:  # start() warm-up은 성공시킨다.
                return ""
            raise RuntimeError("inference failure")

        stream, _ = self.make_stream(results.append, Mock(side_effect=failing_transcribe))
        stream.push_audio(np.full(30, 0.2, dtype=np.float32), sample_rate=1_000)
        stream.flush()
        errors = [result for result in results if result.type == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].text, "")
        self.assertTrue(errors[0].is_final)
        self.assertIn("inference failure", errors[0].error or "")
        self.assertEqual(stream.get_stats()["error_count"], 1)
        stream.stop()

    def test_result_is_frozen(self) -> None:
        result = STTResult("partial", "안녕", 1.0, "2026-08-02T00:00:00+09:00", 1, 1, False)
        with self.assertRaises(FrozenInstanceError):
            result.text = "변경"

    def test_flush_without_utterance_returns(self) -> None:
        stream, _ = self.make_stream(lambda result: None)
        stream.flush()
        stream.stop()

    def test_stats_are_copies_and_contain_required_fields(self) -> None:
        stream, _ = self.make_stream(lambda result: None)
        stream.push_audio(np.full(30, 0.2, dtype=np.float32), sample_rate=1_000)
        stream.flush()
        first = stream.get_stats()
        first["final_count"] = 999
        second = stream.get_stats()
        self.assertNotEqual(second["final_count"], 999)
        self.assertEqual(second["audio_samples_received"], 30)
        self.assertAlmostEqual(second["audio_seconds_received"], 0.03)
        self.assertEqual(len(second), 12)
        stream.stop()

    def test_model_inference_never_overlaps(self) -> None:
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def slow_transcribe(model, audio, rate, beam, partial):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return "부분" if partial else "최종"

        results: list[STTResult] = []
        stream, _ = self.make_stream(results.append, Mock(side_effect=slow_transcribe))
        voice = np.full(25, 0.2, dtype=np.float32)
        for _ in range(8):
            stream.push_audio(voice, sample_rate=1_000)
        stream.flush()
        self.assertEqual(maximum_active, 1)
        self.assertGreater(stream.get_stats()["skipped_partial_requests"], 0)
        stream.stop()

    def test_concurrent_push_audio_is_safe(self) -> None:
        stream, _ = self.make_stream(lambda result: None)
        chunk = np.full(10, 0.2, dtype=np.float32)
        threads = [
            threading.Thread(
                target=lambda: stream.push_audio(chunk, sample_rate=1_000)
            )
            for _ in range(10)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        stream.flush()
        self.assertEqual(stream.get_stats()["audio_samples_received"], 100)
        stream.stop()


if __name__ == "__main__":
    unittest.main()
