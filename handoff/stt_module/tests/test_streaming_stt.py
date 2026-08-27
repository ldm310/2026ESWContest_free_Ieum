"""실제 Whisper와 마이크 없이 StreamingSTT 공개 API를 검증한다."""

from __future__ import annotations

import threading
import time
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from stt import config
from stt.streaming import (
    DEFAULT_NOISE_CALIBRATION_SECONDS,
    DEFAULT_PARTIAL_INTERVAL_SECONDS,
    DEFAULT_PREVIEW_SECONDS,
    StreamingSTT,
    _looks_like_prompt_hallucination,
    _run_model_transcription,
    merge_partial,
    prepare_audio_array,
    resample_linear,
)
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
        self.final_model = object()
        self.partial_model = object()
        self.model_patcher = patch(
            "stt.streaming.get_model", return_value=self.final_model
        )
        self.partial_model_patcher = patch(
            "stt.streaming.ModelManager.get_partial_model",
            return_value=self.partial_model,
        )
        self.unload_patcher = patch("stt.streaming.ModelManager.unload_model")
        self.model_patcher.start()
        self.partial_model_patcher.start()
        self.mock_unload = self.unload_patcher.start()

    def tearDown(self) -> None:
        self.unload_patcher.stop()
        self.partial_model_patcher.stop()
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
            noise_calibration_seconds=0.0,
        )
        stream.start()
        return stream, transcribe

    def test_start_stop_and_duplicate_calls(self) -> None:
        stream, _ = self.make_stream(lambda result: None)
        stream.start()
        stream.stop()
        stream.stop()
        self.mock_unload.assert_called_once()

    def test_realtime_default_settings_match_jetson_profile(self) -> None:
        """Jetson 추천 연산·Partial·Final 기본값의 회귀를 방지한다."""

        self.assertEqual(DEFAULT_PARTIAL_INTERVAL_SECONDS, 0.25)
        self.assertEqual(DEFAULT_PREVIEW_SECONDS, 4.0)
        self.assertEqual(DEFAULT_NOISE_CALIBRATION_SECONDS, 1.0)
        self.assertEqual(config.BEAM_SIZE, 3)
        with patch("stt.config.get_device", return_value="cuda"):
            self.assertEqual(config.get_compute_type(), "int8_float16")
        with patch("stt.config.get_device", return_value="cpu"):
            self.assertEqual(config.get_compute_type(), "int8")

    def test_transcription_uses_final_safety_options_without_hotwords(self) -> None:
        """Final만 제한적 prompt와 VAD를 사용하고 hotwords는 주입하지 않는다."""

        model = Mock()
        model.transcribe.return_value = (
            iter([SimpleNamespace(text=" 전문용어 자막 ")]),
            object(),
        )
        text = _run_model_transcription(
            model,
            np.ones(1_600, dtype=np.float32),
            beam_size=config.BEAM_SIZE,
            is_partial=False,
        )

        self.assertEqual(text, "전문용어 자막")
        options = model.transcribe.call_args.kwargs
        self.assertEqual(options["initial_prompt"], config.INITIAL_PROMPT)
        self.assertNotIn("hotwords", options)
        self.assertEqual(options["beam_size"], 3)
        self.assertTrue(options["vad_filter"])
        self.assertIn("vad_parameters", options)
        self.assertFalse(options["condition_on_previous_text"])

        model.transcribe.return_value = (
            iter([SimpleNamespace(text=" 부분 자막 ")]),
            object(),
        )
        _run_model_transcription(
            model,
            np.ones(1_600, dtype=np.float32),
            beam_size=1,
            is_partial=True,
        )
        partial_options = model.transcribe.call_args.kwargs
        self.assertNotIn("initial_prompt", partial_options)
        self.assertNotIn("hotwords", partial_options)
        self.assertFalse(partial_options["vad_filter"])

    def test_prompt_only_hallucinations_are_rejected(self) -> None:
        """사진에서 확인된 전문용어 나열과 반복을 Final에서 차단한다."""

        self.assertTrue(_looks_like_prompt_hallucination("BEM, MVDR, BEM, BEM"))
        self.assertTrue(_looks_like_prompt_hallucination("CTranslate2"))
        self.assertFalse(
            _looks_like_prompt_hallucination(
                "Jetson Orin Nano에서 한국어 자막을 실행합니다"
            )
        )

        model = Mock()
        model.transcribe.return_value = (
            iter(
                [
                    SimpleNamespace(
                        text=" BEM, MVDR, BEM ",
                        avg_logprob=-0.2,
                        no_speech_prob=0.1,
                    )
                ]
            ),
            object(),
        )
        self.assertEqual(
            _run_model_transcription(
                model,
                np.ones(1_600, dtype=np.float32),
                beam_size=3,
                is_partial=False,
            ),
            "",
        )

    def test_conflicting_partial_is_replaced_instead_of_concatenated(self) -> None:
        """서로 다른 tiny 가설을 이어 붙여 중복 자막을 만들지 않는다."""

        self.assertEqual(
            merge_partial(
                "재생 오린 나노에서 한국어 실시간 자막 시스템을",
                "Jetson Orin Nano에서 한국어 실시간 자막 시스템을 실행합니다",
            ),
            "Jetson Orin Nano에서 한국어 실시간 자막 시스템을 실행합니다",
        )

    def test_audio_conditioning_removes_dc_and_limits_peak(self) -> None:
        """긴 Beamforming chunk의 DC와 범위 초과 peak를 보정한다."""

        positions = np.arange(1_600, dtype=np.float32)
        audio = 0.4 + 2.0 * np.sin(2 * np.pi * positions / 32)
        prepared = prepare_audio_array(audio, 16_000, 16_000)
        self.assertAlmostEqual(float(np.mean(prepared)), 0.0, places=5)
        self.assertLessEqual(float(np.max(np.abs(prepared))), 1.0)

    def test_noise_calibration_does_not_create_false_caption(self) -> None:
        """지속 Beamforming 잡음은 발화로 등록하지 않고 실제 음성만 처리한다."""

        results: list[STTResult] = []
        transcribe = Mock(
            side_effect=lambda model, audio, rate, beam, partial: (
                "부분 자막" if partial else "최종 자막"
            )
        )
        self.addCleanup(patch.stopall)
        patch("stt.streaming._transcribe_audio", transcribe).start()
        stream = StreamingSTT(
            on_result=results.append,
            sample_rate=1_000,
            partial_interval=0.02,
            silence_threshold=0.01,
            silence_duration=0.02,
            preview_seconds=0.2,
            pre_roll_seconds=0.01,
            min_speech_duration=0.01,
            noise_calibration_seconds=0.06,
            noise_threshold_multiplier=2.5,
        )
        stream.start()

        noise = np.tile(np.array([0.02, -0.02], dtype=np.float32), 15)
        for _ in range(4):
            stream.push_audio(noise, sample_rate=1_000)
        stream.flush()
        self.assertFalse(any(result.type in {"partial", "final"} for result in results))

        voice = np.tile(np.array([0.2, -0.2], dtype=np.float32), 15)
        stream.push_audio(voice, sample_rate=1_000)
        stream.push_audio(voice, sample_rate=1_000)
        stream.push_audio(np.zeros(30, dtype=np.float32), sample_rate=1_000)
        wait_until(lambda: any(result.type == "final" for result in results))
        stream.stop()

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

    def test_partial_and_final_use_separate_models(self) -> None:
        transcribe = Mock(
            side_effect=lambda model, audio, rate, beam, partial: (
                "부분 자막" if partial else "최종 자막"
            )
        )
        results: list[STTResult] = []
        stream, _ = self.make_stream(results.append, transcribe)
        transcribe.reset_mock()  # start()의 두 모델 warm-up 호출은 제외한다.

        voice = np.full(30, 0.2, dtype=np.float32)
        stream.push_audio(voice, sample_rate=1_000)
        stream.push_audio(voice, sample_rate=1_000)
        wait_until(lambda: any(result.type == "partial" for result in results))
        stream.flush()

        runtime_calls = transcribe.call_args_list
        partial_calls = [call for call in runtime_calls if call.args[4] is True]
        final_calls = [call for call in runtime_calls if call.args[4] is False]
        self.assertTrue(partial_calls)
        self.assertTrue(final_calls)
        self.assertTrue(all(call.args[0] is self.partial_model for call in partial_calls))
        self.assertTrue(all(call.args[0] is self.final_model for call in final_calls))
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
