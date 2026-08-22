"""한국어 평가 데이터 생성기와 점수 계산을 실제 모델 없이 검증한다."""

from __future__ import annotations

import tempfile
import unittest
import wave
from argparse import Namespace
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from evaluation.evaluate_dataset import (
    canonicalize_terms,
    load_aliases,
    percentile,
    score_prediction,
    summarize_results,
)
from evaluation.generate_evaluation_data import (
    DEFAULT_SENTENCES,
    EVALUATION_ROOT,
    _windows_script,
    generate_dataset,
    load_sentences,
)


class EvaluationToolsTest(unittest.TestCase):
    """30문장 구성, TTS manifest와 CER/WER 집계를 검증한다."""

    def setUp(self) -> None:
        self.aliases = load_aliases(EVALUATION_ROOT / "term_aliases.json")

    def test_sentence_categories_have_ten_items_each(self) -> None:
        sentences = load_sentences(DEFAULT_SENTENCES)
        self.assertEqual(len(sentences), 30)
        self.assertEqual(
            Counter(sentence["category"] for sentence in sentences),
            {"general": 10, "domain": 10, "mixed": 10},
        )
        self.assertEqual(len({sentence["id"] for sentence in sentences}), 30)

    def test_term_aliases_normalize_english_and_korean_spellings(self) -> None:
        english = canonicalize_terms("Jetson Orin Nano와 DOA", self.aliases)
        korean = canonicalize_terms("제트슨 오린 나노와 디오에이", self.aliases)
        self.assertEqual(english, korean)

        metrics = score_prediction(
            "Jetson Orin Nano에서 DOA를 실행합니다.",
            "제트슨 오린 나노에서 디오에이를 실행합니다.",
            self.aliases,
        )
        self.assertGreater(metrics["raw_cer_percent"], 0)
        self.assertEqual(metrics["normalized_cer_percent"], 0)
        self.assertEqual(metrics["expected_domain_terms"], 2)
        self.assertEqual(metrics["correct_domain_terms"], 2)

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([10.0, 20.0, 30.0], 0.5), 20.0)
        self.assertEqual(percentile([10.0, 20.0], 0.5), 15.0)
        self.assertEqual(percentile([], 0.95), 0.0)

    def test_summary_contains_accuracy_latency_and_failure_fields(self) -> None:
        metrics = score_prediction("안녕하세요", "안녕하세요", self.aliases)
        records = [
            {
                "id": "1",
                "category": "general",
                "audio_seconds": 1.0,
                "latency_ms": 200.0,
                "hypothesis": "안녕하세요",
                "metrics": metrics,
            },
            {
                "id": "1",
                "category": "general",
                "audio_seconds": 1.0,
                "latency_ms": 400.0,
                "hypothesis": "안녕하세요",
                "metrics": metrics,
            },
            {
                "id": "2",
                "category": "general",
                "audio_seconds": 1.0,
                "latency_ms": 10.0,
                "hypothesis": "",
                "error": "test error",
            },
        ]
        summary = summarize_results(records, "small", "cpu", "int8", 100.0)
        self.assertEqual(summary["latency_ms"]["average"], 300.0)
        self.assertEqual(summary["rtf"], 0.2)
        self.assertEqual(summary["raw_cer_percent"], 0.0)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["duplicate_result_count"], 1)

    def test_generator_writes_manifest_and_valid_wav_metadata(self) -> None:
        def fake_generator(text: str, output: Path, voice: str | None, rate: int) -> str:
            del text, voice, rate
            with wave.open(str(output), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16_000)
                wav_file.writeframes(b"\x00\x00" * 1_600)
            return "테스트 음성"

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated"
            args = Namespace(
                sentences=DEFAULT_SENTENCES,
                output=output,
                overwrite=False,
                voice=None,
                windows_rate=0,
                macos_rate=180,
            )
            with (
                patch(
                    "evaluation.generate_evaluation_data.platform.system",
                    return_value="Darwin",
                ),
                patch(
                    "evaluation.generate_evaluation_data.generate_macos_wav",
                    side_effect=fake_generator,
                ),
            ):
                manifest_path = generate_dataset(args)

            self.assertTrue(manifest_path.is_file())
            self.assertEqual(len(list((output / "audio").glob("*.wav"))), 30)
            self.assertIn('"purpose": "synthetic-regression-only"', manifest_path.read_text())

    def test_windows_tts_script_requires_korean_and_pcm16_mono(self) -> None:
        """Windows에서 실행할 PowerShell TTS 계약을 정적으로 검증한다."""

        script = _windows_script()
        for required_text in (
            'Culture.Name -eq "ko-KR"',
            "16000",
            "AudioBitsPerSample]::Sixteen",
            "AudioChannel]::Mono",
            "SetOutputToWaveFile",
            "SetOutputToNull",
        ):
            self.assertIn(required_text, script)


if __name__ == "__main__":
    unittest.main()
