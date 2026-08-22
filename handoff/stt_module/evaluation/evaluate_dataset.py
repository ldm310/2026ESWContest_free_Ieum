"""정답 manifest의 WAV를 faster-whisper로 평가하고 JSON 결과를 저장한다."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence, TypeVar

from stt.config import (
    BEAM_SIZE,
    HOTWORDS,
    INITIAL_PROMPT,
    LANGUAGE,
    MODEL_SIZE,
    get_compute_type,
    get_device,
)


EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = EVALUATION_ROOT / "generated" / "ground_truth.json"
DEFAULT_ALIASES = EVALUATION_ROOT / "term_aliases.json"
DEFAULT_RESULTS_ROOT = EVALUATION_ROOT / "results"
T = TypeVar("T")


def edit_distance(reference: Sequence[T], hypothesis: Sequence[T]) -> int:
    """두 문자 또는 단어 sequence 사이의 Levenshtein 거리를 계산한다."""

    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_value in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_value in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_value != hypothesis_value),
                )
            )
        previous = current
    return previous[-1]


def normalize_text(text: str) -> str:
    """Unicode와 연속 공백을 정리하고 영문 대소문자를 통일한다."""

    normalized = unicodedata.normalize("NFC", text).casefold()
    return " ".join(normalized.split())


def character_units(text: str) -> list[str]:
    """문장부호와 공백을 제외한 CER 문자 단위를 반환한다."""

    return [character for character in normalize_text(text) if character.isalnum()]


def word_units(text: str) -> list[str]:
    """문장부호를 공백으로 바꾼 WER 단어 단위를 반환한다."""

    cleaned = "".join(
        character if character.isalnum() else " " for character in normalize_text(text)
    )
    return cleaned.split()


def load_aliases(path: Path) -> list[dict[str, Any]]:
    """전문용어 표기 변형 목록을 길이가 긴 변형부터 반환한다."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("전문용어 alias JSON은 리스트여야 합니다.")
    aliases: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("전문용어 alias 항목은 객체여야 합니다.")
        canonical = str(item.get("canonical", "")).strip()
        variants = [str(value).strip() for value in item.get("variants", [])]
        if not canonical or not variants or any(not value for value in variants):
            raise ValueError(f"잘못된 전문용어 alias입니다: {item}")
        aliases.append(
            {
                "canonical": canonical,
                "variants": sorted(variants, key=len, reverse=True),
            }
        )
    return aliases


def canonicalize_terms(text: str, aliases: list[dict[str, Any]]) -> str:
    """영문·음차 등 허용된 전문용어 표기를 같은 canonical 값으로 바꾼다."""

    canonicalized = unicodedata.normalize("NFC", text)
    for item in aliases:
        for variant in item["variants"]:
            canonicalized = re.sub(
                re.escape(variant),
                item["canonical"],
                canonicalized,
                flags=re.IGNORECASE,
            )
    return canonicalized


def percentile(values: Sequence[float], percentage: float) -> float:
    """선형 보간 방식의 percentile을 계산한다."""

    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def wav_duration_seconds(path: Path) -> float:
    """PCM WAV 길이를 초 단위로 반환한다."""

    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """생성된 ground_truth manifest의 평가 항목을 검증해 반환한다."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise ValueError("ground_truth.json에 평가 항목이 없습니다.")
    identifiers: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("평가 항목은 객체여야 합니다.")
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError(f"비어 있거나 중복된 평가 ID입니다: {identifier!r}")
        identifiers.add(identifier)
        if not str(item.get("audio_path", "")).strip() or not str(
            item.get("text", "")
        ).strip():
            raise ValueError(
                f"평가 항목의 audio_path/text가 비어 있습니다: {identifier}"
            )
    return items


def _contains_variant(text: str, variants: list[str]) -> bool:
    """문장에 허용된 전문용어 표기 중 하나가 포함됐는지 확인한다."""

    normalized = unicodedata.normalize("NFC", text).casefold()
    return any(
        unicodedata.normalize("NFC", variant).casefold() in normalized
        for variant in variants
    )


def score_prediction(
    reference: str,
    hypothesis: str,
    aliases: list[dict[str, Any]],
) -> dict[str, int | float | bool]:
    """한 문장의 원시·전문용어 정규화 정확도 지표를 계산한다."""

    raw_reference_characters = character_units(reference)
    raw_hypothesis_characters = character_units(hypothesis)
    raw_reference_words = word_units(reference)
    raw_hypothesis_words = word_units(hypothesis)
    normalized_reference = canonicalize_terms(reference, aliases)
    normalized_hypothesis = canonicalize_terms(hypothesis, aliases)
    normalized_reference_characters = character_units(normalized_reference)
    normalized_hypothesis_characters = character_units(normalized_hypothesis)

    expected_terms = 0
    correct_terms = 0
    for item in aliases:
        variants = item["variants"]
        if _contains_variant(reference, variants):
            expected_terms += 1
            if _contains_variant(hypothesis, variants):
                correct_terms += 1

    raw_character_edits = edit_distance(
        raw_reference_characters, raw_hypothesis_characters
    )
    raw_word_edits = edit_distance(raw_reference_words, raw_hypothesis_words)
    normalized_character_edits = edit_distance(
        normalized_reference_characters,
        normalized_hypothesis_characters,
    )
    return {
        "raw_character_edits": raw_character_edits,
        "raw_character_count": len(raw_reference_characters),
        "raw_word_edits": raw_word_edits,
        "raw_word_count": len(raw_reference_words),
        "normalized_character_edits": normalized_character_edits,
        "normalized_character_count": len(normalized_reference_characters),
        "raw_cer_percent": round(
            100 * raw_character_edits / max(1, len(raw_reference_characters)), 2
        ),
        "raw_wer_percent": round(
            100 * raw_word_edits / max(1, len(raw_reference_words)), 2
        ),
        "normalized_cer_percent": round(
            100
            * normalized_character_edits
            / max(1, len(normalized_reference_characters)),
            2,
        ),
        "exact_match": normalize_text(reference) == normalize_text(hypothesis),
        "expected_domain_terms": expected_terms,
        "correct_domain_terms": correct_terms,
    }


def summarize_results(
    records: list[dict[str, Any]],
    model_size: str,
    device: str,
    compute_type: str,
    model_load_ms: float,
) -> dict[str, Any]:
    """전체 및 카테고리별 정확도·지연·실패 지표를 집계한다."""

    successful = [record for record in records if "error" not in record]
    latencies = [float(record["latency_ms"]) for record in successful]
    total_audio_seconds = sum(
        float(record["audio_seconds"]) for record in records
    )

    def accuracy_summary(selected: list[dict[str, Any]]) -> dict[str, float | int]:
        raw_character_count = sum(
            int(record["metrics"]["raw_character_count"]) for record in selected
        )
        raw_word_count = sum(
            int(record["metrics"]["raw_word_count"]) for record in selected
        )
        normalized_character_count = sum(
            int(record["metrics"]["normalized_character_count"])
            for record in selected
        )
        expected_terms = sum(
            int(record["metrics"]["expected_domain_terms"]) for record in selected
        )
        correct_terms = sum(
            int(record["metrics"]["correct_domain_terms"]) for record in selected
        )
        return {
            "raw_cer_percent": round(
                100
                * sum(
                    int(record["metrics"]["raw_character_edits"])
                    for record in selected
                )
                / max(1, raw_character_count),
                2,
            ),
            "raw_wer_percent": round(
                100
                * sum(
                    int(record["metrics"]["raw_word_edits"])
                    for record in selected
                )
                / max(1, raw_word_count),
                2,
            ),
            "normalized_cer_percent": round(
                100
                * sum(
                    int(record["metrics"]["normalized_character_edits"])
                    for record in selected
                )
                / max(1, normalized_character_count),
                2,
            ),
            "sentence_accuracy_percent": round(
                100
                * sum(bool(record["metrics"]["exact_match"]) for record in selected)
                / max(1, len(selected)),
                2,
            ),
            "domain_keyword_accuracy_percent": round(
                100 * correct_terms / max(1, expected_terms), 2
            ),
            "expected_domain_terms": expected_terms,
        }

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in successful:
        categories[str(record["category"])].append(record)

    return {
        "model": model_size,
        "device": device,
        "compute_type": compute_type,
        "beam_size": BEAM_SIZE,
        "language": LANGUAGE,
        "audio_count": len(records),
        "total_audio_seconds": round(total_audio_seconds, 3),
        "model_load_ms": round(model_load_ms, 2),
        "latency_ms": {
            "average": round(sum(latencies) / max(1, len(latencies)), 2),
            "minimum": round(min(latencies), 2) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "maximum": round(max(latencies), 2) if latencies else 0.0,
        },
        "rtf": round(sum(latencies) / max(1.0, total_audio_seconds * 1_000), 4),
        **accuracy_summary(successful),
        "empty_count": sum(
            not str(record.get("hypothesis", "")) for record in successful
        ),
        "error_count": len(records) - len(successful),
        "duplicate_result_count": len(records)
        - len({str(record["id"]) for record in records}),
        "categories": {
            category: {"audio_count": len(selected), **accuracy_summary(selected)}
            for category, selected in sorted(categories.items())
        },
    }


def evaluate(args: argparse.Namespace) -> Path:
    """모델을 한 번 로드하고 manifest의 모든 WAV를 순차 평가한다."""

    from faster_whisper import WhisperModel

    manifest_path = args.manifest.resolve()
    items = load_manifest(manifest_path)
    aliases = load_aliases(args.aliases.resolve())
    device = get_device()
    compute_type = get_compute_type()

    load_started = time.perf_counter()
    model = WhisperModel(args.model_size, device=device, compute_type=compute_type)
    model_load_ms = (time.perf_counter() - load_started) * 1_000
    records: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        audio_path = (manifest_path.parent / str(item["audio_path"])).resolve()
        reference = str(item["text"])
        audio_seconds = wav_duration_seconds(audio_path)
        started_at = time.perf_counter()
        try:
            segments, _ = model.transcribe(
                str(audio_path),
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=False,
                without_timestamps=True,
                initial_prompt=INITIAL_PROMPT,
                hotwords=HOTWORDS,
            )
            hypothesis = " ".join(
                "".join(segment.text for segment in segments).split()
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1_000
            record = {
                "id": item["id"],
                "category": item["category"],
                "audio_path": str(item["audio_path"]),
                "reference": reference,
                "hypothesis": "",
                "audio_seconds": round(audio_seconds, 3),
                "latency_ms": round(latency_ms, 2),
                "error": str(exc),
            }
        else:
            latency_ms = (time.perf_counter() - started_at) * 1_000
            record = {
                "id": item["id"],
                "category": item["category"],
                "audio_path": str(item["audio_path"]),
                "reference": reference,
                "hypothesis": hypothesis,
                "audio_seconds": round(audio_seconds, 3),
                "latency_ms": round(latency_ms, 2),
                "metrics": score_prediction(reference, hypothesis, aliases),
            }
        records.append(record)
        print(
            f"[평가 {index:02d}/{len(items)}] {item['id']} "
            f"{record['latency_ms']:.2f} ms"
        )

    summary = summarize_results(
        records,
        args.model_size,
        device,
        compute_type,
        model_load_ms,
    )
    result = {
        "schema_version": 1,
        "notice": "합성 음성 결과는 모델 비교용이며 실제 Jetson 성능이 아닙니다.",
        "summary": summary,
        "records": records,
    }
    output_path = (
        args.output.resolve()
        if args.output
        else (DEFAULT_RESULTS_ROOT / f"{args.model_size}_result.json").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[완료] 평가 결과: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """평가 실행 명령행 인자를 반환한다."""

    parser = argparse.ArgumentParser(description="한국어 faster-whisper 평가")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--model-size", default=MODEL_SIZE)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """명령행 설정으로 STT 평가를 실행한다."""

    evaluate(parse_args())


if __name__ == "__main__":
    main()
