"""Zeroth-Korean과 FLEURS에서 일상 한국어 평가용 WAV 100개를 준비한다.

전체 원본을 내려받지 않고 Hugging Face Dataset Viewer API에서 테스트 split의
메타데이터를 조회한 뒤 선택된 오디오만 다운로드한다. 생성 결과는 기존
``evaluate_dataset.py``가 바로 읽을 수 있는 ground-truth manifest 형식이다.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import urllib.parse
import urllib.request
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np


EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EVALUATION_ROOT / "generated" / "open_korean_100"
DATASET_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
TARGET_SAMPLE_RATE = 16_000
PAGE_SIZE = 100
DEFAULT_PER_SOURCE = 50


@dataclass(frozen=True)
class DatasetSource:
    """공개 데이터셋의 조회 및 출처 정보."""

    key: str
    dataset: str
    config: str
    split: str
    text_field: str
    license_name: str
    original_url: str
    distribution_url: str


@dataclass(frozen=True)
class Candidate:
    """일상 문장 점수가 계산된 단일 음성 후보."""

    source: DatasetSource
    row_index: int
    source_id: str
    speaker_id: str | None
    gender: str | int | None
    text: str
    audio_url: str
    topic: str
    score: float


SOURCES = (
    DatasetSource(
        key="zeroth",
        dataset="kresnik/zeroth_korean",
        config="default",
        split="test",
        text_field="text",
        license_name="CC BY 4.0",
        original_url="https://openslr.org/40/",
        distribution_url="https://huggingface.co/datasets/kresnik/zeroth_korean",
    ),
    DatasetSource(
        key="fleurs",
        dataset="google/fleurs",
        config="ko_kr",
        split="test",
        text_field="transcription",
        license_name="CC BY 4.0",
        original_url="https://huggingface.co/datasets/google/fleurs",
        distribution_url="https://huggingface.co/datasets/google/fleurs",
    ),
)

# 두 공개 말뭉치는 자유 대화가 아니라 낭독 음성이다. 그 안에서 프로젝트의
# 일반 한국어 평가에 가까운 문장을 고르기 위해 생활 주제 어휘를 사용한다.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "family_social": (
        "가족",
        "부모",
        "어머니",
        "아버지",
        "아이",
        "친구",
        "사람",
        "만나",
        "결혼",
        "이야기",
    ),
    "daily_routine": (
        "오늘",
        "내일",
        "어제",
        "아침",
        "점심",
        "저녁",
        "주말",
        "생활",
        "시간",
        "잠을",
        "집에서",
        "집에",
    ),
    "food_shopping": (
        "음식",
        "먹",
        "마시",
        "요리",
        "커피",
        "식당",
        "시장",
        "가게",
        "쇼핑",
        "물건",
    ),
    "travel_transport": (
        "여행",
        "버스",
        "지하철",
        "자동차",
        "운전",
        "기차",
        "비행기",
        "공항",
        "호텔",
        "도로",
    ),
    "school_work": (
        "학교",
        "대학교",
        "수업",
        "학생",
        "공부",
        "회사",
        "직장",
        "업무",
        "일을",
        "일하는",
    ),
    "health": (
        "건강",
        "병원",
        "의사",
        "환자",
        "치료",
        "약을",
        "아프",
        "수면",
        "운동",
    ),
    "leisure": (
        "영화",
        "음악",
        "공연",
        "사진",
        "취미",
        "독서",
        "책을",
        "놀이",
        "휴가",
    ),
    "weather_home": (
        "날씨",
        "계절",
        "비가",
        "눈이",
        "더위",
        "추위",
        "방에서",
        "집안",
        "청소",
    ),
}

# 뉴스·역사·정치 문장은 일반 생활 문장보다 낮은 우선순위를 준다. 완전히
# 제거하지 않는 이유는 각 출처에서 50개의 서로 다른 문장을 안정적으로
# 선택하고 현실적인 고유명사·숫자 발음도 일부 남기기 위해서다.
NON_DAILY_KEYWORDS = (
    "대통령",
    "국회",
    "정부",
    "장관",
    "의원",
    "위원회",
    "법원",
    "선거",
    "전쟁",
    "왕국",
    "황제",
    "세기",
    "연구진",
    "공식 발표",
    "주가",
)


def normalize_text(text: str) -> str:
    """문장 비교를 위해 공백과 양끝을 정리한다."""

    return " ".join(text.split())


def classify_and_score(text: str) -> tuple[str, float] | None:
    """문장의 생활 주제를 분류하고 일상 문장 우선순위 점수를 계산한다.

    Args:
        text: 데이터셋 정답 문장.

    Returns:
        생활 어휘가 있으면 ``(topic, score)``, 없으면 ``None``.
    """

    normalized = normalize_text(text)
    if not 14 <= len(normalized) <= 70:
        return None

    topic_hits = {
        topic: sum(keyword in normalized for keyword in keywords)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    topic, maximum_hits = max(topic_hits.items(), key=lambda item: item[1])
    if maximum_hits == 0:
        return None

    total_hits = sum(topic_hits.values())
    penalty = sum(keyword in normalized for keyword in NON_DAILY_KEYWORDS)
    # 약 36자 문장이 평가·청취에 다루기 쉬우므로 지나치게 짧거나 긴 문장에
    # 작은 감점을 준다. 생활 어휘 포함 여부가 길이보다 더 큰 영향을 갖는다.
    score = total_hits * 3.0 + maximum_hits - penalty * 4.0
    score -= abs(len(normalized) - 36) / 30.0
    return topic, score


def extract_audio_url(audio_value: Any) -> str:
    """Dataset Viewer의 Audio 필드에서 다운로드 URL을 반환한다."""

    if isinstance(audio_value, list) and audio_value:
        audio_value = audio_value[0]
    if isinstance(audio_value, dict):
        source = audio_value.get("src")
        if isinstance(source, str) and source:
            return source
    raise ValueError("데이터셋 행에 다운로드 가능한 audio URL이 없습니다.")


def _request_json(url: str, attempts: int = 3) -> dict[str, Any]:
    """일시적인 네트워크 실패를 재시도하며 JSON 객체를 받는다."""

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "embedded-korean-stt-evaluation/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError("JSON 응답이 객체가 아닙니다.")
            return payload
        except Exception as exc:  # 네트워크 오류의 원문을 마지막에 보존한다.
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"데이터셋 메타데이터 요청 실패: {last_error}") from last_error


def fetch_source_rows(source: DatasetSource) -> list[dict[str, Any]]:
    """한 출처의 지정 split 전체 행을 Dataset Viewer API에서 읽는다."""

    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": source.dataset,
                "config": source.config,
                "split": source.split,
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        payload = _request_json(f"{DATASET_ROWS_ENDPOINT}?{query}")
        page = payload.get("rows", [])
        if not isinstance(page, list):
            raise ValueError(f"{source.key} rows 응답이 리스트가 아닙니다.")
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += len(page)
    return rows


def build_candidates(
    source: DatasetSource,
    entries: Iterable[dict[str, Any]],
) -> list[Candidate]:
    """원본 행에서 생활 어휘가 있는 유효 후보를 만든다."""

    candidates: list[Candidate] = []
    for fallback_index, entry in enumerate(entries):
        row = entry.get("row", {})
        if not isinstance(row, dict):
            continue
        text = normalize_text(str(row.get(source.text_field, "")))
        classification = classify_and_score(text)
        if classification is None:
            continue
        try:
            audio_url = extract_audio_url(row.get("audio"))
        except ValueError:
            continue
        topic, score = classification
        row_index = int(entry.get("row_idx", fallback_index))
        source_id = str(row.get("id", f"row-{row_index}"))
        speaker_value = row.get("speaker_id")
        candidates.append(
            Candidate(
                source=source,
                row_index=row_index,
                source_id=source_id,
                speaker_id=(
                    str(speaker_value) if speaker_value is not None else None
                ),
                gender=row.get("gender"),
                text=text,
                audio_url=audio_url,
                topic=topic,
                score=score,
            )
        )
    return candidates


def select_candidates(candidates: list[Candidate], count: int) -> list[Candidate]:
    """주제와 화자를 분산하면서 점수가 높은 서로 다른 문장을 선택한다."""

    if count <= 0:
        raise ValueError("선택 개수는 0보다 커야 합니다.")

    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in sorted(
        candidates,
        key=lambda item: (-item.score, item.row_index),
    ):
        grouped[candidate.topic].append(candidate)

    selected: list[Candidate] = []
    used_texts: set[str] = set()
    speaker_counts: Counter[str] = Counter()
    # Zeroth test에는 10명의 화자가 있으므로 한 화자에 과도하게 치우치지 않게
    # 기본적으로 목표치의 약 1/8까지만 허용한다. 후보가 부족하면 아래 fallback
    # 단계에서 제한을 완화해 정확한 개수를 채운다.
    speaker_limit = max(2, math.ceil(count / 8))
    topics = list(TOPIC_KEYWORDS)

    while len(selected) < count:
        progressed = False
        for topic in topics:
            group = grouped[topic]
            while group:
                candidate = group.pop(0)
                normalized = normalize_text(candidate.text).casefold()
                if normalized in used_texts:
                    continue
                if (
                    candidate.speaker_id is not None
                    and speaker_counts[candidate.speaker_id] >= speaker_limit
                ):
                    continue
                selected.append(candidate)
                used_texts.add(normalized)
                if candidate.speaker_id is not None:
                    speaker_counts[candidate.speaker_id] += 1
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break

    if len(selected) < count:
        for candidate in sorted(
            candidates,
            key=lambda item: (-item.score, item.row_index),
        ):
            normalized = normalize_text(candidate.text).casefold()
            if normalized in used_texts:
                continue
            selected.append(candidate)
            used_texts.add(normalized)
            if len(selected) >= count:
                break

    if len(selected) != count:
        raise RuntimeError(
            f"서로 다른 일상 문장 {count}개를 선택할 수 없습니다: "
            f"선택 {len(selected)}개, 후보 {len(candidates)}개"
        )
    return selected


def _download_bytes(url: str, attempts: int = 3) -> bytes:
    """선택된 오디오 한 건을 재시도하며 다운로드한다."""

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "embedded-korean-stt-evaluation/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if not data:
                raise ValueError("빈 오디오 응답")
            return data
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"오디오 다운로드 실패: {last_error}") from last_error


def decode_pcm16_mono(audio_data: bytes) -> np.ndarray:
    """PyAV로 임의 입력 오디오를 16kHz mono PCM16 배열로 변환한다."""

    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(audio_data)) as container:
        if not container.streams.audio:
            raise ValueError("다운로드 파일에 오디오 stream이 없습니다.")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=TARGET_SAMPLE_RATE,
        )
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1).astype("<i2"))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1).astype("<i2"))
    if not chunks:
        raise ValueError("오디오를 PCM sample로 디코딩하지 못했습니다.")
    return np.concatenate(chunks).astype("<i2", copy=False)


def write_pcm_wav(path: Path, samples: np.ndarray) -> None:
    """16kHz mono signed PCM16 WAV를 작성한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".wav.tmp")
    with wave.open(str(temporary_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TARGET_SAMPLE_RATE)
        wav_file.writeframes(samples.astype("<i2", copy=False).tobytes())
    temporary_path.replace(path)


def prepare_dataset(output_dir: Path, per_source: int) -> Path:
    """두 출처에서 동일 개수의 WAV를 받고 정답 manifest를 저장한다."""

    output_dir = output_dir.resolve()
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for source in SOURCES:
        print(f"[조회] {source.key}: {source.dataset}/{source.split}", flush=True)
        entries = fetch_source_rows(source)
        candidates = build_candidates(source, entries)
        selected = select_candidates(candidates, per_source)
        print(
            f"[선별] {source.key}: 전체 {len(entries)}개 중 "
            f"일상 후보 {len(candidates)}개, 최종 {len(selected)}개",
            flush=True,
        )

        for source_index, candidate in enumerate(selected, start=1):
            identifier = f"{source.key}_{source_index:03d}"
            audio_path = audio_dir / f"{identifier}.wav"
            if not audio_path.is_file():
                audio_data = _download_bytes(candidate.audio_url)
                samples = decode_pcm16_mono(audio_data)
                write_pcm_wav(audio_path, samples)
            else:
                with wave.open(str(audio_path), "rb") as wav_file:
                    samples = np.frombuffer(
                        wav_file.readframes(wav_file.getnframes()),
                        dtype="<i2",
                    )

            duration_seconds = samples.size / TARGET_SAMPLE_RATE
            checksum = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            records.append(
                {
                    "id": identifier,
                    "category": f"daily_{candidate.topic}",
                    "source": source.key,
                    "source_dataset": source.dataset,
                    "source_split": source.split,
                    "source_row_index": candidate.row_index,
                    "source_record_id": candidate.source_id,
                    "speaker_id": candidate.speaker_id,
                    "gender": candidate.gender,
                    "audio_path": audio_path.relative_to(output_dir).as_posix(),
                    "text": candidate.text,
                    "audio_seconds": round(duration_seconds, 3),
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "license": source.license_name,
                    "sha256": checksum,
                }
            )
            print(
                f"[준비 {len(records):03d}/{len(SOURCES) * per_source:03d}] "
                f"{identifier}.wav ({duration_seconds:.2f}초)",
                flush=True,
            )

    manifest = {
        "schema_version": 1,
        "name": "open_korean_daily_100",
        "purpose": (
            "공개된 실제 한국어 낭독 음성으로 현재 STT의 CER, WER와 "
            "latency를 재현 가능하게 평가"
        ),
        "notice": (
            "일상 어휘를 기준으로 선별했지만 Zeroth-Korean과 FLEURS는 "
            "자유 대화가 아닌 낭독 음성이다. ReSpeaker 현장 성능을 대신하지 않는다."
        ),
        "selection": {
            "count": len(records),
            "per_source": per_source,
            "split": "test",
            "sample_rate": TARGET_SAMPLE_RATE,
            "strategy": "생활 주제 어휘 점수, 문장 중복 제거, Zeroth 화자 분산",
        },
        "sources": [
            {
                "key": source.key,
                "dataset": source.dataset,
                "license": source.license_name,
                "original_url": source.original_url,
                "distribution_url": source.distribution_url,
            }
            for source in SOURCES
        ],
        "items": records,
    }
    manifest_path = output_dir / "ground_truth.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[완료] 정답 manifest: {manifest_path}", flush=True)
    return manifest_path


def parse_args() -> argparse.Namespace:
    """명령행 옵션을 파싱한다."""

    parser = argparse.ArgumentParser(
        description="Zeroth-Korean 50개와 FLEURS Korean 50개 평가셋 준비"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-source", type=int, default=DEFAULT_PER_SOURCE)
    return parser.parse_args()


def main() -> None:
    """CLI 진입점."""

    args = parse_args()
    if args.per_source <= 0:
        raise SystemExit("--per-source는 0보다 커야 합니다.")
    prepare_dataset(args.output_dir, args.per_source)


if __name__ == "__main__":
    main()
