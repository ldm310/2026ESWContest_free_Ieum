"""Streaming STT 공개 결과 타입."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class STTResult:
    """StreamingSTT가 사용자 callback에 전달하는 불변 결과.

    Attributes:
        type: partial, final 또는 error 결과 종류.
        text: 인식된 한국어 문자열. 오류이면 빈 문자열.
        latency_ms: 해당 Whisper 추론 시간(ms).
        timestamp: timezone을 포함한 ISO 8601 생성 시각.
        sequence_id: callback 결과가 생성될 때마다 증가하는 번호.
        utterance_id: 발화 시작 시마다 증가하는 발화 번호.
        is_final: final 결과 또는 final 추론 오류인지 여부.
        error: 오류 결과의 원인. 정상 결과에서는 None.
    """

    type: Literal["partial", "final", "error"]
    text: str
    latency_ms: float
    timestamp: str
    sequence_id: int
    utterance_id: int
    is_final: bool
    error: str | None = None
