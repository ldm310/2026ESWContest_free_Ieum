
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class STTResult:

    type: Literal["partial", "final", "error"]
    text: str
    latency_ms: float
    timestamp: str
    sequence_id: int
    utterance_id: int
    is_final: bool
    error: str | None = None
