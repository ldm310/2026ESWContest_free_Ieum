"""준형님 Beamforming NumPy 출력을 연결하는 최소 예제."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stt import STTResult, StreamingSTT


def handle_result(result: STTResult) -> None:
    """partial은 UI 갱신에, final은 확정 기록에 사용하는 예시 callback."""

    if result.type == "partial":
        print("[PARTIAL]", result.text)
    elif result.type == "final":
        print("[FINAL]", result.text)
    else:
        print("[ERROR]", result.error)


def connect_beamforming_output(beamforming_output_stream) -> None:
    """외부 Beamforming mono float32 chunk stream을 StreamingSTT에 연결한다.

    ``beamforming_output_stream``은 이 패키지에 정의되지 않은 실제 연동
    지점이다. 시간 순서의 mono NumPy 배열 iterable을 전달해야 한다.
    """

    streaming_stt = StreamingSTT(on_result=handle_result)
    try:
        streaming_stt.start()
        for audio_chunk in beamforming_output_stream:
            streaming_stt.push_audio(audio_chunk, sample_rate=16_000)
        streaming_stt.flush()
    finally:
        streaming_stt.stop()


# 실제 사용 예:
# connect_beamforming_output(beamforming_output_stream)
