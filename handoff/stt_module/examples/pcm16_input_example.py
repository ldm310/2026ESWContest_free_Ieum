"""WAV raw PCM bytes를 StreamingSTT.push_pcm16()에 전달하는 예제."""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stt import STTResult, StreamingSTT  # noqa: E402


def handle_result(result: STTResult) -> None:
    """PCM16 예제 결과를 출력한다."""

    if result.type == "error":
        print("[ERROR]", result.error)
    else:
        print(f"[{result.type.upper()}]", result.text)


def main() -> None:
    """PCM WAV frame을 raw bytes 상태로 StreamingSTT에 전달한다."""

    parser = argparse.ArgumentParser(description="PCM16 bytes StreamingSTT 예제")
    parser.add_argument("wav_path", type=Path, help="처리할 PCM16 WAV 경로")
    args = parser.parse_args()

    streaming_stt = StreamingSTT(on_result=handle_result)
    try:
        streaming_stt.start()
        with wave.open(str(args.wav_path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
                raise ValueError("압축되지 않은 signed 16-bit PCM WAV가 필요합니다.")
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            chunk_frames = max(1, round(sample_rate * 0.1))

            while True:
                pcm_bytes = wav_file.readframes(chunk_frames)
                if not pcm_bytes:
                    break
                # 실제 연결에서는 이 지점에 준형님 모듈의 PCM16 buffer를 넣는다.
                streaming_stt.push_pcm16(
                    pcm_bytes,
                    sample_rate=sample_rate,
                    channels=channels,
                )
                time.sleep(len(pcm_bytes) / (2 * channels * sample_rate))
        streaming_stt.flush()
    finally:
        streaming_stt.stop()


if __name__ == "__main__":
    main()
