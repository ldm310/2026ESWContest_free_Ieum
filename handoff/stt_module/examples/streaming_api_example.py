"""PCM WAV 파일을 NumPy chunk로 나눠 StreamingSTT에 전달하는 실행 예제."""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stt import STTResult, StreamingSTT  # noqa: E402


CHUNK_DURATION_SECONDS = 0.1


def handle_result(result: STTResult) -> None:
    """실행 예제의 STT 결과를 터미널에 출력한다."""

    if result.type == "partial":
        print(f"[PARTIAL #{result.utterance_id}] {result.text}")
    elif result.type == "final":
        print(f"[FINAL #{result.utterance_id}] {result.text}")
    else:
        print(f"[ERROR #{result.utterance_id}] {result.error}")


def read_wav_chunks(wav_path: Path) -> tuple[int, list[np.ndarray]]:
    """PCM WAV를 mono float32 100ms chunk 목록으로 읽는다.

    Args:
        wav_path: 읽을 PCM WAV 파일.

    Returns:
        WAV 샘플레이트와 시간 순서의 mono float32 chunk 목록.

    Raises:
        ValueError: 압축 WAV, 16-bit가 아닌 WAV 또는 다채널 WAV인 경우.
    """

    with wave.open(str(wav_path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError("압축되지 않은 PCM WAV만 지원합니다.")
        if wav_file.getsampwidth() != 2:
            raise ValueError("이 예제는 signed 16-bit PCM WAV를 요구합니다.")
        if wav_file.getnchannels() != 1:
            raise ValueError("이 예제는 mono WAV를 권장하고 요구합니다.")

        sample_rate = wav_file.getframerate()
        chunk_frames = max(1, round(sample_rate * CHUNK_DURATION_SECONDS))
        chunks: list[np.ndarray] = []
        while True:
            pcm_bytes = wav_file.readframes(chunk_frames)
            if not pcm_bytes:
                break
            chunk = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
            chunks.append(chunk / 32_768.0)
    return sample_rate, chunks


def main() -> None:
    """명령행 WAV 파일을 실제 시간 간격으로 StreamingSTT에 전달한다."""

    parser = argparse.ArgumentParser(description="WAV 기반 StreamingSTT 예제")
    parser.add_argument("wav_path", type=Path, help="처리할 mono PCM16 WAV 경로")
    args = parser.parse_args()

    sample_rate, chunks = read_wav_chunks(args.wav_path)
    streaming_stt = StreamingSTT(on_result=handle_result)
    try:
        streaming_stt.start()
        for chunk in chunks:
            streaming_stt.push_audio(chunk, sample_rate=sample_rate)
            time.sleep(chunk.size / sample_rate)
        streaming_stt.flush()
    finally:
        streaming_stt.stop()

    print(streaming_stt.get_stats())


if __name__ == "__main__":
    main()
