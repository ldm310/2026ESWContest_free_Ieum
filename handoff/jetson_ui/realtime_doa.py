#!/usr/bin/env python3
"""4채널 마이크의 DOA·빔포밍 결과를 자막 UI로 전송한다."""

from __future__ import annotations

import argparse
import queue
import socket
import threading
from pathlib import Path

import numpy as np

from beamform import (
    apply_mvdr,
    hybrid_weights,
    loaded_mixture_covariance,
    normalize_steering,
)
from bem import load_bem_dictionary, load_bem_steering
from doa import estimate_bem_doa
from runtime_protocol import (
    AUDIO_UDP_ADDR,
    DIRECTION_UDP_ADDR,
    iter_audio_packets,
    pack_direction,
)
from stft import istft_single, normalized_spatial_covariance, stft_multichannel


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BEM_TABLE = PROJECT_ROOT / "bem_table_reduced.h5"

SAMPLE_RATE = 16_000
N_FFT, HOP = 512, 128
DOA_MIN_HZ, DOA_MAX_HZ = 1_000.0, 5_000.0
MVDR_MIN_HZ = 1_250.0
GRID_STEP_DEG = 5.0
DIAGONAL_LOADING = 1e-2
CHUNK_SEC = 0.5
MAX_PENDING_CHUNKS = 2

DEVICE_NAME = "ReSpeaker"
CHANNELS = 6
DOA_CHANNELS = [1, 2, 3, 4]
PANEL_INDICES = np.array([0, 3, 5, 6], dtype=np.int64)

_latest_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_direction_lock = threading.Lock()


def get_latest_direction() -> np.ndarray:
    """현재 프로세스가 마지막으로 계산한 방향 단위벡터를 반환한다."""

    with _direction_lock:
        return _latest_direction.copy()


def parse_args() -> argparse.Namespace:
    """Jetson 실행 옵션을 파싱한다."""

    parser = argparse.ArgumentParser(description="실시간 BEM DOA + MVDR 전송기")
    parser.add_argument(
        "--bem-table",
        type=Path,
        default=DEFAULT_BEM_TABLE,
        help="BEM 테이블 경로(기본값: project_main/bem_table_reduced.h5)",
    )
    parser.add_argument(
        "--device",
        default=DEVICE_NAME,
        help="입력 장치 번호 또는 장치 이름 일부(기본값: ReSpeaker)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="오디오 장치 목록만 출력하고 종료",
    )
    return parser.parse_args()


def _import_sounddevice():
    """오디오 장치가 필요한 시점에만 sounddevice를 불러온다."""

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice가 설치되어 있지 않습니다. requirements.txt를 설치해 주세요."
        ) from exc
    return sd


def resolve_input_device(sd, requested: str) -> int:
    """장치 번호 또는 이름 일부를 실제 입력 장치 번호로 변환한다."""

    if requested.isdigit():
        index = int(requested)
        info = sd.query_devices(index)
        if info["max_input_channels"] < CHANNELS:
            raise RuntimeError(
                f"장치 {index}의 입력 채널은 {info['max_input_channels']}개입니다. "
                f"최소 {CHANNELS}개가 필요합니다."
            )
        return index

    for index, info in enumerate(sd.query_devices()):
        if (
            requested.lower() in info["name"].lower()
            and info["max_input_channels"] >= CHANNELS
        ):
            return index

    raise RuntimeError(
        f"입력 채널 {CHANNELS}개 이상인 '{requested}' 장치를 찾지 못했습니다. "
        "--list-devices로 장치 목록을 확인해 주세요."
    )


class AudioChunkCollector:
    """sounddevice 입력을 최신 0.5초 chunk 중심으로 유지한다."""

    def __init__(self) -> None:
        # 처리 속도가 입력보다 느려져도 오래된 음성이 최대 4초씩 쌓이지 않게
        # 1초 분량만 대기시킨다. 실시간 자막에서는 과거 입력 보존보다 현재
        # 화자의 음성을 빠르게 전달하는 것이 중요하다.
        self.queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=MAX_PENDING_CHUNKS
        )
        self._pending = np.empty((0, len(DOA_CHANNELS)), dtype=np.float32)
        self._frames_per_chunk = int(SAMPLE_RATE * CHUNK_SEC)

    def callback(self, indata, frames, time_info, status) -> None:
        """4개 실제 마이크 채널만 선택해 추론 큐에 넣는다."""

        del frames, time_info
        if status:
            print(f"[오디오 경고] {status}", flush=True)

        selected = np.asarray(indata[:, DOA_CHANNELS], dtype=np.float32).copy()
        self._pending = np.concatenate((self._pending, selected), axis=0)

        while self._pending.shape[0] >= self._frames_per_chunk:
            chunk = self._pending[: self._frames_per_chunk].copy()
            self._pending = self._pending[self._frames_per_chunk :].copy()
            try:
                self.queue.put_nowait(chunk)
            except queue.Full:
                # 가득 찬 경우 새 chunk를 버리면 화면이 계속 과거 발화를
                # 따라가게 된다. 가장 오래된 하나를 제거하고 최신 입력을 넣는다.
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put_nowait(chunk)
                print(
                    "[오디오 경고] 처리 지연으로 가장 오래된 chunk를 버리고 "
                    "최신 입력을 유지합니다.",
                    flush=True,
                )


def send_mono_audio(
    sock: socket.socket,
    mono: np.ndarray,
    sequence_id: int,
) -> None:
    """빔포밍 mono 오디오를 중복 없는 sequence 기반 패킷으로 전송한다."""

    payload = np.asarray(mono, dtype=np.float32).tobytes()
    for packet in iter_audio_packets(sequence_id, payload):
        sock.sendto(packet, AUDIO_UDP_ADDR)


def run(args: argparse.Namespace) -> None:
    """마이크 입력을 계속 처리해 방향과 mono 오디오를 UI에 전달한다."""

    bem_table = args.bem_table.expanduser().resolve()
    if not bem_table.is_file():
        raise FileNotFoundError(
            f"BEM 테이블을 찾을 수 없습니다: {bem_table}\n"
            "project_main 바로 아래에 bem_table_reduced.h5를 두어야 합니다."
        )

    sd = _import_sounddevice()
    if args.list_devices:
        print(sd.query_devices())
        return

    device = resolve_input_device(sd, str(args.device))
    device_info = sd.query_devices(device)
    print(f"[마이크] {device}: {device_info['name']}", flush=True)
    print(f"[BEM 로딩] {bem_table}", flush=True)

    dictionary = load_bem_dictionary(
        bem_table,
        PANEL_INDICES,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        min_hz=DOA_MIN_HZ,
        max_hz=DOA_MAX_HZ,
        grid_step_deg=GRID_STEP_DEG,
    )
    print("[준비 완료] DOA·MVDR 처리를 시작합니다. Ctrl+C로 종료합니다.", flush=True)

    collector = AudioChunkCollector()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sequence_id = 0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        device=device,
        channels=CHANNELS,
        dtype="float32",
        blocksize=HOP,
        callback=collector.callback,
    )

    try:
        with stream:
            while True:
                audio_chunk = collector.queue.get()
                freqs, spectra = stft_multichannel(
                    audio_chunk,
                    SAMPLE_RATE,
                    n_fft=N_FFT,
                    hop=HOP,
                )
                band = (freqs >= DOA_MIN_HZ) & (freqs <= DOA_MAX_HZ)
                csm = normalized_spatial_covariance(spectra[:, band, :])
                estimate = estimate_bem_doa(csm, dictionary)

                direction = np.asarray(estimate.direction, dtype=np.float64)
                with _direction_lock:
                    global _latest_direction
                    _latest_direction = direction

                _, steering = load_bem_steering(
                    bem_table,
                    PANEL_INDICES,
                    estimate.doa_index,
                    sample_rate=SAMPLE_RATE,
                    n_fft=N_FFT,
                )
                mixture_csm = loaded_mixture_covariance(
                    spectra,
                    diagonal_loading=DIAGONAL_LOADING,
                )
                weights = hybrid_weights(
                    normalize_steering(steering),
                    mixture_csm,
                    mvdr_mask=freqs >= MVDR_MIN_HZ,
                )
                mono = istft_single(
                    apply_mvdr(spectra, weights),
                    SAMPLE_RATE,
                    n_fft=N_FFT,
                    hop=HOP,
                    expected_samples=audio_chunk.shape[0],
                )

                sequence_id += 1
                send_mono_audio(sock, mono, sequence_id)
                sock.sendto(
                    pack_direction(tuple(float(value) for value in direction)),
                    DIRECTION_UDP_ADDR,
                )

                print(
                    f"[처리] az={estimate.azimuth_deg:6.1f}° "
                    f"el={estimate.elevation_deg:6.1f}° "
                    f"mono={mono.shape[0]} samples",
                    flush=True,
                )
    finally:
        sock.close()


def main() -> None:
    """CLI 진입점."""

    args = parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[종료] 실시간 DOA·MVDR 처리를 종료합니다.", flush=True)
    except Exception as exc:
        raise SystemExit(f"[실행 오류] {exc}") from exc
