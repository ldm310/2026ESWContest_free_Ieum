#!/usr/bin/env python3
from __future__ import annotations

import socket
import struct
import threading
import queue

import numpy as np
import sounddevice as sd

# ── 준형님 원본 모듈 (같은 폴더에 두고 실행) ──
from stft import stft_multichannel, normalized_spatial_covariance, istft_single
from bem import load_bem_dictionary, load_bem_steering
from doa import estimate_bem_doa
from beamform import (
    hybrid_weights, apply_mvdr, normalize_steering, loaded_mixture_covariance,
)

# ── 설정 (준형님 pipeline.py 상수와 동일하게 맞춤) ──
SAMPLE_RATE = 16000
N_FFT, HOP = 512, 128
DOA_MIN_HZ, DOA_MAX_HZ = 1000.0, 5000.0
MVDR_MIN_HZ = 1250.0
GRID_STEP_DEG = 5.0
DIAGONAL_LOADING = 1e-2
CHUNK_SEC = 0.5                      # 몇 초 조각마다 방향+빔포밍을 갱신할지

# 병현에게 mono 오디오를 보낼 UDP 주소 (같은 노트북 = localhost)
AUDIO_UDP_ADDR = ("127.0.0.1", 50007)

# TODO(1): 마이크 장치 / 채널
#   `py -3.11 -c "import sounddevice as sd; print(sd.query_devices())"` 로 index 확인.
#   4채널 입력 장치여야 하고, 채널 물리 순서가 panel_indices 순서와 같아야 한다.
DEVICE = None                        # 예: 3  (None이면 기본 입력장치)
CHANNELS = 4

# TODO(2): BEM 테이블 경로 / 패널 인덱스 (상자에 붙인 마이크 4개의 채널 순서와 일치!)
BEM_TABLE = "sample_data/bem_table_reduced.h5"
PANEL_INDICES = np.array([0, 1, 2, 3], dtype=np.int64)

# ── 병현이 읽어가는 공유 방향값 ──
_latest_direction = np.array([1.0, 0.0, 0.0])
_dir_lock = threading.Lock()

def get_latest_direction() -> np.ndarray:
    """최신 방향 단위벡터 [x,y,z]. = 준형님 estimate.direction."""
    with _dir_lock:
        return _latest_direction.copy()


# ── 마이크 캡처 → 큐 (0.5초 조각) ──
_audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
_buffer: list[np.ndarray] = []
_frames_per_chunk = int(SAMPLE_RATE * CHUNK_SEC)

def _audio_callback(indata, frames, time_info, status):
    if status:
        print("[audio]", status, flush=True)
    _buffer.append(indata.copy())
    if sum(len(b) for b in _buffer) >= _frames_per_chunk:
        chunk = np.concatenate(_buffer, axis=0)[:_frames_per_chunk]
        _buffer.clear()
        _audio_q.put(chunk)             # [samples, 4] float32


def _send_mono_udp(sock: socket.socket, mono: np.ndarray) -> None:
    """빔포밍된 mono 오디오를 UDP로 전송. 헤더(길이) + float32 payload.
    큰 조각은 여러 패킷으로 쪼갬(UDP 안전 크기)."""
    data = mono.astype(np.float32).tobytes()
    MAX = 1400 * 4                      # 대략 안전한 payload 크기
    offset = 0
    total = len(data)
    while offset < total:
        part = data[offset:offset + MAX]
        sock.sendto(struct.pack("<II", total, offset) + part, AUDIO_UDP_ADDR)
        offset += len(part)


def main() -> None:
    print("BEM 사전 로딩...", flush=True)
    dictionary = load_bem_dictionary(
        BEM_TABLE, PANEL_INDICES,
        sample_rate=SAMPLE_RATE, n_fft=N_FFT,
        min_hz=DOA_MIN_HZ, max_hz=DOA_MAX_HZ, grid_step_deg=GRID_STEP_DEG,
    )
    print("완료. 마이크 스트림 시작. (Ctrl+C 종료)", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, device=DEVICE, channels=CHANNELS,
        dtype="float32", blocksize=HOP, callback=_audio_callback,
    )
    with stream:
        while True:
            audio_chunk = _audio_q.get()     # [samples, 4]

            # ── (A) DOA: 준형님 함수 그대로 ──
            freqs, spectra = stft_multichannel(audio_chunk, SAMPLE_RATE, n_fft=N_FFT, hop=HOP)
            band = (freqs >= DOA_MIN_HZ) & (freqs <= DOA_MAX_HZ)
            csm = normalized_spatial_covariance(spectra[:, band, :])
            est = estimate_bem_doa(csm, dictionary)

            with _dir_lock:
                global _latest_direction
                _latest_direction = np.asarray(est.direction, dtype=np.float64)

            # ── (B) Beamforming: 준형님 함수 그대로 (pipeline.separate 와 동일) ──
            _, steering = load_bem_steering(
                BEM_TABLE, PANEL_INDICES, est.doa_index,
                sample_rate=SAMPLE_RATE, n_fft=N_FFT)
            mixture_csm = loaded_mixture_covariance(spectra, diagonal_loading=DIAGONAL_LOADING)
            mvdr_mask = freqs >= MVDR_MIN_HZ
            weights = hybrid_weights(normalize_steering(steering), mixture_csm, mvdr_mask=mvdr_mask)
            mono = istft_single(apply_mvdr(spectra, weights), SAMPLE_RATE,
                                n_fft=N_FFT, hop=HOP, expected_samples=audio_chunk.shape[0])

            # ── (C) 병현에게 전달: 방향은 공유변수, mono 오디오는 UDP ──
            _send_mono_udp(sock, mono)

            print(f"dir=[{est.direction[0]:+.2f},{est.direction[1]:+.2f},{est.direction[2]:+.2f}] "
                  f"az={est.azimuth_deg:6.1f} el={est.elevation_deg:6.1f}  "
                  f"mono {mono.shape[0]} samples 전송", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료.", flush=True)
