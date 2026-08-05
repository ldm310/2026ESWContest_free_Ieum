#!/usr/bin/env python3
"""
realtime_doa.py  -  실시간 DOA 방향 출력 래퍼 (준형님용 stub)

목적:
  준형님의 기존 함수를 그대로 써서, 4채널 마이크 입력을 실시간으로 받아
  '방향(direction 단위벡터) + 방위각/고도각'을 매 순간 출력한다.
  pipeline.py 는 WAV 파일 통째로 처리하지만, 이건 마이크 스트림을 조각내서
  같은 함수들(estimate_bem_doa 등)을 반복 호출할 뿐이다.

준형님이 채울 부분은 3곳(아래 TODO)뿐:
  1) 마이크 장치 index / 채널 순서(panel 매핑)
  2) BEM 테이블 경로와 panel_indices
  3) 결과를 어디로 보낼지 (지금은 print + 공유변수. 병현 연동은 맨 아래 설명)

병현 연동:
  이 파일의 latest_direction (전역, [x,y,z] 단위벡터)이 곧 estimate.direction 이다.
  병현 stage3_direction_vector.py 에서
        q = query_to_unit(...)   →   q = get_latest_direction()
  로 바꾸면 그대로 이어진다. (같은 프로세스로 합치거나, 소켓/큐로 넘겨도 됨)

설치:  py -3.11 -m pip install sounddevice numpy scipy h5py soundfile
실행:  py -3.11 realtime_doa.py
"""
from __future__ import annotations

import threading
import queue

import numpy as np
import sounddevice as sd

# 준형님 기존 모듈 (같은 폴더에 두고 실행)
from stft import stft_multichannel, normalized_spatial_covariance
from bem import load_bem_dictionary
from doa import estimate_bem_doa

# ── 설정 (pipeline.py 와 동일하게 맞춤) ──────────────────────────
SAMPLE_RATE = 16000
N_FFT, HOP = 512, 128
DOA_MIN_HZ, DOA_MAX_HZ = 1000.0, 5000.0
GRID_STEP_DEG = 5.0
CHUNK_SEC = 0.5                      # 몇 초마다 방향을 새로 추정할지 (0.3~1.0 권장)

# TODO(1): 마이크 장치와 채널
#   - `python -c "import sounddevice as sd; print(sd.query_devices())"` 로 장치 index 확인
#   - 4채널 인터페이스여야 함. 채널 물리 순서가 panel_indices 순서와 같아야 한다.
DEVICE = None                        # 예: 3  (None이면 기본 입력장치)
CHANNELS = 4

# TODO(2): BEM 테이블 경로 / 패널 인덱스 (물리 마이크 ↔ BEM 인덱스 매핑)
BEM_TABLE = "sample_data/bem_table_reduced.h5"
PANEL_INDICES = np.array([0, 1, 2, 3], dtype=np.int64)   # 실물 배선과 반드시 일치시킬 것

# ── 공유 상태 (병현 쪽에서 읽어감) ──
latest_direction = np.array([1.0, 0.0, 0.0])   # [x,y,z] 단위벡터, 기본=정면
latest_lock = threading.Lock()

def get_latest_direction() -> np.ndarray:
    with latest_lock:
        return latest_direction.copy()


# ── 마이크 캡처 → 큐 ──
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
_buffer = []
_frames_per_chunk = int(SAMPLE_RATE * CHUNK_SEC)

def _audio_callback(indata, frames, time_info, status):
    if status:
        print("[audio]", status, flush=True)
    _buffer.append(indata.copy())
    total = sum(len(b) for b in _buffer)
    if total >= _frames_per_chunk:
        chunk = np.concatenate(_buffer, axis=0)[:_frames_per_chunk]
        _buffer.clear()
        audio_q.put(chunk)          # [samples, 4] float32


def main():
    # 사전(dictionary)은 시작할 때 한 번만 로드 (pipeline.py 와 동일 인자)
    print("BEM 사전 로딩...", flush=True)
    dictionary = load_bem_dictionary(
        BEM_TABLE, PANEL_INDICES,
        sample_rate=SAMPLE_RATE, n_fft=N_FFT,
        min_hz=DOA_MIN_HZ, max_hz=DOA_MAX_HZ, grid_step_deg=GRID_STEP_DEG,
    )
    print("완료. 마이크 스트림 시작. (Ctrl+C 종료)", flush=True)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, device=DEVICE, channels=CHANNELS,
        dtype="float32", blocksize=HOP, callback=_audio_callback,
    )
    with stream:
        while True:
            audio_chunk = audio_q.get()          # [samples, 4]
            # ── pipeline.py 의 DOA 부분과 동일 ──
            freqs, spectra = stft_multichannel(audio_chunk, SAMPLE_RATE, n_fft=N_FFT, hop=HOP)
            band = (freqs >= DOA_MIN_HZ) & (freqs <= DOA_MAX_HZ)
            csm = normalized_spatial_covariance(spectra[:, band, :])
            est = estimate_bem_doa(csm, dictionary)

            with latest_lock:
                global latest_direction
                latest_direction = np.asarray(est.direction, dtype=np.float64)

            # TODO(3): 출력. 지금은 콘솔. 병현 연동 시 get_latest_direction() 로 읽거나
            #          소켓/파이프/멀티프로세싱 큐로 est.direction 을 넘긴다.
            print(f"dir=[{est.direction[0]:+.2f},{est.direction[1]:+.2f},{est.direction[2]:+.2f}] "
                  f"az={est.azimuth_deg:6.1f}  el={est.elevation_deg:6.1f}  score={est.score:.3f}",
                  flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료.", flush=True)
