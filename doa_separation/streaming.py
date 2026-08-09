from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import get_window

from beamform import das_weights, hybrid_weights, normalize_steering
from bem import BemDictionary, load_bem_steering
from doa import DOAEstimate, pick_peaks, score_map


@dataclass(frozen=True)
class StreamingConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop: int = 128
    doa_min_hz: float = 1000.0
    doa_max_hz: float = 5000.0
    mvdr_min_hz: float = 1250.0
    csm_forget: float = 0.95
    doa_update_frames: int = 16
    warmup_frames: int = 32
    weight_smooth: float = 0.25
    diagonal_loading: float = 1e-2


class StreamingFrontend:
    def __init__(self, table_path: str | Path, dictionary: BemDictionary,
                 config: StreamingConfig | None = None) -> None:
        self.table_path = Path(table_path)
        self.dictionary = dictionary
        self.config = config or StreamingConfig()

        n_fft, hop = self.config.n_fft, self.config.hop
        self.window = get_window("hann", n_fft).astype(np.float32)
        self.frequencies = np.fft.rfftfreq(n_fft, d=1.0 / self.config.sample_rate)
        self.doa_band = (self.frequencies >= self.config.doa_min_hz) & (self.frequencies <= self.config.doa_max_hz)
        self.mvdr_mask = self.frequencies >= self.config.mvdr_min_hz
        if not np.allclose(self.frequencies[self.doa_band], dictionary.frequencies_hz):
            raise ValueError("dictionary band does not match the streaming DOA band")

        channels = len(dictionary.panel_indices)
        n_bin = len(self.frequencies)
        self._input = np.zeros((0, channels), dtype=np.float32)
        self._tail = np.zeros(n_fft - hop, dtype=np.float32)
        self._norm_tail = np.zeros(n_fft - hop, dtype=np.float32)
        self._csm = np.zeros((n_bin, channels, channels), dtype=np.complex128)
        self._weights = np.zeros((n_bin, channels), dtype=np.complex64)
        self._frames_seen = 0
        self._frames_since_doa = 0
        self.estimate: DOAEstimate | None = None

    def push(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] != self._input.shape[1]:
            raise ValueError(f"expected [samples, {self._input.shape[1]}] audio, got {audio.shape}")
        self._input = np.concatenate([self._input, audio], axis=0)

        n_fft, hop = self.config.n_fft, self.config.hop
        out = []
        while self._input.shape[0] >= n_fft:
            frame = self._input[:n_fft] * self.window[:, None]
            self._input = self._input[hop:]
            spectrum = np.fft.rfft(frame, axis=0)
            out.append(self._process_frame(spectrum))
        if not out:
            return np.zeros(0, dtype=np.float32)
        return self._overlap_add(np.stack(out, axis=1))

    def _process_frame(self, spectrum: np.ndarray) -> np.ndarray:
        alpha = self.config.csm_forget
        outer = spectrum[:, :, None] * np.conj(spectrum)[:, None, :]
        self._csm = alpha * self._csm + (1.0 - alpha) * outer
        self._frames_seen += 1
        self._frames_since_doa += 1

        ready = self._frames_seen >= self.config.warmup_frames
        if ready and self._frames_since_doa >= self.config.doa_update_frames:
            self._frames_since_doa = 0
            self._update_weights()
        if not ready:
            return np.zeros(spectrum.shape[0], dtype=np.complex64)
        return np.einsum("fc,fc->f", np.conj(self._weights), spectrum).astype(np.complex64)

    def _update_weights(self) -> None:
        band = self._csm[self.doa_band]
        norm = np.linalg.norm(band, axis=(1, 2), keepdims=True)
        scores = score_map(band / np.maximum(norm, 1e-12), self.dictionary)
        index = pick_peaks(scores, self.dictionary, max_sources=1)[0]
        doa_index = int(self.dictionary.doa_indices[index])

        if self.estimate is None or self.estimate.doa_index != doa_index:
            _, steering = load_bem_steering(
                self.table_path, self.dictionary.panel_indices, doa_index,
                sample_rate=self.config.sample_rate, n_fft=self.config.n_fft)
            self._steering = normalize_steering(steering)
        self.estimate = DOAEstimate(
            direction=self.dictionary.directions[index],
            azimuth_deg=float(self.dictionary.azimuths_deg[index]),
            elevation_deg=float(self.dictionary.elevations_deg[index]),
            score=float(scores[index]),
            dictionary_index=index,
            doa_index=doa_index,
        )

        loaded = self._loaded_csm()
        target = hybrid_weights(self._steering, loaded, mvdr_mask=self.mvdr_mask)
        if not self._weights.any():
            self._weights = target
        else:
            beta = self.config.weight_smooth
            self._weights = ((1.0 - beta) * self._weights + beta * target).astype(np.complex64)

    def _loaded_csm(self) -> np.ndarray:
        covariance = self._csm.copy()
        channels = covariance.shape[1]
        identity = np.eye(channels, dtype=np.complex128)
        matrix = (covariance + np.conj(covariance.transpose(0, 2, 1))) / 2
        power = np.maximum(np.real(np.trace(matrix, axis1=1, axis2=2)) / channels, 1e-12)
        return matrix + self.config.diagonal_loading * power[:, None, None] * identity

    def _overlap_add(self, spectra: np.ndarray) -> np.ndarray:
        n_fft, hop = self.config.n_fft, self.config.hop
        frames = np.fft.irfft(spectra, n=n_fft, axis=0).astype(np.float32) * self.window[:, None]
        n_frame = frames.shape[1]
        length = (n_frame - 1) * hop + n_fft
        signal = np.zeros(length, dtype=np.float32)
        weight = np.zeros(length, dtype=np.float32)
        square = self.window ** 2
        for index in range(n_frame):
            start = index * hop
            signal[start:start + n_fft] += frames[:, index]
            weight[start:start + n_fft] += square
        signal[: len(self._tail)] += self._tail
        weight[: len(self._norm_tail)] += self._norm_tail
        emit = n_frame * hop
        self._tail = signal[emit:].copy()
        self._norm_tail = weight[emit:].copy()
        return signal[:emit] / np.maximum(weight[:emit], 1e-8)
