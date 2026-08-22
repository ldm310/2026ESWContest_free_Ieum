from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft


EPSILON = 1e-12


def stft_multichannel(audio: np.ndarray, sample_rate: int, n_fft: int = 512, hop: int = 128) -> tuple[np.ndarray, np.ndarray]:
    audio = np.asarray(audio)
    if audio.ndim != 2:
        raise ValueError(f"audio must have shape [samples, channels], got {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError("audio contains non-finite values")
    if not (0 < hop <= n_fft):
        raise ValueError("hop must be in [1, n_fft]")
    frequencies_hz, _, spectra = stft(
        audio.T,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    return np.asarray(frequencies_hz), np.asarray(spectra)


def istft_single(
    spectrum: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 512,
    hop: int = 128,
    expected_samples: int | None = None,
) -> np.ndarray:
    _, waveform = istft(
        spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
    )
    if expected_samples is not None:
        waveform = waveform[:expected_samples]
    return np.asarray(waveform, dtype=np.float32)


def spatial_covariance(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must have shape [channel, frequency, frame], got {spectra.shape}")
    if spectra.shape[-1] == 0:
        raise ValueError("cannot estimate a covariance from zero STFT frames")
    return np.einsum("cft,dft->fcd", spectra, np.conj(spectra), optimize=True) / spectra.shape[-1]


def normalized_spatial_covariance(spectra: np.ndarray) -> np.ndarray:
    covariance = spatial_covariance(spectra)
    norm = np.linalg.norm(covariance, axis=(1, 2), keepdims=True)
    return covariance / np.maximum(norm, EPSILON)
