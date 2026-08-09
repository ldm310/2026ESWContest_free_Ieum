from __future__ import annotations

import numpy as np

from stft import EPSILON, spatial_covariance


def loaded_mixture_covariance(spectra: np.ndarray, *, diagonal_loading: float = 1e-2) -> np.ndarray:
    if not 0 < diagonal_loading:
        raise ValueError("diagonal_loading must be positive")
    covariance = spatial_covariance(spectra).astype(np.complex128, copy=False)
    channels = covariance.shape[1]
    identity = np.eye(channels, dtype=np.complex128)
    for frequency_index in range(covariance.shape[0]):
        # Numerical symmetrization plus trace-relative loading makes each CSM
        # positive definite before solve(R, a).
        matrix = (covariance[frequency_index] + covariance[frequency_index].conj().T) / 2
        average_power = max(float(np.real(np.trace(matrix)) / channels), EPSILON)
        covariance[frequency_index] = matrix + diagonal_loading * average_power * identity
    return covariance


def normalize_steering(steering: np.ndarray) -> np.ndarray:
    steering = np.asarray(steering)
    if steering.ndim < 2:
        raise ValueError(f"steering must have shape [..., frequency, channel], got {steering.shape}")
    norm = np.linalg.norm(steering, axis=-1, keepdims=True)
    if np.any(norm < EPSILON):
        raise ValueError("at least one steering vector has zero norm")
    return steering / norm


def das_weights(steering: np.ndarray) -> np.ndarray:
    steering = np.asarray(steering)
    if steering.ndim < 2:
        raise ValueError(f"steering must have shape [..., frequency, channel], got {steering.shape}")
    power = np.sum(np.abs(steering) ** 2, axis=-1, keepdims=True)
    if np.any(power < EPSILON):
        raise ValueError("at least one steering vector has zero norm")
    return (steering / power).astype(np.complex64)


def white_noise_gain_db(weights: np.ndarray, steering: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights)
    steering = np.asarray(steering)
    if weights.shape != steering.shape:
        raise ValueError(f"weights {weights.shape} and steering {steering.shape} must match")
    response = np.abs(np.sum(np.conj(weights) * steering, axis=-1)) ** 2
    noise_power = np.sum(np.abs(weights) ** 2, axis=-1)
    return 10.0 * np.log10(np.maximum(response, EPSILON) / np.maximum(noise_power, EPSILON))


def hybrid_weights(steering: np.ndarray, noise_csm: np.ndarray, *, mvdr_mask: np.ndarray) -> np.ndarray:
    # MVDR loses white noise gain where the array is small relative to the
    # wavelength, and cancels the target where steering error is large.  Fall
    # back to distortionless DAS on those bins instead of leaving them raw.
    steering = np.asarray(steering)
    mvdr_mask = np.asarray(mvdr_mask, dtype=bool)
    if steering.ndim != 2:
        raise ValueError(f"steering must have shape [frequency, channel], got {steering.shape}")
    if mvdr_mask.shape != (steering.shape[0],):
        raise ValueError(f"mvdr_mask must have shape ({steering.shape[0]},), got {mvdr_mask.shape}")
    weights = das_weights(steering)
    if mvdr_mask.any():
        weights[mvdr_mask] = mvdr_weights(steering[mvdr_mask], np.asarray(noise_csm)[mvdr_mask])
    return weights


def mvdr_weights(steering: np.ndarray, noise_csm: np.ndarray) -> np.ndarray:
    steering = np.asarray(steering)
    noise_csm = np.asarray(noise_csm)
    if noise_csm.ndim != 3 or steering.shape != (noise_csm.shape[0], noise_csm.shape[1]):
        raise ValueError("steering/noise CSM shapes are incompatible")
    if noise_csm.shape[1] != noise_csm.shape[2]:
        raise ValueError("noise CSM must be square over channels")

    weights = np.empty_like(steering, dtype=np.complex128)
    for frequency_index, (direction, covariance) in enumerate(zip(steering, noise_csm, strict=True)):
        inverse_times_direction = np.linalg.solve(covariance, direction)
        denominator = np.vdot(direction, inverse_times_direction)
        if abs(denominator) < EPSILON:
            raise np.linalg.LinAlgError(f"MVDR denominator is zero at frequency index {frequency_index}")
        weights[frequency_index] = inverse_times_direction / denominator
    return weights.astype(np.complex64)


def apply_mvdr(spectra: np.ndarray, weights: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra)
    weights = np.asarray(weights)
    if spectra.shape[:2] != (weights.shape[1], weights.shape[0]):
        raise ValueError("spectra and MVDR weight shapes are incompatible")
    return np.einsum("fc,cft->ft", np.conj(weights), spectra, optimize=True)
