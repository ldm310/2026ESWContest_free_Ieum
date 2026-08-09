from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bem import BemDictionary
from stft import EPSILON


@dataclass(frozen=True)
class DOAEstimate:
    direction: np.ndarray
    azimuth_deg: float
    elevation_deg: float
    score: float
    dictionary_index: int
    doa_index: int


def score_map(observed_normalized_csm: np.ndarray, dictionary: BemDictionary) -> np.ndarray:
    observed = np.asarray(observed_normalized_csm)
    n_directions, n_frequencies, n_channels = dictionary.steering.shape
    if observed.shape != (n_frequencies, n_channels, n_channels):
        raise ValueError(
            "observed CSM must have shape "
            f"({n_frequencies}, {n_channels}, {n_channels}), got {observed.shape}"
        )

    # <a a^H / ||a a^H||_F, R> = a^H R a / ||a||².  Computing this
    # rank-one form directly avoids materializing [direction, frequency, ch, ch]
    # templates and keeps the grid search memory-bounded.
    steering_power = np.sum(np.abs(dictionary.steering) ** 2, axis=2)
    quadratic_form = np.einsum(
        "dfi,fij,dfj->df", np.conj(dictionary.steering), observed, dictionary.steering, optimize=True
    )
    return np.sum(np.real(quadratic_form) / np.maximum(steering_power, EPSILON), axis=1)


def pick_peaks(
    scores: np.ndarray,
    dictionary: BemDictionary,
    *,
    max_sources: int = 1,
    min_separation_deg: float = 30.0,
    score_threshold: float | None = None,
) -> list[int]:
    scores = np.asarray(scores)
    if scores.shape != (dictionary.directions.shape[0],):
        raise ValueError(f"scores must have shape ({dictionary.directions.shape[0]},), got {scores.shape}")
    if max_sources < 1:
        raise ValueError("max_sources must be positive")
    if not 0.0 <= min_separation_deg <= 180.0:
        raise ValueError("min_separation_deg must be in [0, 180]")

    # Neighbouring grid cells of one source score almost identically, so a plain
    # top-K returns the same source K times.  Reject candidates that fall inside
    # min_separation_deg of an already accepted direction.
    cosine_limit = np.cos(np.deg2rad(min_separation_deg))
    picked: list[int] = []
    for index in np.argsort(scores)[::-1]:
        if len(picked) >= max_sources:
            break
        if score_threshold is not None and scores[index] < score_threshold:
            break
        if picked and np.any(dictionary.directions[picked] @ dictionary.directions[index] > cosine_limit):
            continue
        picked.append(int(index))
    return picked


def estimate_bem_doas(
    observed_normalized_csm: np.ndarray,
    dictionary: BemDictionary,
    *,
    max_sources: int = 1,
    min_separation_deg: float = 30.0,
    score_threshold: float | None = None,
) -> list[DOAEstimate]:
    scores = score_map(observed_normalized_csm, dictionary)
    indices = pick_peaks(
        scores,
        dictionary,
        max_sources=max_sources,
        min_separation_deg=min_separation_deg,
        score_threshold=score_threshold,
    )
    return [_estimate_at(index, scores, dictionary) for index in indices]


def estimate_bem_doa(observed_normalized_csm: np.ndarray, dictionary: BemDictionary) -> DOAEstimate:
    scores = score_map(observed_normalized_csm, dictionary)
    return _estimate_at(int(np.argmax(scores)), scores, dictionary)


def angular_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(estimate, truth), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def _estimate_at(index: int, scores: np.ndarray, dictionary: BemDictionary) -> DOAEstimate:
    return DOAEstimate(
        direction=dictionary.directions[index],
        azimuth_deg=float(dictionary.azimuths_deg[index]),
        elevation_deg=float(dictionary.elevations_deg[index]),
        score=float(scores[index]),
        dictionary_index=index,
        doa_index=int(dictionary.doa_indices[index]),
    )
