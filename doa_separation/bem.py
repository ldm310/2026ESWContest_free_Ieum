from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class BemDictionary:
    steering: np.ndarray  # [direction, frequency, channel], STFT convention
    directions: np.ndarray  # [direction, xyz]
    azimuths_deg: np.ndarray
    elevations_deg: np.ndarray
    doa_indices: np.ndarray  # indices in the full HDF5 doa_grid
    frequencies_hz: np.ndarray
    panel_indices: np.ndarray  # WAV-channel order


def directions_to_angles(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = np.asarray(directions, dtype=np.float64)
    azimuths_deg = np.rad2deg(np.arctan2(directions[:, 1], directions[:, 0])) % 360.0
    elevations_deg = np.rad2deg(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
    return azimuths_deg, elevations_deg


def load_bem_dictionary(
    table_path: str | Path,
    panel_indices: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    min_hz: float = 1_000.0,
    max_hz: float = 5_000.0,
    grid_step_deg: float = 5.0,
) -> BemDictionary:
    panel_indices = _validate_panel_indices(panel_indices)
    requested_frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    frequency_mask = (requested_frequencies >= min_hz) & (requested_frequencies <= max_hz)
    if not np.any(frequency_mask):
        raise ValueError("the requested frequency band contains no FFT bins")
    requested_frequencies = requested_frequencies[frequency_mask]

    with h5py.File(Path(table_path), "r") as table:
        _validate_table_format(table, sample_rate, n_fft)
        table_frequencies = np.asarray(table["freqs"], dtype=np.float64)
        frequency_indices = _matching_frequency_indices(table_frequencies, requested_frequencies)
        doa_indices = _grid_indices(table, grid_step_deg)
        table_rows = _table_rows(table, panel_indices)

        # h5py requires increasing fancy indices.  Restore NPZ panel order after
        # reading because it is also the WAV channel order.
        sorted_order = np.argsort(table_rows)
        inverse_order = np.argsort(sorted_order)
        sorted_panels = table_rows[sorted_order]
        raw = table["p"][sorted_panels, :, frequency_indices[0] : frequency_indices[-1] + 1]
        raw = raw[inverse_order, :, :]
        raw = raw[:, doa_indices, :]
        directions = np.asarray(table["doa_grid"], dtype=np.float64)[doa_indices]

    steering = raw.transpose(1, 2, 0).astype(np.complex64, copy=False)
    azimuths_deg, elevations_deg = directions_to_angles(directions)
    return BemDictionary(
        steering=steering,
        directions=directions,
        azimuths_deg=azimuths_deg,
        elevations_deg=elevations_deg,
        doa_indices=doa_indices,
        frequencies_hz=requested_frequencies,
        panel_indices=panel_indices,
    )


def load_bem_steering(
    table_path: str | Path,
    panel_indices: np.ndarray,
    doa_index: int,
    *,
    sample_rate: int,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray]:
    panel_indices = _validate_panel_indices(panel_indices)
    with h5py.File(Path(table_path), "r") as table:
        _validate_table_format(table, sample_rate, n_fft)
        n_doa = int(table.attrs["n_doa"])
        if not 0 <= doa_index < n_doa:
            raise IndexError(f"doa_index must be in [0, {n_doa}), got {doa_index}")
        table_rows = _table_rows(table, panel_indices)
        sorted_order = np.argsort(table_rows)
        inverse_order = np.argsort(sorted_order)
        raw = table["p"][table_rows[sorted_order], doa_index, :]
        raw = raw[inverse_order, :]
        frequencies_hz = np.asarray(table["freqs"], dtype=np.float64)

    return frequencies_hz, raw.T.astype(np.complex64, copy=False)


def _table_rows(table: h5py.File, panel_indices: np.ndarray) -> np.ndarray:
    source = table.attrs.get("source_panel_indices")
    if source is None:
        return panel_indices
    source = np.asarray(source, dtype=np.int64)
    missing = set(panel_indices.tolist()) - set(source.tolist())
    if missing:
        raise ValueError(f"BEM table has panels {source.tolist()}, requested {panel_indices.tolist()}")
    return np.asarray([int(np.flatnonzero(source == p)[0]) for p in panel_indices], dtype=np.int64)


def _validate_panel_indices(panel_indices: np.ndarray) -> np.ndarray:
    panels = np.asarray(panel_indices, dtype=np.int64)
    if panels.ndim != 1 or panels.size == 0:
        raise ValueError(f"panel_indices must be a non-empty vector, got {panels.shape}")
    if np.any(panels < 0) or len(np.unique(panels)) != len(panels):
        raise ValueError("panel_indices must be unique non-negative integers")
    return panels


def _validate_table_format(table: h5py.File, sample_rate: int, n_fft: int) -> None:
    if int(table.attrs["fs"]) != sample_rate:
        raise ValueError(f"BEM fs={table.attrs['fs']} does not match WAV fs={sample_rate}")
    if int(table.attrs["nfft"]) != n_fft:
        raise ValueError(f"BEM nfft={table.attrs['nfft']} does not match requested n_fft={n_fft}")
    if table["p"].shape[0] <= 0 or table["p"].shape[-1] != n_fft // 2 + 1:
        raise ValueError("unexpected BEM response table shape")


def _matching_frequency_indices(table_frequencies: np.ndarray, requested_frequencies: np.ndarray) -> np.ndarray:
    start = int(np.searchsorted(table_frequencies, requested_frequencies[0]))
    end = start + len(requested_frequencies)
    if end > len(table_frequencies) or not np.allclose(
        table_frequencies[start:end], requested_frequencies, atol=1e-5, rtol=0.0
    ):
        raise ValueError("the requested STFT band is not a contiguous BEM frequency slice")
    return np.arange(start, end, dtype=np.int64)


def _grid_indices(table: h5py.File, grid_step_deg: float) -> np.ndarray:
    n_azimuth = int(table.attrs["n_az"])
    n_elevation = int(table.attrs["n_el"])
    native_azimuth_step = 360.0 / n_azimuth
    native_elevation_step = 180.0 / (n_elevation - 1)
    azimuth_stride = int(round(grid_step_deg / native_azimuth_step))
    elevation_stride = int(round(grid_step_deg / native_elevation_step))
    if azimuth_stride < 1 or elevation_stride < 1:
        raise ValueError("grid_step_deg is below the BEM grid resolution")
    if not np.isclose(azimuth_stride * native_azimuth_step, grid_step_deg) or not np.isclose(
        elevation_stride * native_elevation_step, grid_step_deg
    ):
        raise ValueError("grid_step_deg must be an integer multiple of BEM grid spacing")
    azimuth_indices = np.arange(0, n_azimuth, azimuth_stride)
    elevation_indices = np.arange(0, n_elevation, elevation_stride)
    return (azimuth_indices[:, None] * n_elevation + elevation_indices[None, :]).ravel()
