from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt
import soundfile as sf


FloatArray = npt.NDArray[np.float32]


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SampleMetadata:
    sample_id: str
    direction_unit: npt.NDArray[np.float64]
    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    source_position_m: npt.NDArray[np.float64]
    body_center_m: npt.NDArray[np.float64]
    room_size_m: npt.NDArray[np.float64]
    rt60_s: float
    snr_db: float
    body_id: str
    panel_indices: npt.NDArray[np.int64]
    microphone_positions_m: npt.NDArray[np.float64]
    sample_rate: int
    duration_s: float
    wav_scale: float


@dataclass(frozen=True)
class MultichannelSample:
    audio: FloatArray
    metadata: SampleMetadata


class ScatteringDataset:
    _REQUIRED_KEYS = frozenset({
        "wav_scale", "t_sec", "el_deg", "d_unit", "panel_indices", "body_center",
        "distance", "body_id", "panel_pos", "fs", "snr", "az_deg", "src_pos",
        "room_sz", "t60",
    })

    def __init__(
        self,
        split_dir: str | Path,
        *,
        expected_channels: int = 4,
        expected_sample_rate: int | None = 16000,
    ) -> None:
        self.split_dir = Path(split_dir).expanduser().resolve()
        self.expected_channels = expected_channels
        self.expected_sample_rate = expected_sample_rate
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"dataset split directory does not exist: {self.split_dir}")
        if expected_channels < 1:
            raise ValueError("expected_channels must be positive")
        self._sample_ids = self._discover_pairs()

    def _discover_pairs(self) -> tuple[str, ...]:
        wav_ids = {path.stem for path in self.split_dir.glob("*.wav")}
        npz_ids = {path.stem for path in self.split_dir.glob("*.npz")}
        missing_metadata = sorted(wav_ids - npz_ids)
        missing_audio = sorted(npz_ids - wav_ids)
        if missing_metadata or missing_audio:
            details: list[str] = []
            if missing_metadata:
                details.append(f"missing NPZ for {', '.join(missing_metadata[:5])}")
            if missing_audio:
                details.append(f"missing WAV for {', '.join(missing_audio[:5])}")
            raise DatasetValidationError("unpaired dataset files: " + "; ".join(details))
        if not wav_ids:
            raise DatasetValidationError(f"no WAV/NPZ pairs found in {self.split_dir}")
        return tuple(sorted(wav_ids))

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, index: int) -> MultichannelSample:
        return self.load(self._sample_ids[index])

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    def __iter__(self) -> Iterator[MultichannelSample]:
        for sample_id in self._sample_ids:
            yield self.load(sample_id)

    def load(self, sample_id: str) -> MultichannelSample:
        if sample_id not in self._sample_ids:
            raise KeyError(f"unknown sample ID {sample_id!r} in {self.split_dir}")
        metadata = self._load_metadata(sample_id)
        wav_path = self.split_dir / f"{sample_id}.wav"
        pcm, sample_rate = sf.read(wav_path, dtype="int16", always_2d=True)
        if pcm.ndim != 2 or pcm.shape[1] != self.expected_channels:
            raise DatasetValidationError(
                f"{wav_path.name}: expected [samples, {self.expected_channels}] PCM, got {pcm.shape}"
            )
        if sample_rate != metadata.sample_rate:
            raise DatasetValidationError(
                f"{sample_id}: WAV sample rate {sample_rate} disagrees with NPZ fs {metadata.sample_rate}"
            )
        if self.expected_sample_rate is not None and sample_rate != self.expected_sample_rate:
            raise DatasetValidationError(
                f"{sample_id}: expected {self.expected_sample_rate} Hz, got {sample_rate} Hz"
            )

        duration_s = pcm.shape[0] / sample_rate
        if not np.isclose(duration_s, metadata.duration_s, atol=1.0 / sample_rate, rtol=0.0):
            raise DatasetValidationError(
                f"{sample_id}: WAV duration {duration_s:.6f}s disagrees with NPZ t_sec {metadata.duration_s:.6f}s"
            )

        audio = pcm.astype(np.float32) / np.float32(32767.0) / np.float32(metadata.wav_scale)
        if not np.isfinite(audio).all():
            raise DatasetValidationError(f"{sample_id}: restored waveform contains non-finite values")
        return MultichannelSample(audio=audio, metadata=metadata)

    def _load_metadata(self, sample_id: str) -> SampleMetadata:
        npz_path = self.split_dir / f"{sample_id}.npz"
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = self._REQUIRED_KEYS.difference(archive.files)
            if missing:
                raise DatasetValidationError(f"{npz_path.name}: missing keys {sorted(missing)}")

            direction = self._vector(archive["d_unit"], "d_unit", sample_id)
            direction_norm = np.linalg.norm(direction)
            if not np.isclose(direction_norm, 1.0, atol=1e-4):
                raise DatasetValidationError(f"{sample_id}: d_unit has norm {direction_norm:.6f}, not 1")
            panel_indices = np.asarray(archive["panel_indices"], dtype=np.int64)
            if panel_indices.shape != (self.expected_channels,):
                raise DatasetValidationError(
                    f"{sample_id}: panel_indices must have shape ({self.expected_channels},), got {panel_indices.shape}"
                )
            microphone_positions = np.asarray(archive["panel_pos"], dtype=np.float64)
            if microphone_positions.shape != (self.expected_channels, 3):
                raise DatasetValidationError(
                    f"{sample_id}: panel_pos must have shape ({self.expected_channels}, 3), got {microphone_positions.shape}"
                )
            if not np.isfinite(microphone_positions).all():
                raise DatasetValidationError(f"{sample_id}: panel_pos contains non-finite values")

            sample_rate = self._scalar_int(archive["fs"], "fs", sample_id)
            if sample_rate <= 0:
                raise DatasetValidationError(f"{sample_id}: fs must be positive")
            duration_s = self._scalar_float(archive["t_sec"], "t_sec", sample_id)
            wav_scale = self._scalar_float(archive["wav_scale"], "wav_scale", sample_id)
            if not np.isfinite(duration_s) or duration_s <= 0:
                raise DatasetValidationError(f"{sample_id}: t_sec must be finite and positive")
            if not np.isfinite(wav_scale) or wav_scale <= 0:
                raise DatasetValidationError(f"{sample_id}: wav_scale must be finite and positive")

            return SampleMetadata(
                sample_id=sample_id,
                direction_unit=direction,
                azimuth_deg=self._scalar_float(archive["az_deg"], "az_deg", sample_id),
                elevation_deg=self._scalar_float(archive["el_deg"], "el_deg", sample_id),
                distance_m=self._scalar_float(archive["distance"], "distance", sample_id),
                source_position_m=self._vector(archive["src_pos"], "src_pos", sample_id),
                body_center_m=self._vector(archive["body_center"], "body_center", sample_id),
                room_size_m=self._vector(archive["room_sz"], "room_sz", sample_id),
                rt60_s=self._scalar_float(archive["t60"], "t60", sample_id),
                snr_db=self._scalar_float(archive["snr"], "snr", sample_id),
                body_id=self._scalar_text(archive["body_id"], "body_id", sample_id),
                panel_indices=panel_indices,
                microphone_positions_m=microphone_positions,
                sample_rate=sample_rate,
                duration_s=duration_s,
                wav_scale=wav_scale,
            )

    @staticmethod
    def _scalar_float(value: npt.ArrayLike, field: str, sample_id: str) -> float:
        array = np.asarray(value)
        if array.size != 1:
            raise DatasetValidationError(f"{sample_id}: {field} must be scalar, got shape {array.shape}")
        result = float(array.reshape(()))
        if not np.isfinite(result):
            raise DatasetValidationError(f"{sample_id}: {field} must be finite")
        return result

    @staticmethod
    def _scalar_int(value: npt.ArrayLike, field: str, sample_id: str) -> int:
        number = ScatteringDataset._scalar_float(value, field, sample_id)
        if not number.is_integer():
            raise DatasetValidationError(f"{sample_id}: {field} must be an integer, got {number}")
        return int(number)

    @staticmethod
    def _scalar_text(value: npt.ArrayLike, field: str, sample_id: str) -> str:
        array = np.asarray(value)
        if array.size != 1:
            raise DatasetValidationError(f"{sample_id}: {field} must be scalar, got shape {array.shape}")
        text = str(array.reshape(()).item())
        if not text:
            raise DatasetValidationError(f"{sample_id}: {field} must not be empty")
        return text

    @staticmethod
    def _vector(value: npt.ArrayLike, field: str, sample_id: str) -> npt.NDArray[np.float64]:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (3,):
            raise DatasetValidationError(f"{sample_id}: {field} must have shape (3,), got {array.shape}")
        if not np.isfinite(array).all():
            raise DatasetValidationError(f"{sample_id}: {field} contains non-finite values")
        return array


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and preview a scattering-dataset split.")
    parser.add_argument("split_dir", type=Path, help="directory containing matched .wav and .npz files")
    parser.add_argument("--limit", type=int, default=3, help="number of examples to load (default: 3)")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    dataset = ScatteringDataset(args.split_dir)
    print(f"split={dataset.split_dir} pairs={len(dataset)}")
    for sample_id in dataset.sample_ids[:args.limit]:
        sample = dataset.load(sample_id)
        meta = sample.metadata
        print(
            f"{meta.sample_id}: audio={sample.audio.shape} fs={meta.sample_rate} "
            f"az={meta.azimuth_deg:.1f} el={meta.elevation_deg:.1f} "
            f"rt60={meta.rt60_s:.2f}s snr={meta.snr_db:.1f}dB"
        )


if __name__ == "__main__":
    main()
