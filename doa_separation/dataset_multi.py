from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from stft import stft_multichannel


@dataclass(frozen=True)
class LoaderConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop: int = 128
    crop_seconds: float = 2.0
    max_sources: int = 3
    activity_threshold: float = 0.05
    target_key: str = "src_direct"
    band_lo_hz: float = 150.0
    band_hi_hz: float = 7000.0

    def band_mask(self) -> np.ndarray:
        frequencies = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        return (frequencies >= self.band_lo_hz) & (frequencies <= self.band_hi_hz)


class MultiSourceDataset(Dataset):
    def __init__(self, root: str | Path, config: LoaderConfig | None = None,
                 train: bool = True) -> None:
        self.root = Path(root)
        self.config = config or LoaderConfig()
        self.train = train
        self.paths = sorted(self.root.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"클립이 없다: {self.root}")
        self.mask = self.config.band_mask()[None, :, None]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        config = self.config
        path = self.paths[index]
        meta = np.load(path, allow_pickle=True)
        pcm, _ = sf.read(str(path.with_suffix(".wav")), dtype="int16")

        divisor = 32767.0 * float(meta["wav_scale"])
        mixture = pcm.astype(np.float32) / divisor
        direct = meta["src_direct"].astype(np.float32) / divisor
        wanted = (direct if config.target_key == "src_direct"
                  else meta[config.target_key].astype(np.float32) / divisor)

        crop = int(round(config.crop_seconds * config.sample_rate))
        if mixture.shape[0] < crop:
            raise ValueError(f"{path.name}: 클립이 크롭 길이보다 짧다")
        start = int(np.random.randint(0, mixture.shape[0] - crop + 1)) if self.train else 0
        mixture = mixture[start:start + crop]
        direct = direct[:, start:start + crop]
        wanted = wanted[:, start:start + crop]

        _, spectra = stft_multichannel(mixture, config.sample_rate,
                                       n_fft=config.n_fft, hop=config.hop)
        _, direct_spectra = stft_multichannel(direct.T, config.sample_rate,
                                              n_fft=config.n_fft, hop=config.hop)
        target_spectra = (direct_spectra if config.target_key == "src_direct" else
                          stft_multichannel(wanted.T, config.sample_rate,
                                            n_fft=config.n_fft, hop=config.hop)[1])

        spectra = spectra * self.mask
        target_spectra = target_spectra * self.mask
        direct_spectra = direct_spectra * self.mask

        level = max(float(np.abs(spectra[0]).mean()), 1e-12)
        spectra = spectra / level
        target_spectra = target_spectra / level

        reference = np.abs(spectra[0]) * level
        activity = (np.abs(direct_spectra) / (reference[None] + 1e-12)).mean(axis=1)
        activity = (activity > config.activity_threshold).astype(np.float32)

        n_source, n_freq, n_frame = target_spectra.shape
        slots = config.max_sources
        if n_source > slots:
            raise ValueError(f"{path.name}: 음원 {n_source}개 > 슬롯 {slots}개")

        directions = np.zeros((slots, 3), dtype=np.float32)
        directions[:n_source] = np.atleast_2d(meta["d_unit"]).astype(np.float32)
        padded_activity = np.zeros((slots, n_frame), dtype=np.float32)
        padded_activity[:n_source] = activity
        padded_target = np.zeros((slots, n_freq, n_frame), dtype=np.complex64)
        padded_target[:n_source] = target_spectra

        return {
            "mixture": torch.from_numpy(np.stack([spectra.real, spectra.imag]).astype(np.float32)),
            "target": torch.from_numpy(np.stack([padded_target.real,
                                                 padded_target.imag]).astype(np.float32)),
            "direction": torch.from_numpy(directions),
            "activity": torch.from_numpy(padded_activity),
            "n_src": torch.tensor(int(meta["n_src"]), dtype=torch.long),
            "level": torch.tensor(level, dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    mixture = torch.stack([item["mixture"] for item in batch])
    return {
        "mixture": mixture.flatten(1, 2),
        "target": torch.stack([item["target"] for item in batch]).permute(0, 2, 1, 3, 4),
        "direction": torch.stack([item["direction"] for item in batch]),
        "activity": torch.stack([item["activity"] for item in batch]),
        "n_src": torch.stack([item["n_src"] for item in batch]),
        "level": torch.stack([item["level"] for item in batch]),
    }
