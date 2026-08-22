from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from dataset_multi import LoaderConfig
from net import JointNet, NetConfig


@dataclass
class StreamConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop: int = 128
    chunk_frames: int = 16
    level_halflife_sec: float = 2.0
    warmup_frames: int = 64
    device: str = "cuda"


@dataclass
class _State:
    hidden: list = field(default_factory=list)
    tail: np.ndarray | None = None
    overlap: torch.Tensor | None = None
    level: float = 0.0
    frames_seen: int = 0


class StreamingJointNet:

    def __init__(self, checkpoint: str, config: StreamConfig | None = None) -> None:
        self.config = config or StreamConfig()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = JointNet(NetConfig(**state["config"]))
        self.model.load_state_dict(state["model_state"])
        self.model.eval().to(self.config.device)
        self.epoch = int(state.get("epoch", -1))

        loader = LoaderConfig(sample_rate=self.config.sample_rate,
                              n_fft=self.config.n_fft, hop=self.config.hop)
        self.band_mask = torch.as_tensor(
            loader.band_mask(), dtype=torch.float32,
            device=self.config.device)[None, :, None]
        self.window = torch.hann_window(self.config.n_fft, device=self.config.device)

        self.narrow_blocks = [block for block in self.model.trunk.blocks
                              if hasattr(block, "rnn")]

        squared = self.window ** 2
        overlap_gain = torch.zeros_like(squared)
        for shift in range(0, self.config.n_fft, self.config.hop):
            overlap_gain += torch.roll(squared, shift)
        self.overlap_gain = float(overlap_gain[self.config.n_fft // 2])

        self.reset()

    def reset(self) -> None:
        self.state = _State(hidden=[None] * len(self.narrow_blocks))

    def _normalise(self, spectra: torch.Tensor) -> torch.Tensor:
        current = float(spectra[0].abs().mean())
        seconds = self.config.chunk_frames * self.config.hop / self.config.sample_rate
        decay = 0.5 ** (seconds / self.config.level_halflife_sec)
        if self.state.frames_seen == 0:
            self.state.level = current
        else:
            self.state.level = decay * self.state.level + (1.0 - decay) * current
        return spectra / max(self.state.level, 1e-12)

    def _trunk(self, spectra: torch.Tensor) -> torch.Tensor:
        x = self.model.trunk.encode(spectra)
        narrow_index = 0
        for block in self.model.trunk.blocks:
            if hasattr(block, "rnn"):
                batch, hidden, n_freq, n_frame = x.shape
                y = x.permute(0, 2, 3, 1).reshape(batch * n_freq, n_frame, hidden)
                y = block.norm(y)
                y, new_hidden = block.rnn(y, self.state.hidden[narrow_index])
                self.state.hidden[narrow_index] = new_hidden.detach()
                y = block.proj(y)
                y = y.reshape(batch, n_freq, n_frame, hidden).permute(0, 3, 1, 2)
                x = x + y
                narrow_index += 1
            else:
                x = block(x)
        return x

    def _overlap_add(self, spectra: torch.Tensor) -> torch.Tensor:
        config = self.config
        frames = torch.fft.irfft(spectra, n=config.n_fft, dim=1)
        frames = frames * self.window[None, :, None]

        n_track, _, n_frame = frames.shape
        emit = n_frame * config.hop
        buffer = torch.zeros(n_track, emit + config.n_fft - config.hop,
                             device=frames.device, dtype=frames.dtype)
        if self.state.overlap is not None:
            buffer[:, : self.state.overlap.shape[1]] += self.state.overlap
        for index in range(n_frame):
            start = index * config.hop
            buffer[:, start:start + config.n_fft] += frames[:, :, index]

        self.state.overlap = buffer[:, emit:].clone()
        return buffer[:, :emit] / self.overlap_gain

    @torch.no_grad()
    def push(self, samples: np.ndarray) -> dict | None:
        config = self.config
        if self.state.tail is not None:
            samples = np.concatenate([self.state.tail, samples], axis=0)

        need = (config.chunk_frames - 1) * config.hop + config.n_fft
        if samples.shape[0] < need:
            self.state.tail = samples
            return None
        self.state.tail = samples[config.chunk_frames * config.hop:]
        block = samples[:need]

        wave = torch.as_tensor(block.T, dtype=torch.float32, device=config.device)
        spectra = torch.stft(wave, n_fft=config.n_fft, hop_length=config.hop,
                             window=self.window, center=False, return_complex=True)
        spectra = spectra * self.band_mask
        spectra = self._normalise(spectra)
        packed = torch.cat([spectra.real, spectra.imag], dim=0)[None]

        features = self._trunk(packed)
        separated = self.model.separation(features).float()
        doa = self.model.doa(features).float()

        complex_out = torch.complex(separated[0, :, 0], separated[0, :, 1]) * self.band_mask
        audio = self._overlap_add(complex_out)

        direction = doa[0, :, :3].mean(dim=2)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-9)
        activity = doa[0, :, 3].mean(dim=1)

        self.state.frames_seen += config.chunk_frames
        return {
            "direction": direction.cpu().numpy(),
            "activity": activity.cpu().numpy(),
            "audio": (audio * self.state.level).cpu().numpy(),
            "warm": self.state.frames_seen >= config.warmup_frames,
        }


def count_probability(activity: np.ndarray) -> np.ndarray:
    poly = np.array([1.0])
    for p in activity:
        grown = np.zeros(len(poly) + 1)
        grown[:-1] += poly * (1.0 - p)
        grown[1:] += poly * p
        poly = grown
    return poly
