from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn


@dataclass(frozen=True)
class NetConfig:
    channels: int = 4
    n_freq: int = 257
    hidden: int = 96
    blocks: int = 8
    max_sources: int = 3
    cross_kernel: int = 5


class NarrowBandBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, hidden, n_freq, n_frame = x.shape
        y = x.permute(0, 2, 3, 1).reshape(batch * n_freq, n_frame, hidden)
        y = self.norm(y)
        y, _ = self.rnn(y)
        y = self.proj(y)
        y = y.reshape(batch, n_freq, n_frame, hidden).permute(0, 3, 1, 2)
        return x + y


class CrossBandBlock(nn.Module):
    def __init__(self, hidden: int, kernel: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2, groups=1)
        self.act = nn.GELU()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, hidden, n_freq, n_frame = x.shape
        y = x.permute(0, 3, 2, 1).reshape(batch * n_frame, n_freq, hidden)
        y = self.norm(y)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        y = self.proj(self.act(y))
        y = y.reshape(batch, n_frame, n_freq, hidden).permute(0, 3, 2, 1)
        return x + y


class SharedTrunk(nn.Module):
    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.encode = nn.Conv2d(2 * config.channels, config.hidden, 1)
        self.blocks = nn.ModuleList()
        self.checkpoint = False
        for _ in range(config.blocks):
            self.blocks.append(NarrowBandBlock(config.hidden))
            self.blocks.append(CrossBandBlock(config.hidden, config.cross_kernel))

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        x = self.encode(spectra)
        for block in self.blocks:
            if self.checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class SeparationHead(nn.Module):
    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.out = nn.Conv2d(config.hidden, 2 * config.max_sources, 1)
        self.max_sources = config.max_sources

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.out(x)
        batch, _, n_freq, n_frame = y.shape
        return y.reshape(batch, self.max_sources, 2, n_freq, n_frame)


class DoaHead(nn.Module):
    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.score = nn.Conv2d(config.hidden, 1, 1)
        self.value = nn.Conv2d(config.hidden, config.hidden, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(config.hidden, config.hidden, 1),
            nn.GELU(),
            nn.Conv1d(config.hidden, 4 * config.max_sources, 1),
        )
        self.max_sources = config.max_sources

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(self.score(x), dim=2)
        y = (self.value(x) * attention).sum(dim=2)
        y = self.mlp(y)
        batch, _, n_frame = y.shape
        y = y.reshape(batch, self.max_sources, 4, n_frame)
        direction = torch.nn.functional.normalize(y[:, :, :3], dim=2, eps=1e-8)
        activity = torch.sigmoid(y[:, :, 3])
        return torch.cat([direction, activity.unsqueeze(2)], dim=2)


class JointNet(nn.Module):
    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
        self.trunk = SharedTrunk(self.config)
        self.separation = SeparationHead(self.config)
        self.doa = DoaHead(self.config)

    def forward(self, spectra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(spectra)
        return self.separation(features), self.doa(features)


def stack_real_imag(spectra: torch.Tensor) -> torch.Tensor:
    return torch.cat([spectra.real, spectra.imag], dim=1)


def to_complex(separated: torch.Tensor) -> torch.Tensor:
    return torch.complex(separated[:, :, 0], separated[:, :, 1])


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_dummy(path: str, config: NetConfig | None = None) -> JointNet:
    config = config or NetConfig()
    model = JointNet(config)
    torch.save({"config": config.__dict__, "model_state": model.state_dict()}, path)
    return model


def load(path: str, device: str = "cpu") -> JointNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = JointNet(NetConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model.to(device)
