from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn

from mamba_block import NarrowBandBlock, StreamState


@dataclass(frozen=True)
class NetConfigV3:
    channels: int = 4
    n_freq: int = 129
    hidden: int = 128
    blocks: int = 8
    max_sources: int = 3
    freq_kernel: int = 3
    freq_groups: int = 8
    fullband_channels: int = 8
    time_kernel: int = 5
    d_state: int = 16
    headdim: int = 64


class SeparationHead(nn.Module):

    def __init__(self, config: NetConfigV3) -> None:
        super().__init__()
        self.out = nn.Conv2d(config.hidden, 2 * config.max_sources, 1)
        self.max_sources = config.max_sources

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.out(x)
        batch, _, n_freq, n_frame = y.shape
        return y.reshape(batch, self.max_sources, 2, n_freq, n_frame)


class DoaHead(nn.Module):

    def __init__(self, config: NetConfigV3) -> None:
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


class FreqConvModule(nn.Module):

    def __init__(self, hidden: int, kernel: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2,
                              groups=groups)
        self.act = nn.PReLU(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, hidden, n_freq, n_frame = x.shape
        y = x.permute(0, 3, 2, 1).reshape(batch * n_frame, n_freq, hidden)
        y = self.norm(y).transpose(1, 2)
        y = self.act(self.conv(y)).transpose(1, 2)
        y = y.reshape(batch, n_frame, n_freq, hidden).permute(0, 3, 2, 1)
        return x + y


class FullBandLinear(nn.Module):

    def __init__(self, n_freq: int, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, n_freq, n_freq))
        self.bias = nn.Parameter(torch.zeros(channels, n_freq))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = torch.einsum("bfc,cgf->bgc", x, self.weight) + self.bias.t()
        return y


class FullBandModule(nn.Module):
    def __init__(self, hidden: int, channels: int, mapping: FullBandLinear) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.reduce = nn.Linear(hidden, channels)
        self.mapping = mapping
        self.expand = nn.Linear(channels, hidden)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, hidden, n_freq, n_frame = x.shape
        y = x.permute(0, 3, 2, 1).reshape(batch * n_frame, n_freq, hidden)
        y = self.act(self.reduce(self.norm(y)))
        y = self.mapping(y)
        y = self.act(self.expand(y))
        y = y.reshape(batch, n_frame, n_freq, hidden).permute(0, 3, 2, 1)
        return x + y


class CrossBandBlock(nn.Module):
    def __init__(self, config: NetConfigV3, mapping: FullBandLinear) -> None:
        super().__init__()
        self.first = FreqConvModule(config.hidden, config.freq_kernel,
                                    config.freq_groups)
        self.fullband = FullBandModule(config.hidden, config.fullband_channels,
                                       mapping)
        self.second = FreqConvModule(config.hidden, config.freq_kernel,
                                     config.freq_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.fullband(self.first(x)))


class CausalTimeConv(nn.Module):

    _KEY = "encode_tail"

    def __init__(self, in_channels: int, hidden: int, kernel: int) -> None:
        super().__init__()
        self.pad = kernel - 1
        self.conv = nn.Conv2d(in_channels, hidden, (1, kernel))

    def forward(self, x: torch.Tensor,
                state: "StreamState | None" = None) -> torch.Tensor:
        if state is None:
            return self.conv(torch.nn.functional.pad(x, (self.pad, 0)))
        slot = state.key_value_memory_dict.get(self._KEY)
        if slot is None or slot[0].shape[:3] != x.shape[:3]:
            slot = (torch.zeros(*x.shape[:3], self.pad,
                                dtype=x.dtype, device=x.device),)
            state.key_value_memory_dict[self._KEY] = slot
        y = self.conv(torch.cat([slot[0], x], dim=3))
        slot[0].copy_(x[..., -self.pad:])
        return y


class SharedTrunkV3(nn.Module):
    def __init__(self, config: NetConfigV3) -> None:
        super().__init__()
        self.encode = CausalTimeConv(2 * config.channels, config.hidden,
                                     config.time_kernel)
        mapping = FullBandLinear(config.n_freq, config.fullband_channels)
        self.blocks = nn.ModuleList()
        self.checkpoint = False
        for layer in range(config.blocks):
            self.blocks.append(CrossBandBlock(config, mapping))
            self.blocks.append(NarrowBandBlock(config.hidden, config.d_state,
                                               config.headdim, layer_idx=layer))

    def forward(self, spectra: torch.Tensor,
                state: StreamState | None = None) -> torch.Tensor:
        x = self.encode(spectra, state)
        for block in self.blocks:

            if self.checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=True)
            elif isinstance(block, NarrowBandBlock):
                x = block(x, state)
            else:
                x = block(x)
        return x


class JointNetV3(nn.Module):
    def __init__(self, config: NetConfigV3 | None = None) -> None:
        super().__init__()
        self.config = config or NetConfigV3()
        self.trunk = SharedTrunkV3(self.config)
        self.separation = SeparationHead(self.config)
        self.doa = DoaHead(self.config)

    def forward(self, spectra: torch.Tensor,
                state: StreamState | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(spectra, state)
        return self.separation(features), self.doa(features)

    def new_stream_state(self, n_freq: int | None = None,
                         max_seconds: float = 3600.0) -> StreamState:
        freq = self.config.n_freq if n_freq is None else n_freq
        return StreamState(max_seqlen=int(max_seconds * 8000 / 128),
                           max_batch_size=freq)


def load(path: str, device: str = "cpu") -> JointNetV3:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = JointNetV3(NetConfigV3(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model.to(device)
