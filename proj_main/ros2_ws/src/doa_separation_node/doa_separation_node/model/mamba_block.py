from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field

import torch
from torch import nn

D_STATE = 16
HEADDIM = 64
EXPAND = 2
CHUNK_SIZE = 32


def load_mamba3():
    if "mamba_ssm" not in sys.modules:
        spec = importlib.util.find_spec("mamba_ssm")
        if spec is None:
            raise ImportError(
                "mamba_ssm 이 없다: MAMBA_SKIP_CUDA_BUILD=TRUE pip install --user "
                "--no-deps --no-build-isolation mamba-ssm==2.3.2.post1")
        sys.modules["mamba_ssm"] = importlib.util.module_from_spec(spec)
    from mamba_ssm.modules.mamba3 import Mamba3
    return Mamba3


@dataclass
class StreamState:

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample = None

    def reset(self) -> None:
        self.seqlen_offset = 0
        self.key_value_memory_dict.clear()


def _siso_forward(m, u, state_slot):
    import torch.nn.functional as F
    from einops import rearrange
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

    zxBCdtAtrap = m.in_proj(u)
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        zxBCdtAtrap,
        [m.d_inner, m.d_inner,
         m.d_state * m.num_bc_heads * m.mimo_rank,
         m.d_state * m.num_bc_heads * m.mimo_rank,
         m.nheads, m.nheads, m.nheads, m.num_rope_angles], dim=-1)
    z = rearrange(z, "b l (h p) -> b l h p", p=m.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=m.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.num_bc_heads)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.num_bc_heads)
    trap = rearrange(trap, "b l h -> b h l")

    _A = torch.clamp(-F.softplus(dd_A.to(torch.float32)), max=-m.A_floor)
    DT = F.softplus(dd_dt + m.dt_bias)
    ADT = rearrange(_A * DT, "b l n -> b n l")
    DT = rearrange(DT, "b l n -> b n l")
    angles = angles.unsqueeze(-2).expand(-1, -1, m.nheads, -1).to(torch.float32)
    B, C = m.B_norm(B), m.C_norm(C)

    y = mamba3_siso_combined(
        Q=C.squeeze(2), K=B.squeeze(2), V=x, ADT=ADT, DT=DT, Trap=trap,
        Q_bias=m.C_bias.squeeze(1), K_bias=m.B_bias.squeeze(1), Angles=angles,
        D=m.D, Z=z if not m.is_outproj_norm else None, chunk_size=m.chunk_size,
        Input_States=state_slot, return_final_states=state_slot is not None)
    if state_slot is not None:
        y, last_angle, last_state, last_k, last_v, *_ = y
        state_slot[0].copy_(last_angle)
        state_slot[1].copy_(last_state)
        state_slot[2].copy_(last_k)
        state_slot[3].copy_(last_v)
    y = rearrange(y, "b l h p -> b l (h p)")
    if m.is_outproj_norm:
        y = m.norm(y, rearrange(z, "b l h p -> b l (h p)"))
    return m.out_proj(y.to(x.dtype))


class MambaModule(nn.Module):

    def __init__(self, hidden: int, d_state: int = D_STATE, headdim: int = HEADDIM,
                 expand: int = EXPAND, chunk_size: int = CHUNK_SIZE,
                 layer_idx: int | None = None) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)

        self.mamba = load_mamba3()(d_model=hidden, d_state=d_state, headdim=headdim,
                                   expand=expand, chunk_size=chunk_size, is_mimo=False,
                                   layer_idx=layer_idx)

    def _slot(self, state, n_batch, device, dtype):
        m = self.mamba
        key = self.mamba.layer_idx
        if key not in state.key_value_memory_dict:
            state.key_value_memory_dict[key] = (
                torch.zeros(n_batch, m.nheads, m.num_rope_angles, device=device,
                            dtype=torch.float32),
                torch.zeros(n_batch, m.nheads, m.headdim, m.d_state, device=device,
                            dtype=torch.float32),
                torch.zeros(n_batch, m.nheads, m.d_state, device=device, dtype=dtype),
                torch.zeros(n_batch, m.nheads, m.headdim, device=device, dtype=dtype))
        return state.key_value_memory_dict[key]

    def forward(self, x: torch.Tensor, state: "StreamState | None" = None) -> torch.Tensor:
        batch, hidden, n_freq, n_frame = x.shape
        y = x.permute(0, 2, 3, 1).reshape(batch * n_freq, n_frame, hidden)
        if state is None:
            out = self.mamba(self.norm(y))
        else:
            slot = self._slot(state, y.shape[0], y.device, self.mamba.in_proj.weight.dtype)
            out = _siso_forward(self.mamba, self.norm(y), slot)
        y = y + out
        return y.reshape(batch, n_freq, n_frame, hidden).permute(0, 3, 1, 2)


class NarrowBandBlock(nn.Module):

    def __init__(self, hidden: int, d_state: int = D_STATE, headdim: int = HEADDIM,
                 expand: int = EXPAND, chunk_size: int = CHUNK_SIZE,
                 layer_idx: int | None = None) -> None:
        super().__init__()
        a = None if layer_idx is None else 2 * layer_idx
        b = None if layer_idx is None else 2 * layer_idx + 1
        self.mhsa = MambaModule(hidden, d_state, headdim, expand, chunk_size, a)
        self.convffn = MambaModule(hidden, d_state, headdim, expand, chunk_size, b)

    def forward(self, x: torch.Tensor, state: "StreamState | None" = None) -> torch.Tensor:
        return self.convffn(self.mhsa(x, state), state)
