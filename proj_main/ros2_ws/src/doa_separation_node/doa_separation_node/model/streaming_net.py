from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from net_v3 import JointNetV3, NetConfigV3

MODEL_RATE = 8000
BAND_LO_HZ = 150.0


BAND_HI_HZ = 3800.0


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
    resample_tail: np.ndarray | None = None
    tail: np.ndarray | None = None
    overlap: torch.Tensor | None = None
    overlap_env: torch.Tensor | None = None
    level: float = 0.0
    frames_seen: int = 0


def band_mask(n_fft: int, sample_rate: int) -> np.ndarray:
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return (frequencies >= BAND_LO_HZ) & (frequencies <= BAND_HI_HZ)


class StreamingJointNet:
    def __init__(self, checkpoint: str, config: StreamConfig | None = None) -> None:
        self.config = config or StreamConfig()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        net_config = state["config"]
        if "d_state" not in net_config:
            raise ValueError(
                f"{checkpoint}: v3 체크포인트 아님")

        self.input_rate = self.config.sample_rate
        self.config.sample_rate = MODEL_RATE
        self.config.n_fft = (int(net_config["n_freq"]) - 1) * 2
        self.decimate = max(1, self.input_rate // self.config.sample_rate)

        self.model = JointNetV3(NetConfigV3(**net_config))
        self.model.load_state_dict(state["model_state"])
        self.model.eval().to(self.config.device)
        self.epoch = int(state.get("epoch", -1))

        self.mamba_state = self.model.new_stream_state(n_freq=int(net_config["n_freq"]))

        self._graph = None
        self._graph_in = None
        self._graph_out = None
        self._graph_warmups = 0
        self.use_graph = str(self.config.device).startswith("cuda")

        self.band_mask = torch.as_tensor(
            band_mask(self.config.n_fft, self.config.sample_rate),
            dtype=torch.float32, device=self.config.device)[None, :, None]
        self.window = torch.hann_window(self.config.n_fft, device=self.config.device)

        self.window_sq = (self.window ** 2)[None, :, None]

        self.reset()

    def reset(self) -> None:
        self.state = _State()
        if self._graph is None:
            self.mamba_state.reset()
        else:

            self.mamba_state.seqlen_offset = 0
            for slot in self.mamba_state.key_value_memory_dict.values():
                for tensor in slot:
                    tensor.zero_()

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
        if not self.use_graph:
            return self.model.trunk(spectra, self.mamba_state)
        if self._graph is not None:
            self._graph_in.copy_(spectra)
            self._graph.replay()
            return self._graph_out
        out = self.model.trunk(spectra, self.mamba_state)
        self._graph_warmups += 1
        if self._graph_warmups >= 6:
            self._capture(spectra)
        return out

    def _capture(self, sample: torch.Tensor) -> None:
        try:
            self._graph_in = sample.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.model.trunk(self._graph_in, self.mamba_state)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._graph_out = self.model.trunk(self._graph_in, self.mamba_state)
            torch.cuda.synchronize()
            self._graph = graph
        except Exception:
            self._graph = None
            self._graph_in = None
            self._graph_out = None
            self.use_graph = False

    def _overlap_add(self, spectra: torch.Tensor) -> torch.Tensor:
        config = self.config
        frames = torch.fft.irfft(spectra, n=config.n_fft, dim=1)
        frames = frames * self.window[None, :, None]

        n_track, _, n_frame = frames.shape
        emit = n_frame * config.hop
        span = emit + config.n_fft - config.hop
        buffer = torch.zeros(n_track, span, device=frames.device, dtype=frames.dtype)
        envelope = torch.zeros(1, span, device=frames.device, dtype=frames.dtype)
        if self.state.overlap is not None:
            buffer[:, : self.state.overlap.shape[1]] += self.state.overlap
            envelope[:, : self.state.overlap_env.shape[1]] += self.state.overlap_env
        for index in range(n_frame):
            start = index * config.hop
            buffer[:, start:start + config.n_fft] += frames[:, :, index]
            envelope[:, start:start + config.n_fft] += self.window_sq[0, :, 0]

        self.state.overlap = buffer[:, emit:].clone()
        self.state.overlap_env = envelope[:, emit:].clone()
        return buffer[:, :emit] / envelope[:, :emit].clamp(min=1e-8)

    def _decimate(self, samples: np.ndarray) -> np.ndarray:
        from scipy.signal import resample_poly
        pad = self.decimate * 32
        if self.state.resample_tail is None:
            head = np.zeros((pad,) + samples.shape[1:], dtype=samples.dtype)
        else:
            head = self.state.resample_tail
        joined = np.concatenate([head, samples], axis=0)
        self.state.resample_tail = joined[-pad:].copy()
        out = resample_poly(joined, 1, self.decimate, axis=0).astype(np.float32)
        return out[pad // self.decimate:]

    @torch.no_grad()
    def push(self, samples: np.ndarray) -> dict | None:
        config = self.config
        if self.decimate > 1:
            samples = self._decimate(samples)
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
