#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_multi import LoaderConfig, MultiSourceDataset, collate
from net import JointNet, NetConfig, parameter_count

DIRECTION_WEIGHT = 1.0
ACTIVITY_WEIGHT = 1.0
SEPARATION_WEIGHT = 1.0
SDR_SCALE = 10.0
SDR_CLAMP = 30.0
N_FFT, HOP = 512, 128
BAND_MASK = torch.as_tensor(LoaderConfig().band_mask(), dtype=torch.float32)[None, None, :, None]


def to_waveform(spectra: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    batch, slots = spectra.shape[:2]
    complex_spectra = torch.complex(spectra[:, :, 0], spectra[:, :, 1])
    if mask is not None:
        complex_spectra = complex_spectra * mask
    flat = complex_spectra.reshape(batch * slots, *complex_spectra.shape[2:])
    window = torch.hann_window(N_FFT, device=spectra.device, dtype=flat.real.dtype)
    wave = torch.istft(flat, n_fft=N_FFT, hop_length=HOP, window=window, center=True)
    return wave.reshape(batch, slots, -1)


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    alpha = ((estimate * reference).sum(dim=-1, keepdim=True)
             / (reference.pow(2).sum(dim=-1, keepdim=True) + 1e-8))
    projection = alpha * reference
    noise = estimate - projection
    ratio = (projection.pow(2).sum(dim=-1) + 1e-8) / (noise.pow(2).sum(dim=-1) + 1e-8)
    return (10.0 * torch.log10(ratio)).clamp(-SDR_CLAMP, SDR_CLAMP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JointNet 학습 — 분리 + DOA + 음원 개수")
    parser.add_argument("data_root", type=Path, help="box_24x20x16__multi 디렉터리")
    parser.add_argument("-o", "--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-b", "--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop_seconds", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--target", choices=["src_direct", "src_ref"], default="src_direct",
                        help="분리 목표 — 직접경로(잔향제거 포함) 또는 잔향 포함 신호")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint", action="store_true", help="그래디언트 체크포인팅 (VRAM↓)")
    parser.add_argument("--resume", action="store_true", help="last.pt에서 이어서 학습")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def permuted_loss(prediction_doa, prediction_waveform, target_direction,
                  target_activity, target_waveform, present, order):
    doa = prediction_doa[:, order]
    waveform = prediction_waveform[:, order]

    cosine = (doa[:, :, :3] * target_direction.unsqueeze(-1)).sum(dim=2)
    weight = target_activity
    direction = ((1.0 - cosine) * weight).sum(dim=(1, 2)) / (weight.sum(dim=(1, 2)) + 1e-6)

    activity = torch.nn.functional.binary_cross_entropy(
        doa[:, :, 3].clamp(1e-6, 1.0 - 1e-6), target_activity, reduction="none"
    ).mean(dim=(1, 2))

    ratio = si_sdr(waveform, target_waveform) * present
    separation = -ratio.sum(dim=1) / (present.sum(dim=1) + 1e-6) / SDR_SCALE
    return direction, activity, separation


def joint_loss(prediction_doa, prediction_separation, batch):
    slots = prediction_doa.shape[1]
    orders = list(itertools.permutations(range(slots)))
    present = (batch["activity"].sum(dim=2) > 0).float()
    mask = BAND_MASK.to(prediction_separation.device)
    prediction_waveform = to_waveform(prediction_separation, mask)
    target_waveform = to_waveform(batch["target"], mask)
    parts = [permuted_loss(prediction_doa, prediction_waveform, batch["direction"],
                           batch["activity"], target_waveform, present, list(order))
             for order in orders]
    directions = torch.stack([part[0] for part in parts], dim=1)
    activities = torch.stack([part[1] for part in parts], dim=1)
    separations = torch.stack([part[2] for part in parts], dim=1)

    assignment = (DIRECTION_WEIGHT * directions + ACTIVITY_WEIGHT * activities).argmin(dim=1)
    chosen = assignment[:, None]
    direction = directions.gather(1, chosen).squeeze(1)
    activity = activities.gather(1, chosen).squeeze(1)
    separation = separations.gather(1, chosen).squeeze(1)
    total = (DIRECTION_WEIGHT * direction + ACTIVITY_WEIGHT * activity
             + SEPARATION_WEIGHT * separation)
    return total.mean(), {"direction": direction.mean().item(),
                          "activity": activity.mean().item(),
                          "separation": separation.mean().item()}, assignment, orders


def evaluate_batch(prediction_doa, prediction_separation, batch, assignment, orders):
    order_table = torch.as_tensor(orders, device=prediction_doa.device)
    chosen = order_table[assignment]
    index = chosen[:, :, None, None].expand(-1, -1, prediction_doa.shape[2],
                                            prediction_doa.shape[3])
    doa = prediction_doa.gather(1, index)

    weight = batch["activity"]
    present = weight.sum(dim=2) > 0
    cosine = (doa[:, :, :3] * batch["direction"].unsqueeze(-1)).sum(dim=2).clamp(-1.0, 1.0)
    degrees = torch.rad2deg(torch.arccos(cosine))
    per_source = (degrees * weight).sum(dim=2) / weight.sum(dim=2).clamp(min=1e-6)

    detected = ((doa[:, :, 3] > 0.5).float().mean(dim=2) > 0.5).sum(dim=1)
    correct = (detected == present.sum(dim=1)).float()

    separation = prediction_separation.gather(
        1, chosen[:, :, None, None, None].expand(-1, -1, *prediction_separation.shape[2:]))
    estimate = to_waveform(separation)
    truth = to_waveform(batch["target"])
    mixture = batch["mixture"]
    unprocessed = torch.istft(
        torch.complex(mixture[:, 0], mixture[:, mixture.shape[1] // 2]),
        n_fft=N_FFT, hop_length=HOP, center=True,
        window=torch.hann_window(N_FFT, device=mixture.device, dtype=mixture.dtype))
    length = min(estimate.shape[-1], truth.shape[-1], unprocessed.shape[-1])
    after = si_sdr(estimate[..., :length], truth[..., :length])
    before = si_sdr(unprocessed[:, None, :length].expand_as(truth[..., :length]),
                    truth[..., :length])
    return (per_source[present].detach().cpu(), present.sum(dim=1).cpu(), correct.cpu(),
            after[present].detach().cpu(), (after - before)[present].detach().cpu())


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals, count = {}, 0
    degrees, counts, correct, sdr, gain = [], [], [], [], []
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast("cuda", enabled=scaler is not None):
            separation, doa = model(batch["mixture"])
        doa, separation = doa.float(), separation.float()
        loss, parts, assignment, orders = joint_loss(doa, separation, batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        parts["total"] = loss.item()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        with torch.no_grad():
            error, present, hit, ratio, delta = evaluate_batch(
                doa, separation, batch, assignment, orders)
        degrees.append(error)
        counts.append(present)
        correct.append(hit)
        sdr.append(ratio)
        gain.append(delta)

    degrees = torch.cat(degrees)
    counts = torch.cat(counts)
    correct = torch.cat(correct)
    sdr = torch.cat(sdr)
    gain = torch.cat(gain)
    per_count = {}
    offset = 0
    for present in counts.tolist():
        per_count.setdefault(present, []).extend(degrees[offset:offset + present].tolist())
        offset += present
    error = degrees.numpy()
    metrics = {"mae": float(error.mean()) if len(error) else float("nan"),
               "med": float(np.median(error)) if len(error) else float("nan"),
               "p90": float(np.percentile(error, 90)) if len(error) else float("nan"),
               "count_accuracy": float(correct.mean()),
               "si_sdr": float(sdr.mean()),
               "si_sdr_gain": float(gain.mean()),
               "per_count": {k: (float(np.mean(v)), float(np.median(v)),
                                 float(np.percentile(v, 90)))
                             for k, v in sorted(per_count.items()) if v}}
    return {key: value / count for key, value in totals.items()}, metrics


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = LoaderConfig(crop_seconds=args.crop_seconds, target_key=args.target)

    datasets = {}
    for split, train in (("train", True), ("dev", False)):
        dataset = MultiSourceDataset(args.data_root / split, config, train=train)
        if args.limit:
            dataset.paths = dataset.paths[: args.limit]
        datasets[split] = dataset
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch, shuffle=(split == "train"),
                          num_workers=args.workers, collate_fn=collate,
                          pin_memory=True, drop_last=(split == "train"))
        for split, dataset in datasets.items()
    }

    model = JointNet(NetConfig(hidden=args.hidden, blocks=args.blocks,
                               max_sources=config.max_sources)).to(device)
    model.trunk.checkpoint = args.checkpoint
    decayed = [p for n, p in model.named_parameters() if p.ndim > 1]
    plain = [p for n, p in model.named_parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [{"params": decayed, "weight_decay": 1e-2}, {"params": plain, "weight_decay": 0.0}],
        lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device == "cuda" else None

    args.output.mkdir(parents=True, exist_ok=True)
    start_epoch, best = 1, float("inf")
    last_path = args.output / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        schedule.load_state_dict(state["schedule_state"])
        if scaler is not None and state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        print(f"{last_path}에서 재개 — epoch {start_epoch}부터, 최고 dev {best:.4f}")

    print(f"{device}  파라미터 {parameter_count(model) / 1e6:.2f}M  "
          f"train {len(datasets['train'])} / dev {len(datasets['dev'])}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_parts, _ = run_epoch(model, loaders["train"], device, optimizer, scaler)
        with torch.no_grad():
            dev_parts, dev_metrics = run_epoch(model, loaders["dev"], device)
        schedule.step()

        breakdown = "  ".join(f"{k}음원 {v[0]:.1f}/{v[1]:.1f}/{v[2]:.1f}"
                              for k, v in dev_metrics["per_count"].items())
        print(f"epoch {epoch:3d}  {time.time() - started:5.0f}s  "
              f"train {train_parts['total']:.4f} (방향 {train_parts['direction']:.4f} "
              f"활성 {train_parts['activity']:.4f} 분리 {train_parts['separation']:.4f})  "
              f"dev {dev_parts['total']:.4f}\n"
              f"            MAE {dev_metrics['mae']:5.1f}도  MED {dev_metrics['med']:5.1f}도  "
              f"P90 {dev_metrics['p90']:5.1f}도  [{breakdown}]  "
              f"개수 {100 * dev_metrics['count_accuracy']:.1f}%  "
              f"SI-SDR {dev_metrics['si_sdr']:+.2f}dB (개선 {dev_metrics['si_sdr_gain']:+.2f}dB)",
              flush=True)

        if dev_parts["total"] < best:
            best = dev_parts["total"]
            torch.save({"config": model.config.__dict__, "model_state": model.state_dict(),
                        "epoch": epoch, "dev_loss": best, "metrics": dev_metrics},
                       args.output / "best.pt")
        torch.save({"config": model.config.__dict__, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "schedule_state": schedule.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler is not None else None,
                    "epoch": epoch, "best": best}, last_path)
    print(f"-> {args.output / 'best.pt'}  최고 dev {best:.4f}")


if __name__ == "__main__":
    main()
