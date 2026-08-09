#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from beamform import apply_mvdr, hybrid_weights, loaded_mixture_covariance, normalize_steering
from bem import BemDictionary, load_bem_dictionary, load_bem_steering
from dataset import MultichannelSample, ScatteringDataset
from doa import DOAEstimate, angular_error_deg, estimate_bem_doas
from stft import istft_single, normalized_spatial_covariance, stft_multichannel


BEM_TABLE = Path("sample_data/bem_table_reduced.h5")
N_FFT, HOP = 512, 128
GRID_STEP_DEG = 5.0
DOA_MIN_HZ, DOA_MAX_HZ = 1000.0, 5000.0
MVDR_MIN_HZ = 1250.0
DIAGONAL_LOADING = 1e-2
MIN_SEPARATION_DEG = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4채널 WAV -> DOA 추정 -> 빔포밍 -> 모노 WAV")
    parser.add_argument("input", type=Path, help="WAV 파일 또는 WAV/NPZ 쌍이 있는 디렉터리")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs"))
    parser.add_argument("-n", "--sources", type=int, default=1, help="찾을 음원 개수")
    parser.add_argument("-l", "--limit", type=int, default=0, help="디렉터리 입력일 때 개수 제한")
    return parser.parse_args()


def separate(sample: MultichannelSample, dictionary: BemDictionary, n_sources: int
             ) -> list[tuple[DOAEstimate, np.ndarray]]:
    sample_rate = sample.metadata.sample_rate
    frequencies, spectra = stft_multichannel(sample.audio, sample_rate, n_fft=N_FFT, hop=HOP)
    doa_band = (frequencies >= DOA_MIN_HZ) & (frequencies <= DOA_MAX_HZ)
    estimates = estimate_bem_doas(
        normalized_spatial_covariance(spectra[:, doa_band, :]),
        dictionary,
        max_sources=n_sources,
        min_separation_deg=MIN_SEPARATION_DEG,
    )

    mixture_csm = loaded_mixture_covariance(spectra, diagonal_loading=DIAGONAL_LOADING)
    mvdr_mask = frequencies >= MVDR_MIN_HZ

    outputs = []
    for estimate in estimates:
        _, steering = load_bem_steering(BEM_TABLE, sample.metadata.panel_indices,
                                        estimate.doa_index, sample_rate=sample_rate, n_fft=N_FFT)
        weights = hybrid_weights(normalize_steering(steering), mixture_csm, mvdr_mask=mvdr_mask)
        waveform = istft_single(apply_mvdr(spectra, weights), sample_rate,
                                n_fft=N_FFT, hop=HOP, expected_samples=sample.audio.shape[0])
        outputs.append((estimate, waveform))
    return outputs


def main() -> None:
    args = parse_args()
    if args.input.is_dir():
        dataset = ScatteringDataset(args.input)
        sample_ids = list(dataset.sample_ids)
    elif args.input.suffix.lower() == ".wav":
        dataset = ScatteringDataset(args.input.parent)
        sample_ids = [args.input.stem]
    else:
        raise SystemExit(f"WAV 파일이나 디렉터리를 지정해라: {args.input}")
    if args.limit:
        sample_ids = sample_ids[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)

    first = dataset.load(sample_ids[0])
    dictionary = load_bem_dictionary(BEM_TABLE, first.metadata.panel_indices,
                                     sample_rate=first.metadata.sample_rate, n_fft=N_FFT,
                                     min_hz=DOA_MIN_HZ, max_hz=DOA_MAX_HZ,
                                     grid_step_deg=GRID_STEP_DEG)

    for sample_id in sample_ids:
        sample = dataset.load(sample_id)
        if not np.array_equal(sample.metadata.panel_indices, dictionary.panel_indices):
            raise SystemExit(f"{sample_id}: 채널 배치가 사전과 다르다 {sample.metadata.panel_indices}")

        results = separate(sample, dictionary, args.sources)
        record = {"sample_id": sample_id, "sample_rate": sample.metadata.sample_rate,
                  "n_sources": len(results), "sources": []}
        print(f"{sample_id}  음원 {len(results)}개")
        for index, (estimate, waveform) in enumerate(results):
            error = angular_error_deg(estimate.direction, sample.metadata.direction_unit)
            record["sources"].append({
                "index": index,
                "azimuth_deg": estimate.azimuth_deg,
                "elevation_deg": estimate.elevation_deg,
                "score": estimate.score,
                "doa_index": estimate.doa_index,
                "angular_error_deg": error,
            })
            suffix = "" if args.sources == 1 else f"_src{index}"
            sf.write(args.output / f"{sample_id}{suffix}.wav", waveform,
                     sample.metadata.sample_rate, subtype="FLOAT")
            print(f"  src{index}  az={estimate.azimuth_deg:6.1f}  el={estimate.elevation_deg:6.1f}  "
                  f"오차 {error:5.1f}도")
        (args.output / f"{sample_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
