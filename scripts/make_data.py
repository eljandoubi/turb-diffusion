"""Generate the trajectories with the pseudo-spectral solver and describe them.

Writes data/turb.npz (not tracked) and results/solver.json.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from turbdiff import spectra
from turbdiff.data import SEEDS, SNAPSHOTS, merge_json, run_trajectories
from turbdiff.solver import Params

ROOT = Path(__file__).resolve().parents[1]
PARAMS = Params()
SPIN_TIME = 100.0
MEASURE_TIME = 20.0


def lag_correlation(fields: np.ndarray, lag: int) -> float:
    a, b = fields[:, :-lag], fields[:, lag:]
    return float((a * b).mean() / (fields**2).mean())


def describe(runs: dict, split: dict, sigma: float, seconds: float) -> dict:
    train = torch.from_numpy(split["train"])
    energy = runs["energy"]
    half = energy.shape[1] // 2
    first, second = energy[:, :half].mean(), energy[:, half:].mean()
    e_spec = spectra.energy_spectrum(train.reshape(-1, *train.shape[-2:])).mean(0)
    peak = int(e_spec.argmax()) + 1
    coeff = torch.fft.fft2(train.double(), norm="forward")
    forced = coeff[..., PARAMS.kf, 0].reshape(-1)
    window = 20.0
    per_window = round(window / 0.5)
    spin_energy = runs["spin"][:, 0].mean(axis=1)
    return {
        "parameters": {
            "n": PARAMS.n,
            "nu": PARAMS.nu,
            "mu": PARAMS.mu,
            "forcing_amplitude": PARAMS.amp,
            "forcing_wavenumber": PARAMS.kf,
            "dt": PARAMS.dt,
            "scheme": "RK4 with integrating factor",
            "dealiasing": "2/3 rule, modes with round(|k|) <= 21 kept",
        },
        "turnover": {
            "omega_rms": runs["omega_rms"],
            "eddy_turnover_time": 1 / runs["omega_rms"],
            "snapshot_time": runs["snapshot_time"],
            "snapshot_steps": runs["snapshot_steps"],
            "spin_up_time": SPIN_TIME,
            "measurement_window": MEASURE_TIME,
        },
        "stationarity": {
            "energy_mean": float(energy.mean()),
            "energy_std_over_snapshots": float(energy.std()),
            "first_half_mean": float(first),
            "second_half_mean": float(second),
            "relative_drift": float(abs(second - first) / energy.mean()),
            "spin_up_energy_by_20_time_units": [
                float(spin_energy[i : i + per_window].mean())
                for i in range(0, len(spin_energy), per_window)
            ],
        },
        "dataset": {
            "train": list(split["train"].shape[:2]),
            "test": list(split["test"].shape[:2]),
            "val": list(split["val"].shape[:2]),
            "sigma": sigma,
            "mean_over_sigma": float(train.mean() / sigma),
            "seeds": {k: list(v) for k, v in SEEDS.items()},
        },
        "spectrum": {
            "peak_shell": peak,
            "energy_at_peak": float(e_spec[peak - 1]),
            "energy_at_k21_over_peak": float(e_spec[20] / e_spec[peak - 1]),
            "mean_square_outside_cutoff": float(spectra.outside_fraction(train).mean()),
        },
        "forced_mode": {
            "mean_amplitude": float(abs(forced.mean())),
            "rms_amplitude": float(forced.abs().pow(2).mean().sqrt()),
            "coherence": float(abs(forced.mean()) / forced.abs().pow(2).mean().sqrt()),
        },
        "correlation_between_snapshots": {
            f"lag_{lag}": lag_correlation(split["train"], lag)
            for lag in (1, 2, 3, 5, 10)
        },
        "generation_seconds": seconds,
        "torch_threads": torch.get_num_threads(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "turb.npz")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--spin-time", type=float, default=SPIN_TIME)
    args = parser.parse_args()

    seeds = sum(SEEDS.values(), ())
    most = max(SNAPSHOTS.values())
    start = time.time()
    runs = run_trajectories(PARAMS, seeds, args.spin_time, MEASURE_TIME, most)
    seconds = time.time() - start
    fields = runs["fields"]
    split, row = {}, 0
    for name, seeds_here in SEEDS.items():
        split[name] = fields[row : row + len(seeds_here), : SNAPSHOTS[name]]
        row += len(seeds_here)
    sigma = float(split["train"].std())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, sigma=sigma, **split)
    runs["energy"] = runs["energy"][: len(SEEDS["train"])]
    info = describe(runs, split, sigma, seconds)
    merge_json(args.results / "solver.json", info)
    print(f"wrote {args.out} and {args.results / 'solver.json'}")


if __name__ == "__main__":
    main()
