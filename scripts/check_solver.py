"""Check the solver on the generated data: time step and grid resolution.

Adds a "checks" section to results/solver.json. The resolution check runs a
second simulation on a 128 x 128 grid and takes several minutes.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from turbdiff import spectra
from turbdiff.data import load, merge_json, run_trajectories
from turbdiff.solver import Params, Solver

ROOT = Path(__file__).resolve().parents[1]
PARAMS = Params()


def time_step_check(fields: torch.Tensor, span: float) -> dict:
    """Evolve solver output fields with dt and dt / 2 and compare."""
    coarse = Solver(PARAMS).evolve(fields, span)
    fine = Solver(replace(PARAMS, dt=PARAMS.dt / 2)).evolve(fields, span)
    rel = (coarse - fine).flatten(1).norm(dim=1) / fine.flatten(1).norm(dim=1)
    e_c, e_f = spectra.energy_spectrum(coarse), spectra.energy_spectrum(fine)
    return {
        "fields": len(fields),
        "evolution_time": span,
        "max_relative_l2_difference": rel.max().item(),
        "max_relative_difference_of_E_k": ((e_c - e_f).abs() / e_f)[:, :21]
        .max()
        .item(),
    }


def resolution_check(reference: torch.Tensor, snapshots: int, spin: float) -> dict:
    """Compare the mean E(k) of the data with a run on a grid twice as fine."""
    fine = replace(PARAMS, n=2 * PARAMS.n)
    runs = run_trajectories(fine, (300, 301), spin, 20.0, snapshots)
    fields = torch.from_numpy(runs["fields"]).reshape(-1, fine.n, fine.n)
    e_fine = spectra.energy_spectrum(fields.double()).mean(0)[:21]
    e_data = spectra.energy_spectrum(reference.double()).mean(0)[:21]
    ratio = e_data / e_fine
    return {
        "fine_grid": fine.n,
        "fine_trajectories": 2,
        "fine_snapshots_each": snapshots,
        "fine_omega_rms": runs["omega_rms"],
        "ratio_data_over_fine_by_shell": ratio.tolist(),
        "max_abs_log_ratio_k_le_10": ratio[:10].log().abs().max().item(),
        "max_abs_log_ratio_k_le_14": ratio[:14].log().abs().max().item(),
        "ratio_at_k18": ratio[17].item(),
        "ratio_at_k21": ratio[20].item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "turb.npz")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--fine-snapshots", type=int, default=150)
    parser.add_argument("--fine-spin", type=float, default=60.0)
    args = parser.parse_args()

    data = load(args.data)
    n = PARAMS.n
    train = data["train"].reshape(-1, n, n)
    test = data["test"].reshape(-1, n, n)
    span = 1.0 / float(torch.sqrt(2 * spectra.enstrophy(train.double()).mean()))
    checks = {"time_step": time_step_check(test[:: len(test) // 4][:4].double(), span)}
    merge_json(args.results / "solver.json", {"checks": checks})
    print(checks["time_step"], flush=True)
    checks["resolution"] = resolution_check(train, args.fine_snapshots, args.fine_spin)
    merge_json(args.results / "solver.json", {"checks": checks})
    print(checks["resolution"])


if __name__ == "__main__":
    main()
