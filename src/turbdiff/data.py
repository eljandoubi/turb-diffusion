"""Generation and loading of the vorticity trajectories."""

import json
from pathlib import Path

import numpy as np
import torch

from turbdiff import spectra
from turbdiff.solver import Params, Solver, random_vorticity

SEEDS = {"train": (0, 1, 2, 3), "test": (100, 101), "val": (200,)}
SNAPSHOTS = {"train": 500, "test": 250, "val": 250}


def merge_json(path: Path, updates: dict) -> None:
    """Add or replace top-level keys of a JSON file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(updates)
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_trajectories(
    params: Params,
    seeds: tuple[int, ...],
    spin_time: float,
    measure_time: float,
    snapshots: int,
) -> dict:
    """Spin up from random fields, measure the turnover time, then take snapshots.

    Snapshots are one eddy turnover time 1 / omega_rms apart, with omega_rms the
    root-mean-square vorticity over the measurement window at the end of the
    spin-up. Returns the fields (trajectory, snapshot, n, n) and the time series.
    """
    solver = Solver(params)
    start = [
        random_vorticity((1,), params.n, 2.0, 6.0, torch.Generator().manual_seed(s))
        for s in seeds
    ]
    w_hat = solver.to_hat(torch.cat(start))
    chunk = round(0.5 / params.dt)
    series = []
    for _ in range(round((spin_time + measure_time) / (chunk * params.dt))):
        w_hat = solver.advance(w_hat, chunk)
        w = solver.to_phys(w_hat)
        series.append((spectra.energy(w), spectra.enstrophy(w)))
    spin = torch.stack([torch.stack(pair) for pair in series])
    measured = spin[-round(measure_time / (chunk * params.dt)) :, 1]
    omega_rms = float(torch.sqrt(2 * measured.mean()))
    every = round(1 / omega_rms / params.dt)
    fields = torch.zeros(len(seeds), snapshots, params.n, params.n)
    energy = torch.zeros(len(seeds), snapshots, dtype=torch.float64)
    for i in range(snapshots):
        w_hat = solver.advance(w_hat, every)
        w = solver.to_phys(w_hat)
        fields[:, i] = w.float()
        energy[:, i] = spectra.energy(w)
    return {
        "fields": fields.numpy(),
        "energy": energy.numpy(),
        "spin": spin.numpy(),
        "omega_rms": omega_rms,
        "snapshot_steps": every,
        "snapshot_time": every * params.dt,
    }


def load(path: Path) -> dict:
    """Load the npz written by scripts/make_data.py into torch tensors."""
    with np.load(path) as f:
        out = {k: torch.from_numpy(f[k]) for k in ("train", "test", "val")}
        out["sigma"] = float(f["sigma"])
    return out


def random_shifts(
    x: torch.Tensor, period_y: int, generator: torch.Generator
) -> torch.Tensor:
    """Shift each field of (B, 1, n, n) by a random circular amount.

    The flow statistics are invariant to any shift along x and to shifts along y
    by multiples of the forcing period, so only those shifts are drawn.
    """
    b, _, n, _ = x.shape
    dev = x.device
    sy = torch.randint(0, n // period_y, (b,), generator=generator, device=dev)
    sx = torch.randint(0, n, (b,), generator=generator, device=dev)
    ar = torch.arange(n, device=dev)
    rows = (ar[None, :] - (sy * period_y)[:, None]) % n
    cols = (ar[None, :] - sx[:, None]) % n
    picked = x[
        torch.arange(b, device=dev)[:, None, None],
        0,
        rows[:, :, None],
        cols[:, None, :],
    ]
    return picked[:, None]
