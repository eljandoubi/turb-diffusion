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


def reflect_x(x: torch.Tensor) -> torch.Tensor:
    """Fields w(x, y) -> w(-x, y)."""
    return torch.roll(torch.flip(x, dims=(-1,)), 1, dims=-1)


def half_turn(x: torch.Tensor) -> torch.Tensor:
    """Fields w(x, y) -> w(-x, -y)."""
    return torch.roll(torch.flip(x, dims=(-2, -1)), (1, 1), dims=(-2, -1))


def shift_reflect(x: torch.Tensor, half_period: int) -> torch.Tensor:
    """Fields w(x, y) -> -w(-x, y + half_period), half the forcing period in y."""
    return -torch.roll(reflect_x(x), -half_period, dims=-2)


def symmetry_cosets(period_y: int) -> list:
    """The four maps that, with the shifts, generate the symmetries of the flow.

    The forcing cos(kf y) is unchanged by the half turn and by reflecting x while
    shifting y by half a forcing period (which also flips the sign of w), so the
    equations and their statistics are invariant under both, as under the shifts.
    """
    half = period_y // 2
    return [
        lambda x: x,
        half_turn,
        lambda x: shift_reflect(x, half),
        lambda x: shift_reflect(half_turn(x), half),
    ]


def random_symmetries(
    x: torch.Tensor, period_y: int, generator: torch.Generator
) -> torch.Tensor:
    """Apply a random symmetry of the flow to each field of (B, 1, n, n).

    The symmetries are the shifts along x, the shifts along y by multiples of
    the forcing period, the half turn, and the reflection with a half-period
    shift in y.
    """
    b, _, n, _ = x.shape
    dev = x.device
    coset = torch.randint(0, 4, (b,), generator=generator, device=dev)
    out = x
    for i, fn in enumerate(symmetry_cosets(period_y)[1:], start=1):
        out = torch.where((coset == i)[:, None, None, None], fn(x), out)
    sy = torch.randint(0, n // period_y, (b,), generator=generator, device=dev)
    sx = torch.randint(0, n, (b,), generator=generator, device=dev)
    ar = torch.arange(n, device=dev)
    rows = (ar[None, :] - (sy * period_y)[:, None]) % n
    cols = (ar[None, :] - sx[:, None]) % n
    index = torch.arange(b, device=dev)[:, None, None]
    return out[index, 0, rows[:, :, None], cols[:, None, :]][:, None]
