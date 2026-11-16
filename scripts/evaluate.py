"""Sample from the baselines and from the trained model, then run the diagnostics.

Writes results/diagnostics.json, results/consistency.json and
results/example_fields.npz. Needs data/turb.npz and work/ema.pt (see train.py).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from turbdiff import diagnostics as dg
from turbdiff import spectra
from turbdiff.data import load, merge_json
from turbdiff.diffusion import Schedule, sample_batches
from turbdiff.gaussian import (
    GaussianDenoiser,
    GaussianField,
    noise_prediction_error,
)
from turbdiff.solver import Params, Solver
from turbdiff.unet import UNet

ROOT = Path(__file__).resolve().parents[1]
PARAMS = Params()


def sample_chain(model, sched, count, n, sigma, device, steps, seed):
    """DDIM samples in physical units from a noise predictor."""
    return sample_batches(model, sched, count, n, device, steps, seed) * sigma


def load_model(path, base, forcing, device):
    """The EMA weights of a trained network, in eval mode."""
    model = UNet(base=base, forcing_wavenumber=PARAMS.kf if forcing else 0)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def describe(x, real, train, ref, device, in_train=False):
    """Diagnostics of a set of fields against the real held-out set."""
    x = x.double()
    spectrum = spectra.energy_spectrum(x).mean(0)
    sf = dg.structure_functions(x)
    ens = spectra.enstrophy(x)
    centres, density = dg.histogram(x, ref["sigma"])
    out = {
        "spectrum": spectrum.tolist(),
        "band_log_ratio": spectra.band_errors(spectrum, ref["spectrum"]),
        "mean_square_outside_cutoff": spectra.outside_fraction(x).mean().item(),
        **dg.moments(x),
        "ks_distance": dg.ks_distance(x, real),
        "enstrophy_mean": ens.mean().item(),
        "enstrophy_std": ens.std().item(),
        "structure_functions": sf,
        "structure_log_ratio": dg.structure_error(sf, ref["structure"]),
        "forced_mode_coherence": dg.forced_mode_coherence(x, PARAMS.kf),
        "histogram": {"centres": centres, "density": density},
    }
    if not in_train:
        plain, shifted = dg.nearest_distance(x.to(device), train, PARAMS.n // PARAMS.kf)
        out["nearest_training_distance"] = {
            "plain_median": plain.median().item(),
            "plain_p05": plain.quantile(0.05).item(),
            "shifted_median": shifted.median().item(),
            "shifted_p05": shifted.quantile(0.05).item(),
        }
    return out


def relaxation(x, solver, time_span, batch=250):
    """Change of the ensemble spectrum when the fields are evolved by the solver."""
    after = torch.cat(
        [
            solver.evolve(x[i : i + batch], time_span).cpu()
            for i in range(0, len(x), batch)
        ]
    )
    e0 = spectra.energy_spectrum(x).mean(0)
    e1 = spectra.energy_spectrum(after).mean(0)
    z0, z1 = spectra.enstrophy(x).mean(), spectra.enstrophy(after).mean()
    k0, k1 = spectra.energy(x).mean(), spectra.energy(after).mean()
    return {
        "band_log_ratio": spectra.band_errors(e1, e0),
        "spectrum_ratio_after_over_before": (e1 / e0).tolist(),
        "enstrophy_relative_change": ((z1 - z0) / z0).item(),
        "energy_relative_change": ((k1 - k0) / k0).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "turb.npz")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--long-steps", type=int, default=250)
    parser.add_argument("--base", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forcing", action="store_true", help="forcing channel")
    parser.add_argument("--early", type=Path, default=None, help="earlier checkpoint")
    parser.add_argument("--plain", type=Path, default=None, help="no-forcing model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(args.data)
    sigma, n = data["sigma"], data["train"].shape[-1]
    train = data["train"].reshape(-1, n, n).double()
    real = data["test"].reshape(-1, n, n).double()[: args.samples]
    sched = Schedule()
    start = time.time()

    sources = {
        "real_test": real,
        "real_train": train[:: max(1, len(train) // len(real))],
    }
    gens = {
        "g_shell": GaussianField.fit(train, shell=True),
        "g_mode": GaussianField.fit(train),
    }
    for i, (name, field) in enumerate(gens.items()):
        g = torch.Generator().manual_seed(args.seed + 1 + i)
        sources[name] = field.sample(args.samples, g)
    oracle = GaussianDenoiser(gens["g_mode"].power / sigma**2, sched)
    val = data["val"].reshape(-1, n, n).double() / sigma
    val = torch.cat([val, val])
    noise = torch.randn(
        (len(val), 1, n, n), generator=torch.Generator().manual_seed(12345)
    )
    t = (torch.arange(len(val)) * 397) % sched.steps
    gaussian_loss = noise_prediction_error(
        gens["g_mode"].power / sigma**2, sched, val, t, noise[:, 0]
    )
    model = load_model(args.work / "ema.pt", args.base, args.forcing, device)
    for name, net, steps in (
        ("g_mode_ddim", oracle, args.steps),
        ("g_mode_ddim_long", oracle, args.long_steps),
        ("diffusion", model, args.steps),
        ("diffusion_long", model, args.long_steps),
    ):
        sources[name] = sample_chain(
            net, sched, args.samples, n, sigma, device, steps, args.seed
        )
    others = {"diffusion_early": (args.early, args.forcing)}
    others["diffusion_plain"] = (args.plain, False)
    for name, (path, forcing) in others.items():
        if path is not None:
            net = load_model(path, args.base, forcing, device)
            sources[name] = sample_chain(
                net, sched, args.samples, n, sigma, device, args.steps, args.seed
            )
    sampling_seconds = time.time() - start

    ref_spectrum = spectra.energy_spectrum(real).mean(0)
    ref = {
        "sigma": sigma,
        "spectrum": ref_spectrum,
        "structure": dg.structure_functions(real),
    }
    train_dev = train.to(device)
    results = {
        name: describe(x, real, train_dev, ref, device, in_train=name == "real_train")
        for name, x in sources.items()
    }
    solver = Solver(PARAMS, device)
    solver_info = json.loads((args.results / "solver.json").read_text())
    span = float(solver_info["turnover"]["snapshot_time"])
    consistency = {name: relaxation(x, solver, span) for name, x in sources.items()}

    setup = {
        "samples": args.samples,
        "ddim_steps": args.steps,
        "long_ddim_steps": args.long_steps,
        "sampler_seed": args.seed,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "sigma": sigma,
        "gaussian_validation_loss": gaussian_loss,
        "sampling_seconds": sampling_seconds,
        "evolution_time": span,
        "sources": {
            "real_test": "held-out trajectories",
            "real_train": "every fourth training field",
            "g_shell": "Gaussian field, shell-averaged power of the training set",
            "g_mode": "Gaussian field, per-mode power of the training set",
            "g_mode_ddim": "g_mode through the exact Gaussian denoiser and DDIM",
            "g_mode_ddim_long": "the same with more DDIM steps",
            "diffusion": "periodic UNet, EMA weights, DDIM",
            "diffusion_long": "the same network with more DDIM steps",
            "diffusion_early": f"EMA weights of {args.early}, DDIM",
            "diffusion_plain": f"network without the forcing channel, {args.plain}",
        },
        "forcing_channel": args.forcing,
    }
    merge_json(args.results / "diagnostics.json", {"setup": setup, "sources": results})
    merge_json(
        args.results / "consistency.json", {"setup": setup, "sources": consistency}
    )
    keep = torch.linspace(0, len(real) - 1, 6).long()
    examples = {
        name: x[keep].float().numpy().astype(np.float16) for name, x in sources.items()
    }
    np.savez_compressed(args.results / "example_fields.npz", **examples)
    print(f"wrote diagnostics, consistency and example fields in {args.results}")


if __name__ == "__main__":
    main()
