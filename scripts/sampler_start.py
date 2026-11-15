"""Compare where the DDIM chain starts, on the validation trajectory.

Writes results/sampler_start.json. The start is the last timestep whose signal
level alpha_bar is at least the given value; 0 starts at the end of the schedule.
"""

import argparse
import math
from pathlib import Path

import torch

from turbdiff import spectra
from turbdiff.data import load, merge_json
from turbdiff.diffusion import Schedule, sample_batches, start_index
from turbdiff.unet import UNet

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (0.0, 1e-6, 1e-4, 1e-3, 1e-2)


@torch.no_grad()
def first_estimate_rms(model, sched, level, n, device, seed):
    """Rms of the clean estimate at the first step of a chain from pure noise."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((100, 1, n, n), generator=g, device=device)
    t = start_index(sched, level)
    ab = sched.alphas_bar[t].item()
    eps = model(x, torch.full((100,), t, device=device)).float()
    x0 = (x - math.sqrt(1 - ab) * eps) / math.sqrt(ab)
    return x0.pow(2).mean().sqrt().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "turb.npz")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--base", type=int, default=48)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(args.data)
    n = data["train"].shape[-1]
    val = data["val"].reshape(-1, n, n).double() / data["sigma"]
    ref = spectra.energy_spectrum(val).mean(0)
    sched = Schedule()
    model = UNet(base=args.base).to(device)
    model.load_state_dict(torch.load(args.work / "ema.pt", map_location=device))
    model.eval()

    rows = {}
    for level in LEVELS:
        x = sample_batches(
            model, sched, args.samples, n, device, args.steps, 0, min_alpha_bar=level
        )
        e = spectra.energy_spectrum(x).mean(0)
        rows[str(level)] = {
            "start_timestep": start_index(sched, level),
            "first_clean_estimate_rms": first_estimate_rms(
                model, sched, level, n, device, 0
            ),
            "enstrophy_over_validation": (
                spectra.enstrophy(x).mean() / spectra.enstrophy(val).mean()
            ).item(),
            "mean_square_outside_cutoff": spectra.outside_fraction(x).mean().item(),
            "band_log_ratio_all": spectra.band_errors(e, ref)["all"],
            "max_abs_value": x.abs().max().item(),
        }
        print(level, rows[str(level)], flush=True)
    setup = {
        "samples": args.samples,
        "ddim_steps": args.steps,
        "reference": "validation",
    }
    merge_json(args.results / "sampler_start.json", {"setup": setup, "start": rows})


if __name__ == "__main__":
    main()
