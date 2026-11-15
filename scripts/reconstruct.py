"""Reconstruct held-out fields from their values at a few random grid points.

Compares conditioning by replacement during DDIM sampling with Gaussian
conditioning (kriging) that uses the per-mode power of the training set.
Writes results/reconstruction.json and results/reconstruction_example.npz.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from turbdiff.data import load, merge_json
from turbdiff.diffusion import Schedule, ddim_sample
from turbdiff.gaussian import GaussianField, condition
from turbdiff.reconstruct import point_mask, random_points, relative_l2, spectrum_error
from turbdiff.unet import UNet

ROOT = Path(__file__).resolve().parents[1]
PERCENTS = (1, 2, 5, 10)
SEEDS = (0, 1, 2)


def run_diffusion(model, sched, truth, masks, sigma, steps, seed, fresh, device):
    """Reconstruct fields by replacement conditioning, in physical units."""
    g = torch.Generator(device=device).manual_seed(seed)
    values = (truth / sigma).float()[:, None].to(device)
    known = (masks[:, None].to(device), values)
    shape = tuple(values.shape)
    out = ddim_sample(
        model, sched, shape, g, steps=steps, known=known, fresh_noise=fresh
    )
    return out[:, 0].cpu().double() * sigma


def make_masks(n, count, fields, seed):
    g = torch.Generator().manual_seed(seed)
    index = [random_points(n, count, g) for _ in range(fields)]
    return index, torch.stack([point_mask(n, i) for i in index])


def summarise(values):
    t = torch.tensor(values)
    return {"mean": t.mean().item(), "std": t.std().item(), "per_seed": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "turb.npz")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--fields", type=int, default=100)
    parser.add_argument("--val-fields", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--base", type=int, default=48)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(args.data)
    sigma, n = data["sigma"], data["train"].shape[-1]
    train = data["train"].reshape(-1, n, n).double()
    test = data["test"].reshape(-1, n, n).double()
    val = data["val"].reshape(-1, n, n).double()
    test = test[:: len(test) // args.fields][: args.fields]
    val = val[:: len(val) // args.val_fields][: args.val_fields]
    gauss = GaussianField.fit(train)
    sched = Schedule()
    model = UNet(base=args.base).to(device)
    model.load_state_dict(torch.load(args.work / "ema.pt", map_location=device))
    model.eval()

    variants = {}
    for fresh, name in ((True, "fresh_noise"), (False, "predicted_noise")):
        errors = []
        for k, percent in enumerate(PERCENTS):
            count = round(percent / 100 * n * n)
            _, masks = make_masks(n, count, len(val), 9000 + k)
            est = run_diffusion(
                model, sched, val, masks, sigma, args.steps, 9000, fresh, device
            )
            errors.append(relative_l2(est, val).mean().item())
        variants[name] = {
            "relative_l2_by_percent": errors,
            "mean": float(np.mean(errors)),
        }
    chosen = min(variants, key=lambda v: variants[v]["mean"])
    variants["chosen"] = chosen

    results = {}
    examples = {}
    for k, percent in enumerate(PERCENTS):
        count = round(percent / 100 * n * n)
        rows = {m: {"relative_l2": [], "spectrum_error": []} for m in METHODS}
        for seed in SEEDS:
            index, masks = make_masks(n, count, len(test), 10 * seed + k)
            prior = gauss.sample(len(test), torch.Generator().manual_seed(500 + seed))
            est = {
                "kriging_mean": torch.stack(
                    [condition(gauss, i, t) for i, t in zip(index, test, strict=True)]
                ),
                "kriging_sample": torch.stack(
                    [
                        condition(gauss, i, t, p)
                        for i, t, p in zip(index, test, prior, strict=True)
                    ]
                ),
                "diffusion": run_diffusion(
                    model,
                    sched,
                    test,
                    masks,
                    sigma,
                    args.steps,
                    seed,
                    chosen == "fresh_noise",
                    device,
                ),
            }
            for name, field in est.items():
                rows[name]["relative_l2"].append(relative_l2(field, test).mean().item())
                rows[name]["spectrum_error"].append(
                    spectrum_error(field, test).mean().item()
                )
            if percent == 5 and seed == 0:
                keep = slice(0, 3)
                examples = {
                    "truth": test[keep],
                    "mask": masks[keep],
                    **{k: v[keep] for k, v in est.items()},
                }
        results[str(percent)] = {
            "points": count,
            **{m: {key: summarise(v) for key, v in rows[m].items()} for m in METHODS},
        }

    setup = {
        "test_fields": len(test),
        "seeds": list(SEEDS),
        "percent_of_grid_points": list(PERCENTS),
        "ddim_steps": args.steps,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "replacement": chosen,
        "validation_fields_for_variant": len(val),
    }
    merge_json(
        args.results / "reconstruction.json",
        {"setup": setup, "variant_selection": variants, "results": results},
    )
    np.savez_compressed(
        args.results / "reconstruction_example.npz",
        **{k: v.float().numpy().astype(np.float16) for k, v in examples.items()},
    )
    print(f"wrote {args.results / 'reconstruction.json'}")


METHODS = ("kriging_mean", "kriging_sample", "diffusion")

if __name__ == "__main__":
    main()
