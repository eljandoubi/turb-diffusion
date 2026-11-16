"""Control with the training values: a Gaussian field mapped onto them.

Draws fields with the per-mode power of the training set (the "G mode" sample of
evaluate.py, same seed), maps their values onto the pooled distribution of the
training values (not the values of any network) and runs the diagnostics and the
solver test of evaluate.py on them. Writes results/copula.json. Needs
data/turb.npz and results/solver.json.
"""

import argparse
import json
from pathlib import Path

import torch
from evaluate import PARAMS, describe, relaxation

from turbdiff import diagnostics as dg
from turbdiff import spectra
from turbdiff.data import load, merge_json
from turbdiff.gaussian import GaussianField, match_values
from turbdiff.solver import Solver

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "turb.npz")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    data = load(args.data)
    n = data["train"].shape[-1]
    train = data["train"].reshape(-1, n, n).double()
    real = data["test"].reshape(-1, n, n).double()[: args.samples]
    ref = {
        "sigma": data["sigma"],
        "spectrum": spectra.energy_spectrum(real).mean(0),
        "structure": dg.structure_functions(real),
    }
    field = GaussianField.fit(train)
    gauss = field.sample(args.samples, torch.Generator().manual_seed(args.seed))
    copula = match_values(gauss, train)
    info = json.loads((args.results / "solver.json").read_text())
    span = float(info["turnover"]["snapshot_time"])
    merge_json(
        args.results / "copula.json",
        {
            "setup": {
                "samples": args.samples,
                "sampler_seed": args.seed,
                "values": "pooled values of the training fields",
                "evolution_time": span,
            },
            "sources": {
                "g_mode_copula": describe(copula, real, train, ref, "cpu", True)
            },
            "consistency": {
                "g_mode_copula": relaxation(copula, Solver(PARAMS, "cpu"), span)
            },
        },
    )
    print(f"wrote {args.results / 'copula.json'}")


if __name__ == "__main__":
    main()
