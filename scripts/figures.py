"""Draw the figures of the README from the files in results/.

usage: python scripts/figures.py [results_dir]
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
STYLE = {
    "real_test": ("real held-out", "black", "-"),
    "g_shell": ("G shell", "tab:blue", "-"),
    "g_mode": ("G mode", "tab:green", "-"),
    "g_mode_ddim": ("G mode, exact denoiser, DDIM 50", "tab:gray", "--"),
    "diffusion": ("diffusion, DDIM 50", "tab:red", "-"),
    "diffusion_long": ("diffusion, DDIM 250", "tab:orange", "--"),
}
KMAX = 21


def load(name):
    return json.loads((RESULTS / name).read_text())


def spectra_figure(diag):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    k = np.arange(1, 33)
    real = np.array(diag["sources"]["real_test"]["spectrum"])
    for name, (label, color, ls) in STYLE.items():
        e = np.array(diag["sources"][name]["spectrum"])
        axes[0].loglog(k, e, color=color, ls=ls, label=label)
        if name != "real_test":
            axes[1].semilogy(
                k[:KMAX], (e / real)[:KMAX], color=color, ls=ls, label=label
            )
    axes[0].loglog(
        k[3:21], 0.06 * (k[3:21] / 4.0) ** -3.0, "k:", lw=0.8, label="$k^{-3}$"
    )
    for ax in axes:
        ax.axvline(KMAX, color="0.7", lw=0.8)
        ax.set_xlabel("wavenumber k")
    axes[0].set_ylabel("E(k)")
    axes[0].set_ylim(1e-12, 1)
    axes[1].axhline(1.0, color="black", lw=0.8)
    axes[1].set_ylabel("E(k) / E(k) of the held-out set")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "spectra.png", dpi=100)


def fields_figure():
    ex = np.load(RESULTS / "example_fields.npz")
    rows = [("real_test", "real"), ("g_mode", "G mode"), ("diffusion", "diffusion")]
    scale = 3.5 * float(ex["real_test"].astype(np.float32).std())
    fig, axes = plt.subplots(3, 5, figsize=(9, 5.8))
    for r, (name, label) in enumerate(rows):
        for c in range(5):
            ax = axes[r, c]
            ax.imshow(
                ex[name][c].astype(np.float32), cmap="RdBu_r", vmin=-scale, vmax=scale
            )
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(label)
    fig.tight_layout()
    fig.savefig(RESULTS / "fields.png", dpi=80)


def statistics_figure(diag):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    seps = np.array([1, 2, 4, 8, 16])
    for name, (label, color, ls) in STYLE.items():
        if name in ("g_shell", "g_mode_ddim"):
            continue
        src = diag["sources"][name]
        h = src["histogram"]
        axes[0].semilogy(
            h["centres"],
            np.maximum(h["density"], 1e-8),
            color=color,
            ls=ls,
            label=label,
        )
        s2 = (
            np.array(src["structure_functions"]["S2"]["x"])
            + np.array(src["structure_functions"]["S2"]["y"])
        ) / 2
        s4 = (
            np.array(src["structure_functions"]["S4"]["x"])
            + np.array(src["structure_functions"]["S4"]["y"])
        ) / 2
        axes[1].semilogx(
            seps, s4 / s2**2, color=color, ls=ls, marker="o", ms=3, label=label
        )
    axes[0].set_xlabel("vorticity / rms of the training set")
    axes[0].set_ylabel("probability density")
    axes[0].set_ylim(1e-6, 2)
    axes[1].set_xlabel("separation r (grid points)")
    axes[1].set_ylabel("flatness S4 / S2$^2$ of increments")
    axes[1].axhline(3.0, color="0.7", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "statistics.png", dpi=100)


def consistency_figure(cons):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    k = np.arange(1, KMAX + 1)
    for name, (label, color, ls) in STYLE.items():
        ratio = np.array(cons["sources"][name]["spectrum_ratio_after_over_before"])[
            :KMAX
        ]
        ax.semilogy(k, ratio, color=color, ls=ls, label=label)
    ax.axhline(1.0, color="0.7", lw=0.8)
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("E(k) after / before one turnover time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "consistency.png", dpi=100)


def reconstruction_figure(rec):
    ex = np.load(RESULTS / "reconstruction_example.npz")
    fig = plt.figure(figsize=(11, 6.2))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.0])
    truth = ex["truth"][0].astype(np.float32)
    scale = 3.5 * truth.std()
    panels = [
        ("truth", truth),
        ("5% of the points", np.where(ex["mask"][0], truth, np.nan)),
        ("kriging mean", ex["kriging_mean"][0].astype(np.float32)),
        ("kriging sample", ex["kriging_sample"][0].astype(np.float32)),
        ("diffusion", ex["diffusion"][0].astype(np.float32)),
    ]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.92")
    for i, (title, img) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(img, cmap=cmap, vmin=-scale, vmax=scale)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    percents = [int(p) for p in rec["results"]]
    for j, (key, label) in enumerate(
        (
            ("relative_l2", "relative L2 error"),
            ("spectrum_error", "E(k) mean abs log ratio"),
        )
    ):
        ax = fig.add_subplot(grid[1, 1 + 2 * j : 3 + 2 * j] if j == 0 else grid[1, 3:5])
        for m, color in (
            ("kriging_mean", "tab:blue"),
            ("kriging_sample", "tab:green"),
            ("diffusion", "tab:red"),
        ):
            mean = [rec["results"][str(p)][m][key]["mean"] for p in percents]
            std = [rec["results"][str(p)][m][key]["std"] for p in percents]
            ax.errorbar(
                percents,
                mean,
                yerr=std,
                color=color,
                marker="o",
                ms=3,
                label=m.replace("_", " "),
            )
        ax.set_xscale("log")
        ax.set_xticks(percents, [str(p) for p in percents])
        ax.set_xlabel("observed grid points (%)")
        ax.set_ylabel(label)
        if j == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "reconstruction.png", dpi=100)


def training_figure():
    with open(RESULTS / "train_log.csv") as f:
        rows = list(csv.DictReader(f))
    step = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(step, [float(r["train_loss"]) for r in rows], label="train")
    ax.plot(
        step, [float(r["val_loss_ema"]) for r in rows], label="held-out, EMA weights"
    )
    ax.set_xlabel("step")
    ax.set_ylabel("noise prediction error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "training.png", dpi=100)


def main() -> None:
    if (RESULTS / "diagnostics.json").exists():
        diag = load("diagnostics.json")
        spectra_figure(diag)
        statistics_figure(diag)
        fields_figure()
    if (RESULTS / "consistency.json").exists():
        consistency_figure(load("consistency.json"))
    if (RESULTS / "reconstruction.json").exists():
        reconstruction_figure(load("reconstruction.json"))
    if (RESULTS / "train_log.csv").exists():
        training_figure()
    print(f"figures in {RESULTS}")


if __name__ == "__main__":
    main()
