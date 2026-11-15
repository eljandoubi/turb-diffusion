"""Print the markdown tables of the README from the files in results/.

usage: python scripts/report.py [results_dir]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
SOURCES = ["real_test", "real_train", "g_shell", "g_mode", "g_mode_ddim", "diffusion"]
STEP_SOURCES = [
    "g_mode_ddim",
    "g_mode_ddim_long",
    "diffusion",
    "diffusion_long",
    "diffusion_last",
]
TITLES = {
    "real_test": "real held-out",
    "real_train": "real train",
    "g_shell": "G shell",
    "g_mode": "G mode",
    "g_mode_ddim": "G mode, DDIM oracle",
    "g_mode_ddim_long": "G mode, DDIM oracle long",
    "diffusion": "diffusion",
    "diffusion_long": "diffusion, DDIM long",
    "diffusion_last": "diffusion, last step",
}


def g(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}g}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def solver_table(info: dict) -> str:
    p, t, s = info["parameters"], info["turnover"], info["stationarity"]
    rows = [
        ["grid", f"{p['n']} x {p['n']}, 2/3 rule (round(|k|) <= 21 kept)"],
        ["viscosity, drag", f"{g(p['nu'])}, {g(p['mu'])}"],
        ["forcing", f"{g(p['forcing_amplitude'])} cos({p['forcing_wavenumber']} y)"],
        ["time step", f"{g(p['dt'])} ({p['scheme']})"],
        ["rms vorticity", g(t["omega_rms"])],
        ["eddy turnover time", g(t["eddy_turnover_time"])],
        ["snapshot spacing", f"{g(t['snapshot_time'])} ({t['snapshot_steps']} steps)"],
        [
            "mean energy",
            f"{g(s['energy_mean'])} (std {g(s['energy_std_over_snapshots'])})",
        ],
        [
            "energy, first and second half",
            f"{g(s['first_half_mean'])}, {g(s['second_half_mean'])}",
        ],
    ]
    return table(["quantity", "value"], rows)


def diagnostics_table(diag: dict) -> str:
    src = diag["sources"]
    names = [s for s in SOURCES if s in src]
    rows = []
    for band in ("low", "mid", "high", "all"):
        rows.append(
            [f"E(k) mean abs log ratio, {band}"]
            + [g(src[s]["band_log_ratio"][band]) for s in names]
        )
    rows.append(["enstrophy mean"] + [g(src[s]["enstrophy_mean"]) for s in names])
    rows.append(["enstrophy std"] + [g(src[s]["enstrophy_std"]) for s in names])
    rows.append(["skewness"] + [g(src[s]["skewness"]) for s in names])
    rows.append(["excess kurtosis"] + [g(src[s]["excess_kurtosis"]) for s in names])
    rows.append(
        ["KS distance of the values"] + [g(src[s]["ks_distance"]) for s in names]
    )
    for order in ("S2", "S4"):
        rows.append(
            [f"{order} mean abs log ratio"]
            + [g(src[s]["structure_log_ratio"][order]) for s in names]
        )
    rows.append(
        ["forced-mode coherence"] + [g(src[s]["forced_mode_coherence"]) for s in names]
    )
    rows.append(
        ["mean square outside k=21"]
        + [g(src[s]["mean_square_outside_cutoff"]) for s in names]
    )
    for key, label in (
        ("plain_median", "nearest train, plain, median"),
        ("shifted_median", "nearest train, shifted, median"),
        ("shifted_p05", "nearest train, shifted, 5th pct"),
    ):
        rows.append(
            [label]
            + [
                g(src[s]["nearest_training_distance"][key])
                if "nearest_training_distance" in src[s]
                else "-"
                for s in names
            ]
        )
    return table(["metric"] + [TITLES[s] for s in names], rows)


def consistency_table(cons: dict) -> str:
    src = cons["sources"]
    names = [s for s in SOURCES if s in src]
    rows = []
    for s in names:
        c = src[s]
        rows.append(
            [TITLES[s]]
            + [g(c["band_log_ratio"][b]) for b in ("low", "mid", "high", "all")]
            + [g(c["enstrophy_relative_change"]), g(c["energy_relative_change"])]
        )
    header = ["set", "low", "mid", "high", "all", "enstrophy change", "energy change"]
    return table(header, rows)


def reconstruction_table(rec: dict) -> str:
    names = [
        ("kriging_mean", "kriging mean"),
        ("kriging_sample", "kriging sample"),
        ("diffusion", "diffusion"),
    ]
    rows = []
    for percent, entry in rec["results"].items():
        row = [f"{percent}% ({entry['points']} points)"]
        for key in ("relative_l2", "spectrum_error"):
            for m, _ in names:
                v = entry[m][key]
                row.append(f"{g(v['mean'])} +- {g(v['std'], 2)}")
        rows.append(row)
    header = (
        ["observed"] + [f"L2 {t}" for _, t in names] + [f"E(k) {t}" for _, t in names]
    )
    return table(header, rows)


def steps_table(diag: dict, cons: dict) -> str:
    rows = []
    for s in STEP_SOURCES:
        if s not in diag["sources"]:
            continue
        d, c = diag["sources"][s], cons["sources"][s]
        steps = diag["setup"]["long_ddim_steps" if s.endswith("long") else "ddim_steps"]
        rows.append(
            [
                TITLES[s].replace("long", str(steps)),
                g(d["band_log_ratio"]["all"]),
                g(d["enstrophy_mean"]),
                g(d["structure_log_ratio"]["S4"]),
                g(d["forced_mode_coherence"]),
                g(c["band_log_ratio"]["all"]),
            ]
        )
    header = ["set", "E(k) all", "enstrophy mean", "S4", "coherence", "relaxation"]
    return table(header, rows)


def main() -> None:
    for title, name, fn in (
        ("Solver", "solver.json", solver_table),
        ("Diagnostics", "diagnostics.json", diagnostics_table),
        ("Consistency", "consistency.json", consistency_table),
        ("Reconstruction", "reconstruction.json", reconstruction_table),
    ):
        path = RESULTS / name
        if path.exists():
            print(f"### {title}\n")
            print(fn(json.loads(path.read_text())))
            print()
    both = (RESULTS / "diagnostics.json", RESULTS / "consistency.json")
    if all(p.exists() for p in both):
        print("### DDIM steps\n")
        print(steps_table(*(json.loads(p.read_text()) for p in both)))
        print()
    train = RESULTS / "train.json"
    if train.exists():
        t = json.loads(train.read_text())
        print("### Training\n")
        print(
            f"{t['parameters']:,} parameters, {t['steps']} steps of batch "
            f"{t['batch']} on {t['device']} in "
            f"{g(t['wall_clock_seconds'] / 60)} minutes, final train loss "
            f"{g(t['final_train_loss'])}, held-out loss "
            f"{g(t['final_val_loss_ema'])}."
        )


if __name__ == "__main__":
    main()
