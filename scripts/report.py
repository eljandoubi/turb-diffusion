"""Print the markdown tables of the README from the files in results/.

usage: python scripts/report.py [results_dir]
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
SOURCES = [
    "real_test",
    "real_train",
    "g_shell",
    "g_mode",
    "g_mode_ddim",
    "diffusion",
    "diffusion_plain",
]
STEP_SOURCES = [
    "g_mode_ddim",
    "g_mode_ddim_long",
    "diffusion",
    "diffusion_long",
    "diffusion_early",
    "diffusion_plain",
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
    "diffusion_early": "diffusion, early checkpoint",
    "diffusion_plain": "diffusion, no forcing channel",
}


def titles(setup: dict) -> dict[str, str]:
    """Column titles, with the main diffusion network named after its input."""
    out = dict(TITLES)
    if setup.get("forcing_channel"):
        for key in ("diffusion", "diffusion_long", "diffusion_early"):
            out[key] = out[key].replace("diffusion", "diffusion, forcing channel", 1)
    return out


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
        ["grid", f"{p['n']} x {p['n']}, 2/3 rule (modes with round(abs(k)) <= 21)"],
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
        [
            "forced-mode coherence of the data",
            g(info["forced_mode"]["coherence"]),
        ],
        [
            "correlation of snapshots, lag 1 and 5",
            ", ".join(
                g(info["correlation_between_snapshots"][f"lag_{k}"]) for k in (1, 5)
            ),
        ],
    ]
    checks = info.get("checks", {})
    if "time_step" in checks:
        ts = checks["time_step"]
        rows.append(
            [
                "dt against dt / 2 after one turnover time",
                f"relative L2 {g(ts['max_relative_l2_difference'])}, "
                f"E(k) {g(ts['max_relative_difference_of_E_k'])}",
            ]
        )
    if "resolution" in checks:
        ratio = checks["resolution"]["ratio_data_over_fine_by_shell"]
        rows.append(
            [
                "E(k) of the data over E(k) of a 128 x 128 run",
                ", ".join(f"k={k}: {g(ratio[k - 1], 2)}" for k in (5, 10, 14, 18, 21)),
            ]
        )
    return table(["quantity", "value"], rows)


def flatness(sf: dict) -> float:
    """S4 / S2^2 of the increments at the smallest separation, mean of x and y."""
    s2 = (sf["S2"]["x"][0] + sf["S2"]["y"][0]) / 2
    s4 = (sf["S4"]["x"][0] + sf["S4"]["y"][0]) / 2
    return s4 / s2**2


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
        ["flatness S4 / S2^2 at r = 1"]
        + [g(flatness(src[s]["structure_functions"])) for s in names]
    )
    rows.append(
        ["forced-mode coherence"] + [g(src[s]["forced_mode_coherence"]) for s in names]
    )
    rows.append(
        ["share of mean square outside k=21"]
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
    return table(["metric"] + [titles(diag["setup"])[s] for s in names], rows)


def consistency_table(cons: dict) -> str:
    src = cons["sources"]
    names = [s for s in SOURCES if s in src]
    rows = []
    for s in names:
        c = src[s]
        rows.append(
            [titles(cons["setup"])[s]]
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
                titles(diag["setup"])[s].replace("long", str(steps)),
                g(d["band_log_ratio"]["all"]),
                g(d["enstrophy_mean"]),
                g(d["structure_log_ratio"]["S4"]),
                g(d["forced_mode_coherence"]),
                g(c["band_log_ratio"]["all"]),
            ]
        )
    header = ["set", "E(k) all", "enstrophy mean", "S4", "coherence", "relaxation"]
    return table(header, rows)


def start_table(start: dict) -> str:
    rows = []
    for level, s in start["start"].items():
        rows.append(
            [
                level,
                str(s["start_timestep"]),
                g(s["first_clean_estimate_rms"]),
                g(s["enstrophy_over_validation"]),
                g(s["mean_square_outside_cutoff"]),
                g(s["band_log_ratio_all"]),
            ]
        )
    header = [
        "alpha_bar at the start",
        "start timestep",
        "first clean estimate, rms",
        "enstrophy / validation",
        "share of mean square outside k=21",
        "E(k) all",
    ]
    return table(header, rows)


def read(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def networks_table() -> str | None:
    """The trained networks side by side, one column per network and seed."""
    diag, cons = read("diagnostics.json"), read("consistency.json")
    other = (
        read("diagnostics_forcing_seed1.json"),
        read("consistency_forcing_seed1.json"),
    )
    if not (diag and cons and all(other) and "diffusion_plain" in diag["sources"]):
        return None
    columns = (
        (
            "no forcing channel, seed 0",
            diag,
            cons,
            "diffusion_plain",
            "train.json",
            "reconstruction_plain.json",
        ),
        (
            "forcing channel, seed 0",
            diag,
            cons,
            "diffusion",
            "train_forcing.json",
            "reconstruction.json",
        ),
        (
            "forcing channel, seed 1",
            *other,
            "diffusion",
            "train_forcing_seed1.json",
            "reconstruction_forcing_seed1.json",
        ),
    )
    cells = []
    for _, dg, cs, key, train, rec in columns:
        d, c, t, r = dg["sources"][key], cs["sources"][key], read(train), read(rec)
        column = [g(t["selected_val_loss_ema"])]
        column += [g(d["band_log_ratio"][b]) for b in ("low", "mid", "high", "all")]
        column += [g(d[k]) for k in ("enstrophy_mean", "enstrophy_std")]
        column += [g(d["excess_kurtosis"]), g(d["ks_distance"])]
        column += [g(d["structure_log_ratio"][o]) for o in ("S2", "S4")]
        column += [g(flatness(d["structure_functions"]))]
        column += [g(d["forced_mode_coherence"]), g(c["band_log_ratio"]["all"])]
        column += [g(c["enstrophy_relative_change"]), g(c["energy_relative_change"])]
        column += [
            g(r["results"][p]["diffusion"]["relative_l2"]["mean"])
            for p in ("1", "2", "5", "10")
        ]
        cells.append(column)
    labels = [
        "validation noise error",
        "E(k) mean abs log ratio, low",
        "E(k) mean abs log ratio, mid",
        "E(k) mean abs log ratio, high",
        "E(k) mean abs log ratio, all",
        "enstrophy mean",
        "enstrophy std",
        "excess kurtosis",
        "KS distance of the values",
        "S2 mean abs log ratio",
        "S4 mean abs log ratio",
        "flatness S4 / S2^2 at r = 1",
        "forced-mode coherence",
        "E(k) change after one turnover time",
        "enstrophy change after one turnover time",
        "energy change after one turnover time",
        "reconstruction L2, 1% of the points",
        "reconstruction L2, 2% of the points",
        "reconstruction L2, 5% of the points",
        "reconstruction L2, 10% of the points",
    ]
    rows = [[label, *values] for label, *values in zip(labels, *cells, strict=True)]
    return table(["metric"] + [c[0] for c in columns], rows)


def read_log(name: str) -> list[dict[str, float]]:
    with open(RESULTS / name, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def crossing_line(label: str, rows: list[dict[str, float]]) -> str:
    """Where the training error lies above the validation error, in words."""
    steps = [int(r["step"]) for r in rows]
    above = [int(r["step"]) for r in rows if r["train_loss"] > r["val_loss_ema"]]
    head = f"- {label}: training error"
    if not above:
        return f"{head} below the validation error at every evaluation."
    last = max(above)
    inside = [s for s in steps if s < last and s not in above]
    text = f"{head} above the validation error up to step {last}"
    if inside:
        word = "steps" if len(inside) > 1 else "step"
        text += f", except at {word} " + ", ".join(map(str, inside))
    after = [s for s in steps if s > last]
    return text + (f", below it from step {after[0]} on." if after else ".")


def training_lines() -> list[str]:
    out, crossings = [], []
    for name, log, label in (
        ("train.json", "train_log.csv", "no forcing channel"),
        ("train_forcing.json", "train_log_forcing.csv", "forcing channel"),
        (
            "train_forcing_seed1.json",
            "train_log_forcing_seed1.csv",
            "forcing channel, seed 1",
        ),
    ):
        path = RESULTS / name
        if not path.exists():
            continue
        t = json.loads(path.read_text())
        out.append(
            f"- {label}: {t['parameters']:,} parameters, {t['steps']} steps of "
            f"batch {t['batch']} on {t['device']} in "
            f"{g(t['wall_clock_seconds'] / 60)} minutes, final train loss "
            f"{g(t['final_train_loss'])}, validation loss "
            f"{g(t['final_val_loss_ema'])}, selected step {t['selected_step']} "
            f"(validation loss {g(t['selected_val_loss_ema'])})."
        )
        if (RESULTS / log).exists():
            crossings.append(crossing_line(label, read_log(log)))
    diag = RESULTS / "diagnostics.json"
    if out and diag.exists():
        setup = json.loads(diag.read_text())["setup"]
        if "gaussian_validation_loss" in setup:
            out.append(
                "- exact Gaussian denoiser, same validation fields, timesteps and "
                f"noise: loss {g(setup['gaussian_validation_loss'])}."
            )
    return out + crossings


def main() -> None:
    for title, name, fn in (
        ("Solver", "solver.json", solver_table),
        ("Diagnostics", "diagnostics.json", diagnostics_table),
        ("Consistency", "consistency.json", consistency_table),
        ("Reconstruction", "reconstruction.json", reconstruction_table),
        ("Reconstruction", "reconstruction_plain.json", reconstruction_table),
        ("Sampler start, no forcing channel", "sampler_start.json", start_table),
        (
            "Sampler start, forcing channel",
            "sampler_start_forcing.json",
            start_table,
        ),
    ):
        path = RESULTS / name
        if path.exists():
            data = json.loads(path.read_text())
            if fn is reconstruction_table:
                title += ", " + (
                    "forcing channel"
                    if data["setup"].get("forcing_channel")
                    else "no forcing channel"
                )
            print(f"### {title}\n")
            print(fn(data))
            print()
    both = (RESULTS / "diagnostics.json", RESULTS / "consistency.json")
    if all(p.exists() for p in both):
        print("### Sampler and network variants\n")
        print(steps_table(*(json.loads(p.read_text()) for p in both)))
        print()
    networks = networks_table()
    if networks:
        print("### Networks and seeds\n")
        print(networks)
        print()
    lines = training_lines()
    if lines:
        print("### Training\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
