"""The tables and lists of the README are the ones scripts/report.py prints."""

import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import pytest

from turbdiff.diffusion import ddim_sample

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report(monkeypatch):
    """The report script as a module, reading the results folder of the repo."""
    monkeypatch.setattr(sys, "argv", ["report.py"])
    spec = importlib.util.spec_from_file_location(
        "report", ROOT / "scripts" / "report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_tables_are_what_the_report_prints(report, capsys):
    report.main()
    printed = capsys.readouterr().out.splitlines()
    readme = set((ROOT / "README.md").read_text().splitlines())
    rows = [line for line in printed if line.startswith(("|", "- "))]
    assert len(rows) > 100
    assert [line for line in rows if line not in readme] == []


def log_rows(pairs):
    return [
        {"step": 250.0 * (i + 1), "train_loss": t, "val_loss_ema": v}
        for i, (t, v) in enumerate(pairs)
    ]


def test_crossing_line_names_the_steps_that_break_the_pattern(report):
    rows = log_rows([(3, 2), (1, 2), (3, 2), (1, 2), (1, 2)])
    assert report.crossing_line("run", rows) == (
        "- run: training error above the validation error up to step 750, "
        "except at step 500, below it from step 1000 on."
    )
    rows = log_rows([(3, 2), (1, 2), (1, 2), (3, 2), (3, 2)])
    assert report.crossing_line("run", rows) == (
        "- run: training error above the validation error up to step 1250, "
        "except at steps 500, 750."
    )


def test_crossing_line_handles_runs_that_never_cross(report):
    assert report.crossing_line("run", log_rows([(1, 2), (1, 2)])) == (
        "- run: training error below the validation error at every evaluation."
    )
    assert report.crossing_line("run", log_rows([(3, 2), (3, 2)])) == (
        "- run: training error above the validation error up to step 500."
    )


def test_crossing_lines_of_the_committed_runs(report):
    lines = [line for line in report.training_lines() if "training error" in line]
    assert len(lines) == 3
    for line in lines[:2]:
        assert line.endswith(
            "up to step 3500, except at step 2750, below it from step 3750 on."
        )
    assert lines[2].endswith(
        "up to step 3750, except at steps 2750, 3250, 3500, below it from step 4000 on."
    )


def start_with(errors, forcing=False):
    rows = {k: {"band_log_ratio_all": e} for k, e in errors.items()}
    return {"setup": {"forcing_channel": forcing}, "start": rows}


def test_start_summary_leaves_out_the_start_that_blows_up(report):
    errors = {"0.0": 11.0, "1e-06": 0.2, "0.001": 0.5, "0.01": 0.4}
    lines = report.start_summary([start_with(errors)]).splitlines()
    assert lines[2] == "| no forcing channel | 0.2 | 1e-06 | 0.5 | 0.001 | 0.3 |"


def test_the_start_used_gives_the_highest_validation_error(report):
    used = str(inspect.signature(ddim_sample).parameters["min_alpha_bar"].default)
    for name in ("sampler_start.json", "sampler_start_forcing.json"):
        start = json.loads((ROOT / "results" / name).read_text())["start"]
        errors = {
            level: s["band_log_ratio_all"]
            for level, s in start.items()
            if level != report.END_OF_SCHEDULE
        }
        assert max(errors, key=errors.get) == used
        assert min(errors, key=errors.get) == "1e-06"


def test_early_checkpoint_is_better_only_in_the_high_band(report):
    sources = json.loads((ROOT / "results" / "diagnostics.json").read_text())["sources"]
    early, final = sources["diffusion_early"], sources["diffusion"]
    assert early["band_log_ratio"]["high"] < final["band_log_ratio"]["high"]
    for band in ("low", "mid"):
        assert early["band_log_ratio"][band] > final["band_log_ratio"][band]
    for key in ("excess_kurtosis", "ks_distance", "mean_square_outside_cutoff"):
        assert early[key] > final[key]
    assert early["forced_mode_coherence"] < final["forced_mode_coherence"]
    assert early["enstrophy_mean"] < final["enstrophy_mean"]


def test_mean_abs_log_averages_the_first_shells(report):
    ratio = [2.0, 0.5, 1.0, 4.0]
    assert report.mean_abs_log(ratio, 3) == pytest.approx(2 * math.log(2) / 3)
    assert report.mean_abs_log(ratio, 4) == pytest.approx(4 * math.log(2) / 4)


def test_relaxation_score_over_shells_1_to_20_keeps_the_ranking(report):
    path = ROOT / "results" / "consistency.json"
    sources = json.loads(path.read_text())["sources"]
    score = {
        name: report.mean_abs_log(s["spectrum_ratio_after_over_before"], 20)
        for name, s in sources.items()
    }
    for s in sources.values():
        full = report.mean_abs_log(s["spectrum_ratio_after_over_before"], 21)
        assert full == pytest.approx(s["band_log_ratio"]["all"], rel=1e-9)
    networks = (score["diffusion"], score["diffusion_plain"])
    gaussians = (score["g_shell"], score["g_mode"], score["g_mode_ddim"])
    assert max(networks) < min(gaussians)
    assert score["diffusion"] < score["diffusion_plain"]


def solver_info():
    return json.loads((ROOT / "results" / "solver.json").read_text())


def test_spin_up_windows_are_reported_against_the_snapshot_mean(report):
    s = solver_info()["stationarity"]
    ratio = [w / s["energy_mean"] for w in s["spin_up_energy_by_20_time_units"]]
    assert len(ratio) == 6
    assert all(0.03 < 1 - r < 0.04 for r in ratio[:2])
    assert all(0.03 < r - 1 < 0.06 for r in ratio[4:])
    assert "| 0.965, 0.962, 0.994, 1.01, 1.05, 1.04 |" in report.solver_table(
        solver_info()
    )


def test_copula_control_keeps_the_gaussian_flatness_and_the_training_values(report):
    """The control has the training values, not those of the networks."""
    results = ROOT / "results"
    control = json.loads((results / "copula.json").read_text())["sources"]
    control = control["g_mode_copula"]
    sources = json.loads((results / "diagnostics.json").read_text())["sources"]
    flat = {k: report.flatness(v["structure_functions"]) for k, v in sources.items()}
    mapped = report.flatness(control["structure_functions"])
    assert flat["g_mode"] < mapped < flat["g_mode"] + 0.2
    assert mapped + 0.5 < min(flat["diffusion"], flat["diffusion_plain"])
    train_kurtosis = sources["real_train"]["excess_kurtosis"]
    assert abs(control["excess_kurtosis"] - train_kurtosis) < 0.01
    networks = (sources["diffusion"], sources["diffusion_plain"])
    assert min(n["excess_kurtosis"] for n in networks) > train_kurtosis + 0.1


def test_time_step_check_names_its_own_time_span(report):
    info = solver_info()
    row = next(
        line
        for line in report.solver_table(info).splitlines()
        if line.startswith("| dt against")
    )
    assert row.startswith("| dt against dt / 2 after 0.505 time units |")
    spans = (
        info["turnover"]["eddy_turnover_time"],
        info["turnover"]["snapshot_time"],
        info["checks"]["time_step"]["evolution_time"],
    )
    assert len({round(span, 3) for span in spans}) == 3
