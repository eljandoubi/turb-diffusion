"""The tables and lists of the README are the ones scripts/report.py prints."""

import importlib.util
import inspect
import json
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
