"""The tables and lists of the README are the ones scripts/report.py prints."""

import importlib.util
import sys
from pathlib import Path

import pytest

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
