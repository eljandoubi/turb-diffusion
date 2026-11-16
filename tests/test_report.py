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
