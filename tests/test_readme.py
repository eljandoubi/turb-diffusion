"""The commands of the README name options and files that exist."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def readme_commands() -> list[str]:
    text = (ROOT / "README.md").read_text()
    lines = [
        line
        for block in re.findall(r"```\n(.*?)```", text, re.S)
        for line in block.splitlines()
    ]
    return [
        command.split("#")[0].strip()
        for line in lines
        for command in line.split("&&")
        if command.split("#")[0].strip()
    ]


def test_script_commands_use_options_that_the_scripts_accept():
    seen = 0
    for command in readme_commands():
        parts = command.split()
        if parts[0] == "python" and parts[1].startswith("scripts/"):
            source = (ROOT / parts[1]).read_text()
            for option in (p for p in parts[2:] if p.startswith("--")):
                assert f'"{option}"' in source, command
            seen += 1
    assert seen >= 12


def test_copy_commands_name_files_that_are_in_results():
    seen = 0
    for command in readme_commands():
        if command.startswith("cp results_") and "*" not in command:
            *sources, target = command.split()[1:]
            for source in sources:
                name = Path(source).name if target.endswith("/") else Path(target).name
                assert (ROOT / "results" / name).exists(), command
                seen += 1
    assert seen >= 8
