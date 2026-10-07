import os
import subprocess
import sys

import pytest
from click.testing import CliRunner

from dcmtether import __version__
from dcmtether.cli import EXIT_NOT_IMPLEMENTED, main

COMMANDS = ["scan", "to-nii", "to-dcm", "compare"]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_help_lists_all_commands(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_command_shows_help_as_usage_error(runner):
    result = runner.invoke(main, [])
    assert result.exit_code == 2
    assert "Usage" in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help(runner, command):
    assert runner.invoke(main, [command, "--help"]).exit_code == 0


@pytest.mark.parametrize(
    "args",
    [
        ["scan", "{dir}", "{out}"],
        ["to-nii", "{dir}"],
        ["to-dcm", "{file}", "{out}"],
        ["compare", "{dir}", "{dir}"],
    ],
)
def test_stub_commands_report_not_implemented(runner, tmp_path, args):
    nifti = tmp_path / "x.nii.gz"
    nifti.touch()
    paths = {"dir": tmp_path, "file": nifti, "out": tmp_path / "out"}
    result = runner.invoke(main, [a.format(**paths) for a in args])
    assert result.exit_code == EXIT_NOT_IMPLEMENTED
    assert "not implemented" in result.stderr


@pytest.mark.parametrize(
    "args", [["scan", "missing", "out"], ["to-nii", "missing"], ["compare", "missing", "missing"]]
)
def test_missing_input_paths_are_usage_errors(runner, tmp_path, args):
    result = runner.invoke(main, [str(tmp_path / a) if a == "missing" else a for a in args])
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_help_works_on_a_legacy_console_encoding():
    # Windows consoles may default to cp1252; help text must not crash there.
    result = subprocess.run(
        [sys.executable, "-m", "dcmtether", "--help"],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"Usage: dcmtether" in result.stdout
