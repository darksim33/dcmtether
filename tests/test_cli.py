import os
import subprocess
import sys

import pytest
from click.testing import CliRunner
from synthetic import SeriesSpec, make_localizer, make_series, write_series

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


def test_scan_writes_manifest_and_summary(runner, tmp_path):
    write_series(make_series(SeriesSpec(series_number=1)), tmp_path / "src" / "a")
    write_series(make_localizer(), tmp_path / "src" / "b")
    (tmp_path / "src" / "notes.txt").write_text("x")
    dataset = tmp_path / "dataset"

    result = runner.invoke(main, ["scan", str(tmp_path / "src"), str(dataset)])
    assert result.exit_code == 0, result.output
    assert (dataset / "manifest.sqlite").exists()
    assert "Scanned 9 files: 8 DICOM instances in 2 series, 1 file skipped." in result.output
    assert "supported:   1" in result.output
    assert "split:       1" in result.output
    assert "series #2 (3 instances) split: 3 distinct image planes" in result.output
    assert "skipped 1 file: not a DICOM file" in result.output
    # PHI rules: no UIDs, names or descriptions in the output.
    assert "2.25." not in result.output
    assert "SYNTHETIC" not in result.output
    assert "Synthetic" not in result.output


def test_scan_refuses_existing_manifest(runner, tmp_path):
    write_series(make_series(), tmp_path / "src")
    args = ["scan", str(tmp_path / "src"), str(tmp_path / "dataset")]
    assert runner.invoke(main, args).exit_code == 0
    result = runner.invoke(main, args)
    assert result.exit_code == 1
    assert "manifest already exists" in result.output


def test_scan_skips_dataset_inside_source(runner, tmp_path):
    write_series(make_series(), tmp_path / "src" / "a")
    dataset = tmp_path / "src" / "dataset"
    dataset.mkdir()
    (dataset / "README.md").write_text("earlier output")  # would be skipped if walked
    result = runner.invoke(main, ["scan", str(tmp_path / "src"), str(dataset)])
    assert result.exit_code == 0, result.output
    assert "0 files skipped" in result.output
