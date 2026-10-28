import os
import subprocess
import sys

import pytest
from click.testing import CliRunner
from synthetic import SeriesSpec, make_localizer, make_series, write_series

from dcmtether import __version__
from dcmtether.cli import main

COMMANDS = ["scan", "to-nii", "to-dcm", "compare", "to-seg", "to-pmap", "to-mr", "merge"]


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
    assert "Copyright (C) 2026 Thomas Thiel" in result.output
    assert "NO WARRANTY" in result.output and "GPL-3.0-or-later" in result.output


def test_no_command_shows_help_as_usage_error(runner):
    result = runner.invoke(main, [])
    assert result.exit_code == 2
    assert "Usage" in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help(runner, command):
    assert runner.invoke(main, [command, "--help"]).exit_code == 0


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


def test_body_part_lists_every_series(runner, tmp_path):
    spec = SeriesSpec(series_number=1, extra={"BodyPartExamined": "ABDOMEN"})
    write_series(make_series(spec), tmp_path / "src" / "a")
    localizer = make_localizer()
    for ds in localizer.datasets:
        ds.pop("BodyPartExamined", None)
    write_series(localizer, tmp_path / "src" / "b")

    result = runner.invoke(main, ["scan", str(tmp_path / "src"), str(tmp_path / "d1"), "-b"])
    assert result.exit_code == 0, result.output
    assert "series #1 (5 instances, body part ABDOMEN) supported\n" in result.output
    assert (
        "series #2 (3 instances, body part not set) split: 3 distinct image planes" in result.output
    )
    without = runner.invoke(main, ["scan", str(tmp_path / "src"), str(tmp_path / "d2")])
    assert "body part" not in without.output and "series #1" not in without.output

    result = runner.invoke(main, ["run", str(tmp_path / "src"), "-o", str(tmp_path / "d3"), "-b"])
    assert result.exit_code == 0, result.output
    assert "series #1 (5 instances, body part ABDOMEN) supported\n" in result.output
    again = runner.invoke(main, ["run", str(tmp_path / "src"), "-o", str(tmp_path / "d3"), "-b"])
    assert "body part" not in again.output  # nothing new or rescanned


def test_scan_refuses_existing_manifest(runner, tmp_path):
    write_series(make_series(), tmp_path / "src")
    args = ["scan", str(tmp_path / "src"), str(tmp_path / "dataset")]
    assert runner.invoke(main, args).exit_code == 0
    result = runner.invoke(main, args)
    assert result.exit_code == 1
    assert result.output == "Error: manifest already exists in DATASET\n"


def test_output_never_names_the_dataset_folder(runner, tmp_path):
    # The folder name may identify a patient or study (PHI rules).
    write_series(make_series(), tmp_path / "src" / "a")
    dataset = tmp_path / "Mueller_Thomas_study"
    outputs = [
        runner.invoke(main, ["to-nii", str(tmp_path / "src")]).output,  # no manifest
        runner.invoke(main, ["scan", str(tmp_path / "src"), str(dataset)]).output,
        runner.invoke(main, ["scan", str(tmp_path / "src"), str(dataset)]).output,  # exists
        runner.invoke(main, ["to-nii", str(dataset)]).output,
        runner.invoke(main, ["run", str(tmp_path / "src"), "-o", str(dataset)]).output,
    ]
    assert outputs[0] == "Error: no manifest in DATASET (run 'dcmtether scan' first)\n"
    assert outputs[1].splitlines()[-1] == "Manifest written."
    for output in outputs:
        assert "Mueller" not in output and tmp_path.name not in output


def test_scan_skips_dataset_inside_source(runner, tmp_path):
    write_series(make_series(), tmp_path / "src" / "a")
    dataset = tmp_path / "src" / "dataset"
    dataset.mkdir()
    (dataset / "README.md").write_text("earlier output")  # would be skipped if walked
    result = runner.invoke(main, ["scan", str(tmp_path / "src"), str(dataset)])
    assert result.exit_code == 0, result.output
    assert "0 files skipped" in result.output
