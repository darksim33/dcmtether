"""Phase 6: `dcmtether run SOURCE DATASET` (scan + to-nii, rerunnable) and manifest updates."""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from roundtrip import manifest_rows, output_rows
from synthetic import SeriesSpec, make_4d, make_series, write_series

from dcmtether.cli import main
from dcmtether.forward import convert_dataset
from dcmtether.manifest import ManifestError, update_manifest, write_manifest
from dcmtether.scan import scan


def _run(source: Path, dataset: Path, *extra: str):
    return CliRunner().invoke(main, ["run", str(source), "-o", str(dataset), *extra])


def _series(number: int, **kw):
    return make_series(SeriesSpec(series_number=number, **kw))


def test_first_run_scans_and_converts(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    write_series(_series(2), tmp_path / "src" / "b")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert "Scanned 10 files: 10 DICOM instances in 2 series, 0 files skipped." in result.output
    assert "Manifest: 2 new series, 0 rescanned, 0 unchanged." in result.output
    assert "Converted 2 series, 0 failed." in result.output
    assert len(output_rows(tmp_path / "dataset")) == 2


def test_without_output_the_dataset_is_written_into_source(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    (tmp_path / "src" / "notes.txt").write_text("x")
    result = CliRunner().invoke(main, ["run", str(tmp_path / "src")])
    assert result.exit_code == 0, result.output
    assert "Converted 1 series, 0 failed." in result.output
    assert (tmp_path / "src" / "manifest.sqlite").is_file()
    assert len(output_rows(tmp_path / "src")) == 1

    # the dataset's own files are not read back as source files
    result = CliRunner().invoke(main, ["run", str(tmp_path / "src")])
    assert result.exit_code == 0, result.output
    assert "Scanned 6 files: 5 DICOM instances in 1 series, 1 file skipped." in result.output
    assert "Manifest: 0 new series, 0 rescanned, 1 unchanged." in result.output


def test_rerun_without_changes_does_nothing(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    _run(tmp_path / "src", tmp_path / "dataset")
    nifti = tmp_path / "dataset" / output_rows(tmp_path / "dataset")[0]["nifti_path"]
    before = nifti.stat().st_mtime_ns
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert "Manifest: 0 new series, 0 rescanned, 1 unchanged." in result.output
    assert "Converted 0 series, 0 failed." in result.output
    assert nifti.stat().st_mtime_ns == before


def test_rerun_adds_new_series_with_new_ids(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    _run(tmp_path / "src", tmp_path / "dataset")
    write_series(_series(2), tmp_path / "src" / "b")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert "Manifest: 1 new series, 0 rescanned, 1 unchanged." in result.output
    assert "Converted 1 series, 0 failed." in result.output
    rows = manifest_rows(tmp_path / "dataset")
    assert [(r["id"], r["source_dir"], r["conversion_status"]) for r in rows] == [
        (1, "a", "converted"),
        (2, "b", "converted"),
    ]


def test_failed_series_is_rescanned_and_retried(tmp_path):
    series = _series(1)
    for ds in series.datasets:
        ds.PatientName = ""  # naming needs a PatientName: the series fails
    write_series(series, tmp_path / "src" / "a")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 1
    assert "series #1 failed: PatientName is missing" in result.output

    shutil.rmtree(tmp_path / "src" / "a")
    write_series(_series(1), tmp_path / "src" / "a")  # fixed export, same series UID
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert "Manifest: 0 new series, 1 rescanned, 0 unchanged." in result.output
    assert "Converted 1 series, 0 failed." in result.output


def test_unsupported_series_is_reclassified_on_rerun(tmp_path):
    s = make_4d(n_volumes=2, kind="none")  # unsupported: volumes cannot be told apart
    write_series(s, tmp_path / "src" / "a")
    _run(tmp_path / "src", tmp_path / "dataset")
    assert manifest_rows(tmp_path / "dataset")[0]["status"] == "unsupported"
    shutil.rmtree(tmp_path / "src" / "a")
    fixed = make_4d(n_volumes=2, kind="echo")
    for ds in fixed.datasets:
        ds.SeriesInstanceUID = s.series_instance_uid
    write_series(fixed, tmp_path / "src" / "a")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert "1 rescanned" in result.output
    assert manifest_rows(tmp_path / "dataset")[0]["status"] == "supported"


def test_changed_files_of_a_converted_series_are_reported_not_reconverted(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    _run(tmp_path / "src", tmp_path / "dataset")
    (tmp_path / "src" / "a" / "IM_0005").unlink()
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert (
        "  series #1: source files changed after conversion (not reconverted; "
        "remove its outputs and manifest entry to redo it)"
    ) in result.output
    assert manifest_rows(tmp_path / "dataset")[0]["conversion_status"] == "converted"


def test_series_missing_from_source_is_reported(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    write_series(_series(2), tmp_path / "src" / "b")
    _run(tmp_path / "src", tmp_path / "dataset")
    shutil.rmtree(tmp_path / "src" / "b")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert "  series #2: no longer found in SOURCE (kept in the manifest)" in result.output


def test_other_source_folder_is_refused(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    _run(tmp_path / "src", tmp_path / "dataset")
    write_series(_series(2), tmp_path / "other" / "b")
    result = _run(tmp_path / "other", tmp_path / "dataset")
    assert result.exit_code == 1
    assert "SOURCE is not the folder this dataset was scanned from" in result.output
    assert len(manifest_rows(tmp_path / "dataset")) == 1


def test_update_manifest_keeps_ids_and_replaces_skipped_files(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    (tmp_path / "src" / "junk.txt").write_text("x")
    dataset = tmp_path / "dataset"
    write_manifest(dataset, scan(tmp_path / "src"))
    convert_dataset(dataset)
    (tmp_path / "src" / "junk.txt").unlink()
    write_series(_series(2), tmp_path / "src" / "b")
    update = update_manifest(dataset, scan(tmp_path / "src"))
    assert (update.new, update.rescanned, update.unchanged) == ([2], [], [1])
    import sqlite3

    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM skipped_files").fetchone() == (0,)


def test_update_manifest_needs_a_manifest(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    with pytest.raises(ManifestError, match="no manifest"):
        update_manifest(tmp_path / "dataset", scan(tmp_path / "src"))


def test_verbose_lists_every_series(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    result = _run(tmp_path / "src", tmp_path / "dataset", "--verbose")
    assert "  series #1 converted (1 output)" in result.output
    for secret in ("Synth", "Phantom", "2.25.", "20260101", "sub-"):
        assert secret not in result.output
