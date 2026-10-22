"""`dcmtether run SOURCE [-o DATASET]` (scan + to-nii, rerunnable) and manifest updates."""

import shutil
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner
from roundtrip import manifest_rows, output_rows
from synthetic import SeriesSpec, make_4d, make_series, write_series

from dcmtether import naming
from dcmtether.cli import main
from dcmtether.forward import convert_dataset
from dcmtether.manifest import ManifestError, update_manifest, write_manifest
from dcmtether.scanning import scan


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


def test_full_name_is_set_on_the_first_run_and_kept(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    result = _run(tmp_path / "src", tmp_path / "dataset", "-fv")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "dataset" / "sub-SyntheticPhantom").is_dir()
    write_series(_series(2), tmp_path / "src" / "b")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert [r["subject"] for r in manifest_rows(tmp_path / "dataset")] == ["SyntheticPhantom"] * 2
    assert not (tmp_path / "dataset" / "sub-SynthPhant").exists()


def test_full_name_is_refused_for_a_short_label_dataset(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    _run(tmp_path / "src", tmp_path / "dataset")
    result = _run(tmp_path / "src", tmp_path / "dataset", "--full-name")
    assert result.exit_code == 1
    assert "this dataset uses short subject labels" in result.output
    assert "Synth" not in result.output


# --- converted series stay converted -------------------------------------------------------------


def _twin(seed: int, description: str = "SYNTHETIC"):
    """Series of one patient and study that differ in their content only."""
    spec = SeriesSpec(seed=seed, series_description=description, extra={"PatientID": "P1"})
    series = make_series(spec)
    for ds in series.datasets:
        ds.StudyInstanceUID = "2.25.999"
    return series


def _names(dataset: Path) -> dict[int, str]:
    return {r["series_id"]: Path(r["nifti_path"]).name for r in output_rows(dataset)}


def _status(dataset: Path) -> dict[int, str | None]:
    return {r["id"]: r["conversion_status"] for r in manifest_rows(dataset)}


def test_converted_series_survives_an_unreadable_source(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    _run(tmp_path / "src", tmp_path / "dataset")
    shutil.move(tmp_path / "src" / "a", tmp_path / "away")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert "Converted 0 series, 0 failed." in result.output
    assert _status(tmp_path / "dataset") == {1: "converted"}
    assert len(output_rows(tmp_path / "dataset")) == 1

    shutil.move(tmp_path / "away", tmp_path / "src" / "a")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert "Manifest: 0 new series, 0 rescanned, 1 unchanged." in result.output
    assert "Converted 0 series, 0 failed." in result.output


def test_converted_series_keeps_its_name_when_naming_rules_change(tmp_path, monkeypatch):
    write_series(_twin(1), tmp_path / "src" / "a")
    _run(tmp_path / "src", tmp_path / "dataset")
    before = _names(tmp_path / "dataset")
    monkeypatch.setattr(naming, "contrast_suffix", lambda *a: naming.Suffix("NEW", "test"))
    write_series(_twin(2), tmp_path / "src" / "b")  # same number + name
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert "warning" not in result.output
    names = _names(tmp_path / "dataset")
    assert names[1] == before[1]
    assert names[2].endswith("_ser-01_NEW.nii.gz")


def _downgrade_to_v3(dataset: Path) -> None:
    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        for column in ("stem", "dup", "patient_id", "subject_base", "merged_into"):
            db.execute(f"ALTER TABLE series DROP COLUMN {column}")
        db.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")


def test_v3_manifest_is_upgraded_and_backfilled(tmp_path):
    for n, folder in enumerate("ab", start=1):  # same name: converted as _01 / _02
        write_series(_twin(n), tmp_path / "src" / folder)
    _run(tmp_path / "src", tmp_path / "dataset")
    expected = manifest_rows(tmp_path / "dataset")
    assert [(r["stem"], r["dup"]) for r in expected] == [
        ("sub-SynthPhant_ses-20260101_acq-SYNTHETIC_ser-01_T1w", 1),
        ("sub-SynthPhant_ses-20260101_acq-SYNTHETIC_ser-01_T1w", 2),
    ]
    assert [r["subject_base"] for r in expected] == ["SynthPhant"] * 2
    assert all(r["patient_id"] is not None for r in expected)

    _downgrade_to_v3(tmp_path / "dataset")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert manifest_rows(tmp_path / "dataset") == expected


def test_upgraded_manifest_without_source_cannot_vouch_for_a_subject_label(tmp_path):
    write_series(_series(1), tmp_path / "src" / "a")
    _run(tmp_path / "src", tmp_path / "dataset")
    _downgrade_to_v3(tmp_path / "dataset")
    shutil.move(tmp_path / "src" / "a", tmp_path / "away")
    other = make_series(SeriesSpec(series_number=2, extra={"PatientID": "OTHER"}))
    write_series(other, tmp_path / "src" / "b")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 1
    assert (
        "series #2 failed: cannot check subject label against series #1 (source not readable)"
    ) in result.output
    assert _status(tmp_path / "dataset") == {1: "converted", 2: "failed"}
    assert len(output_rows(tmp_path / "dataset")) == 1


def test_late_series_gets_the_next_free_duplicate_counter(tmp_path):
    for n, folder in enumerate("abc", start=1):
        series = _twin(n)
        if folder == "b":
            broken_uid = series.series_instance_uid
            for ds in series.datasets:
                ds.StudyDate = ""  # fails for now
        write_series(series, tmp_path / "src" / folder)
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 1
    assert _status(tmp_path / "dataset") == {1: "converted", 2: "failed", 3: "converted"}
    before = _names(tmp_path / "dataset")
    assert [n.split("_ser-")[1] for n in before.values()] == [
        "01_01_T1w.nii.gz",
        "01_02_T1w.nii.gz",
    ]

    shutil.rmtree(tmp_path / "src" / "b")
    fixed = _twin(2)
    for ds in fixed.datasets:
        ds.SeriesInstanceUID = broken_uid
    write_series(fixed, tmp_path / "src" / "b")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    names = _names(tmp_path / "dataset")
    assert {sid: names[sid] for sid in before} == before
    assert names[2].endswith("_ser-01_03_T1w.nii.gz")
    assert "series #2 warning: duplicate name: series #1, #2, #3" in result.output


def test_series_converted_without_counter_keeps_the_bare_name(tmp_path):
    write_series(_twin(1), tmp_path / "src" / "a")
    _run(tmp_path / "src", tmp_path / "dataset")
    before = _names(tmp_path / "dataset")
    write_series(_twin(2), tmp_path / "src" / "b")
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    names = _names(tmp_path / "dataset")
    assert names[1] == before[1] and "_ser-01_T1w" in names[1]
    assert names[2].endswith("_ser-01_02_T1w.nii.gz")


def test_names_that_differ_only_in_case_are_numbered(tmp_path):
    for folder, description in (("a", "sag T1"), ("b", "Sag T1")):
        write_series(_twin(ord(folder), description), tmp_path / "src" / folder)
    result = _run(tmp_path / "src", tmp_path / "dataset")
    assert result.exit_code == 0, result.output
    assert sorted(_names(tmp_path / "dataset").values()) == [
        "sub-SynthPhant_ses-20260101_acq-SagT1_ser-01_02_T1w.nii.gz",
        "sub-SynthPhant_ses-20260101_acq-sagT1_ser-01_01_T1w.nii.gz",
    ]
    assert "duplicate name: series #1, #2 map to the same name" in result.output


# --- in-place runs: what belongs to the dataset -------------------------------------------------


def _in_place(source: Path, command: str = "run"):
    args = ["run", str(source)] if command == "run" else ["scan", str(source), str(source)]
    return CliRunner().invoke(main, args)


@pytest.mark.parametrize("command", ["run", "scan"])
def test_first_in_place_run_refuses_source_folders_named_like_subjects(tmp_path, command):
    src = tmp_path / "src"
    write_series(_series(1), src / "sub-01" / "a")
    write_series(_series(2), src / "sub-02")
    write_series(_series(3), src / "other")
    before = sorted(src.rglob("*"))
    result = _in_place(src, command)
    assert result.exit_code == 1
    assert result.output == "Error: SOURCE already has sub-* folders (2); use --output\n"
    assert sorted(src.rglob("*")) == before  # nothing written

    result = _run(src, tmp_path / "dataset")  # elsewhere they are ordinary source folders
    assert result.exit_code == 0, result.output
    assert "Converted 3 series, 0 failed." in result.output


def test_first_in_place_run_reads_a_source_folder_named_logs(tmp_path):
    src = tmp_path / "src"
    write_series(_series(1), src / "logs")
    write_series(_series(2), src / "logs" / "deeper")
    (src / "README.md").write_text("mine", "utf-8")
    result = _in_place(src)
    assert result.exit_code == 0, result.output
    assert "Scanned 10 files: 10 DICOM instances in 2 series, 0 files skipped." in result.output
    assert "Converted 2 series, 0 failed." in result.output
    assert (src / "README.md").read_text("utf-8") == "mine"

    result = _in_place(src)  # the log files next to the DICOM files are not source files
    assert "Scanned 10 files: 10 DICOM instances in 2 series, 0 files skipped." in result.output
    assert "Manifest: 0 new series, 0 rescanned, 2 unchanged." in result.output
    assert len(list((src / "logs").glob("dcmtether_*.jsonl"))) == 2


def test_later_in_place_runs_skip_the_subject_folders(tmp_path):
    src = tmp_path / "src"
    write_series(_series(1), src / "a")
    assert _in_place(src).exit_code == 0
    assert any(src.glob("sub-*"))
    result = _in_place(src)
    assert result.exit_code == 0, result.output
    assert "Scanned 5 files: 5 DICOM instances in 1 series, 0 files skipped." in result.output


# --- a path that changes its series -------------------------------------------------------------


def _replace_folder(folder: Path, series) -> None:
    shutil.rmtree(folder)
    write_series(series, folder)


def test_file_of_an_unconverted_series_replaced_by_a_new_series(tmp_path):
    src, dataset = tmp_path / "src", tmp_path / "dataset"
    write_series(_series(1), src / "a")
    write_series(_series(2), src / "b")
    write_manifest(dataset, scan(src))  # nothing converted yet
    _replace_folder(src / "a", _series(3, seed=3))  # same file names, another series
    shutil.copy(src / "a" / "IM_0001", src / "b" / "IM_0001")  # and one file inside series 2

    update = update_manifest(dataset, scan(src))

    assert (update.new, update.rescanned, update.missing) == ([3], [2], [1])
    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        counts = dict(db.execute("SELECT series_id, COUNT(*) FROM instances GROUP BY 1"))
    assert counts == {2: 4, 3: 6}  # series 1 lost the paths that are no longer its files


def test_series_that_lost_all_its_files_fails_cleanly(tmp_path):
    src, dataset = tmp_path / "src", tmp_path / "dataset"
    write_series(_series(1), src / "a")
    write_manifest(dataset, scan(src))  # scanned, not converted
    _replace_folder(src / "a", _series(2, seed=2))  # every file is another series now

    result = _run(src, dataset)

    assert isinstance(result.exception, SystemExit), repr(result.exception)  # no traceback
    assert "  series #1 failed: source files no longer found" in result.output
    assert _status(dataset) == {1: "failed", 2: "converted"}


def test_file_of_a_converted_series_replaced_by_another_series(tmp_path):
    src, dataset = tmp_path / "src", tmp_path / "dataset"
    write_series(_series(1), src / "a")
    assert _run(src, dataset).exit_code == 0
    _replace_folder(src / "a", _series(2, seed=2))
    before = manifest_rows(dataset)

    result = _run(src, dataset)

    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exit_code == 1
    assert result.output.splitlines()[-1] == (
        "Error: a source file of converted series #1 now belongs to a new series (restore "
        "it, or remove the outputs and manifest entry of series #1)"
    )
    assert manifest_rows(dataset) == before  # rolled back
