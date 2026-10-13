"""Structured JSON-lines log in the dataset (no PHI: ids, keywords and counts only)."""

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from click.testing import CliRunner
from roundtrip import output_rows
from synthetic import SeriesSpec, make_localizer, make_series, write_series
from test_seg import META as SEG_META

from dcmtether import log
from dcmtether.cli import main

#: Values that identify the synthetic patient/study or the output files.
SECRETS = ("Synthetic", "Phantom", "SYNTH-", "2.25.", "20260101", "sub-", "IM_0", "/")


def _log_files(dataset: Path) -> list[Path]:
    """Oldest first."""
    return log.log_files(dataset)


def _records(dataset: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in _log_files(dataset)
        for line in path.read_text("utf-8").splitlines()
    ]


def _assert_no_phi(dataset: Path) -> None:
    for path in _log_files(dataset):
        text = path.read_text("utf-8")
        for secret in SECRETS:
            assert secret not in text, secret


@pytest.fixture
def dataset(tmp_path) -> Path:
    write_series(make_series(SeriesSpec(series_number=1)), tmp_path / "src" / "a")
    write_series(make_localizer(), tmp_path / "src" / "loc")
    series = make_series(SeriesSpec(series_number=3))
    for ds in series.datasets:
        ds.PatientName = ""  # fails naming
    write_series(series, tmp_path / "src" / "c")
    result = CliRunner().invoke(
        main, ["run", str(tmp_path / "src"), "-o", str(tmp_path / "dataset")]
    )
    assert result.exit_code == 1, result.output
    return tmp_path / "dataset"


def test_run_writes_structured_records(dataset):
    records = _records(dataset)
    assert all(r["command"] == "run" for r in records)
    assert all(set(r) <= log.ALLOWED_FIELDS for r in records)
    events = [r["event"] for r in records]
    assert events[0] == "start" and events[-1] == "end"
    scanned = next(r for r in records if r["event"] == "scanned")
    assert (scanned["n_files"], scanned["n_series"]) == (13, 3)
    converted = [r for r in records if r["event"] == "series_converted"]
    failed = [r for r in records if r["event"] == "series_failed"]
    # ids follow the scan order (folders a, c, loc): c is the one without a PatientName
    assert sorted(r["series_id"] for r in converted) == [1, 3]
    assert [(r["series_id"], r["reason"]) for r in failed] == [(2, "PatientName is missing")]
    assert {r["outputs"] for r in converted} == {1, 3}  # localizer: 3 chunks
    assert records[-1]["n_failed"] == 1
    assert all("time" in r and "version" in r for r in records)


def test_log_contains_no_identifying_values(dataset):
    _assert_no_phi(dataset)


def test_log_file_name_has_a_timestamp(dataset):
    [path] = _log_files(dataset)
    assert re.fullmatch(r"dcmtether_\d{8}T\d{6}\.jsonl", path.name)


def _commands(path: Path) -> set[str]:
    return {json.loads(line)["command"] for line in path.read_text("utf-8").splitlines()}


def test_run_scan_and_to_nii_each_start_a_log_file(dataset, tmp_path):
    CliRunner().invoke(main, ["to-nii", str(dataset)])
    CliRunner().invoke(main, ["run", str(tmp_path / "src"), "-o", str(dataset)])
    files = _log_files(dataset)
    assert len(files) == 3  # also within one second: the names never collide
    assert [_commands(f) for f in files] == [{"run"}, {"to-nii"}, {"run"}]

    CliRunner().invoke(main, ["scan", str(tmp_path / "src"), str(tmp_path / "other")])
    [scan_log] = _log_files(tmp_path / "other")
    assert scan_log.name.startswith("dcmtether_")


def test_single_series_commands_append_to_the_newest_log_file(dataset, tmp_path):
    CliRunner().invoke(main, ["to-nii", str(dataset)])
    older, newest = _log_files(dataset)
    before = older.read_text("utf-8")
    nifti = dataset / output_rows(dataset)[0]["nifti_path"]
    CliRunner().invoke(main, ["to-dcm", str(nifti), str(tmp_path / "rebuilt")])
    assert _log_files(dataset) == [older, newest]
    assert older.read_text("utf-8") == before
    assert json.loads(newest.read_text("utf-8").splitlines()[-1])["command"] == "to-dcm"


def test_old_log_file_is_left_alone(dataset, tmp_path):
    [path] = _log_files(dataset)
    old = path.with_name("dcmtether.jsonl")  # as written before the names had a timestamp
    path.rename(old)
    before = old.read_text("utf-8")
    nifti = dataset / output_rows(dataset)[0]["nifti_path"]
    CliRunner().invoke(main, ["to-dcm", str(nifti), str(tmp_path / "rebuilt")])
    assert old.read_text("utf-8") == before
    assert len(_log_files(dataset)) == 2  # a new, timestamped one was started
    assert _log_files(dataset)[0] == old  # and the old one counts as the oldest


def test_scan_logs_classification(tmp_path):
    write_series(make_localizer(), tmp_path / "src" / "loc")
    CliRunner().invoke(main, ["scan", str(tmp_path / "src"), str(tmp_path / "dataset")])
    [classified] = [r for r in _records(tmp_path / "dataset") if r["event"] == "series_classified"]
    assert classified["status"] == "split"
    assert classified["reason"] == "3 distinct image planes (orientation, matrix or pixel spacing)"


def test_commands_on_a_dataset_nifti_log_into_that_dataset(dataset, tmp_path):
    row = output_rows(dataset)[0]
    nifti = dataset / row["nifti_path"]
    result = CliRunner().invoke(main, ["to-dcm", str(nifti), str(tmp_path / "rebuilt")])
    assert result.exit_code == 0, result.output
    record = _records(dataset)[-2]
    assert (record["command"], record["event"], record["series_id"], record["count"]) == (
        "to-dcm",
        "written",
        row["series_id"],
        5,
    )

    img = nib.load(nifti)
    labels = np.zeros(img.shape, np.uint8)
    labels[1, 1, 1] = 1
    nib.save(nib.Nifti1Image(labels, img.affine), tmp_path / "labels.nii.gz")
    (tmp_path / "meta.json").write_text(json.dumps(SEG_META))
    args = [str(nifti), str(tmp_path / "labels.nii.gz"), str(tmp_path / "meta.json")]
    result = CliRunner().invoke(main, ["to-seg", *args, str(tmp_path / "seg.dcm")])
    assert result.exit_code == 0, result.output
    record = _records(dataset)[-2]
    assert (record["command"], record["event"], record["series_id"]) == (
        "to-seg",
        "written",
        row["series_id"],
    )
    _assert_no_phi(dataset)


def test_failures_are_logged_with_the_safe_message(dataset, tmp_path):
    nifti = dataset / output_rows(dataset)[0]["nifti_path"]
    CliRunner().invoke(main, ["to-dcm", str(nifti), str(tmp_path / "rebuilt")])
    CliRunner().invoke(main, ["to-dcm", str(nifti), str(tmp_path / "rebuilt")])  # exists
    record = _records(dataset)[-2]
    assert record["event"] == "error"
    assert record["reason"].startswith("output file already exists")


def test_unknown_fields_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not allowed"):
        log.DatasetLog(tmp_path, "scan").event("start", patient_name="x")


def test_non_dataset_nifti_does_not_log(tmp_path):
    assert log.find_dataset(tmp_path / "x.nii.gz") is None
