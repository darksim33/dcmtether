import json
import sqlite3

import pytest
from synthetic import SeriesSpec, make_localizer, make_series, write_series

from dcmtether import __version__
from dcmtether.manifest import MANIFEST_NAME, SCHEMA_VERSION, write_manifest
from dcmtether.scan import scan


@pytest.fixture
def scanned(tmp_path):
    write_series(make_series(SeriesSpec(series_number=1)), tmp_path / "src" / "patient" / "t1")
    write_series(make_localizer(), tmp_path / "src" / "patient" / "loc")
    (tmp_path / "src" / "readme.txt").write_text("not dicom")
    return scan(tmp_path / "src")


def _rows(db, sql):
    db.row_factory = sqlite3.Row
    return [dict(r) for r in db.execute(sql)]


def test_manifest_contents(tmp_path, scanned):
    path = write_manifest(tmp_path / "dataset", scanned)
    assert path == tmp_path / "dataset" / MANIFEST_NAME
    assert not path.with_suffix(".sqlite.partial").exists()

    with sqlite3.connect(path) as db:
        meta = dict(db.execute("SELECT key, value FROM meta"))
        series = _rows(db, "SELECT * FROM series ORDER BY id")
        instances = _rows(db, "SELECT * FROM instances ORDER BY path")
        skipped = _rows(db, "SELECT * FROM skipped_files")

    assert meta["schema_version"] == str(SCHEMA_VERSION)
    assert meta["tool_version"] == __version__
    assert meta["source_root"] == str((tmp_path / "src").resolve())

    assert [s["id"] for s in series] == [1, 2]
    by_dir = {s["source_dir"]: s for s in series}
    assert set(by_dir) == {"patient/t1", "patient/loc"}
    t1, loc = by_dir["patient/t1"], by_dir["patient/loc"]
    assert (t1["status"], json.loads(t1["reasons"])) == ("supported", [])
    assert loc["status"] == "split"
    assert t1["n_instances"] == 5
    assert t1["modality"] == "MR"
    assert t1["sop_class_uid"] == "1.2.840.10008.5.1.4.1.1.4"
    assert t1["transfer_syntax_uid"] == "1.2.840.10008.1.2.1"
    assert t1["manufacturer"] == "dcmtether-synthetic"
    assert t1["implementation_class_uid"]
    assert json.loads(t1["geometry"])["n_positions"] == 5

    assert len(instances) == 8
    assert {i["path"] for i in instances if i["series_id"] == t1["id"]} == {
        f"patient/t1/IM_000{n}" for n in range(1, 6)
    }
    assert skipped == [{"path": "readme.txt", "reason": "not a DICOM file"}]


def test_series_uid_is_unique(tmp_path, scanned):
    scanned.series.append(scanned.series[0])
    with pytest.raises(sqlite3.IntegrityError):
        write_manifest(tmp_path / "dataset", scanned)
    assert not (tmp_path / "dataset" / MANIFEST_NAME).exists()


def test_existing_manifest_is_never_overwritten(tmp_path, scanned):
    path = write_manifest(tmp_path / "dataset", scanned)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_manifest(tmp_path / "dataset", scanned)
    assert path.read_bytes() == before
