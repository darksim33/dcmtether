import json
import sqlite3
from contextlib import closing

import pytest
from synthetic import SeriesSpec, make_localizer, make_series, write_series

from dcmtether import __version__, manifest
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


def test_upgrade_from_v2_moves_outputs(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with closing(sqlite3.connect(dataset / MANIFEST_NAME)) as db, db:
        db.executescript(manifest._SCHEMA_V1)
        db.executescript(manifest._UPGRADES[2])
        db.execute("INSERT INTO meta VALUES ('schema_version', '2')")
        db.execute(
            "INSERT INTO series (id, series_instance_uid, status, reasons, n_instances, "
            "source_dir, geometry, nifti_path, tether_path, sidecar_path, conversion_status) "
            "VALUES (1, '2.25.1', 'supported', '[]', 5, 'a', '{}', 'x.nii.gz', 'x.tether.json', "
            "'x.json', 'converted')"
        )
    db = manifest.open_manifest(dataset)
    try:
        outputs = _rows(db, "SELECT * FROM outputs")
        columns = {r["name"] for r in _rows(db, "PRAGMA table_info(series)")}
    finally:
        db.close()
    assert outputs == [
        {
            "series_id": 1,
            "chunk": 1,
            "nifti_path": "x.nii.gz",
            "tether_path": "x.tether.json",
            "sidecar_path": "x.json",
            "n_volumes": 1,
        }
    ]
    assert "nifti_path" not in columns
    assert "conversion_status" in columns


def test_a_converted_series_cannot_be_marked_failed(tmp_path):
    write_series(make_series(), tmp_path / "src")
    dataset = tmp_path / "dataset"
    write_manifest(dataset, scan(tmp_path / "src"))
    db = manifest.open_manifest(dataset)
    try:
        output = manifest.Output(1, "a.nii.gz", "a.tether.json", "a.json", 1)
        names = manifest.Names("Sub", "20260101", "stem", None, "P1", "Sub")
        manifest.record_converted(db, 1, names, [output])
        manifest.record_failed(db, 1, "cannot read header")
        assert _rows(db, "SELECT conversion_status, conversion_error FROM series") == [
            {"conversion_status": "converted", "conversion_error": None}
        ]
        assert len(_rows(db, "SELECT * FROM outputs")) == 1
    finally:
        db.close()
