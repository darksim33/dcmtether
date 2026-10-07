"""Dataset manifest: one SQLite file per dataset (PLAN.md §2).

The manifest holds Study/Series UIDs and source paths, so it is as sensitive as the source
DICOM headers and lives inside the (non-shareable) dataset directory.
"""

import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from dcmtether import __version__
from dcmtether.scan import ScanResult

MANIFEST_NAME = "manifest.sqlite"
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE series (
    id                          INTEGER PRIMARY KEY,
    series_instance_uid         TEXT NOT NULL UNIQUE,
    study_instance_uid          TEXT,
    status                      TEXT NOT NULL
                                CHECK (status IN ('supported', 'split', 'unsupported')),
    reasons                     TEXT NOT NULL,  -- JSON list of human-readable strings
    modality                    TEXT,
    sop_class_uid               TEXT,
    transfer_syntax_uid         TEXT,
    implementation_class_uid    TEXT,
    implementation_version_name TEXT,
    manufacturer                TEXT,
    manufacturer_model_name     TEXT,
    software_versions           TEXT,
    body_part_examined          TEXT,
    series_number               INTEGER,
    series_description          TEXT,
    n_instances                 INTEGER NOT NULL,
    source_dir                  TEXT NOT NULL,  -- POSIX, relative to meta.source_root
    geometry                    TEXT NOT NULL   -- JSON summary from the scan
);
CREATE TABLE instances (
    path             TEXT PRIMARY KEY,           -- POSIX, relative to meta.source_root
    series_id        INTEGER NOT NULL REFERENCES series (id),
    sop_instance_uid TEXT,
    instance_number  INTEGER
);
CREATE INDEX instances_series ON instances (series_id);
CREATE TABLE skipped_files (
    path   TEXT PRIMARY KEY,                     -- POSIX, relative to meta.source_root
    reason TEXT NOT NULL
);
"""


def manifest_path(dataset: Path) -> Path:
    return dataset / MANIFEST_NAME


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_manifest(dataset: Path, result: ScanResult) -> Path:
    """Create the manifest for ``result``. Refuses to overwrite an existing one.

    Series ids are 1..n in ``result.series`` order.
    """
    target = manifest_path(dataset)
    if target.exists():
        raise FileExistsError(f"manifest already exists: {target}")
    dataset.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".sqlite.partial")
    partial.unlink(missing_ok=True)

    with closing(sqlite3.connect(partial)) as db, db:
        db.executescript(_SCHEMA)
        db.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", str(SCHEMA_VERSION)),
                ("tool_version", __version__),
                ("source_root", str(result.root)),
                ("scanned_at", datetime.now(UTC).isoformat(timespec="seconds")),
            ],
        )
        for series_id, series in enumerate(result.series, start=1):
            first = series.first
            source_dir = Path(os.path.commonpath([i.path.parent for i in series.instances]))
            db.execute(
                """INSERT INTO series (
                    id, series_instance_uid, study_instance_uid, status, reasons, modality,
                    sop_class_uid, transfer_syntax_uid, implementation_class_uid,
                    implementation_version_name, manufacturer, manufacturer_model_name,
                    software_versions, body_part_examined, series_number, series_description,
                    n_instances, source_dir, geometry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    series_id,
                    series.series_instance_uid,
                    first.study_instance_uid,
                    series.status.value,
                    json.dumps(series.reasons),
                    first.modality,
                    first.sop_class_uid,
                    first.transfer_syntax_uid,
                    first.implementation_class_uid,
                    first.implementation_version_name,
                    first.manufacturer,
                    first.manufacturer_model_name,
                    first.software_versions,
                    first.body_part_examined,
                    first.series_number,
                    first.series_description,
                    len(series.instances),
                    _relative(source_dir, result.root),
                    json.dumps(series.geometry),
                ),
            )
            db.executemany(
                "INSERT INTO instances (path, series_id, sop_instance_uid, instance_number) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        _relative(i.path, result.root),
                        series_id,
                        i.sop_instance_uid,
                        i.instance_number,
                    )
                    for i in series.instances
                ],
            )
        db.executemany(
            "INSERT INTO skipped_files (path, reason) VALUES (?, ?)",
            [(_relative(s.path, result.root), s.reason) for s in result.skipped],
        )
    partial.replace(target)
    return target
