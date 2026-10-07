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
SCHEMA_VERSION = 2

# Version 1 as written by Phase 1; newer versions are reached via _UPGRADES only, so new
# manifests and upgraded old ones always end up identical.
_SCHEMA_V1 = """
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

_UPGRADES = {
    # 1 → 2 (Phase 2): labels and outputs of the NIfTI conversion; paths relative to the dataset.
    2: """
ALTER TABLE series ADD COLUMN subject TEXT;
ALTER TABLE series ADD COLUMN session TEXT;
ALTER TABLE series ADD COLUMN nifti_path TEXT;
ALTER TABLE series ADD COLUMN tether_path TEXT;
ALTER TABLE series ADD COLUMN sidecar_path TEXT;
ALTER TABLE series ADD COLUMN conversion_status TEXT
    CHECK (conversion_status IN ('converted', 'failed'));
ALTER TABLE series ADD COLUMN conversion_error TEXT;
""",
}


class ManifestError(RuntimeError):
    pass


def manifest_path(dataset: Path) -> Path:
    return dataset / MANIFEST_NAME


def _schema_version(db: sqlite3.Connection) -> int:
    return int(db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0])


def _upgrade(db: sqlite3.Connection) -> None:
    version = _schema_version(db)
    if version > SCHEMA_VERSION:
        raise ManifestError(
            f"manifest schema {version} is newer than this dcmtether supports ({SCHEMA_VERSION})"
        )
    for target in range(version + 1, SCHEMA_VERSION + 1):
        db.executescript(_UPGRADES[target])
        db.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(target),))


def open_manifest(dataset: Path) -> sqlite3.Connection:
    """Open an existing manifest, upgrading old schema versions in place."""
    path = manifest_path(dataset)
    if not path.exists():
        raise ManifestError(f"no manifest in {dataset} (run 'dcmtether scan' first)")
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    with db:
        _upgrade(db)
    return db


def source_root(db: sqlite3.Connection) -> Path:
    return Path(db.execute("SELECT value FROM meta WHERE key = 'source_root'").fetchone()[0])


def instance_paths(db: sqlite3.Connection, series_id: int) -> list[str]:
    rows = db.execute("SELECT path FROM instances WHERE series_id = ? ORDER BY path", (series_id,))
    return [r[0] for r in rows]


def record_converted(
    db: sqlite3.Connection,
    series_id: int,
    subject: str,
    session: str,
    nifti_path: str,
    tether_path: str,
    sidecar_path: str,
) -> None:
    with db:
        db.execute(
            """UPDATE series SET subject = ?, session = ?, nifti_path = ?, tether_path = ?,
               sidecar_path = ?, conversion_status = 'converted', conversion_error = NULL
               WHERE id = ?""",
            (subject, session, nifti_path, tether_path, sidecar_path, series_id),
        )


def record_failed(db: sqlite3.Connection, series_id: int, error: str) -> None:
    with db:
        db.execute(
            "UPDATE series SET conversion_status = 'failed', conversion_error = ? WHERE id = ?",
            (error, series_id),
        )


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
        db.executescript(_SCHEMA_V1)
        db.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", "1"),
                ("tool_version", __version__),
                ("source_root", str(result.root)),
                ("scanned_at", datetime.now(UTC).isoformat(timespec="seconds")),
            ],
        )
        _upgrade(db)
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
