"""Dataset manifest: one SQLite file per dataset (PLAN.md §2).

The manifest holds Study/Series UIDs and source paths, so it is as sensitive as the source
DICOM headers and lives inside the (non-shareable) dataset directory.
"""

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dcmtether import __version__
from dcmtether.scan import ScanResult

MANIFEST_NAME = "manifest.sqlite"
SCHEMA_VERSION = 3

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
    # 2 → 3 (Phase 4): a series can have several outputs (chunks of a split series).
    3: """
CREATE TABLE outputs (
    series_id    INTEGER NOT NULL REFERENCES series (id),
    chunk        INTEGER NOT NULL,          -- 1..n; one NIfTI per image plane of a series
    nifti_path   TEXT NOT NULL UNIQUE,      -- POSIX, relative to the dataset
    tether_path  TEXT NOT NULL,
    sidecar_path TEXT NOT NULL,
    n_volumes    INTEGER NOT NULL,
    PRIMARY KEY (series_id, chunk)
);
INSERT INTO outputs (series_id, chunk, nifti_path, tether_path, sidecar_path, n_volumes)
    SELECT id, 1, nifti_path, tether_path, sidecar_path, 1 FROM series
    WHERE nifti_path IS NOT NULL;
ALTER TABLE series DROP COLUMN nifti_path;
ALTER TABLE series DROP COLUMN tether_path;
ALTER TABLE series DROP COLUMN sidecar_path;
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


@dataclass(frozen=True)
class Output:
    chunk: int
    nifti_path: str
    tether_path: str
    sidecar_path: str
    n_volumes: int


def record_converted(
    db: sqlite3.Connection, series_id: int, subject: str, session: str, outputs: list[Output]
) -> None:
    with db:
        db.execute(
            """UPDATE series SET subject = ?, session = ?, conversion_status = 'converted',
               conversion_error = NULL WHERE id = ?""",
            (subject, session, series_id),
        )
        db.execute("DELETE FROM outputs WHERE series_id = ?", (series_id,))
        db.executemany(
            "INSERT INTO outputs (series_id, chunk, nifti_path, tether_path, sidecar_path, "
            "n_volumes) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (series_id, o.chunk, o.nifti_path, o.tether_path, o.sidecar_path, o.n_volumes)
                for o in outputs
            ],
        )


def record_failed(db: sqlite3.Connection, series_id: int, error: str) -> None:
    with db:
        db.execute(
            "UPDATE series SET conversion_status = 'failed', conversion_error = ? WHERE id = ?",
            (error, series_id),
        )
        db.execute("DELETE FROM outputs WHERE series_id = ?", (series_id,))


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


_SERIES_COLUMNS = (
    "series_instance_uid, study_instance_uid, status, reasons, modality, sop_class_uid, "
    "transfer_syntax_uid, implementation_class_uid, implementation_version_name, manufacturer, "
    "manufacturer_model_name, software_versions, body_part_examined, series_number, "
    "series_description, n_instances, source_dir, geometry"
)


def _files(series) -> dict[Path, object]:
    """One instance per file (multi-frame files hold several images)."""
    files: dict[Path, object] = {}
    for inst in series.instances:
        files.setdefault(inst.path, inst)
    return files


def _series_values(series, root: Path) -> tuple:
    first = series.first
    source_dir = Path(os.path.commonpath([path.parent for path in series.files]))
    return (
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
        len(_files(series)),
        _relative(source_dir, root),
        json.dumps(series.geometry),
    )


def _insert_instances(db: sqlite3.Connection, series_id: int, series, root: Path) -> None:
    db.executemany(
        "INSERT INTO instances (path, series_id, sop_instance_uid, instance_number) "
        "VALUES (?, ?, ?, ?)",
        [
            (_relative(i.path, root), series_id, i.sop_instance_uid, i.instance_number)
            for i in _files(series).values()
        ],
    )


def _insert_series(db: sqlite3.Connection, series_id: int, series, root: Path) -> None:
    db.execute(
        f"INSERT INTO series (id, {_SERIES_COLUMNS}) VALUES ({', '.join('?' * 19)})",
        (series_id, *_series_values(series, root)),
    )
    _insert_instances(db, series_id, series, root)


def _replace_skipped(db: sqlite3.Connection, result: ScanResult) -> None:
    db.execute("DELETE FROM skipped_files")
    db.executemany(
        "INSERT INTO skipped_files (path, reason) VALUES (?, ?)",
        [(_relative(s.path, result.root), s.reason) for s in result.skipped],
    )


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
            _insert_series(db, series_id, series, result.root)
        _replace_skipped(db, result)
    partial.replace(target)
    return target


@dataclass
class ManifestUpdate:
    """What :func:`update_manifest` did, by series id."""

    new: list[int] = field(default_factory=list)
    #: Not converted yet (or failed): classification and files replaced by the new scan.
    rescanned: list[int] = field(default_factory=list)
    #: Converted, files unchanged.
    unchanged: list[int] = field(default_factory=list)
    #: Converted, but the source files differ now: reported, never reconverted automatically.
    changed_after_conversion: list[int] = field(default_factory=list)
    #: In the manifest but not found by the new scan (kept).
    missing: list[int] = field(default_factory=list)


def update_manifest(dataset: Path, result: ScanResult) -> ManifestUpdate:
    """Merge a new scan of the same SOURCE into an existing manifest (``dcmtether run``).

    Ids of known series never change; new series get the next ids. Converted series are left
    alone. The whole update is one transaction.
    """
    db = open_manifest(dataset)
    update = ManifestUpdate()
    try:
        if Path(source_root(db)) != Path(result.root):
            raise ManifestError("SOURCE is not the folder this dataset was scanned from")
        known = {
            row["series_instance_uid"]: (row["id"], row["conversion_status"])
            for row in db.execute("SELECT id, series_instance_uid, conversion_status FROM series")
        }
        next_id = 1 + max((sid for sid, _ in known.values()), default=0)
        with db:
            for series in result.series:
                entry = known.get(series.series_instance_uid)
                if entry is None:
                    _insert_series(db, next_id, series, result.root)
                    update.new.append(next_id)
                    next_id += 1
                    continue
                series_id, conversion_status = entry
                if conversion_status == "converted":
                    old = set(instance_paths(db, series_id))
                    new = {_relative(path, result.root) for path in series.files}
                    target = update.unchanged if old == new else update.changed_after_conversion
                    target.append(series_id)
                    continue
                db.execute(
                    f"UPDATE series SET ({_SERIES_COLUMNS}) = ({', '.join('?' * 18)}) WHERE id = ?",
                    (*_series_values(series, result.root), series_id),
                )
                db.execute("DELETE FROM instances WHERE series_id = ?", (series_id,))
                _insert_instances(db, series_id, series, result.root)
                update.rescanned.append(series_id)
            scanned = {s.series_instance_uid for s in result.series}
            update.missing = sorted(sid for uid, (sid, _) in known.items() if uid not in scanned)
            _replace_skipped(db, result)
            db.execute(
                "UPDATE meta SET value = ? WHERE key = 'scanned_at'",
                (datetime.now(UTC).isoformat(timespec="seconds"),),
            )
    finally:
        db.close()
    return update


def series_for_output(dataset: Path, nifti: Path) -> int | None:
    """Manifest series id of an output NIfTI of ``dataset`` (None if it is not one)."""
    try:
        rel = Path(nifti).resolve().relative_to(Path(dataset).resolve()).as_posix()
    except ValueError:
        return None
    with closing(sqlite3.connect(manifest_path(dataset))) as db:
        row = db.execute("SELECT series_id FROM outputs WHERE nifti_path = ?", (rel,)).fetchone()
    return row[0] if row else None


def series_ids(dataset: Path) -> dict[str, int]:
    """SeriesInstanceUID -> manifest series id (for reports by id; never print the UIDs)."""
    with closing(sqlite3.connect(manifest_path(dataset))) as db:
        return dict(db.execute("SELECT series_instance_uid, id FROM series"))
