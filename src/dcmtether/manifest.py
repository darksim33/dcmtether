"""Dataset manifest: one SQLite file per dataset.

The manifest holds Study/Series UIDs and source paths, so it is as sensitive as the source
DICOM headers and lives inside the (non-shareable) dataset directory.
"""

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import astuple, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dcmtether import __version__
from dcmtether.errors import DcmtetherError
from dcmtether.scanning import ScanResult

MANIFEST_NAME = "manifest.sqlite"
SCHEMA_VERSION = 5

# Version 1; newer versions are reached via _UPGRADES only, so new
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
    # 1 → 2: labels and outputs of the NIfTI conversion; paths relative to the dataset.
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
    # 2 → 3: a series can have several outputs (chunks of a split series).
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
    # 3 → 4: the names a series was converted under, so later runs never rebuild them from the
    # source (which may be gone) or from naming rules (which may have changed).
    4: """
ALTER TABLE series ADD COLUMN stem TEXT;          -- file stem without counter and chunk
ALTER TABLE series ADD COLUMN dup INTEGER;        -- the _XX counter, NULL if none
ALTER TABLE series ADD COLUMN patient_id TEXT;    -- '' if the header has none; NULL: unknown
ALTER TABLE series ADD COLUMN subject_base TEXT;  -- subject label before clash numbering
""",
    # 4 → 5: series replaced by a merged file (merge --replace) point to the series that holds it.
    5: """
ALTER TABLE series ADD COLUMN merged_into INTEGER;  -- id of the series with the merged output
""",
}


class ManifestError(DcmtetherError, RuntimeError):
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
        raise ManifestError("no manifest in DATASET (run 'dcmtether scan' first)")
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    with db:
        _upgrade(db)
    return db


def source_root(db: sqlite3.Connection) -> Path:
    return Path(db.execute("SELECT value FROM meta WHERE key = 'source_root'").fetchone()[0])


#: ``meta.subject_names``: subject labels from 5+5 letters of the name, or from the full name.
BRIEF_NAMES = "brief"
FULL_NAMES = "full"


def subject_names(db: sqlite3.Connection) -> str | None:
    """How the dataset builds subject labels; ``None`` until its first conversion."""
    row = db.execute("SELECT value FROM meta WHERE key = 'subject_names'").fetchone()
    return row[0] if row else None


def set_subject_names(db: sqlite3.Connection, value: str) -> None:
    with db:
        db.execute("DELETE FROM meta WHERE key = 'subject_names'")
        db.execute("INSERT INTO meta (key, value) VALUES ('subject_names', ?)", (value,))


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


@dataclass(frozen=True)
class Names:
    """The names a series is converted under. Once stored they are fixed: later runs read
    them from here instead of rebuilding them."""

    subject: str
    session: str
    #: File stem without counter and chunk.
    stem: str
    #: The ``_XX`` counter of series sharing a stem, if any.
    dup: int | None
    #: ``""`` if the header has none; ``None`` if unknown (converted by an older version and
    #: the source could not be read since).
    patient_id: str | None
    #: Subject label before clash numbering (``PhantQa`` for ``PhantQa02``).
    subject_base: str


_NAME_COLUMNS = "subject, session, stem, dup, patient_id, subject_base"


def set_names(db: sqlite3.Connection, series_id: int, names: Names) -> None:
    db.execute(
        f"UPDATE series SET ({_NAME_COLUMNS}) = (?, ?, ?, ?, ?, ?) WHERE id = ?",
        (*astuple(names), series_id),
    )


def record_converted(
    db: sqlite3.Connection, series_id: int, names: Names, outputs: list[Output]
) -> None:
    with db:
        set_names(db, series_id, names)
        db.execute(
            """UPDATE series SET conversion_status = 'converted', conversion_error = NULL
               WHERE id = ?""",
            (series_id,),
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


def record_merged(
    db: sqlite3.Connection, first_id: int, other_ids: list[int], output: Output
) -> None:
    """A merged file replaces the outputs of several series: it becomes the output of
    ``first_id``; the others keep no output of their own and point to ``first_id``. All stay
    converted, so they are never converted again."""
    with db:
        db.execute(
            "UPDATE outputs SET nifti_path = ?, tether_path = ?, sidecar_path = ?, "
            "n_volumes = ? WHERE series_id = ? AND chunk = ?",
            (
                output.nifti_path,
                output.tether_path,
                output.sidecar_path,
                output.n_volumes,
                first_id,
                output.chunk,
            ),
        )
        for series_id in other_ids:
            db.execute("DELETE FROM outputs WHERE series_id = ?", (series_id,))
            db.execute("UPDATE series SET merged_into = ? WHERE id = ?", (first_id, series_id))


def record_failed(db: sqlite3.Connection, series_id: int, error: str) -> None:
    """Mark a series as failed. A converted series is left alone: its files exist, and nothing
    that goes wrong later (e.g. an unreadable source) may take that record away."""
    with db:
        updated = db.execute(
            "UPDATE series SET conversion_status = 'failed', conversion_error = ? "
            "WHERE id = ? AND conversion_status IS NOT 'converted'",
            (error, series_id),
        )
        if updated.rowcount:
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
        raise FileExistsError("manifest already exists in DATASET")
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


def _release_paths(db: sqlite3.Connection, result: ScanResult, known: dict) -> None:
    """Free the paths the new scan gives to series that are (re)written.

    A path belongs to one series (primary key), but files get replaced between scans. Rows of
    series that are rescanned anyway go first; a series the scan no longer finds loses the
    paths that are another series' files now. A converted series keeps its rows: a file of
    it that belongs to another series now is an error (by id).
    """
    claimed: dict[str, str] = {}  # path -> the series that has the file now
    for series in result.series:
        series_id, conversion_status = known.get(series.series_instance_uid, (None, None))
        if conversion_status == "converted":
            continue
        if series_id is not None:
            db.execute("DELETE FROM instances WHERE series_id = ?", (series_id,))
        owner = "a new series" if series_id is None else f"series #{series_id}"
        claimed.update((_relative(path, result.root), owner) for path in series.files)
    converted = {sid for sid, status in known.values() if status == "converted"}
    for row in db.execute("SELECT path, series_id FROM instances").fetchall():
        if row["path"] not in claimed:
            continue
        if row["series_id"] in converted:
            raise ManifestError(
                f"a source file of converted series #{row['series_id']} now belongs to "
                f"{claimed[row['path']]} (restore it, or remove the outputs and manifest entry "
                f"of series #{row['series_id']})"
            )
        db.execute("DELETE FROM instances WHERE path = ?", (row["path"],))


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
            _release_paths(db, result, known)
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
