"""Shared helpers for forward + reverse round-trip tests."""

import sqlite3
from pathlib import Path

from synthetic import (
    AXIAL,
    CORONAL,
    SAGITTAL,
    SeriesSpec,
    flipped,
    rotated,
)

from dcmtether.compare import Report, compare_series, dicom_files
from dcmtether.forward import convert_dataset
from dcmtether.manifest import write_manifest
from dcmtether.reverse import to_dicom
from dcmtether.scan import scan

#: Geometry variants every conversion test runs over.
SPECS = {
    "axial": SeriesSpec(),
    "coronal": SeriesSpec(orientation=CORONAL),
    "sagittal": SeriesSpec(orientation=SAGITTAL),
    "flip-rows": SeriesSpec(orientation=flipped(AXIAL, rows=True)),
    "flip-both-sagittal": SeriesSpec(orientation=flipped(SAGITTAL, rows=True, cols=True)),
    "oblique": SeriesSpec(orientation=rotated(CORONAL, (1.0, 1.0, 0.0), 17.0)),
    "oblique-40": SeriesSpec(orientation=rotated(SAGITTAL, (0.0, 0.3, 1.0), 40.0)),
    "descending": SeriesSpec(descending=True),
    "shuffled": SeriesSpec(shuffle=True, seed=3),
    "signed": SeriesSpec(signed=True),
    "single-slice": SeriesSpec(n_slices=1),
}


def manifest_rows(dataset: Path) -> list[dict]:
    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM series ORDER BY id")]


def output_rows(dataset: Path) -> list[dict]:
    """One row per written NIfTI, ordered by series id and chunk."""
    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM outputs ORDER BY series_id, chunk")]


def forward(source: Path, dataset: Path) -> list[dict]:
    """scan + to-nii; every series must convert. Returns the output rows."""
    write_manifest(dataset, scan(source))
    outcomes = convert_dataset(dataset)
    assert outcomes and all(o.ok for o in outcomes), [o.message for o in outcomes]
    return output_rows(dataset)


def round_trip(source: Path, work: Path) -> Report:
    """One series in ``source`` → NIfTI(s) + tether → DICOM; compare with the source files."""
    rebuilt = work / "rebuilt"
    written = []
    for row in forward(source, work / "dataset"):
        written += to_dicom(work / "dataset" / row["nifti_path"], rebuilt)
    assert sorted(written) == dicom_files(rebuilt)
    return compare_series(dicom_files(source), dicom_files(rebuilt))
