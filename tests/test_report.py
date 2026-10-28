"""Report lines shared by the CLI and the GUI (the CLI tests check the full console text)."""

from pathlib import Path

from dcmtether import report
from dcmtether.api import ConversionResult, DcmResult, ScanSummary, SeriesInfo
from dcmtether.derived_mr import DerivedMRResult
from dcmtether.forward import Outcome
from dcmtether.manifest import ManifestUpdate
from dcmtether.scanning import Status

SUMMARY = ScanSummary(
    n_files=12,
    n_instances=11,
    series=(
        SeriesInfo(1, Status.SUPPORTED, (), 5),
        SeriesInfo(2, Status.SPLIT, ("3 distinct image planes",), 3),
        SeriesInfo(3, Status.UNSUPPORTED, ("missing slices",), 3),
    ),
    skipped={"not a DICOM file": 1},
)


def test_scan_lines():
    assert report.scan_lines(SUMMARY) == [
        "Scanned 12 files: 11 DICOM instances in 3 series, 1 file skipped.",
        "  supported:   1",
        "  split:       1",
        "  unsupported: 1",
        "  series #2 (3 instances) split: 3 distinct image planes",
        "  series #3 (3 instances) unsupported: missing slices",
        "  skipped 1 file: not a DICOM file",
    ]


def test_update_lines_list_only_new_and_rescanned_series():
    update = ManifestUpdate(new=[3], unchanged=[1, 2], changed_after_conversion=[1], missing=[4])
    lines = report.update_lines(SUMMARY, update)
    assert "  series #2 (3 instances) split: 3 distinct image planes" not in lines
    assert "  series #3 (3 instances) unsupported: missing slices" in lines
    assert lines[-3:] == [
        "Manifest: 1 new series, 0 rescanned, 2 unchanged.",
        "  series #1: source files changed after conversion (not reconverted; remove its "
        "outputs and manifest entry to redo it)",
        "  series #4: no longer found in SOURCE (kept in the manifest)",
    ]


def test_conversion_lines():
    result = ConversionResult(
        (
            Outcome(1, True, outputs=1),
            Outcome(2, True, outputs=3, warning="duplicate name"),
            Outcome(3, False, "PatientName is missing"),
        )
    )
    assert report.conversion_lines(result) == [
        "Converted 2 series, 1 failed.",
        "  series #2 warning: duplicate name",
        "  series #3 failed: PatientName is missing",
    ]
    assert "  series #2 converted (3 outputs)" in report.conversion_lines(result, verbose=True)


def test_dcm_and_mr_lines():
    paths = (Path("a"), Path("b"))
    assert report.dcm_lines(DcmResult(paths, ("RLE Lossless", "Explicit VR Little Endian"))) == [
        "Wrote 2 DICOM files.",
        "Transfer syntax changed from RLE Lossless to Explicit VR Little Endian (lossless).",
    ]
    assert report.mr_lines(DerivedMRResult(list(paths), "map", 1e-9, rounded=True)) == [
        "Wrote 2 derived MR images (map).",
        "Rescale rounded to DS precision: max error 1e-09 (to-pmap is exact).",
    ]
    assert report.mr_lines(DerivedMRResult(list(paths), "labels", 0.0)) == [
        "Wrote 2 derived MR images (labels)."
    ]
