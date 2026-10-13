import shutil
from pathlib import Path

import numpy as np
import pydicom
import pytest
from click.testing import CliRunner
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import ImplicitVRLittleEndian, RLELossless
from pydicom.valuerep import DS
from synthetic import SeriesSpec, compressed, make_series, write_series

from dcmtether.cli import main
from dcmtether.comparison import compare_files, compare_series, dicom_files

SECRETS = ("Synthetic", "Phantom", "SYNTH", "2.25.", "20260101", "IM_0")


def _pair(tmp_path: Path, **extra) -> tuple[Path, Path]:
    series = make_series(SeriesSpec(n_slices=3, extra=extra))
    for ds in series.datasets:
        item = Dataset()
        item.SliceThickness = DS("3.0")
        item.ReferencedSOPInstanceUID = ds.SOPInstanceUID
        ds.ReferencedImageSequence = Sequence([item])
    original = tmp_path / "original"
    write_series(series, original)
    rebuilt = tmp_path / "rebuilt"
    shutil.copytree(original, rebuilt)
    return original, rebuilt


def _edit(path: Path, change) -> None:
    ds = pydicom.dcmread(path)
    change(ds)
    ds.save_as(path)


def _report(original: Path, rebuilt: Path):
    return compare_series(dicom_files(original), dicom_files(rebuilt))


def _assert_no_values(lines: list[str]) -> None:
    text = "\n".join(lines)
    for secret in SECRETS:
        assert secret not in text


def test_identical_copies(tmp_path):
    original, rebuilt = _pair(tmp_path)
    report = _report(original, rebuilt)
    assert report.identical, report.lines()
    assert report.n_compared == 3
    assert report.notes == []


def _set_pixel(ds):
    pixels = ds.pixel_array.copy()
    pixels[0, 0] += 1
    ds.PixelData = pixels.tobytes()


def _set_nested(ds):
    ds.ReferencedImageSequence[0].SliceThickness = DS("4.0")


def _drop_nested_item(ds):
    ds.ReferencedImageSequence = Sequence([])


def _set_private(ds):
    ds.add_new(0x00091010, "LO", "changed")


CHANGES = {
    "value": (lambda ds: setattr(ds, "PatientName", "Other^Name"), "PatientName: value differs"),
    "ds-string": (
        lambda ds: setattr(ds, "RepetitionTime", DS("500.0")),
        "RepetitionTime: value differs",
    ),
    "missing": (lambda ds: delattr(ds, "EchoTrainLength"), "EchoTrainLength: missing in rebuilt"),
    "extra": (lambda ds: setattr(ds, "ImageComments", "x"), "ImageComments: extra in rebuilt"),
    "vr": (lambda ds: setattr(ds["EchoTrainLength"], "VR", "US"), "EchoTrainLength: VR differs"),
    "nested": (_set_nested, "ReferencedImageSequence[0].SliceThickness: value differs"),
    "items": (_drop_nested_item, "ReferencedImageSequence: number of items differs"),
    "private": (_set_private, "(0009,1010): extra in rebuilt"),
    "pixel": (_set_pixel, "PixelData: decoded pixels differ"),
    "meta": (
        lambda ds: setattr(ds.file_meta, "ImplementationVersionName", "OTHER"),
        "ImplementationVersionName: value differs",
    ),
}


@pytest.mark.parametrize(("change", "expected"), CHANGES.values(), ids=CHANGES.keys())
def test_detects_difference(tmp_path, change, expected):
    original, rebuilt = _pair(tmp_path)
    _edit(rebuilt / "IM_0002", change)
    report = _report(original, rebuilt)
    assert not report.identical
    assert [f"instance 2: {expected}"] == report.lines()
    _assert_no_values(report.lines())


def test_file_names_do_not_matter_instances_pair_by_sop_uid(tmp_path):
    original, rebuilt = _pair(tmp_path)
    (rebuilt / "IM_0001").rename(rebuilt / "renamed")
    assert _report(original, rebuilt).identical


def test_missing_and_extra_instances(tmp_path):
    original, rebuilt = _pair(tmp_path)
    (rebuilt / "IM_0003").unlink()
    other = make_series(SeriesSpec(n_slices=1, seed=9))
    write_series(other, tmp_path / "other")
    shutil.copy(tmp_path / "other" / "IM_0001", rebuilt / "stray")
    report = _report(original, rebuilt)
    assert sorted(report.lines()) == [
        "instance 1: extra instance (no matching SOPInstanceUID in original)",
        "instance 3: missing in rebuilt",
    ]


def test_unreadable_file_is_a_difference(tmp_path):
    original, rebuilt = _pair(tmp_path)
    (rebuilt / "notes.txt").write_text("hello")
    report = _report(original, rebuilt)
    assert report.lines() == ["rebuilt file #4: not a readable DICOM file"]


def test_raw_pixel_bytes_differing_with_equal_decoded_values_is_a_note(tmp_path):
    # BitsStored 12: bits 12-15 are unused, so they don't change the decoded value.
    original, rebuilt = _pair(tmp_path, BitsStored=12, HighBit=11)
    _edit(original / "IM_0001", lambda ds: setattr(ds, "PixelData", _junk_high_bits(ds)))
    report = _report(original, rebuilt)
    assert report.identical, report.lines()
    assert [f"{n.instance}: {n.where}: {n.kind}" for n in report.notes] == [
        "instance 1: PixelData: raw bytes differ (decoded pixels equal)"
    ]


def _junk_high_bits(ds) -> bytes:
    raw = np.frombuffer(ds.PixelData, dtype="<u2")
    return (raw | 0xF000).astype("<u2").tobytes()


def test_compare_files_on_identical_file(tmp_path):
    original, _ = _pair(tmp_path)
    assert compare_files(original / "IM_0001", original / "IM_0001") == ([], [])


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_compare_identical(tmp_path):
    original, rebuilt = _pair(tmp_path)
    result = CliRunner().invoke(main, ["compare", str(original), str(rebuilt)])
    assert result.exit_code == 0, result.output
    assert "Identical: 3 instances compared." in result.output


def test_cli_compare_differences(tmp_path):
    original, rebuilt = _pair(tmp_path)
    _edit(rebuilt / "IM_0002", lambda ds: setattr(ds, "PatientName", "Other^Name"))
    result = CliRunner().invoke(main, ["compare", str(original), str(rebuilt)])
    assert result.exit_code == 1
    assert "1 difference in 3 instances compared:" in result.output
    assert "  instance 2: PatientName: value differs" in result.output
    _assert_no_values(result.output.splitlines())
    result.output.encode("ascii")


# --- transfer syntax changes -------------------------------------------------------------------


def _compressed_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Original RLE-compressed; 'rebuilt' = the same files decompressed (Explicit VR LE)."""
    series = compressed(make_series(SeriesSpec(n_slices=2)), RLELossless)
    original = tmp_path / "original"
    write_series(series, original)
    rebuilt = tmp_path / "rebuilt"
    shutil.copytree(original, rebuilt)
    for path in dicom_files(rebuilt):
        _edit(path, lambda ds: ds.decompress(generate_instance_uid=False))
    return original, rebuilt


def test_lossless_to_uncompressed_is_a_note(tmp_path):
    original, rebuilt = _compressed_pair(tmp_path)
    report = _report(original, rebuilt)
    assert report.identical, report.lines()
    assert sorted({f"{n.where}: {n.kind}" for n in report.notes}) == [
        "TransferSyntaxUID: changed from RLE Lossless to Explicit VR Little Endian (lossless)"
    ]


def test_pixel_change_across_transfer_syntax_change_is_a_difference(tmp_path):
    original, rebuilt = _compressed_pair(tmp_path)
    _edit(rebuilt / "IM_0001", _set_pixel)
    assert _report(original, rebuilt).lines() == ["instance 1: PixelData: decoded pixels differ"]


def test_other_transfer_syntax_changes_are_differences(tmp_path):
    original, rebuilt = _pair(tmp_path)
    for path in dicom_files(rebuilt):
        _edit(path, lambda ds: setattr(ds.file_meta, "TransferSyntaxUID", ImplicitVRLittleEndian))
    assert _report(original, rebuilt).lines() == [
        f"instance {n}: TransferSyntaxUID: value differs" for n in (1, 2, 3)
    ]


def test_cli_compare_groups_notes(tmp_path):
    original, rebuilt = _compressed_pair(tmp_path)
    result = CliRunner().invoke(main, ["compare", str(original), str(rebuilt)])
    assert result.exit_code == 0, result.output
    assert result.output == (
        "Identical: 2 instances compared.\n"
        "  note: TransferSyntaxUID: changed from RLE Lossless to Explicit VR Little Endian "
        "(lossless) (2 instances)\n"
    )
