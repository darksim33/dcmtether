import io
import json

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.valuerep import DS
from synthetic import SeriesSpec, make_series

from dcmtether import tether


def _raw(ds: Dataset) -> Dataset:
    """Write and re-read: elements come back as raw (unconverted) data elements."""
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    buffer.seek(0)
    return pydicom.dcmread(buffer)


def _forced_ds(text: str) -> DS:
    """A DS that pydicom writes exactly as ``text`` (DS() itself strips spaces)."""
    value = DS(text.strip())
    value.original_string = text
    return value


def _series(n: int = 2, **extra) -> list[Dataset]:
    return [_raw(ds) for ds in make_series(SeriesSpec(n_slices=n, extra=extra)).datasets]


def _document(datasets: list[Dataset]) -> dict:
    strings = [tether.capture_strings(ds) for ds in datasets]
    return tether.build(
        datasets,
        [f"IM_{k}" for k in range(len(datasets))],
        [[k] for k in range(len(datasets))],
        np.dtype("uint16"),
        (6, 8, len(datasets)),
        np.array([[0, 1], [1, 1], [2, 1]]),
        strings,
    )


# --- DS/IS strings -----------------------------------------------------------------------------


def test_canonical_strings_are_not_recorded():
    [ds] = _series(1, RepetitionTime=DS("500.0"), EchoTrainLength=1)
    strings = tether.capture_strings(ds)
    assert "00180080" not in strings  # RepetitionTime
    assert "00180091" not in strings  # EchoTrainLength


@pytest.mark.parametrize(
    ("keyword", "value", "expected"),
    [
        ("RepetitionTime", DS("500.000000"), ["500.000000"]),
        ("RepetitionTime", _forced_ds(" 1.0"), [" 1.0"]),
        ("EchoTrainLength", "0005", ["0005"]),
        ("EchoTrainLength", "+05", ["+05 "]),  # odd length: the pad space is part of the value
        ("PixelSpacing", [DS("0.800"), DS("0.9")], ["0.800", "0.9 "]),
    ],
)
def test_non_canonical_strings_are_recorded(keyword, value, expected):
    [ds] = _series(1, **{keyword: value})
    tag = f"{pydicom.datadict.tag_for_keyword(keyword):08X}"
    assert tether.capture_strings(ds)[tag] == expected


@pytest.mark.parametrize(
    ("keyword", "value", "expected"),
    [
        ("ReferringPhysicianName", "  ", ["  "]),  # JSON: empty element
        ("ImageComments", "text   ", ["text    "]),  # JSON drops trailing spaces; + pad
    ],
)
def test_other_text_vrs_are_recorded(keyword, value, expected):
    [ds] = _series(1, **{keyword: value})
    tag = f"{pydicom.datadict.tag_for_keyword(keyword):08X}"
    assert tether.capture_strings(ds)[tag] == expected


def test_strings_inside_sequences_use_item_paths():
    item = Dataset()
    item.SliceThickness = DS("3.00")
    [ds] = _series(1, ReferencedImageSequence=Sequence([Dataset(), item]))
    assert tether.capture_strings(ds)["00081140/1/00180050"] == ["3.00"]


def test_strings_must_be_captured_before_elements_are_converted():
    [ds] = _series(1, RepetitionTime=_forced_ds(" 1.0"))
    _ = ds.RepetitionTime  # converts the element; the leading space is gone
    with pytest.raises(tether.TetherError, match="RepetitionTime"):
        tether.capture_strings(ds)


def test_equal_numbers_with_different_strings_are_not_shared():
    datasets = _series(2, RepetitionTime=DS("500.0"))  # the canonical form: not recorded
    datasets[1].RepetitionTime = DS("500.00")
    datasets = [_raw(ds) for ds in datasets]
    doc = _document(datasets)
    assert "00180080" not in doc["shared"]
    assert "00180080" not in doc.get("original_strings", {})
    assert [i["original_strings"].get("00180080") for i in doc["instances"]] == [None, ["500.00"]]
    assert all("00180080" in i["elements"] for i in doc["instances"])


def test_equal_strings_are_shared():
    doc = _document(_series(2, RepetitionTime=DS("500.000")))
    assert "00180080" in doc["shared"]
    assert doc["original_strings"]["00180080"] == ["500.000 "]
    assert all("00180080" not in i["original_strings"] for i in doc["instances"])


def test_rebuild_restores_strings(tmp_path):
    item = Dataset()
    item.SliceThickness = DS("3.00")
    datasets = _series(
        2,
        RepetitionTime=_forced_ds(" 1.0"),
        EchoTrainLength="0005",
        ReferencedImageSequence=Sequence([item]),
    )
    doc = _document(datasets)
    path = tmp_path / "x.tether.json"
    tether.write(doc, path)
    rebuilt = tether.rebuild_datasets(tether.load(path))
    for ds in rebuilt:
        assert ds["RepetitionTime"].value.original_string == " 1.0"
        assert ds["EchoTrainLength"].value.original_string == "0005"
        assert ds.ReferencedImageSequence[0]["SliceThickness"].value.original_string == "3.00"
        assert "PixelData" not in ds
    assert [ds.SOPInstanceUID for ds in rebuilt] == [ds.SOPInstanceUID for ds in datasets]
    assert [ds.file_meta.MediaStorageSOPInstanceUID for ds in rebuilt] == [
        ds.SOPInstanceUID for ds in datasets
    ]


# --- document ----------------------------------------------------------------------------------


def test_document_records_format_and_pixel_data_vr():
    doc = _document(_series(2))
    assert doc["format_version"] == 2
    assert doc["pixel"]["pixel_data_vr"] == "OW"


def test_nan_cannot_be_stored(tmp_path):
    [ds] = _series(1)
    ds.add_new(0x00189087, "FD", float("nan"))  # DiffusionBValue
    with pytest.raises(tether.TetherError, match="NaN or infinite"):
        tether.write(_document([_raw(ds)]), tmp_path / "x.tether.json")
    assert not (tmp_path / "x.tether.json").exists()


@pytest.mark.parametrize("version", [1, 3])
def test_load_refuses_other_format_versions(tmp_path, version):
    path = tmp_path / "x.tether.json"
    tether.write(_document(_series(1)), path)
    doc = json.loads(path.read_text("utf-8"))
    doc["format_version"] = version
    path.write_text(json.dumps(doc), "utf-8")
    with pytest.raises(tether.TetherError, match=f"format version {version}"):
        tether.load(path)


def test_load_refuses_non_tether_json(tmp_path):
    path = tmp_path / "x.tether.json"
    path.write_text("[]", "utf-8")
    with pytest.raises(tether.TetherError, match="not a tether document"):
        tether.load(path)
