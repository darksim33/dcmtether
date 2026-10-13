"""Phase 5c: derived MR images (DERIVED\\SECONDARY) as a fallback for PACS without SEG/PM."""

import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from click.testing import CliRunner
from dicom_validator.validator.validation_result import ErrorCode
from roundtrip import SPECS, forward, output_rows
from synthetic import make_4d, make_enhanced, make_series, write_series
from test_pmap import META as PMAP_META
from test_seg import META as SEG_META

from dcmtether import __version__, derived_mr
from dcmtether.cli import main
from dcmtether.forward import convert_dataset
from dcmtether.manifest import write_manifest
from dcmtether.scan import Status, scan


def _source(tmp_path: Path, series) -> tuple[Path, Path]:
    src = tmp_path / "src" / "series"
    write_series(series, src)
    [row] = forward(tmp_path / "src", tmp_path / "dataset")
    return tmp_path / "dataset" / row["nifti_path"], src


def _metadata(tmp_path: Path, meta: dict) -> Path:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta), "utf-8")
    return path


def _save(nifti: Path, data: np.ndarray, path: Path, scaling=None) -> None:
    img = nib.Nifti1Image(data, nib.load(nifti).affine)
    if scaling is not None:
        img.header.set_slope_inter(*scaling)
    nib.save(img, path)


def _float_map(nifti: Path, path: Path) -> np.ndarray:
    i, j, k = np.indices(nib.load(nifti).shape[:3])
    values = (1e-3 * (i + 10 * j + 100 * k) + 1e-4).astype(np.float32)
    _save(nifti, values, path)
    return values


def _labels(nifti: Path, path: Path) -> np.ndarray:
    i, j, k = np.indices(nib.load(nifti).shape[:3])
    labels = np.zeros(i.shape, np.uint8)
    labels[(i + 2 * j + 3 * k) % 4 == 1] = 1
    labels[(i + 2 * j + 3 * k) % 4 == 2] = 2
    _save(nifti, labels, path)
    return labels


def _reconvert(tmp_path: Path, output_dir: Path) -> nib.Nifti1Image:
    """What a PACS user would get back: scan + convert the derived series with dcmtether."""
    dataset = tmp_path / "rescan"
    result = scan(output_dir)
    [series] = result.series
    assert series.status is Status.SUPPORTED, series.reasons
    write_manifest(dataset, result)
    assert all(o.ok for o in convert_dataset(dataset))
    [row] = output_rows(dataset)
    return nib.load(dataset / row["nifti_path"])


# --- maps --------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["axial", "oblique", "flip-both-sagittal", "descending"])
def test_float_map_round_trips_through_derived_series(tmp_path, spec):
    nifti, _ = _source(tmp_path, make_series(SPECS[spec]))
    values = _float_map(nifti, tmp_path / "map.nii.gz")
    meta = derived_mr.load_metadata(_metadata(tmp_path, PMAP_META))
    result = derived_mr.to_derived_mr(nifti, tmp_path / "map.nii.gz", meta, tmp_path / "out")
    assert len(result.paths) == 5
    assert 0 < result.max_error <= (values.max() - values.min()) / 65535
    back = _reconvert(tmp_path, tmp_path / "out")
    np.testing.assert_allclose(back.affine, nib.load(nifti).affine, atol=1e-3)
    np.testing.assert_allclose(back.get_fdata(), values, atol=result.max_error * 1.001)


def test_integer_map_is_exact(tmp_path):
    nifti, _ = _source(tmp_path, make_series(SPECS["oblique"]))
    i, j, k = np.indices(nib.load(nifti).shape[:3])
    stored = (i + 10 * j + 100 * k - 50).astype(np.int16)
    _save(nifti, stored, tmp_path / "map.nii.gz", scaling=(0.5, 2.0))
    meta = derived_mr.load_metadata(_metadata(tmp_path, PMAP_META))
    result = derived_mr.to_derived_mr(nifti, tmp_path / "map.nii.gz", meta, tmp_path / "out")
    assert result.max_error == 0
    back = _reconvert(tmp_path, tmp_path / "out")
    np.testing.assert_array_equal(back.get_fdata(), stored * 0.5 + 2.0)


def test_label_map_pixels_are_label_values(tmp_path):
    nifti, _ = _source(tmp_path, make_series(SPECS["sagittal"]))
    labels = _labels(nifti, tmp_path / "labels.nii.gz")
    meta = derived_mr.load_metadata(_metadata(tmp_path, SEG_META))
    result = derived_mr.to_derived_mr(nifti, tmp_path / "labels.nii.gz", meta, tmp_path / "out")
    assert result.kind == "labels" and result.max_error == 0
    back = _reconvert(tmp_path, tmp_path / "out")
    np.testing.assert_array_equal(np.asarray(back.dataobj.get_unscaled()), labels)
    ds = pydicom.dcmread(result.paths[0])
    assert "RescaleSlope" not in ds and ds.BitsAllocated == 16


# --- header ------------------------------------------------------------------------------------


COPIED = (
    "PatientID",
    "StudyInstanceUID",
    "FrameOfReferenceUID",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
)


@pytest.fixture
def derived(tmp_path) -> tuple[list[Path], Path]:
    """Derived map images of a source with private tags and an overlay; (paths, source dir)."""
    series = make_series()
    for ds in series.datasets:
        block = ds.private_block(0x0029, "SIEMENS CSA HEADER", create=True)
        block.add_new(0x10, "OB", b"\x01\x02")
        ds.add_new(0x60000010, "US", 6)  # OverlayRows
        ds.WindowCenter, ds.WindowWidth = 100, 200
    nifti, src = _source(tmp_path, series)
    _float_map(nifti, tmp_path / "map.nii.gz")
    meta = derived_mr.load_metadata(_metadata(tmp_path, PMAP_META))
    result = derived_mr.to_derived_mr(nifti, tmp_path / "map.nii.gz", meta, tmp_path / "out")
    return result.paths, src


def test_derived_header(derived):
    paths, src = derived
    sources = {pydicom.dcmread(p).SOPInstanceUID: pydicom.dcmread(p) for p in src.iterdir()}
    series_uids = set()
    for path in paths:
        ds = pydicom.dcmread(path)
        source = sources[ds.SourceImageSequence[0].ReferencedSOPInstanceUID]
        assert ds.SourceImageSequence[0].ReferencedSOPClassUID == source.SOPClassUID
        assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
        assert list(ds.ImageType)[:2] == ["DERIVED", "SECONDARY"]
        for keyword in COPIED:
            assert ds[keyword].value == source[keyword].value, keyword
        assert ds.SOPInstanceUID.startswith("2.25.") and ds.SOPInstanceUID != source.SOPInstanceUID
        assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
        assert ds.SeriesInstanceUID != source.SeriesInstanceUID
        series_uids.add(ds.SeriesInstanceUID)
        assert ds.SeriesNumber == 701
        assert ds.SeriesDescription == "ADC map RESEARCH USE ONLY"
        assert ds.ImageComments.startswith("RESEARCH USE ONLY")
        assert "Apparent Diffusion Coefficient" in ds.DerivationDescription
        assert ds.RescaleType == "mm2/s"
        assert ds.Manufacturer == "dcmtether" and ds.SoftwareVersions == __version__
        assert ds.BurnedInAnnotation == "NO"
        assert not any(elem.tag.is_private for elem in ds)
        assert 0x60000010 not in ds  # overlays describe the source pixels
        assert float(ds.WindowWidth) != 200  # recomputed for the new pixels
    assert len(series_uids) == 1
    assert sorted(int(pydicom.dcmread(p).InstanceNumber) for p in paths) == [1, 2, 3, 4, 5]


def test_uids_are_deterministic(tmp_path, derived):
    paths, _ = derived
    nifti = next((tmp_path / "dataset").rglob("*.nii.gz"))
    meta = derived_mr.load_metadata(_metadata(tmp_path, PMAP_META))
    again = derived_mr.to_derived_mr(nifti, tmp_path / "map.nii.gz", meta, tmp_path / "again")
    first = sorted(pydicom.dcmread(p).SOPInstanceUID for p in paths)
    assert sorted(pydicom.dcmread(p).SOPInstanceUID for p in again.paths) == first


#: The classic MR Image IOD has no Modality LUT module; like Philips, derived maps still carry
#: RescaleSlope/Intercept/Type so PACS show real values (deliberate, PLAN.md §2).
RESCALE_DEVIATION = {
    ("General", keyword, ErrorCode.TagUnexpected)
    for keyword in ("RescaleSlope", "RescaleIntercept", "RescaleType")
}


@pytest.mark.network
def test_derived_images_validate_like_the_source(derived, iod_errors):
    paths, src = derived
    source_errors = set(iod_errors(next(src.iterdir())))
    for path in paths:
        assert set(iod_errors(path)) <= source_errors | RESCALE_DEVIATION


# --- errors ------------------------------------------------------------------------------------


def test_existing_output_is_never_overwritten(tmp_path, derived):
    paths, _ = derived
    nifti = next((tmp_path / "dataset").rglob("*.nii.gz"))
    meta = derived_mr.load_metadata(_metadata(tmp_path, PMAP_META))
    with pytest.raises(derived_mr.DerivedMRError, match="already exists"):
        derived_mr.to_derived_mr(nifti, tmp_path / "map.nii.gz", meta, tmp_path / "out")
    assert sorted(p for p in (tmp_path / "out").iterdir()) == sorted(paths)


def test_enhanced_source_is_rejected(tmp_path):
    nifti, _ = _source(tmp_path, make_enhanced(make_series()))
    _float_map(nifti, tmp_path / "map.nii.gz")
    meta = derived_mr.load_metadata(_metadata(tmp_path, PMAP_META))
    with pytest.raises(derived_mr.DerivedMRError, match="enhanced"):
        derived_mr.to_derived_mr(nifti, tmp_path / "map.nii.gz", meta, tmp_path / "out")


def test_4d_source_is_rejected(tmp_path):
    nifti, _ = _source(tmp_path, make_4d())
    img = nib.load(nifti)
    _save(nifti, np.ones(img.shape[:3], np.float32), tmp_path / "map.nii.gz")
    meta = derived_mr.load_metadata(_metadata(tmp_path, PMAP_META))
    with pytest.raises(derived_mr.DerivedMRError, match="4D"):
        derived_mr.to_derived_mr(nifti, tmp_path / "map.nii.gz", meta, tmp_path / "out")


def test_undeclared_label_is_rejected(tmp_path):
    nifti, _ = _source(tmp_path, make_series())
    labels = _labels(nifti, tmp_path / "labels.nii.gz")
    labels[0, 0, 0] = 9
    _save(nifti, labels, tmp_path / "labels.nii.gz")
    meta = derived_mr.load_metadata(_metadata(tmp_path, SEG_META))
    with pytest.raises(derived_mr.DerivedMRError, match="not described"):
        derived_mr.to_derived_mr(nifti, tmp_path / "labels.nii.gz", meta, tmp_path / "out")


def test_metadata_must_be_a_seg_or_pmap_template(tmp_path):
    with pytest.raises(derived_mr.DerivedMRError, match="dcmqi"):
        derived_mr.load_metadata(_metadata(tmp_path, {"SeriesNumber": "1"}))
    both = copy.deepcopy(PMAP_META) | {"segmentAttributes": SEG_META["segmentAttributes"]}
    with pytest.raises(derived_mr.DerivedMRError, match="dcmqi"):
        derived_mr.load_metadata(_metadata(tmp_path, both))


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_to_mr(tmp_path):
    nifti, _ = _source(tmp_path, make_series())
    _float_map(nifti, tmp_path / "map.nii.gz")
    args = [
        "to-mr",
        str(nifti),
        str(tmp_path / "map.nii.gz"),
        str(_metadata(tmp_path, PMAP_META)),
        str(tmp_path / "out"),
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "Wrote 5 derived MR images (map)."
    assert lines[1].startswith("Quantized to 16 bits: max error ")
    assert "sub-" not in result.output and "Synth" not in result.output


@pytest.mark.network
def test_real_ge_adc_series_as_derived_mr(tmp_path, test_file):
    src = tmp_path / "src" / "adc"
    src.mkdir(parents=True)
    for i in range(20):  # the public copy is anonymized; naming needs a PatientName
        ds = pydicom.dcmread(test_file(f"dcmqi/paramaps/701-ADCb500/{i:06d}.dcm"))
        ds.PatientName = "Public^Adc"
        ds.save_as(src / f"{i:06d}.dcm")
    [row] = forward(tmp_path / "src", tmp_path / "dataset")
    nifti = tmp_path / "dataset" / row["nifti_path"]
    raw = np.asarray(nib.load(nifti).dataobj.get_unscaled())
    _save(nifti, raw, tmp_path / "map.nii.gz", scaling=(1e-6, 0.0))  # GE ADC: 1e-6 mm2/s
    meta = derived_mr.load_metadata(_metadata(tmp_path, PMAP_META))
    result = derived_mr.to_derived_mr(nifti, tmp_path / "map.nii.gz", meta, tmp_path / "out")
    assert len(result.paths) == 20
    back = _reconvert(tmp_path, tmp_path / "out")
    np.testing.assert_allclose(back.affine, nib.load(nifti).affine, atol=1e-3)
    expected = nib.load(tmp_path / "map.nii.gz").get_fdata()
    np.testing.assert_allclose(back.get_fdata(), expected, rtol=1e-6, atol=result.max_error + 1e-12)


# --- old or damaged tether.json (all single-series commands) -----------------------------------


def _set_version_1(doc):
    doc["format_version"] = 1


def _index_out_of_range(doc):
    doc["instances"][0]["index"] = [99]


def _index_used_twice(doc):
    doc["instances"][1]["index"] = doc["instances"][0]["index"]


def _index_not_a_number(doc):
    doc["instances"][0]["index"] = ["first"]


def _no_pixel(doc):
    del doc["pixel"]


def _no_instances(doc):
    doc["instances"] = []


def _instance_without_elements(doc):
    del doc["instances"][0]["elements"]


DAMAGED_TETHER = {
    "version-1": (_set_version_1, "tether format version 1 is not supported"),
    "broken-json": (None, "tether.json is not readable"),
    "index-out-of-range": (_index_out_of_range, "instance indices do not match"),
    "index-used-twice": (_index_used_twice, "instance indices do not match"),
    "index-not-a-number": (_index_not_a_number, "not a tether document"),
    "no-pixel": (_no_pixel, "not a tether document"),
    "no-instances": (_no_instances, "not a tether document"),
    "instance-without-elements": (_instance_without_elements, "not a tether document"),
}


@pytest.mark.parametrize("command", ["to-dcm", "to-seg", "to-pmap", "to-mr"])
@pytest.mark.parametrize("damage", DAMAGED_TETHER.values(), ids=DAMAGED_TETHER.keys())
def test_damaged_tether_json_is_a_one_line_error(tmp_path, command, damage):
    change, message = damage
    nifti, _ = _source(tmp_path, make_series())
    _float_map(nifti, tmp_path / "map.nii.gz")
    _labels(nifti, tmp_path / "labels.nii.gz")
    tether_path = nifti.with_name(nifti.name.removesuffix(".nii.gz") + ".tether.json")
    if change is None:
        tether_path.write_text(tether_path.read_text("utf-8")[:200], "utf-8")
    else:
        doc = json.loads(tether_path.read_text("utf-8"))
        change(doc)
        tether_path.write_text(json.dumps(doc), "utf-8")
    seg_meta = tmp_path / "seg.json"
    seg_meta.write_text(json.dumps(SEG_META), "utf-8")
    pmap_meta = tmp_path / "pmap.json"
    pmap_meta.write_text(json.dumps(PMAP_META), "utf-8")
    args = {
        "to-dcm": [tmp_path / "out"],
        "to-seg": [tmp_path / "labels.nii.gz", seg_meta, tmp_path / "seg.dcm"],
        "to-pmap": [tmp_path / "map.nii.gz", pmap_meta, tmp_path / "pm.dcm"],
        "to-mr": [tmp_path / "map.nii.gz", pmap_meta, tmp_path / "out"],
    }[command]

    result = CliRunner().invoke(main, [command, str(nifti), *map(str, args)])

    assert isinstance(result.exception, SystemExit), repr(result.exception)  # no traceback
    assert result.exit_code == 1
    [line] = result.output.splitlines()
    assert line.startswith("Error: ") and message in line
    assert "sub-" not in line and "Synth" not in line and "IM_0" not in line
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
    assert not (tmp_path / "seg.dcm").exists() and not (tmp_path / "pm.dcm").exists()


def test_failed_write_leaves_nothing_behind(tmp_path, monkeypatch):
    nifti, _ = _source(tmp_path, make_series())
    _float_map(nifti, tmp_path / "map.nii.gz")
    meta = derived_mr.load_metadata(_metadata(tmp_path, PMAP_META))
    save_as = pydicom.Dataset.save_as
    calls = []

    def failing(self, path, *args, **kwargs):
        calls.append(path)
        save_as(self, path, *args, **kwargs)
        if len(calls) == 3:
            raise OSError("disk full")  # after the file was created

    monkeypatch.setattr(pydicom.Dataset, "save_as", failing)
    with pytest.raises(OSError, match="disk full"):
        derived_mr.to_derived_mr(nifti, tmp_path / "map.nii.gz", meta, tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_integer_map_with_long_slope_is_reported_as_rounded_not_quantized(tmp_path):
    nifti, _ = _source(tmp_path, make_series())
    i, j, k = np.indices(nib.load(nifti).shape[:3])
    stored = (60000 + i + 10 * j + 100 * k).astype(np.uint16)
    # a float32 slope: its exact value does not fit a 16-character DS
    _save(nifti, stored, tmp_path / "map.nii.gz", scaling=(np.float32(1 / 3), 0.0))
    args = [
        "to-mr",
        str(nifti),
        str(tmp_path / "map.nii.gz"),
        str(_metadata(tmp_path, PMAP_META)),
        str(tmp_path / "out"),
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[1].startswith("Rescale rounded to DS precision: max error ")
    assert "uantized" not in result.output
    comments = pydicom.dcmread(tmp_path / "out" / "MR_0001.dcm").ImageComments
    assert "rescale rounded to DS precision, max error " in comments
    assert "quantized" not in comments
