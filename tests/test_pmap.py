"""Float/integer map NIfTI + source pair -> DICOM Parametric Map."""

import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from click.testing import CliRunner
from roundtrip import SPECS, forward
from synthetic import make_4d, make_enhanced, make_series, write_series

from dcmtether import __version__, orientation, pmap
from dcmtether.cli import main

#: dcmqi parametric map metadata template (https://qiicr.org/dcmqi/#/pm).
META = {
    "QuantityValueCode": {
        "CodeValue": "113041",
        "CodingSchemeDesignator": "DCM",
        "CodeMeaning": "Apparent Diffusion Coefficient",
    },
    "MeasurementUnitsCode": {
        "CodeValue": "mm2/s",
        "CodingSchemeDesignator": "UCUM",
        "CodeMeaning": "mm2/s",
    },
    "MeasurementMethodCode": {
        "CodeValue": "113250",
        "CodingSchemeDesignator": "DCM",
        "CodeMeaning": "Mono-exponential ADC model",
    },
    "SeriesNumber": "701",
    "InstanceNumber": "1",
    "SeriesDescription": "ADC map",
}


def _source(tmp_path: Path, series) -> tuple[Path, Path]:
    src = tmp_path / "src" / "series"
    write_series(series, src)
    [row] = forward(tmp_path / "src", tmp_path / "dataset")
    return tmp_path / "dataset" / row["nifti_path"], src


def _map(nifti: Path, path: Path, dtype=np.float32, scaling=None) -> np.ndarray:
    """A map on the source grid whose values differ along every axis; returns real values."""
    img = nib.load(nifti)
    i, j, k = np.indices(img.shape[:3])
    if np.dtype(dtype).kind == "f":
        values = (1e-3 * (i + 10 * j + 100 * k) + 1e-4).astype(dtype)
        stored = values
    else:
        stored = (i + 10 * j + 100 * k - 50).astype(dtype)
        values = stored.astype(np.float64)
    out = nib.Nifti1Image(stored, img.affine)
    if scaling is not None:
        out.header.set_slope_inter(*scaling)
    nib.save(out, path)
    if scaling is not None:  # as stored: NIfTI keeps scl_slope/scl_inter as float32
        saved = nib.load(path).dataobj
        values = stored * float(saved.slope) + float(saved.inter)
    return values


def _metadata(tmp_path: Path, meta: dict = META) -> Path:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta), "utf-8")
    return path


def _expected_frame(source: pydicom.Dataset, values: np.ndarray, affine: np.ndarray):
    """Map values at every pixel of a source image, looked up in world space."""
    iop = np.asarray(source.ImageOrientationPatient, float)
    ipp = np.asarray(source.ImagePositionPatient, float)
    dr, dc = (float(v) for v in source.PixelSpacing)
    r, c = np.indices((source.Rows, source.Columns))
    lps = ipp + np.outer(c.ravel() * dc, iop[:3]) + np.outer(r.ravel() * dr, iop[3:])
    ras = orientation.LPS_TO_RAS[:3, :3] @ lps.T
    ijk = np.linalg.solve(affine[:3, :3], ras - affine[:3, 3:4])
    np.testing.assert_allclose(ijk, np.rint(ijk), atol=1e-3)
    return values[tuple(np.rint(ijk).astype(int))].reshape(source.Rows, source.Columns)


def _real_values(pm: pydicom.Dataset) -> np.ndarray:
    """Frames of a Parametric Map as real-world values (frames, rows, columns)."""
    rwvm = pm.SharedFunctionalGroupsSequence[0].RealWorldValueMappingSequence[0]
    stored = pm.pixel_array.astype(np.float64)
    if stored.ndim == 2:
        stored = stored[np.newaxis]
    return stored * float(rwvm.RealWorldValueSlope) + float(rwvm.RealWorldValueIntercept)


def _check_values(pm_path: Path, src: Path, values: np.ndarray, affine: np.ndarray) -> None:
    pm = pydicom.dcmread(pm_path)
    sources = {ds.SOPInstanceUID: ds for ds in (pydicom.dcmread(p) for p in src.iterdir())}
    real = _real_values(pm)
    assert len(real) == len(sources)
    for f, item in enumerate(pm.PerFrameFunctionalGroupsSequence):
        ref = item.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        expected = _expected_frame(sources[ref], values, affine)
        np.testing.assert_array_equal(real[f], expected)


# --- values and geometry -----------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["axial", "sagittal", "oblique", "oblique-40", "descending"])
def test_float_map_values_in_world_space(tmp_path, spec):
    nifti, src = _source(tmp_path, make_series(SPECS[spec]))
    values = _map(nifti, tmp_path / "map.nii.gz")
    out = tmp_path / "pm.dcm"
    pmap.to_pmap(nifti, tmp_path / "map.nii.gz", pmap.load_metadata(_metadata(tmp_path)), out)
    _check_values(out, src, values, nib.load(nifti).affine)
    assert "FloatPixelData" in pydicom.dcmread(out)


def test_float64_map_keeps_double_precision(tmp_path):
    nifti, src = _source(tmp_path, make_series())
    values = _map(nifti, tmp_path / "map.nii.gz", dtype=np.float64)
    out = tmp_path / "pm.dcm"
    pmap.to_pmap(nifti, tmp_path / "map.nii.gz", pmap.load_metadata(_metadata(tmp_path)), out)
    assert "DoubleFloatPixelData" in pydicom.dcmread(out)
    _check_values(out, src, values, nib.load(nifti).affine)


@pytest.mark.parametrize(
    ("dtype", "scaling"),
    [
        (np.uint16, None),
        (np.uint16, (0.5, 1.0)),
        (np.int16, (2e-6, 0.0)),  # signed: shifted into uint16, intercept adjusted
        (np.int32, None),
        (np.uint8, (0.25, -3.0)),
    ],
)
def test_integer_map_is_stored_losslessly(tmp_path, dtype, scaling):
    nifti, src = _source(tmp_path, make_series())
    values = _map(nifti, tmp_path / "map.nii.gz", dtype=dtype, scaling=scaling)
    out = tmp_path / "pm.dcm"
    pmap.to_pmap(nifti, tmp_path / "map.nii.gz", pmap.load_metadata(_metadata(tmp_path)), out)
    pm = pydicom.dcmread(out)
    assert "PixelData" in pm and pm.PixelRepresentation == 0
    _check_values(out, src, values, nib.load(nifti).affine)


def test_enhanced_source_frames_are_referenced(tmp_path):
    nifti, src = _source(tmp_path, make_enhanced(make_series(SPECS["oblique"])))
    _map(nifti, tmp_path / "map.nii.gz")
    out = tmp_path / "pm.dcm"
    pmap.to_pmap(nifti, tmp_path / "map.nii.gz", pmap.load_metadata(_metadata(tmp_path)), out)
    pm = pydicom.dcmread(out)
    [source] = [pydicom.dcmread(p) for p in src.iterdir()]
    refs = [
        item.DerivationImageSequence[0].SourceImageSequence[0]
        for item in pm.PerFrameFunctionalGroupsSequence
    ]
    assert {r.ReferencedSOPInstanceUID for r in refs} == {source.SOPInstanceUID}
    assert sorted(int(r.ReferencedFrameNumber) for r in refs) == [1, 2, 3, 4, 5]


@pytest.mark.network
def test_pmap_on_real_ge_adc_series(tmp_path, test_file):
    src = tmp_path / "src" / "adc"
    src.mkdir(parents=True)
    for i in range(20):  # the public copy is anonymized; naming needs a PatientName
        ds = pydicom.dcmread(test_file(f"dcmqi/paramaps/701-ADCb500/{i:06d}.dcm"))
        ds.PatientName = "Public^Adc"
        ds.save_as(src / f"{i:06d}.dcm")
    [row] = forward(tmp_path / "src", tmp_path / "dataset")
    nifti = tmp_path / "dataset" / row["nifti_path"]
    # The GE series is an ADC map itself: write its own voxels (x 1e-6 mm2/s) as the map.
    raw = np.asarray(nib.load(nifti).dataobj.get_unscaled())
    values = raw.astype(np.float32) * np.float32(1e-6)
    nib.save(nib.Nifti1Image(values, nib.load(nifti).affine), tmp_path / "map.nii.gz")
    out = tmp_path / "pm.dcm"
    pmap.to_pmap(nifti, tmp_path / "map.nii.gz", pmap.load_metadata(_metadata(tmp_path)), out)
    pm = pydicom.dcmread(out)
    sources = {ds.SOPInstanceUID: ds for ds in (pydicom.dcmread(p) for p in src.iterdir())}
    real = _real_values(pm)
    for f, item in enumerate(pm.PerFrameFunctionalGroupsSequence):
        ref = item.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        expected = sources[ref].pixel_array.astype(np.float32) * np.float32(1e-6)
        np.testing.assert_array_equal(real[f].astype(np.float32), expected)


# --- content -----------------------------------------------------------------------------------


@pytest.fixture
def written(tmp_path) -> tuple[Path, Path, Path]:
    nifti, _ = _source(tmp_path, make_series())
    _map(nifti, tmp_path / "map.nii.gz")
    out = tmp_path / "pm.dcm"
    pmap.to_pmap(nifti, tmp_path / "map.nii.gz", pmap.load_metadata(_metadata(tmp_path)), out)
    return out, nifti, tmp_path / "map.nii.gz"


def test_quantity_units_and_labeling(written):
    pm = pydicom.dcmread(written[0])
    rwvm = pm.SharedFunctionalGroupsSequence[0].RealWorldValueMappingSequence[0]
    assert rwvm.MeasurementUnitsCodeSequence[0].CodeValue == "mm2/s"
    concepts = {
        item.ConceptNameCodeSequence[0].CodeMeaning: item.ConceptCodeSequence[0].CodeValue
        for item in rwvm.QuantityDefinitionSequence
    }
    assert concepts["Quantity"] == "113041"
    assert concepts["Measurement Method"] == "113250"
    assert pm.SeriesDescription == "ADC map RESEARCH USE ONLY"
    assert pm.ContentQualification == "RESEARCH"
    assert "RESEARCH USE ONLY" in pm.ContentDescription
    assert pm.Manufacturer == "dcmtether"
    assert pm.SoftwareVersions == __version__
    assert pm.SeriesNumber == 701


def test_uids_are_deterministic_and_content_dependent(tmp_path, written):
    first = pydicom.dcmread(written[0])
    assert first.SOPInstanceUID.startswith("2.25.")
    again = tmp_path / "again.dcm"
    pmap.to_pmap(written[1], written[2], pmap.load_metadata(_metadata(tmp_path)), again)
    assert pydicom.dcmread(again).SOPInstanceUID == first.SOPInstanceUID

    img = nib.load(written[2])
    changed = np.asarray(img.dataobj).copy()
    changed[0, 0, 0] += 1
    nib.save(nib.Nifti1Image(changed, img.affine), tmp_path / "changed.nii.gz")
    other = tmp_path / "other.dcm"
    pmap.to_pmap(
        written[1], tmp_path / "changed.nii.gz", pmap.load_metadata(_metadata(tmp_path)), other
    )
    assert pydicom.dcmread(other).SOPInstanceUID != first.SOPInstanceUID


@pytest.mark.network
def test_pmap_validates_apart_from_functional_group_false_positives(written, iod_errors):
    # dicom-validator flags the functional group macros of every multi-frame derived object as
    # unexpected (see test_seg: also dcmqi's reference SEG); nothing else may be reported.
    errors = iod_errors(written[0])
    assert {(module, code.name) for module, _, code in errors} <= {
        ("Multi-frame Functional Groups", "TagUnexpected")
    }


# --- errors ------------------------------------------------------------------------------------


def _expect_error(tmp_path, nifti, path, match):
    with pytest.raises(pmap.PmapError, match=match):
        pmap.to_pmap(nifti, path, pmap.load_metadata(_metadata(tmp_path)), tmp_path / "x.dcm")
    assert not (tmp_path / "x.dcm").exists()


def test_map_on_another_grid_is_rejected(tmp_path, written):
    img = nib.load(written[2])
    data = np.asarray(img.dataobj)
    nib.save(nib.Nifti1Image(data[:-1], img.affine), tmp_path / "small.nii.gz")
    _expect_error(tmp_path, written[1], tmp_path / "small.nii.gz", "shape")


def test_nan_values_are_rejected(tmp_path, written):
    img = nib.load(written[2])
    data = np.asarray(img.dataobj).copy()
    data[0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(data, img.affine), tmp_path / "nan.nii.gz")
    _expect_error(tmp_path, written[1], tmp_path / "nan.nii.gz", "NaN or infinite")


def test_integer_range_too_wide_for_16_bits_is_rejected(tmp_path, written):
    img = nib.load(written[2])
    data = np.zeros(img.shape, dtype=np.int32)
    data[0, 0, 0] = 70000
    nib.save(nib.Nifti1Image(data, img.affine), tmp_path / "wide.nii.gz")
    _expect_error(tmp_path, written[1], tmp_path / "wide.nii.gz", "16-bit")


def test_existing_output_is_never_overwritten(tmp_path, written):
    with pytest.raises(pmap.PmapError, match="already exists"):
        pmap.to_pmap(written[1], written[2], pmap.load_metadata(_metadata(tmp_path)), written[0])


def test_4d_source_is_rejected(tmp_path):
    nifti, _ = _source(tmp_path, make_4d())
    img = nib.load(nifti)
    nib.save(nib.Nifti1Image(np.ones(img.shape[:3], np.float32), img.affine), tmp_path / "m.nii.gz")
    _expect_error(tmp_path, nifti, tmp_path / "m.nii.gz", "4D")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda m: m.pop("QuantityValueCode"), "QuantityValueCode"),
        (lambda m: m.pop("MeasurementUnitsCode"), "MeasurementUnitsCode"),
        (lambda m: m.pop("SeriesNumber"), "SeriesNumber"),
        (lambda m: m["MeasurementUnitsCode"].pop("CodeMeaning"), "CodeMeaning"),
        (lambda m: m.update(InstanceNumber="first"), "metadata: InstanceNumber is not an integer"),
        (lambda m: m.update(InstanceNumber=[1]), "metadata: InstanceNumber is not an integer"),
        (lambda m: m.update(QuantityValueCode="ADC"), "QuantityValueCode needs CodeValue"),
    ],
)
def test_bad_metadata_is_rejected(tmp_path, change, match):
    meta = copy.deepcopy(META)
    change(meta)
    with pytest.raises(pmap.PmapError, match=match):
        pmap.load_metadata(_metadata(tmp_path, meta))


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_to_pmap(tmp_path):
    nifti, _ = _source(tmp_path, make_series())
    _map(nifti, tmp_path / "map.nii.gz")
    out = tmp_path / "pm.dcm"
    args = ["to-pmap", str(nifti), str(tmp_path / "map.nii.gz"), str(_metadata(tmp_path))]
    result = CliRunner().invoke(main, [*args, str(out)])
    assert result.exit_code == 0, result.output
    assert result.output == "Wrote DICOM Parametric Map (float32) on 5 source images.\n"
    result = CliRunner().invoke(main, [*args, str(out)])
    assert result.exit_code == 1
    assert "already exists" in result.output
