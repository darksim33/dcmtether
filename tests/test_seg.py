"""Label map NIfTI + source pair -> DICOM SEG."""

import copy
import json
from pathlib import Path

import highdicom as hd
import nibabel as nib
import numpy as np
import pydicom
import pytest
from click.testing import CliRunner
from pydicom.uid import RLELossless
from roundtrip import SPECS, forward
from synthetic import SeriesSpec, compressed, make_4d, make_enhanced, make_series, write_series

from dcmtether import __version__, orientation, seg
from dcmtether.cli import main

#: dcmqi segmentation metadata template (https://qiicr.org/dcmqi/#/seg).
META = {
    "ContentCreatorName": "Test^Reader",
    "SeriesDescription": "Liver and tumor",
    "SeriesNumber": "300",
    "InstanceNumber": "1",
    "segmentAttributes": [
        [
            {
                "labelID": 1,
                "SegmentLabel": "liver",
                "SegmentDescription": "Liver",
                "SegmentAlgorithmType": "AUTOMATIC",
                "SegmentAlgorithmName": "nnU-Net",
                "SegmentedPropertyCategoryCodeSequence": {
                    "CodeValue": "123037004",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": "Anatomical Structure",
                },
                "SegmentedPropertyTypeCodeSequence": {
                    "CodeValue": "10200004",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": "Liver",
                },
                "recommendedDisplayRGBValue": [221, 130, 101],
            },
            {
                "labelID": 2,
                "SegmentDescription": "Tumor",
                "SegmentAlgorithmType": "MANUAL",
                "SegmentedPropertyCategoryCodeSequence": {
                    "CodeValue": "49755003",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": "Morphologically Abnormal Structure",
                },
                "SegmentedPropertyTypeCodeSequence": {
                    "CodeValue": "108369006",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": "Neoplasm",
                },
                "recommendedDisplayRGBValue": [255, 0, 0],
            },
        ]
    ],
}


def _source(tmp_path: Path, series) -> tuple[Path, Path]:
    """Convert ``series``; return (source NIfTI, source DICOM directory)."""
    src = tmp_path / "src" / "series"
    write_series(series, src)
    [row] = forward(tmp_path / "src", tmp_path / "dataset")
    return tmp_path / "dataset" / row["nifti_path"], src


def _label_map(nifti: Path, path: Path, values=(1, 2), dtype=np.uint8) -> np.ndarray:
    """A label map on the source grid: a pattern that differs along every axis."""
    img = nib.load(nifti)
    shape = img.shape[:3]
    i, j, k = np.indices(shape)
    labels = np.zeros(shape, dtype=dtype)
    labels[(i + 2 * j + 3 * k) % 4 == 1] = values[0]
    labels[(i + 2 * j + 3 * k) % 4 == 2] = values[1]
    nib.save(nib.Nifti1Image(labels, img.affine), path)
    return labels


def _metadata(tmp_path: Path, meta: dict = META) -> Path:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta), "utf-8")
    return path


def _expected_frame(source: pydicom.Dataset, labels: np.ndarray, affine: np.ndarray, atol=1e-3):
    """Labels at every pixel of a source image, looked up in world space (independent of the
    reorientation code): DICOM pixel (row, col) -> LPS -> RAS -> label map voxel."""
    iop = np.asarray(source.ImageOrientationPatient, float)
    ipp = np.asarray(source.ImagePositionPatient, float)
    dr, dc = (float(v) for v in source.PixelSpacing)
    r, c = np.indices((source.Rows, source.Columns))
    lps = ipp + np.outer(c.ravel() * dc, iop[:3]) + np.outer(r.ravel() * dr, iop[3:])
    ras = orientation.LPS_TO_RAS[:3, :3] @ lps.T
    ijk = np.linalg.solve(affine[:3, :3], ras - affine[:3, 3:4])
    np.testing.assert_allclose(ijk, np.rint(ijk), atol=atol)
    idx = tuple(np.rint(ijk).astype(int))
    return labels[idx].reshape(source.Rows, source.Columns)


def _check_pixels(seg_path: Path, src: Path, labels: np.ndarray, affine, numbering=None, atol=1e-3):
    segmentation = hd.seg.segread(seg_path)
    numbering = numbering or {1: 1, 2: 2}
    for path in sorted(src.iterdir()):
        source = pydicom.dcmread(path)
        expected = _expected_frame(source, labels, affine, atol)
        expected = np.vectorize(lambda v: numbering.get(int(v), 0))(expected)
        got = segmentation.get_pixels_by_source_instance(
            [source.SOPInstanceUID],
            combine_segments=True,
            ignore_spatial_locations=True,
            assert_missing_frames_are_empty=True,
        )[0]
        np.testing.assert_array_equal(got, expected)


# --- pixels and geometry -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    ["axial", "sagittal", "flip-both-sagittal", "oblique", "oblique-40", "descending", "shuffled"],
)
def test_seg_frames_match_label_map_in_world_space(tmp_path, spec):
    nifti, src = _source(tmp_path, make_series(SPECS[spec]))
    labels = _label_map(nifti, tmp_path / "labels.nii.gz")
    out = tmp_path / "seg.dcm"
    seg.to_seg(nifti, tmp_path / "labels.nii.gz", seg.load_metadata(_metadata(tmp_path)), out)
    _check_pixels(out, src, labels, nib.load(nifti).affine)


def _check_volume(seg_path: Path, labels: np.ndarray, affine: np.ndarray, atol=1e-3) -> None:
    """Rebuild the volume from the SEG's own plane positions (as viewers do) and compare every
    voxel in world space with the label map. ``atol`` in voxels (real data: orientation noise)."""
    volume = hd.seg.segread(seg_path).get_volume(combine_segments=True)
    assert volume.coordinate_system.name == "PATIENT"  # LPS
    ijk = np.indices(volume.array.shape).reshape(3, -1)
    lps = volume.affine @ np.vstack([ijk, np.ones(ijk.shape[1])])
    nifti_ijk = np.linalg.inv(affine) @ (orientation.LPS_TO_RAS @ lps)
    np.testing.assert_allclose(nifti_ijk[:3], np.rint(nifti_ijk[:3]), atol=atol)
    idx = tuple(np.rint(nifti_ijk[:3]).astype(int))
    np.testing.assert_array_equal(volume.array.reshape(-1), labels[idx])


@pytest.mark.parametrize("spec", ["oblique", "flip-both-sagittal", "descending"])
def test_seg_volume_from_plane_positions_matches_label_map(tmp_path, spec):
    nifti, _ = _source(tmp_path, make_series(SPECS[spec]))
    labels = _label_map(nifti, tmp_path / "labels.nii.gz")
    out = tmp_path / "seg.dcm"
    seg.to_seg(nifti, tmp_path / "labels.nii.gz", seg.load_metadata(_metadata(tmp_path)), out)
    _check_volume(out, labels, nib.load(nifti).affine)


@pytest.mark.network
def test_seg_on_real_ge_series(tmp_path, test_file):
    src = tmp_path / "src" / "adc"
    src.mkdir(parents=True)
    for i in range(20):  # the public copy is anonymized; naming needs a PatientName
        ds = pydicom.dcmread(test_file(f"dcmqi/paramaps/701-ADCb500/{i:06d}.dcm"))
        ds.PatientName = "Public^Adc"
        ds.save_as(src / f"{i:06d}.dcm")
    [row] = forward(tmp_path / "src", tmp_path / "dataset")
    nifti = tmp_path / "dataset" / row["nifti_path"]
    labels = _label_map(nifti, tmp_path / "labels.nii.gz")
    out = tmp_path / "seg.dcm"
    seg.to_seg(nifti, tmp_path / "labels.nii.gz", seg.load_metadata(_metadata(tmp_path)), out)
    # GE writes orientation rounding noise per slice: allow 0.01 voxel.
    _check_pixels(out, src, labels, nib.load(nifti).affine, atol=1e-2)
    _check_volume(out, labels, nib.load(nifti).affine, atol=1e-2)


def test_seg_from_compressed_source(tmp_path):
    nifti, src = _source(tmp_path, compressed(make_series(SPECS["oblique"]), RLELossless))
    labels = _label_map(nifti, tmp_path / "labels.nii.gz")
    out = tmp_path / "seg.dcm"
    seg.to_seg(nifti, tmp_path / "labels.nii.gz", seg.load_metadata(_metadata(tmp_path)), out)
    _check_pixels(out, src, labels, nib.load(nifti).affine)


def test_seg_from_enhanced_source_references_frames(tmp_path):
    nifti, src = _source(tmp_path, make_enhanced(make_series(SPECS["oblique"])))
    labels = _label_map(nifti, tmp_path / "labels.nii.gz")
    out = tmp_path / "seg.dcm"
    seg.to_seg(nifti, tmp_path / "labels.nii.gz", seg.load_metadata(_metadata(tmp_path)), out)
    segmentation = hd.seg.segread(out)
    [source] = [pydicom.dcmread(p) for p in src.iterdir()]
    got = segmentation.get_pixels_by_source_frame(
        source.SOPInstanceUID,
        list(range(1, 6)),
        combine_segments=True,
        ignore_spatial_locations=True,
        assert_missing_frames_are_empty=True,
    )
    labels_dicom = orientation.from_closest_ras(
        labels, nib.orientations.io_orientation(orientation.LPS_TO_RAS @ _lps_affine(source))
    )
    for f in range(5):
        np.testing.assert_array_equal(got[f], labels_dicom[:, :, f])


def test_seg_from_enhanced_source_with_per_frame_geometry(tmp_path):
    # Philips writes orientation and pixel measures per frame, not in the shared groups.
    series = make_enhanced(make_series(SPECS["oblique"]))
    [ds] = series.datasets
    shared = ds.SharedFunctionalGroupsSequence[0]
    for item in ds.PerFrameFunctionalGroupsSequence:
        item.PlaneOrientationSequence = copy.deepcopy(shared.PlaneOrientationSequence)
        item.PixelMeasuresSequence = copy.deepcopy(shared.PixelMeasuresSequence)
    del shared.PlaneOrientationSequence
    del shared.PixelMeasuresSequence
    nifti, _ = _source(tmp_path, series)
    labels = _label_map(nifti, tmp_path / "labels.nii.gz")
    out = tmp_path / "seg.dcm"
    seg.to_seg(nifti, tmp_path / "labels.nii.gz", seg.load_metadata(_metadata(tmp_path)), out)
    _check_volume(out, labels, nib.load(nifti).affine)


def test_orientation_noise_between_source_images_is_accepted(tmp_path):
    # GE writes ImageOrientationPatient with rounding noise per slice (within scan tolerance).
    series = make_series(SPECS["oblique"])
    for n, ds in enumerate(series.datasets):
        ds.ImageOrientationPatient = [float(v) + n * 1e-6 for v in ds.ImageOrientationPatient]
    nifti, _ = _source(tmp_path, series)
    labels = _label_map(nifti, tmp_path / "labels.nii.gz")
    out = tmp_path / "seg.dcm"
    seg.to_seg(nifti, tmp_path / "labels.nii.gz", seg.load_metadata(_metadata(tmp_path)), out)
    _check_volume(out, labels, nib.load(nifti).affine)


def _lps_affine(enhanced: pydicom.Dataset) -> np.ndarray:
    from dcmtether import forward as fw
    from dcmtether import frames

    return fw.volume_affine(frames.frame_views(enhanced))


def test_float_label_map_with_integer_values_is_accepted(tmp_path):
    nifti, src = _source(tmp_path, make_series())
    labels = _label_map(nifti, tmp_path / "labels.nii.gz", dtype=np.float32)
    out = tmp_path / "seg.dcm"
    seg.to_seg(nifti, tmp_path / "labels.nii.gz", seg.load_metadata(_metadata(tmp_path)), out)
    _check_pixels(out, src, labels, nib.load(nifti).affine)


def test_label_ids_are_renumbered_consecutively(tmp_path):
    nifti, src = _source(tmp_path, make_series())
    labels = _label_map(nifti, tmp_path / "labels.nii.gz", values=(3, 7))
    meta = copy.deepcopy(META)
    meta["segmentAttributes"][0][0]["labelID"] = 3
    meta["segmentAttributes"][0][1]["labelID"] = 7
    out = tmp_path / "seg.dcm"
    result = seg.to_seg(
        nifti, tmp_path / "labels.nii.gz", seg.load_metadata(_metadata(tmp_path, meta)), out
    )
    assert result.segment_numbers == {3: 1, 7: 2}
    _check_pixels(out, src, labels, nib.load(nifti).affine, numbering={3: 1, 7: 2})


# --- content -----------------------------------------------------------------------------------


@pytest.fixture
def written(tmp_path) -> tuple[Path, Path, Path]:
    """(SEG file, source NIfTI, label map) for a small synthetic series."""
    nifti, _ = _source(tmp_path, make_series())
    _label_map(nifti, tmp_path / "labels.nii.gz")
    out = tmp_path / "seg.dcm"
    seg.to_seg(nifti, tmp_path / "labels.nii.gz", seg.load_metadata(_metadata(tmp_path)), out)
    return out, nifti, tmp_path / "labels.nii.gz"


def test_segment_descriptions(written):
    segmentation = hd.seg.segread(written[0])
    liver, tumor = (segmentation.get_segment_description(n) for n in (1, 2))
    assert liver.segment_label == "liver"
    assert tumor.segment_label == "Tumor"  # no SegmentLabel: falls back to SegmentDescription
    assert liver.segmented_property_type.value == "10200004"
    assert tumor.segmented_property_category.value == "49755003"
    assert liver.algorithm_type.value == "AUTOMATIC"
    assert liver.algorithm_identification[0].AlgorithmName == "nnU-Net"
    assert tumor.algorithm_type.value == "MANUAL"
    np.testing.assert_allclose(
        hd.color.CIELabColor.from_dicom_value(liver.RecommendedDisplayCIELabValue).to_rgb(),
        (221, 130, 101),
        atol=1,
    )


def test_research_labeling_and_references(written):
    ds = pydicom.dcmread(written[0])
    assert ds.SeriesDescription == "Liver and tumor RESEARCH USE ONLY"
    assert "RESEARCH USE ONLY" in ds.ContentDescription
    assert ds.Manufacturer == "dcmtether"
    assert ds.SoftwareVersions == __version__
    assert ds.SeriesNumber == 300
    assert ds.ContentCreatorName == "Test^Reader"
    doc = json.loads(written[1].with_suffix("").with_suffix(".tether.json").read_text("utf-8"))
    referenced = ds.ReferencedSeriesSequence[0]
    assert referenced.SeriesInstanceUID == doc["source"]["series_instance_uid"]
    assert ds.StudyInstanceUID == doc["shared"]["0020000D"]["Value"][0]
    assert ds.FrameOfReferenceUID == doc["shared"]["00200052"]["Value"][0]
    assert len(referenced.ReferencedInstanceSequence) == 5


def test_uids_are_deterministic_and_content_dependent(tmp_path, written):
    first = pydicom.dcmread(written[0])
    assert first.SOPInstanceUID.startswith("2.25.")
    assert first.SeriesInstanceUID.startswith("2.25.")

    again = tmp_path / "again.dcm"
    seg.to_seg(written[1], written[2], seg.load_metadata(_metadata(tmp_path)), again)
    second = pydicom.dcmread(again)
    assert (second.SOPInstanceUID, second.SeriesInstanceUID) == (
        first.SOPInstanceUID,
        first.SeriesInstanceUID,
    )

    img = nib.load(written[2])
    changed = np.asarray(img.dataobj).copy()
    changed[0, 0, 0] = 2 if changed[0, 0, 0] != 2 else 1
    nib.save(nib.Nifti1Image(changed, img.affine), tmp_path / "changed.nii.gz")
    other = tmp_path / "other.dcm"
    seg.to_seg(
        written[1], tmp_path / "changed.nii.gz", seg.load_metadata(_metadata(tmp_path)), other
    )
    third = pydicom.dcmread(other)
    assert third.SOPInstanceUID != first.SOPInstanceUID
    assert third.SeriesInstanceUID != first.SeriesInstanceUID


@pytest.mark.network
def test_seg_validates_like_dcmqi_reference_seg(written, iod_errors, test_file):
    """dicom-validator flags the functional group macros of every SEG as unexpected (also
    dcmqi's own reference SEG, which 3D Slicer reads); apart from those: no errors."""
    reference = iod_errors(test_file("dcmqi/segmentations/liver.dcm"))
    assert {code.name for module, _, code in reference} == {"TagUnexpected"}
    assert {module for module, _, _ in reference} == {"Multi-frame Functional Groups"}
    assert sorted(iod_errors(written[0])) == sorted(reference)


# --- errors ------------------------------------------------------------------------------------


def test_existing_output_is_never_overwritten(tmp_path, written):
    before = written[0].read_bytes()
    with pytest.raises(seg.SegError, match="already exists"):
        seg.to_seg(written[1], written[2], seg.load_metadata(_metadata(tmp_path)), written[0])
    assert written[0].read_bytes() == before


def _labels_from(tmp_path, nifti, data, affine=None) -> Path:
    path = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(data, nib.load(nifti).affine if affine is None else affine), path)
    return path


def _expect_error(tmp_path, nifti, labels, match, meta=META):
    with pytest.raises(seg.SegError, match=match):
        seg.to_seg(nifti, labels, seg.load_metadata(_metadata(tmp_path, meta)), tmp_path / "x.dcm")
    assert not (tmp_path / "x.dcm").exists()


def test_label_map_on_another_grid_is_rejected(tmp_path, written):
    data = np.asarray(nib.load(written[2]).dataobj)
    _expect_error(tmp_path, written[1], _labels_from(tmp_path, written[1], data[:-1]), "shape")
    shifted = nib.load(written[1]).affine.copy()
    shifted[0, 3] += 1.0
    path = _labels_from(tmp_path, written[1], data, shifted)
    _expect_error(tmp_path, written[1], path, "affine")


def test_bad_label_values_are_rejected(tmp_path, written):
    data = np.asarray(nib.load(written[2]).dataobj).astype(np.float32)
    undeclared = data.copy()
    undeclared[0, 0, 0] = 9
    _expect_error(
        tmp_path, written[1], _labels_from(tmp_path, written[1], undeclared), "not described"
    )
    fractional = data.copy()
    fractional[0, 0, 0] = 0.5
    _expect_error(
        tmp_path, written[1], _labels_from(tmp_path, written[1], fractional), "non-integer"
    )
    negative = data.copy()
    negative[0, 0, 0] = -1
    _expect_error(tmp_path, written[1], _labels_from(tmp_path, written[1], negative), "negative")
    empty = np.zeros_like(data)
    _expect_error(tmp_path, written[1], _labels_from(tmp_path, written[1], empty), "empty")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda m: m.pop("SeriesNumber"), "SeriesNumber"),
        (lambda m: m["segmentAttributes"].append([]), "one label map"),
        (lambda m: m["segmentAttributes"][0][1].update(labelID=1), "duplicate labelID"),
        (
            lambda m: m["segmentAttributes"][0][0].pop("SegmentedPropertyTypeCodeSequence"),
            "SegmentedPropertyTypeCodeSequence",
        ),
        (lambda m: m["segmentAttributes"][0][0].update(SegmentAlgorithmType="GUESS"), "GUESS"),
        (lambda m: m["segmentAttributes"][0][0].pop("SegmentAlgorithmName"), "AlgorithmName"),
        (lambda m: m.update(InstanceNumber="first"), "metadata: InstanceNumber is not an integer"),
        (lambda m: m["segmentAttributes"][0].__setitem__(1, "tumor"), "segment 2: not an object"),
        (lambda m: m["segmentAttributes"].__setitem__(0, {"labelID": 1}), "lists no segments"),
        *(
            (
                lambda m, rgb=rgb: m["segmentAttributes"][0][1].update(
                    recommendedDisplayRGBValue=rgb
                ),
                "segment 2: recommendedDisplayRGBValue must be three integers 0-255",
            )
            for rgb in ("red", 255, [255, 0], [255, 0, "x"], [0.5, 0, 0], [256, 0, 0], [None] * 3)
        ),
    ],
)
def test_bad_metadata_is_rejected(tmp_path, written, change, match):
    meta = copy.deepcopy(META)
    change(meta)
    with pytest.raises(seg.SegError, match=match):
        seg.load_metadata(_metadata(tmp_path, meta))


def test_4d_source_is_rejected(tmp_path):
    nifti, _ = _source(tmp_path, make_4d())
    img = nib.load(nifti)
    labels = np.ones(img.shape[:3], dtype=np.uint8)
    path = tmp_path / "labels.nii.gz"
    nib.save(nib.Nifti1Image(labels, img.affine), path)
    _expect_error(tmp_path, nifti, path, "4D")


def test_multi_file_enhanced_source_is_rejected(tmp_path):
    nifti, _ = _source(tmp_path, make_enhanced(make_series(SeriesSpec(n_slices=4)), n_files=2))
    labels = _label_map(nifti, tmp_path / "labels.nii.gz")
    assert labels.any()
    _expect_error(tmp_path, nifti, tmp_path / "labels.nii.gz", "several multi-frame")


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_to_seg(tmp_path):
    nifti, _ = _source(tmp_path, make_series())
    _label_map(nifti, tmp_path / "labels.nii.gz")
    out = tmp_path / "seg.dcm"
    args = ["to-seg", str(nifti), str(tmp_path / "labels.nii.gz"), str(_metadata(tmp_path))]
    result = CliRunner().invoke(main, [*args, str(out)])
    assert result.exit_code == 0, result.output
    assert result.output == "Wrote DICOM SEG with 2 segments on 5 source images.\n"
    assert out.exists()

    result = CliRunner().invoke(main, [*args, str(out)])
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert "sub-" not in result.output and "Synth" not in result.output
