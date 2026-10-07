"""The synthetic generator is ground truth for later phases, so it is tested itself."""

from dataclasses import replace

import numpy as np
import pydicom
import pytest
from dicom_validator.validator.validation_result import ErrorCode
from synthetic import (
    AXIAL,
    CORONAL,
    SAGITTAL,
    SeriesSpec,
    flipped,
    make_localizer,
    make_series,
    rotated,
    write_series,
)

SPECS = {
    "axial": SeriesSpec(),
    "coronal": SeriesSpec(orientation=CORONAL),
    "sagittal": SeriesSpec(orientation=SAGITTAL),
    "flip-rows": SeriesSpec(orientation=flipped(AXIAL, rows=True)),
    "flip-both": SeriesSpec(orientation=flipped(SAGITTAL, rows=True, cols=True)),
    "oblique": SeriesSpec(orientation=rotated(CORONAL, (1.0, 1.0, 0.0), 17.0)),
    "descending": SeriesSpec(descending=True),
    "shuffled": SeriesSpec(shuffle=True, seed=3),
    "signed": SeriesSpec(signed=True),
}


def _normal(ds: pydicom.Dataset) -> np.ndarray:
    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    return np.cross(iop[:3], iop[3:])


def _spatially_sorted(datasets: list[pydicom.Dataset]) -> list[pydicom.Dataset]:
    normal = _normal(datasets[0])
    return sorted(datasets, key=lambda ds: float(np.dot(ds.ImagePositionPatient, normal)))


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_affine_matches_per_slice_geometry(spec):
    series = make_series(spec)
    for k, ds in enumerate(_spatially_sorted(series.datasets)):
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        row_dir, col_dir = iop[:3], iop[3:]
        ps = np.asarray(ds.PixelSpacing, dtype=float)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        for i, j in [(0, 0), (spec.rows - 1, 0), (0, spec.columns - 1), (2, 3)]:
            expected = ipp + j * ps[1] * row_dir + i * ps[0] * col_dir
            np.testing.assert_allclose((series.affine_lps @ [i, j, k, 1])[:3], expected, atol=1e-4)


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_volume_matches_pixel_arrays_in_spatial_order(spec):
    series = make_series(spec)
    stacked = np.stack([ds.pixel_array for ds in _spatially_sorted(series.datasets)], axis=-1)
    np.testing.assert_array_equal(stacked, series.volume)
    assert series.volume.shape == (spec.rows, spec.columns, spec.n_slices)


def test_voxel_values_are_unique():
    volume = make_series().volume
    assert np.unique(volume).size == volume.size


def test_descending_reverses_instance_numbers_along_normal():
    datasets = _spatially_sorted(make_series(SeriesSpec(descending=True)).datasets)
    assert [ds.InstanceNumber for ds in datasets] == [5, 4, 3, 2, 1]


def test_shuffle_changes_list_order_only():
    plain = make_series(SeriesSpec(seed=3))
    shuffled = make_series(SeriesSpec(seed=3, shuffle=True))
    assert [ds.InstanceNumber for ds in shuffled.datasets] != [1, 2, 3, 4, 5]
    np.testing.assert_array_equal(plain.volume, shuffled.volume)


def test_generation_is_deterministic():
    a, b = make_series(SeriesSpec(seed=7)), make_series(SeriesSpec(seed=7))
    assert [ds.SOPInstanceUID for ds in a.datasets] == [ds.SOPInstanceUID for ds in b.datasets]
    assert [ds.PixelData for ds in a.datasets] == [ds.PixelData for ds in b.datasets]
    assert make_series(SeriesSpec(seed=8)).series_instance_uid != a.series_instance_uid


def test_uids_are_valid_and_unique():
    datasets = make_series().datasets
    uids = [ds.SOPInstanceUID for ds in datasets]
    assert len(set(uids)) == len(uids)
    for uid in [*uids, datasets[0].SeriesInstanceUID, datasets[0].StudyInstanceUID]:
        assert uid.startswith("2.25.")
        assert len(uid) <= 64
        assert pydicom.uid.UID(uid).is_valid


def test_gaps_have_no_affine():
    series = make_series(SeriesSpec(slice_offsets=[0.0, 3.0, 6.0, 12.0, 15.0]))
    assert series.affine_lps is None
    offsets = [float(np.dot(ds.ImagePositionPatient, [0, 0, 1])) - 30.0 for ds in series.datasets]
    np.testing.assert_allclose(offsets, [0.0, 3.0, 6.0, 12.0, 15.0])


def test_per_slice_thickness_and_rescale():
    spec = SeriesSpec(
        n_slices=3, slice_thickness=[3.0, 3.0, 5.0], rescale=[(1.0, 0.0), (2.0, 0.0), (2.5, -10.0)]
    )
    datasets = _spatially_sorted(make_series(spec).datasets)
    assert [float(ds.SliceThickness) for ds in datasets] == [3.0, 3.0, 5.0]
    assert [float(ds.RescaleSlope) for ds in datasets] == [1.0, 2.0, 2.5]
    assert [float(ds.RescaleIntercept) for ds in datasets] == [0.0, 0.0, -10.0]


def test_per_slice_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="slice_thickness"):
        make_series(SeriesSpec(n_slices=3, slice_thickness=[3.0, 3.0]))


def test_localizer_has_three_orientations_in_one_series():
    series = make_localizer()
    assert len({ds.SeriesInstanceUID for ds in series.datasets}) == 1
    assert len({tuple(ds.ImageOrientationPatient) for ds in series.datasets}) == 3
    assert len({ds.SOPInstanceUID for ds in series.datasets}) == 3
    for ds in series.datasets:
        assert ds.SOPInstanceUID == ds.file_meta.MediaStorageSOPInstanceUID


def test_write_and_read_back(tmp_path):
    series = make_series(SeriesSpec(rescale=(2.0, -5.0)))
    paths = write_series(series, tmp_path)
    for ds, path in zip(series.datasets, paths, strict=True):
        read = pydicom.dcmread(path)
        assert read.file_meta.TransferSyntaxUID == ds.file_meta.TransferSyntaxUID
        assert read.SOPInstanceUID == ds.SOPInstanceUID
        assert read.ImagePositionPatient == ds.ImagePositionPatient
        np.testing.assert_array_equal(read.pixel_array, ds.pixel_array)


@pytest.mark.network
def test_default_series_passes_iod_validation(tmp_path, iod_errors):
    for spec in (SeriesSpec(), replace(SPECS["oblique"], signed=True)):
        for path in write_series(make_series(spec), tmp_path / str(id(spec))):
            assert iod_errors(path) == []


@pytest.mark.network
def test_rescale_is_the_only_iod_deviation(tmp_path, iod_errors):
    # Real MR data (e.g. Philips) carries rescale although the MR Image IOD has no
    # Modality LUT module; the generator reproduces that deliberately.
    path = write_series(make_series(SeriesSpec(rescale=(2.0, -5.0))), tmp_path)[0]
    errors = {(name, code) for _module, name, code in iod_errors(path)}
    assert errors == {
        ("RescaleIntercept", ErrorCode.TagUnexpected),
        ("RescaleSlope", ErrorCode.TagUnexpected),
        ("RescaleType", ErrorCode.TagUnexpected),
    }
