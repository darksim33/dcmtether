import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataelem import DataElement
from pydicom.uid import JPEG2000Lossless, JPEGLSLossless, RLELossless
from pydicom.valuerep import DS
from synthetic import (
    CORONAL,
    SAGITTAL,
    SeriesSpec,
    SyntheticSeries,
    compressed,
    flipped,
    make_4d,
    make_enhanced,
    make_localizer,
    make_series,
    make_two_planes,
    rotated,
    write_series,
)

from dcmtether import scan as scan_module
from dcmtether.scan import Status, scan

SPECS_OBLIQUE = SeriesSpec(orientation=rotated(CORONAL, (1.0, 1.0, 0.0), 17.0))


def _scan(tmp_path: Path, *series: SyntheticSeries):
    for n, s in enumerate(series):
        write_series(s, tmp_path / "src" / f"series{n}")
    return scan(tmp_path / "src")


def _only(result):
    assert len(result.series) == 1, [s.reasons for s in result.series]
    return result.series[0]


# --- supported ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        SeriesSpec(),
        SeriesSpec(orientation=CORONAL),
        SeriesSpec(orientation=flipped(SAGITTAL, rows=True)),
        SeriesSpec(orientation=rotated(CORONAL, (1.0, 1.0, 0.0), 17.0)),
        SeriesSpec(descending=True),
        SeriesSpec(shuffle=True, seed=3),
        SeriesSpec(signed=True),
        SeriesSpec(rescale=(2.0, -5.0)),
    ],
    ids=["axial", "coronal", "sagittal-flipped", "oblique", "descending", "shuffled", "signed",
         "uniform-rescale"],
)  # fmt: skip
def test_regular_stacks_are_supported(tmp_path, spec):
    series = _only(_scan(tmp_path, make_series(spec)))
    assert series.status is Status.SUPPORTED, series.reasons
    assert series.reasons == []
    assert series.geometry["n_positions"] == 5
    assert series.geometry["slice_spacing"] == pytest.approx(3.0)


def test_single_slice_is_supported(tmp_path):
    series = _only(_scan(tmp_path, make_series(SeriesSpec(n_slices=1))))
    assert series.status is Status.SUPPORTED
    assert series.geometry["slice_spacing"] is None


def test_position_jitter_within_tolerance_is_supported(tmp_path):
    s = make_series(SeriesSpec(slice_offsets=[0.0, 3.004, 5.997, 9.0, 12.003]))
    series = _only(_scan(tmp_path, s))
    assert series.status is Status.SUPPORTED, series.reasons


def test_series_are_grouped_by_uid_not_directory(tmp_path):
    a, b = make_series(SeriesSpec(series_number=1)), make_series(SeriesSpec(series_number=2))
    write_series(a, tmp_path / "src" / "mixed")
    for ds in b.datasets:  # same directory, different file names
        ds.save_as(tmp_path / "src" / "mixed" / f"B_{ds.InstanceNumber}", enforce_file_format=True)
    result = scan(tmp_path / "src")
    assert sorted(len(s.instances) for s in result.series) == [5, 5]
    assert {s.status for s in result.series} == {Status.SUPPORTED}


# --- split -------------------------------------------------------------------------------------


def test_localizer_is_split_by_plane(tmp_path):
    series = _only(_scan(tmp_path, make_localizer()))
    assert series.status is Status.SPLIT
    assert series.reasons == ["3 distinct image planes (orientation, matrix or pixel spacing)"]


def test_two_stacks_in_one_series_are_split(tmp_path):
    series = _only(_scan(tmp_path, make_two_planes()))
    assert series.status is Status.SPLIT
    assert [p["n_positions"] for p in series.geometry["parts"]] == [4, 3]
    assert [len(p.instances) for p in series.parts] == [4, 3]


def test_split_parts_follow_instance_numbers(tmp_path):
    series = _only(_scan(tmp_path, make_localizer()))
    assert [p.instances[0].instance_number for p in series.parts] == [1, 2, 3]
    assert [p.indices for p in series.parts] == [[(0,)]] * 3


def test_split_series_with_an_irregular_part_is_unsupported(tmp_path):
    s = make_two_planes()
    s.datasets.pop(1)  # second axial slice: the axial part now has a gap
    reasons = _unsupported(tmp_path, s)
    assert reasons[0] == "2 distinct image planes (orientation, matrix or pixel spacing)"
    assert reasons[1].startswith("plane 1: missing slices")


# --- enhanced multi-frame ----------------------------------------------------------------------

ENHANCED_KEYS = {
    "echo": ["EffectiveEchoTime"],
    "dynamic": ["TemporalPositionIndex"],
    "diffusion": ["DiffusionBValue", "DiffusionGradientOrientation"],
}


def test_enhanced_3d_is_supported(tmp_path):
    result = _scan(tmp_path, make_enhanced(make_series(SPECS_OBLIQUE)))
    series = _only(result)
    assert series.status is Status.SUPPORTED, series.reasons
    assert len(series.files) == 1
    assert len(series.instances) == 5  # one per frame
    [part] = series.parts
    assert [i.frame for i in part.instances] == [0, 1, 2, 3, 4]
    assert series.geometry["n_positions"] == 5


@pytest.mark.parametrize(("kind", "key"), ENHANCED_KEYS.items(), ids=ENHANCED_KEYS.keys())
@pytest.mark.parametrize("n_files", [1, 3])
def test_enhanced_4d_is_supported(tmp_path, kind, key, n_files):
    series = _only(_scan(tmp_path, make_enhanced(make_4d(kind=kind), n_files=n_files)))
    assert series.status is Status.SUPPORTED, series.reasons
    assert series.geometry["volume_key"] == key
    assert len(series.files) == n_files
    [part] = series.parts
    assert part.indices == [(k, v) for v in range(3) for k in range(5)]


def test_enhanced_without_functional_groups_is_unsupported(tmp_path):
    s = make_enhanced(make_series())
    for ds in s.datasets:
        del ds.SharedFunctionalGroupsSequence
        del ds.PerFrameFunctionalGroupsSequence
    assert _unsupported(tmp_path, s) == [
        "missing or invalid ImagePositionPatient, ImageOrientationPatient, PixelSpacing"
    ]


def test_multi_frame_classic_mr_is_still_unsupported(tmp_path):
    s = make_series(SeriesSpec(n_slices=1))
    s.datasets[0].NumberOfFrames = 1
    assert "multi-frame instances" in _unsupported(tmp_path, s)


def test_volumes_by_combined_key(tmp_path):
    # 2 time points x 2 echoes: no single attribute tells the 4 volumes apart.
    s = make_4d(n_volumes=4, kind="none")
    for ds in s.datasets:
        v = (ds.InstanceNumber - 1) // 5
        ds.TemporalPositionIdentifier = v // 2 + 1
        ds.EchoTime = DS(15 * (v % 2 + 1))
    series = _only(_scan(tmp_path, s))
    assert series.status is Status.SUPPORTED, series.reasons
    assert series.geometry["volume_key"] == ["TemporalPositionIdentifier", "EchoTime"]
    [part] = series.parts
    assert part.indices == [(k, v) for v in range(4) for k in range(5)]


# --- compressed --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transfer_syntax", [RLELossless, JPEGLSLossless, JPEG2000Lossless], ids=["rle", "jls", "j2k"]
)
def test_lossless_compressed_series_is_supported(tmp_path, transfer_syntax):
    spec = SeriesSpec(rows=32, columns=40, n_slices=3)  # JPEG 2000 needs >= 32 px
    series = _only(_scan(tmp_path, compressed(make_series(spec), transfer_syntax)))
    assert series.status is Status.SUPPORTED, series.reasons


# --- 4D ----------------------------------------------------------------------------------------

VOLUME_KEYS = {
    "echo": ["EchoNumbers"],
    "dynamic": ["TemporalPositionIdentifier"],
    "diffusion": ["DiffusionBValue", "DiffusionGradientOrientation"],
    # b-values only in vendor private tags (older classic exports)
    "diffusion_siemens": ["DiffusionBValue", "DiffusionGradientOrientation"],
    "diffusion_ge": ["DiffusionBValue", "DiffusionGradientOrientation"],
    "diffusion_philips": ["DiffusionBValue", "DiffusionGradientOrientation"],
}


@pytest.mark.parametrize(("kind", "key"), VOLUME_KEYS.items(), ids=VOLUME_KEYS.keys())
def test_multi_volume_series_is_supported(tmp_path, kind, key):
    series = _only(_scan(tmp_path, make_4d(n_volumes=3, kind=kind)))
    assert series.status is Status.SUPPORTED, series.reasons
    assert series.geometry["n_volumes"] == 3
    assert series.geometry["n_positions"] == 5
    assert series.geometry["volume_key"] == key
    [part] = series.parts
    assert part.volume_key == tuple(key)
    # instances ordered volume by volume, slices ascending along the normal
    assert part.indices == [(k, v) for v in range(3) for k in range(5)]
    assert [i.instance_number for i in part.instances] == list(range(1, 16))


def test_volumes_follow_acquisition_order_not_key_order(tmp_path):
    s = make_4d(n_volumes=3, kind="echo")
    for ds in s.datasets:  # echo numbers 3, 2, 1 in acquisition order
        ds.EchoNumbers = 4 - ds.EchoNumbers
    [part] = _only(_scan(tmp_path, s)).parts
    assert [part.instances[n].echo_number for n in (0, 5, 10)] == [3, 2, 1]


def test_volumes_of_a_shuffled_series(tmp_path):
    [part] = _only(_scan(tmp_path, make_4d(SeriesSpec(shuffle=True, seed=4)))).parts
    assert [i.instance_number for i in part.instances] == list(range(1, 16))


def test_indistinguishable_volumes_are_unsupported(tmp_path):
    assert _unsupported(tmp_path, make_4d(n_volumes=2, kind="none")) == [
        "2 volumes at the same 5 slice positions, but no attribute tells them apart"
    ]


def test_volume_attribute_must_be_constant_within_a_volume(tmp_path):
    s = make_4d(n_volumes=2, kind="none")
    for ds in s.datasets:  # distinct per instance, so it does not group slices into volumes
        ds.TriggerTime = DS(10 * ds.InstanceNumber)
    assert _unsupported(tmp_path, s) == [
        "2 volumes at the same 5 slice positions, but no attribute tells them apart"
    ]


def test_rescale_may_differ_between_volumes(tmp_path):
    s = make_4d(n_volumes=3, volume_rescale=[(1.0, 0.0), (2.0, 0.0), (2.0, -5.0)])
    series = _only(_scan(tmp_path, s))
    assert series.status is Status.SUPPORTED, series.reasons


def test_rescale_must_be_uniform_within_a_volume(tmp_path):
    s = make_4d(n_volumes=2)
    s.datasets[0].RescaleSlope = DS("2")
    s.datasets[0].RescaleIntercept = DS("0")
    assert _unsupported(tmp_path, s) == [
        "rescale slope/intercept differ within a volume (2 distinct)"
    ]


# --- unsupported -------------------------------------------------------------------------------


def _unsupported(tmp_path, s: SyntheticSeries) -> list[str]:
    series = _only(_scan(tmp_path, s))
    assert series.status is Status.UNSUPPORTED
    return series.reasons


def test_missing_slices(tmp_path):
    s = make_series(SeriesSpec(slice_offsets=[0.0, 3.0, 6.0, 12.0, 15.0, 24.0]))
    reasons = _unsupported(tmp_path, s)
    assert reasons == ["missing slices (3 missing in the stack)"]


def test_non_uniform_spacing(tmp_path):
    reasons = _unsupported(tmp_path, make_series(SeriesSpec(slice_offsets=[0.0, 3.0, 7.0, 10.0])))
    assert reasons == ["non-uniform slice spacing (3 to 4 mm)"]


def test_mixed_thickness(tmp_path):
    s = make_series(SeriesSpec(n_slices=3, slice_thickness=[3.0, 3.0, 5.0]))
    assert _unsupported(tmp_path, s) == ["mixed slice thickness (3, 5 mm)"]


def test_per_slice_rescale(tmp_path):
    s = make_series(SeriesSpec(n_slices=2, rescale=[(1.0, 0.0), (2.0, 0.0)]))
    assert _unsupported(tmp_path, s) == [
        "rescale slope/intercept differ within a volume (2 distinct)"
    ]


def test_mosaic(tmp_path):
    s = make_series(SeriesSpec(extra={"ImageType": ["ORIGINAL", "PRIMARY", "M", "MOSAIC"]}))
    assert _unsupported(tmp_path, s) == ["Siemens mosaic images"]


def test_sheared_stack(tmp_path):
    s = make_series()
    for ds in s.datasets:
        x, y, z = (float(v) for v in ds.ImagePositionPatient)
        ds.ImagePositionPatient = [x + 0.5 * ds.InstanceNumber, y, z]  # in-plane drift
    assert _unsupported(tmp_path, s) == [
        "slices are not stacked along the slice normal (sheared stack)"
    ]


def test_uneven_multi_volume(tmp_path):
    s = make_4d(n_volumes=2)
    s.datasets.pop()
    assert _unsupported(tmp_path, s) == ["uneven number of instances per slice position"]


def test_duplicate_file(tmp_path):
    s = make_series()
    paths = write_series(s, tmp_path / "src" / "a")
    shutil.copy(paths[0], tmp_path / "src" / "a" / "copy_of_first")
    series = _only(scan(tmp_path / "src"))
    assert series.status is Status.UNSUPPORTED
    assert series.reasons[0] == "duplicate SOPInstanceUID (1 extra file)"


def test_missing_geometry(tmp_path):
    s = make_series()
    del s.datasets[2].ImagePositionPatient
    assert _unsupported(tmp_path, s) == ["missing or invalid ImagePositionPatient"]


def test_mixed_frame_of_reference(tmp_path):
    s = make_series()
    s.datasets[0].FrameOfReferenceUID = "2.25.1"
    assert _unsupported(tmp_path, s) == ["mixed frames of reference (2 distinct)"]


def test_unsupported_sop_class_skips_geometry_reasons(tmp_path):
    s = make_series()
    for ds in s.datasets:
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        ds.Modality = "CT"
    assert _unsupported(tmp_path, s) == ["SOP class not supported: CT Image Storage"]


def test_all_problems_are_reported(tmp_path):
    s = make_series(SeriesSpec(slice_offsets=[0.0, 3.0, 9.0], slice_thickness=[3.0, 3.0, 4.0]))
    reasons = _unsupported(tmp_path, s)
    assert set(reasons) == {
        "mixed slice thickness (3, 4 mm)",
        "missing slices (1 missing in the stack)",
    }


# --- skipped files, privacy, header-only ------------------------------------------------------


def test_non_dicom_files_are_skipped(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "notes.txt").write_text("hello")
    write_series(make_series(), tmp_path / "src" / "s")
    result = scan(tmp_path / "src")
    assert [s.reason for s in result.skipped] == ["not a DICOM file"]
    assert result.n_files == 6


def test_excluded_directory_is_not_walked(tmp_path):
    write_series(make_series(), tmp_path / "src" / "s")
    out = tmp_path / "src" / "dataset"
    write_series(make_series(SeriesSpec(seed=9)), out)
    result = scan(tmp_path / "src", exclude=out)
    assert len(result.series) == 1
    assert result.skipped == []


def test_reasons_never_contain_identifying_values(tmp_path):
    cases = [
        make_localizer(),
        make_4d(),
        make_series(SeriesSpec(seed=1, slice_offsets=[0.0, 3.0, 9.0], slice_thickness=[1.0, 2, 3])),
    ]
    result = _scan(tmp_path, *cases)
    text = " ".join(r for s in result.series for r in s.reasons)
    assert text
    for s in result.series:
        for value in [
            s.series_instance_uid,
            s.first.study_instance_uid,
            s.first.sop_instance_uid,
            s.first.series_description,
            "Synthetic",
            "SYNTH-",
            "2.25.",
            str(tmp_path),
        ]:
            assert value not in text


def test_pixel_data_is_never_read(tmp_path, monkeypatch):
    paths = write_series(make_series(), tmp_path / "src" / "s")
    for path in paths:  # corrupt the pixel data; a pixel read would now fail or differ
        data = path.read_bytes()
        path.write_bytes(data[: len(data) - 20])

    calls = []
    real_dcmread = pydicom.dcmread

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_dcmread(*args, **kwargs)

    monkeypatch.setattr(scan_module.pydicom, "dcmread", spy)
    series = _only(scan(tmp_path / "src"))
    assert series.status is Status.SUPPORTED
    assert calls
    assert all(c.get("stop_before_pixels") is True for c in calls)


def test_pydicom_value_warnings_are_suppressed(tmp_path, caplog):
    # pydicom validates values on access and quotes them in warnings/log records; use
    # elements the scanner actually reads.
    s = make_series(SeriesSpec(n_slices=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.datasets[0].add(DataElement(0x00080070, "LO", "SECRET" + "x" * 80))  # Manufacturer
        s.datasets[0].add(DataElement(0x00200052, "UI", "SECRET.UID"))  # FrameOfReferenceUID
        write_series(s, tmp_path / "src")
    caplog.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = scan(tmp_path / "src")
    assert _only(result).status is Status.SUPPORTED
    assert [str(w.message) for w in caught] == []
    assert "SECRET" not in caplog.text
    assert pydicom.config.settings.reading_validation_mode != pydicom.config.IGNORE  # restored


# --- real public test data ---------------------------------------------------------------------


def _copy(test_file, names: list[str], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy(test_file(name), directory / Path(name).name)
    return directory


@pytest.mark.network
def test_real_ge_adc_series_is_supported(tmp_path, test_file):
    names = [f"dcmqi/paramaps/701-ADCb500/{i:06d}.dcm" for i in range(20)]
    series = _only(scan(_copy(test_file, names, tmp_path / "adc")))
    assert series.status is Status.SUPPORTED, series.reasons
    assert series.geometry["n_positions"] == 20
    assert series.geometry["slice_spacing"] == pytest.approx(3.0, abs=1e-3)
    assert series.first.manufacturer == "GE MEDICAL SYSTEMS"


@pytest.mark.network
@pytest.mark.parametrize(
    ("name", "status", "reason"),
    [
        ("pydicom/MR_small.dcm", Status.SUPPORTED, None),
        ("pydicom/MR-SIEMENS-DICOM-WithOverlays.dcm", Status.SUPPORTED, None),
        ("pydicom/MR_small_bigendian.dcm", Status.SUPPORTED, None),
        ("pydicom/emri_small.dcm", Status.UNSUPPORTED,  # enhanced, but no functional groups
         "missing or invalid ImagePositionPatient, ImageOrientationPatient, PixelSpacing"),
        ("WG04/J2KR/MR1_J2KR", Status.SUPPORTED, None),
        ("WG04/RLE/MR2_RLE", Status.SUPPORTED, None),
        ("WG04/JPLL/MR3_JPLL", Status.SUPPORTED, None),
        ("WG04/JLSL/MR4_JLSL", Status.SUPPORTED, None),
        ("WG04/J2KI/MR1_J2KI", Status.UNSUPPORTED,
         "lossy transfer syntax not supported: JPEG 2000 Image Compression"),
        ("WG04/JPLY/MR2_JPLY", Status.UNSUPPORTED,
         "lossy transfer syntax not supported: JPEG Extended (Process 2 and 4)"),
        ("WG04/JLSN/MR3_JLSN", Status.UNSUPPORTED,
         "lossy transfer syntax not supported: JPEG-LS Lossy (Near-Lossless) Image Compression"),
    ],
)  # fmt: skip
def test_real_single_files(tmp_path, test_file, name, status, reason):
    series = _only(scan(_copy(test_file, [name], tmp_path / "f")))
    assert series.status is status, series.reasons
    if reason:
        assert reason in series.reasons


@pytest.mark.network
def test_bare_dataset_without_file_meta_is_skipped_with_its_own_reason(tmp_path, test_file):
    result = scan(_copy(test_file, ["pydicom/ExplVR_LitEndNoMeta.dcm"], tmp_path / "f"))
    assert result.series == []
    assert [s.reason for s in result.skipped] == [
        "DICOM dataset without preamble/file meta (not supported)"
    ]


def test_geometry_summary_is_json_friendly(tmp_path):
    import json

    series = _only(_scan(tmp_path, make_series(replace(SeriesSpec(), n_slices=3))))
    geometry = json.loads(json.dumps(series.geometry))
    assert geometry["orientation"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert geometry["pixel_spacing"] == [0.8, 0.9]
    assert np.isclose(geometry["slice_spacing"], 3.0)
