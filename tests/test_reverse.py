import json
import shutil
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from click.testing import CliRunner
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    UID,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGLSLossless,
    RLELossless,
)
from pydicom.valuerep import DS
from roundtrip import SPECS, forward, round_trip
from synthetic import (
    VOLUME_KINDS,
    SeriesSpec,
    SyntheticSeries,
    compressed,
    make_4d,
    make_enhanced,
    make_localizer,
    make_series,
    make_two_planes,
    write_series,
)

from dcmtether import tether
from dcmtether.cli import main
from dcmtether.compare import compare_series, dicom_files
from dcmtether.forward import convert_dataset
from dcmtether.manifest import write_manifest
from dcmtether.reverse import ReverseError, to_dicom
from dcmtether.scan import scan


def _assert_identical(report) -> None:
    assert report.identical, report.lines()
    assert report.notes == [], [f"{n.instance}: {n.where}: {n.kind}" for n in report.notes]


def _write(series: SyntheticSeries, tmp_path: Path) -> Path:
    source = tmp_path / "src" / "series"
    write_series(series, source)
    return tmp_path / "src"


# --- synthetic geometry ------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_round_trip_is_identical(tmp_path, spec):
    report = round_trip(_write(make_series(spec), tmp_path), tmp_path)
    _assert_identical(report)
    assert report.n_compared == spec.n_slices


# --- 4D and split series -----------------------------------------------------------------------

KINDS = [k for k in VOLUME_KINDS if k != "none"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "spec",
    [SeriesSpec(), SeriesSpec(orientation=SPECS["oblique"].orientation), SeriesSpec(shuffle=True)],
    ids=["axial", "oblique", "shuffled"],
)
def test_round_trip_4d(tmp_path, kind, spec):
    report = round_trip(_write(make_4d(spec, n_volumes=3, kind=kind), tmp_path), tmp_path)
    _assert_identical(report)
    assert report.n_compared == 15


def test_round_trip_4d_with_per_volume_rescale(tmp_path):
    series = make_4d(n_volumes=3, volume_rescale=[(1.0, 0.0), (2.0, 0.0), (2.0, -5.0)])
    _assert_identical(round_trip(_write(series, tmp_path), tmp_path))


@pytest.mark.parametrize("make", [make_localizer, make_two_planes], ids=["localizer", "two-planes"])
def test_round_trip_split_series(tmp_path, make):
    series = make()
    report = round_trip(_write(series, tmp_path), tmp_path)
    _assert_identical(report)
    assert report.n_compared == len(series.datasets)


# --- compressed sources (written back as Explicit VR Little Endian) -----------------------------

CODECS = {"rle": RLELossless, "jls": JPEGLSLossless, "j2k": JPEG2000Lossless}
TS_NOTE = "changed from {} to Explicit VR Little Endian (lossless)"


def _assert_identical_but_decompressed(report, transfer_syntax) -> None:
    assert report.identical, report.lines()
    assert {f"{n.where}: {n.kind}" for n in report.notes} == {
        "TransferSyntaxUID: " + TS_NOTE.format(UID(transfer_syntax).name)
    }


@pytest.mark.parametrize("codec", CODECS.values(), ids=CODECS.keys())
@pytest.mark.parametrize("spec", ["axial", "oblique", "signed", "shuffled"])
def test_round_trip_compressed(tmp_path, codec, spec):
    matrix = {"rows": 32, "columns": 40, "n_slices": 3} if codec == JPEG2000Lossless else {}
    series = compressed(make_series(replace(SPECS[spec], **matrix)), codec)
    report = round_trip(_write(series, tmp_path), tmp_path)
    _assert_identical_but_decompressed(report, codec)
    for path in dicom_files(tmp_path / "rebuilt"):
        meta = pydicom.dcmread(path, stop_before_pixels=True).file_meta
        assert meta.TransferSyntaxUID == ExplicitVRLittleEndian


def test_round_trip_compressed_4d(tmp_path):
    series = compressed(make_4d(n_volumes=2, kind="echo"), RLELossless)
    _assert_identical_but_decompressed(round_trip(_write(series, tmp_path), tmp_path), RLELossless)


COMPRESSED_REAL_FILES = [
    *(
        f"WG04/{codec}/MR{n}_{codec}"
        for codec in ("RLE", "JPLL", "JLSL", "J2KR")
        for n in range(1, 5)
    ),
    "pydicom/MR_small_RLE.dcm",
    "pydicom/MR_small_jp2klossless.dcm",
    "pydicom/MR_small_jpeg_ls_lossless.dcm",
    "pydicom/MR2_J2KR.dcm",
]


@pytest.mark.network
@pytest.mark.parametrize("name", COMPRESSED_REAL_FILES)
def test_round_trip_compressed_real_file(tmp_path, test_file, name):
    source = tmp_path / "src" / "series"
    source.mkdir(parents=True)
    shutil.copy(test_file(name), source / "file.dcm")
    transfer_syntax = pydicom.dcmread(source / "file.dcm", stop_before_pixels=True).file_meta
    report = round_trip(tmp_path / "src", tmp_path)
    _assert_identical_but_decompressed(report, transfer_syntax.TransferSyntaxUID)


def test_cli_to_dcm_reports_the_transfer_syntax_change(tmp_path):
    series = compressed(make_series(SeriesSpec(n_slices=2)), RLELossless)
    [row] = forward(_write(series, tmp_path), tmp_path / "dataset")
    result = CliRunner().invoke(
        main, ["to-dcm", str(tmp_path / "dataset" / row["nifti_path"]), str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    assert result.output == (
        "Wrote 2 DICOM files.\n"
        "Transfer syntax changed from RLE Lossless to Explicit VR Little Endian (lossless).\n"
    )


# --- enhanced multi-frame ----------------------------------------------------------------------


ENHANCED_CASES = [(None, 1)] + [(kind, n) for kind in KINDS for n in (1, 3)]  # 4D: 3 volumes


@pytest.mark.parametrize(("kind", "n_files"), ENHANCED_CASES)
def test_round_trip_enhanced(tmp_path, kind, n_files):
    classic = make_series() if kind is None else make_4d(kind=kind)
    series = make_enhanced(classic, n_files=n_files)
    report = round_trip(_write(series, tmp_path), tmp_path)
    _assert_identical(report)
    assert report.n_compared == n_files


def test_round_trip_enhanced_with_per_volume_rescale(tmp_path):
    classic = make_4d(n_volumes=3, volume_rescale=[(1.0, 0.0), (2.0, 0.0), (2.0, -5.0)])
    _assert_identical(round_trip(_write(make_enhanced(classic), tmp_path), tmp_path))


def test_round_trip_enhanced_compressed(tmp_path):
    series = compressed(make_enhanced(make_4d(kind="dynamic")), RLELossless)
    _assert_identical_but_decompressed(round_trip(_write(series, tmp_path), tmp_path), RLELossless)


def test_round_trip_with_uniform_rescale(tmp_path):
    series = make_series(SeriesSpec(rescale=(2.0, -5.0)))
    _assert_identical(round_trip(_write(series, tmp_path), tmp_path))


def test_round_trip_with_12_bit_data(tmp_path):
    series = make_series(SeriesSpec(extra={"BitsStored": 12, "HighBit": 11}))
    _assert_identical(round_trip(_write(series, tmp_path), tmp_path))


def test_round_trip_8_bit_odd_length(tmp_path):
    series = make_series(SeriesSpec(rows=3, columns=5, n_slices=2))
    for k, ds in enumerate(sorted(series.datasets, key=lambda d: d.InstanceNumber)):
        ds.BitsAllocated = ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelData = (series.volume[:, :, k] % 256).astype(np.uint8).tobytes() + b"\0"
        ds["PixelData"].VR = "OB"
    _assert_identical(round_trip(_write(series, tmp_path), tmp_path))


def test_round_trip_keeps_bytes_after_the_last_pixel(tmp_path):
    series = make_series(SeriesSpec(n_slices=2))
    for ds in series.datasets:
        ds.PixelData += b"\x01\x02\x03\x04"
    _assert_identical(round_trip(_write(series, tmp_path), tmp_path))


def test_unused_high_bits_are_not_kept(tmp_path):
    # Accepted (PLAN.md §6): bits above BitsStored are not part of the pixel value.
    series = make_series(SeriesSpec(n_slices=2, extra={"BitsStored": 12, "HighBit": 11}))
    for ds in series.datasets:
        ds.PixelData = (np.frombuffer(ds.PixelData, "<u2") | 0xF000).astype("<u2").tobytes()
    report = round_trip(_write(series, tmp_path), tmp_path)
    assert report.identical, report.lines()
    assert {n.kind for n in report.notes} == {"raw bytes differ (decoded pixels equal)"}


# --- known JSON-model losses (PLAN.md §6, caveats 3-5) -----------------------------------------


def _forced_ds(text: str) -> DS:
    value = DS(text.strip())
    value.original_string = text
    return value


def _nested(ds: Dataset) -> None:
    item = Dataset()
    item.SliceThickness = DS("3.00")
    item.ReferencedSOPInstanceUID = ds.SOPInstanceUID
    ds.ReferencedImageSequence = Sequence([item, Dataset()])
    ds.add_new(0x00082112, "SQ", Sequence([]))  # SourceImageSequence, empty


def _private(ds: Dataset) -> None:
    block = ds.private_block(0x0009, "DCMTETHER TEST", create=True)
    block.add_new(0x01, "LO", "private text")
    block.add_new(0x02, "OB", b"\x01\x02\x03\x04")
    block.add_new(0x03, "DS", "1.50")


def _implicit(ds: Dataset) -> None:
    _private(ds)
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian


def _charset(charset, **values):
    def change(ds: Dataset) -> None:
        ds.SpecificCharacterSet = charset
        for keyword, value in values.items():
            setattr(ds, keyword, value)

    return change


def _smallest(ds: Dataset) -> None:
    pixels = ds.pixel_array
    ds.SmallestImagePixelValue = int(pixels.min())
    ds.LargestImagePixelValue = int(pixels.max())


EDGE_CASES = {
    "ds-is-strings": lambda ds: (
        setattr(ds, "RepetitionTime", _forced_ds(" 1.0")),
        setattr(ds, "EchoTrainLength", "0005"),
        setattr(ds, "SliceThickness", DS("3." + "0" * ds.InstanceNumber)),  # equal numbers
    ),
    "nested-and-empty-sequences": _nested,
    "empty-vs-absent": lambda ds: (setattr(ds, "ImageComments", ""), delattr(ds, "ScanOptions")),
    "text-padding": lambda ds: (
        setattr(ds, "ReferringPhysicianName", "  "),
        setattr(ds, "ImageComments", "text   "),
    ),
    "private-explicit": _private,
    "private-implicit": _implicit,
    "latin1": _charset("ISO_IR 100", ReferringPhysicianName="Müller^Jörg"),
    "utf8": _charset("ISO_IR 192", ReferringPhysicianName="Müller^Jörg", ImageComments="日本"),
    "iso2022-japanese": _charset(
        ["", "ISO 2022 IR 87"],
        ReferringPhysicianName="Yamada^Tarou=山田^太郎",
    ),
    "us-or-ss-unsigned": _smallest,
}


@pytest.mark.parametrize("change", EDGE_CASES.values(), ids=EDGE_CASES.keys())
def test_round_trip_edge_cases(tmp_path, change):
    series = make_series(SeriesSpec(n_slices=3))
    for ds in series.datasets:
        change(ds)
    _assert_identical(round_trip(_write(series, tmp_path), tmp_path))


def test_round_trip_us_or_ss_signed(tmp_path):
    series = make_series(SeriesSpec(n_slices=3, signed=True))
    for ds in series.datasets:
        _smallest(ds)
    _assert_identical(round_trip(_write(series, tmp_path), tmp_path))


def test_nan_in_header_fails_the_series(tmp_path):
    series = make_series(SeriesSpec(n_slices=2))
    for ds in series.datasets:
        ds.add_new(0x00189087, "FD", float("nan"))  # DiffusionBValue
    dataset = tmp_path / "dataset"
    write_manifest(dataset, scan(_write(series, tmp_path)))
    [outcome] = convert_dataset(dataset)
    assert not outcome.ok
    assert outcome.message == "header contains NaN or infinite values (not storable as JSON)"
    assert list(dataset.rglob("*.nii.gz")) == list(dataset.rglob("*.json")) == []


# --- real public files -------------------------------------------------------------------------

REAL_FILES = [
    "pydicom/MR_small.dcm",
    "pydicom/MR_small_bigendian.dcm",
    "pydicom/MR_small_expb.dcm",
    "pydicom/MR_small_implicit.dcm",
    "pydicom/MR_small_padded.dcm",
    "pydicom/MR-SIEMENS-DICOM-WithOverlays.dcm",
    "pydicom/MR2_UNCI.dcm",
    "pydicom/MR2_UNCR.dcm",
    "WG04/REF/MR1_UNC",
    "WG04/REF/MR2_UNC",
    "WG04/REF/MR3_UNC",
    "WG04/REF/MR4_UNC",
]


@pytest.mark.network
@pytest.mark.parametrize("name", REAL_FILES)
def test_round_trip_real_file(tmp_path, test_file, name):
    source = tmp_path / "src" / "series"
    source.mkdir(parents=True)
    shutil.copy(test_file(name), source / "file.dcm")
    _assert_identical(round_trip(tmp_path / "src", tmp_path))


def _ge_adc_source(tmp_path: Path, test_file) -> Path:
    source = tmp_path / "src" / "adc"
    source.mkdir(parents=True)
    for i in range(20):  # the public copy is anonymized; naming needs a PatientName
        ds = pydicom.dcmread(test_file(f"dcmqi/paramaps/701-ADCb500/{i:06d}.dcm"))
        ds.PatientName = "Public^Adc"
        ds.save_as(source / f"{i:06d}.dcm")
    return tmp_path / "src"


@pytest.mark.network
def test_round_trip_real_ge_adc_series(tmp_path, test_file):
    report = round_trip(_ge_adc_source(tmp_path, test_file), tmp_path)
    _assert_identical(report)
    assert report.n_compared == 20


@pytest.mark.network
def test_truncated_pixel_data_fails_cleanly(tmp_path, test_file):
    source = tmp_path / "src" / "series"
    source.mkdir(parents=True)
    shutil.copy(test_file("pydicom/MR_truncated.dcm"), source / "file.dcm")
    dataset = tmp_path / "dataset"
    write_manifest(dataset, scan(tmp_path / "src"))
    [outcome] = convert_dataset(dataset)
    assert not outcome.ok


# --- validation --------------------------------------------------------------------------------


def _rebuilt_pairs(source: Path, work: Path) -> list[tuple[Path, Path]]:
    [row] = forward(source, work / "dataset")
    to_dicom(work / "dataset" / row["nifti_path"], work / "rebuilt")
    by_uid = {pydicom.dcmread(p).SOPInstanceUID: p for p in dicom_files(work / "rebuilt")}
    return [(p, by_uid[pydicom.dcmread(p).SOPInstanceUID]) for p in dicom_files(source)]


@pytest.mark.network
def test_rebuilt_synthetic_files_validate_like_originals(tmp_path, iod_errors):
    series = make_series(SeriesSpec(n_slices=2, rescale=(2.0, -5.0)))
    for original, rebuilt in _rebuilt_pairs(_write(series, tmp_path), tmp_path):
        assert iod_errors(rebuilt) == iod_errors(original)


@pytest.mark.network
def test_rebuilt_ge_files_validate_like_originals(tmp_path, test_file, iod_errors):
    pairs = _rebuilt_pairs(_ge_adc_source(tmp_path, test_file), tmp_path)
    for original, rebuilt in pairs[:3]:
        assert iod_errors(rebuilt) == iod_errors(original)


@pytest.mark.network
@pytest.mark.parametrize("name", ["WG04/JPLL/MR1_JPLL", "WG04/J2KR/MR3_J2KR"])
def test_decompressed_files_validate_like_originals(tmp_path, test_file, iod_errors, name):
    source = tmp_path / "src" / "series"
    source.mkdir(parents=True)
    shutil.copy(test_file(name), source / "file.dcm")
    [(original, rebuilt)] = _rebuilt_pairs(tmp_path / "src", tmp_path)
    assert iod_errors(rebuilt) == iod_errors(original)


# --- safety ------------------------------------------------------------------------------------


@pytest.fixture
def converted(tmp_path) -> Path:
    """A converted synthetic series; returns the NIfTI path."""
    [row] = forward(_write(make_series(SeriesSpec(n_slices=3)), tmp_path), tmp_path / "dataset")
    return tmp_path / "dataset" / row["nifti_path"]


def _tether_of(nifti: Path) -> Path:
    return nifti.with_name(nifti.name.removesuffix(".nii.gz") + tether.TETHER_SUFFIX)


def _edit_tether(nifti: Path, change) -> None:
    path = _tether_of(nifti)
    doc = json.loads(path.read_text("utf-8"))
    change(doc)
    path.write_text(json.dumps(doc), "utf-8")


def _replace_nifti(nifti: Path, data: np.ndarray, affine: np.ndarray) -> None:
    img = nib.Nifti1Image(data, affine)
    img.header.set_slope_inter(1.0, 0.0)
    nib.save(img, nifti)


def test_existing_target_is_never_overwritten(tmp_path, converted):
    out = tmp_path / "out"
    to_dicom(converted, out)
    before = {p: p.read_bytes() for p in dicom_files(out)}
    with pytest.raises(ReverseError, match="already exists"):
        to_dicom(converted, out)
    assert {p: p.read_bytes() for p in dicom_files(out)} == before


def test_partial_existing_target_leaves_nothing_new(tmp_path, converted):
    out = tmp_path / "out"
    out.mkdir()
    (out / "IM_0002").write_text("keep me")
    with pytest.raises(ReverseError, match="already exists"):
        to_dicom(converted, out)
    assert [p.name for p in out.rglob("*") if p.is_file()] == ["IM_0002"]


@pytest.mark.parametrize("name", ["../escape", "/abs/path", "a/../../b", ""])
def test_unsafe_file_names_are_rejected(tmp_path, converted, name):
    _edit_tether(converted, lambda doc: doc["instances"][0].update(file_name=name))
    with pytest.raises(ReverseError, match="unsafe file name"):
        to_dicom(converted, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").rglob("*"))


def test_duplicate_file_names_are_rejected(tmp_path, converted):
    _edit_tether(converted, lambda doc: doc["instances"][1].update(file_name="IM_0001"))
    with pytest.raises(ReverseError, match="duplicate file name"):
        to_dicom(converted, tmp_path / "out")


def test_missing_tether_is_an_error(tmp_path, converted):
    _tether_of(converted).unlink()
    with pytest.raises(ReverseError, match=r"no \*\.tether\.json next to the NIfTI"):
        to_dicom(converted, tmp_path / "out")


def test_old_format_version_is_an_error(tmp_path, converted):
    _edit_tether(converted, lambda doc: doc.update(format_version=1))
    with pytest.raises(tether.TetherError, match="re-run to-nii"):
        to_dicom(converted, tmp_path / "out")


def test_shape_mismatch_is_an_error(tmp_path, converted):
    img = nib.load(converted)
    data = np.asarray(img.dataobj.get_unscaled())
    _replace_nifti(converted, data[:, :, :-1], img.affine)
    with pytest.raises(ReverseError, match="shape"):
        to_dicom(converted, tmp_path / "out")


def test_dtype_mismatch_is_an_error(tmp_path, converted):
    img = nib.load(converted)
    data = np.asarray(img.dataobj.get_unscaled()).astype(np.int32)
    _replace_nifti(converted, data, img.affine)
    with pytest.raises(ReverseError, match="data type"):
        to_dicom(converted, tmp_path / "out")


def test_affine_mismatch_is_an_error(tmp_path, converted):
    img = nib.load(converted)
    affine = img.affine.copy()
    affine[0, 3] += 1.0
    _replace_nifti(converted, np.asarray(img.dataobj.get_unscaled()), affine)
    with pytest.raises(ReverseError, match="affine"):
        to_dicom(converted, tmp_path / "out")


def test_4d_volume_count_mismatch_is_an_error(tmp_path):
    [row] = forward(_write(make_4d(n_volumes=3), tmp_path), tmp_path / "dataset")
    nifti = tmp_path / "dataset" / row["nifti_path"]
    img = nib.load(nifti)
    _replace_nifti(nifti, np.asarray(img.dataobj.get_unscaled())[..., :2], img.affine)
    with pytest.raises(ReverseError, match="shape"):
        to_dicom(nifti, tmp_path / "out")


def test_scaling_mismatch_is_an_error(tmp_path, converted):
    img = nib.load(converted)
    new = nib.Nifti1Image(np.asarray(img.dataobj.get_unscaled()), img.affine)
    new.header.set_slope_inter(2.0, 0.0)
    nib.save(new, converted)
    with pytest.raises(ReverseError, match="scaling"):
        to_dicom(converted, tmp_path / "out")


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_to_dcm_and_compare(tmp_path, converted):
    out = tmp_path / "out"
    result = CliRunner().invoke(main, ["to-dcm", str(converted), str(out)])
    assert result.exit_code == 0, result.output
    assert result.output == "Wrote 3 DICOM files.\n"

    result = CliRunner().invoke(main, ["compare", str(tmp_path / "src"), str(out)])
    assert result.exit_code == 0, result.output
    assert compare_series(dicom_files(tmp_path / "src"), dicom_files(out)).identical


def test_cli_to_dcm_error(tmp_path, converted):
    out = tmp_path / "out"
    CliRunner().invoke(main, ["to-dcm", str(converted), str(out)])
    result = CliRunner().invoke(main, ["to-dcm", str(converted), str(out)])
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert "sub-" not in result.output and "IM_0" not in result.output
