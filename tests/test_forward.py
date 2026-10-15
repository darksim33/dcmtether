import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from click.testing import CliRunner
from roundtrip import SPECS, output_rows
from synthetic import (
    SAGITTAL,
    SeriesSpec,
    SyntheticSeries,
    make_4d,
    make_enhanced,
    make_localizer,
    make_series,
    make_two_planes,
    write_series,
)

from dcmtether import orientation
from dcmtether.cli import main
from dcmtether.forward import convert_dataset
from dcmtether.manifest import ManifestError, update_manifest, write_manifest
from dcmtether.scanning import scan

# Synthetic defaults: PatientName "Synthetic^Phantom", StudyDate 20260101, description
# "SYNTHETIC", TR 500 / TE 15 → T1w by parameters.
DEFAULT_STEM = "sub-SynthPhant_ses-20260101_acq-SYNTHETIC_ser-01_T1w"
DEFAULT_DIR = Path("sub-SynthPhant/ses-20260101/anat")


def _dataset(tmp_path: Path, *series: SyntheticSeries, names=None) -> Path:
    for n, s in enumerate(series):
        write_series(s, tmp_path / "src" / (names[n] if names else f"s{n}"))
    dataset = tmp_path / "dataset"
    write_manifest(dataset, scan(tmp_path / "src"))
    return dataset


def _manifest_rows(dataset: Path) -> list[dict]:
    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM series ORDER BY id")]


def _world_space_check(nifti_path: Path, series: SyntheticSeries) -> None:
    """Every NIfTI voxel must hold the ground-truth value at the same LPS position.

    4D: the volume axis is not reoriented, so volume ``v`` is checked against truth ``[..., v]``.
    """
    img = nib.load(nifti_path)
    data = np.asarray(img.dataobj.get_unscaled())
    assert data.ndim == series.volume.ndim
    volumes = [data] if data.ndim == 3 else [data[..., v] for v in range(data.shape[3])]
    truths = (
        [series.volume]
        if series.volume.ndim == 3
        else [series.volume[..., v] for v in range(series.volume.shape[3])]
    )
    for volume, truth in zip(volumes, truths, strict=True):
        ijk = np.indices(volume.shape).reshape(3, -1)
        ras = img.affine @ np.vstack([ijk, np.ones(ijk.shape[1])])
        lps = orientation.LPS_TO_RAS @ ras
        truth_ijk = np.linalg.solve(series.affine_lps, lps)[:3]
        np.testing.assert_allclose(truth_ijk, np.rint(truth_ijk), atol=1e-3)
        idx = tuple(np.rint(truth_ijk).astype(int))
        np.testing.assert_array_equal(volume.reshape(-1), truth[idx])


# --- geometry and voxels -----------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_voxels_match_ground_truth_in_world_space(tmp_path, spec):
    series = make_series(spec)
    dataset = _dataset(tmp_path, series)
    outcomes = convert_dataset(dataset)
    assert [o.ok for o in outcomes] == [True], [o.message for o in outcomes]

    nifti = dataset / DEFAULT_DIR / f"{DEFAULT_STEM}.nii.gz"
    _world_space_check(nifti, series)
    img = nib.load(nifti)
    assert nib.aff2axcodes(img.affine) == ("R", "A", "S")
    assert img.get_data_dtype() == series.volume.dtype
    assert img.header["sform_code"] == 1 and img.header["qform_code"] == 1


# --- 4D and split series -----------------------------------------------------------------------


def _single_output(dataset: Path) -> Path:
    [row] = output_rows(dataset)
    return dataset / row["nifti_path"]


@pytest.mark.parametrize("kind", ["echo", "dynamic", "diffusion"])
@pytest.mark.parametrize(
    "spec",
    [SeriesSpec(), SPECS["oblique-40"], SPECS["descending"], SeriesSpec(shuffle=True, seed=6)],
    ids=["axial", "oblique-40", "descending", "shuffled"],
)
def test_4d_voxels_match_ground_truth(tmp_path, kind, spec):
    series = make_4d(spec, n_volumes=3, kind=kind)
    dataset = _dataset(tmp_path, series)
    outcomes = convert_dataset(dataset)
    assert [o.ok for o in outcomes] == [True], [o.message for o in outcomes]
    nifti = _single_output(dataset)
    _world_space_check(nifti, series)
    assert nib.load(nifti).shape[3] == 3


def test_4d_tether_indices(tmp_path):
    dataset = _dataset(tmp_path, make_4d(n_volumes=2))
    convert_dataset(dataset)
    doc = json.loads(
        _single_output(dataset).with_suffix("").with_suffix(".tether.json").read_text()
    )
    assert doc["pixel"]["shape_dicom"] == [6, 8, 5, 2]
    assert [i["index"] for i in doc["instances"]] == [[k, v] for v in range(2) for k in range(5)]


def test_4d_uniform_rescale_goes_into_header(tmp_path):
    dataset = _dataset(tmp_path, make_4d(SeriesSpec(rescale=(2.0, -5.0))))
    convert_dataset(dataset)
    img = nib.load(_single_output(dataset))
    assert (img.dataobj.slope, img.dataobj.inter) == (2.0, -5.0)


def test_4d_per_volume_rescale_leaves_header_unscaled(tmp_path):
    series = make_4d(n_volumes=3, volume_rescale=[(1.0, 0.0), (2.0, 0.0), (2.0, -5.0)])
    dataset = _dataset(tmp_path, series)
    convert_dataset(dataset)
    nifti = _single_output(dataset)
    img = nib.load(nifti)
    assert (img.dataobj.slope, img.dataobj.inter) == (1.0, 0.0)
    sidecar = json.loads(nifti.with_suffix("").with_suffix(".json").read_text("utf-8"))
    assert "RescaleSlope" not in sidecar
    assert sidecar["DcmTether"]["PerVolume"]["RescaleSlope"] == [1.0, 2.0, 2.0]
    assert sidecar["DcmTether"]["PerVolume"]["RescaleIntercept"] == [0.0, 0.0, -5.0]


def test_multi_echo_sidecar(tmp_path):
    dataset = _dataset(tmp_path, make_4d(n_volumes=3, kind="echo"))
    convert_dataset(dataset)
    nifti = _single_output(dataset)
    sidecar = json.loads(nifti.with_suffix("").with_suffix(".json").read_text("utf-8"))
    assert "EchoTime" not in sidecar  # differs per volume
    assert sidecar["RepetitionTime"] == 0.5
    assert sidecar["DcmTether"]["VolumeKey"] == ["EchoNumbers"]
    assert sidecar["DcmTether"]["PerVolume"]["EchoTime"] == [0.015, 0.03, 0.045]
    assert sidecar["DcmTether"]["PerVolume"]["EchoNumbers"] == [1, 2, 3]


def _per_volume(tmp_path, series) -> tuple[list[str], dict]:
    dataset = _dataset(tmp_path, series)
    assert [o.ok for o in convert_dataset(dataset)] == [True]
    nifti = _single_output(dataset)
    info = json.loads(nifti.with_suffix("").with_suffix(".json").read_text("utf-8"))["DcmTether"]
    return info["VolumeKey"], info["PerVolume"]


def test_per_volume_echo_time_in_seconds_without_echo_numbers(tmp_path):
    series = make_4d(n_volumes=3, kind="echo")
    for ds in series.datasets:
        del ds.EchoNumbers
    volume_key, per_volume = _per_volume(tmp_path, series)
    assert volume_key == ["EchoTime"]  # the DICOM keyword
    assert per_volume == {"EchoTime": [0.015, 0.03, 0.045]}


def test_per_volume_inversion_time_in_seconds(tmp_path):
    series = make_4d(n_volumes=3, kind="none")
    for ds in series.datasets:
        ds.InversionTime = pydicom.valuerep.DS(100 * ((ds.InstanceNumber - 1) // 5 + 1))
    volume_key, per_volume = _per_volume(tmp_path, series)
    assert volume_key == ["InversionTime"]
    assert per_volume == {"InversionTime": [0.1, 0.2, 0.3]}


def test_per_volume_keeps_keys_without_unit_rule(tmp_path):
    volume_key, per_volume = _per_volume(tmp_path, make_4d(n_volumes=3, kind="dynamic"))
    assert volume_key == ["TemporalPositionIdentifier"]
    assert per_volume["TemporalPositionIdentifier"] == [1, 2, 3]


def test_diffusion_goes_to_dwi(tmp_path):
    dataset = _dataset(tmp_path, make_4d(n_volumes=3, kind="diffusion"))
    convert_dataset(dataset)
    nifti = _single_output(dataset)
    assert nifti.relative_to(dataset).parts[:3] == ("sub-SynthPhant", "ses-20260101", "dwi")
    assert nifti.name == "sub-SynthPhant_ses-20260101_acq-SYNTHETIC_ser-01_dwi.nii.gz"
    sidecar = json.loads(nifti.with_suffix("").with_suffix(".json").read_text("utf-8"))
    assert sidecar["DcmTether"]["SuffixRule"] == "diffusion volumes"
    per_volume = sidecar["DcmTether"]["PerVolume"]
    assert per_volume["DiffusionBValue"] == [0.0, 1000.0, 1000.0]
    assert per_volume["DiffusionGradientOrientation"] == [None, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert "DiffusionSource" not in sidecar["DcmTether"]


@pytest.mark.parametrize(
    ("vendor", "gradient_key", "b0_gradient"),
    [
        ("siemens", "DiffusionGradientOrientation", None),
        ("ge", "DiffusionGradientVendor", [0.0, 0.0, 0.0]),
        ("philips", "DiffusionGradientVendor", [0.0, 0.0, 0.0]),
    ],
)
def test_private_tag_diffusion_goes_to_dwi(tmp_path, vendor, gradient_key, b0_gradient):
    dataset = _dataset(tmp_path, make_4d(n_volumes=3, kind=f"diffusion_{vendor}"))
    assert [o.ok for o in convert_dataset(dataset)] == [True]
    nifti = _single_output(dataset)
    assert nifti.name == "sub-SynthPhant_ses-20260101_acq-SYNTHETIC_ser-01_dwi.nii.gz"
    sidecar = json.loads(nifti.with_suffix("").with_suffix(".json").read_text("utf-8"))
    info = sidecar["DcmTether"]
    assert info["DiffusionSource"].lower().startswith(vendor)
    per_volume = info["PerVolume"]
    assert per_volume["DiffusionBValue"] == [0.0, 1000.0, 1000.0]
    assert per_volume[gradient_key] == [b0_gradient, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    other = {"DiffusionGradientOrientation", "DiffusionGradientVendor"} - {gradient_key}
    assert not other & per_volume.keys()


@pytest.mark.parametrize(
    ("kind", "gradient_key", "source"),
    [
        ("diffusion", "DiffusionGradientOrientation", None),
        ("diffusion_siemens", "DiffusionGradientOrientation", "siemens"),
        ("diffusion_ge", "DiffusionGradientVendor", "ge"),
        ("diffusion_philips", "DiffusionGradientVendor", "philips"),
    ],
)
def test_single_b_value_series_has_it_in_the_sidecar(tmp_path, kind, gradient_key, source):
    """One series per b-value (e.g. IVIM): listed under PerVolume like in a 4D series."""
    series = make_4d(n_volumes=3, kind=kind)
    series.datasets[:] = [ds for ds in series.datasets if (ds.InstanceNumber - 1) // 5 == 1]
    for ds in series.datasets:
        ds.ImageType = ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE"]
    dataset = _dataset(tmp_path, series)
    assert [o.ok for o in convert_dataset(dataset)] == [True]
    nifti = _single_output(dataset)
    assert nifti.parts[-2] == "dwi"
    sidecar = json.loads(nifti.with_suffix("").with_suffix(".json").read_text("utf-8"))
    info = sidecar["DcmTether"]
    assert info["PerVolume"] == {"DiffusionBValue": [1000.0], gradient_key: [[1.0, 0.0, 0.0]]}
    assert "VolumeKey" not in info
    assert "DiffusionBValue" not in sidecar
    assert info.get("DiffusionSource", "standard").lower().startswith(source or "standard")


def test_zero_b_value_of_a_non_diffusion_series_is_not_listed(tmp_path):
    """Philips writes b = 0 into anatomical images too: only diffusion series list b-values."""
    series = make_series(SeriesSpec(series_description="t1_tse"))
    for ds in series.datasets:
        ds.Manufacturer = "Philips Medical Systems"
        ds.private_block(0x2001, "Philips Imaging DD 001", create=True).add_new(0x03, "FL", 0.0)
    dataset = _dataset(tmp_path, series)
    assert [o.ok for o in convert_dataset(dataset)] == [True]
    nifti = _single_output(dataset)
    assert nifti.parts[-2] == "anat"
    sidecar = json.loads(nifti.with_suffix("").with_suffix(".json").read_text("utf-8"))
    assert "PerVolume" not in sidecar["DcmTether"]
    assert "DiffusionSource" not in sidecar["DcmTether"]


@pytest.mark.parametrize("make", [make_localizer, make_two_planes], ids=["localizer", "two-planes"])
def test_split_series_become_chunks(tmp_path, make):
    series = make()
    dataset = _dataset(tmp_path, series)
    outcomes = convert_dataset(dataset)
    assert [o.ok for o in outcomes] == [True], [o.message for o in outcomes]
    rows = output_rows(dataset)
    assert [r["chunk"] for r in rows] == list(range(1, len(series.parts) + 1))
    for row, part in zip(rows, series.parts, strict=True):
        assert f"_chunk-{row['chunk']}_" in row["nifti_path"]
        _world_space_check(dataset / row["nifti_path"], part)
        sidecar = json.loads((dataset / row["sidecar_path"]).read_text("utf-8"))
        assert sidecar["DcmTether"]["Chunk"] == row["chunk"]


def test_uniform_rescale_goes_into_header_not_voxels(tmp_path):
    series = make_series(SeriesSpec(rescale=(2.0, -5.0)))
    dataset = _dataset(tmp_path, series)
    convert_dataset(dataset)
    img = nib.load(dataset / DEFAULT_DIR / f"{DEFAULT_STEM}.nii.gz")
    assert (img.dataobj.slope, img.dataobj.inter) == (2.0, -5.0)
    _world_space_check(dataset / DEFAULT_DIR / f"{DEFAULT_STEM}.nii.gz", series)


def test_no_rescale_means_identity_scaling(tmp_path):
    dataset = _dataset(tmp_path, make_series())
    convert_dataset(dataset)
    img = nib.load(dataset / DEFAULT_DIR / f"{DEFAULT_STEM}.nii.gz")
    assert (img.dataobj.slope, img.dataobj.inter) == (1.0, 0.0)


# --- tether.json -------------------------------------------------------------------------------


def test_tether_document(tmp_path):
    series = make_series(SeriesSpec(orientation=SAGITTAL, shuffle=True, seed=2))
    dataset = _dataset(tmp_path, series)
    convert_dataset(dataset)
    doc = json.loads((dataset / DEFAULT_DIR / f"{DEFAULT_STEM}.tether.json").read_text("utf-8"))

    assert doc["format_version"] == 3
    assert doc["source"]["series_instance_uid"] == series.series_instance_uid
    assert doc["source"]["n_instances"] == 5
    assert doc["pixel"]["dtype"] == "uint16"
    assert doc["pixel"]["shape_dicom"] == [6, 8, 5]
    assert doc["pixel"]["orientation"]["from"] == "dicom_rows_columns_slices"
    ornt = np.asarray(doc["pixel"]["orientation"]["ornt_transform"])
    expected = nib.orientations.io_orientation(orientation.LPS_TO_RAS @ series.affine_lps)
    np.testing.assert_array_equal(ornt, expected)

    # Instances in DICOM order: ascending position along the normal, index 0..n-1.
    normal = orientation.slice_normal(series.datasets[0].ImageOrientationPatient)
    positions = [
        np.dot([float(v) for v in inst["elements"]["00200032"]["Value"]], normal)
        for inst in doc["instances"]
    ]
    assert positions == sorted(positions)
    assert [inst["index"] for inst in doc["instances"]] == [[k] for k in range(5)]
    assert all(inst["file_name"].startswith("IM_") for inst in doc["instances"])

    # Shared vs per-instance split; PixelData nowhere.
    assert "00100010" in doc["shared"]  # PatientName
    assert "00080018" not in doc["shared"]  # SOPInstanceUID
    assert all("00080018" in inst["elements"] for inst in doc["instances"])
    assert "00020003" not in doc["file_meta"]  # MediaStorageSOPInstanceUID differs
    assert all("00020003" in inst["file_meta"] for inst in doc["instances"])
    assert "00020010" in doc["file_meta"]  # TransferSyntaxUID shared
    text = json.dumps(doc)
    assert "7FE00010" not in text


def test_tether_holds_every_header_element(tmp_path):
    series = make_series()
    dataset = _dataset(tmp_path, series)
    convert_dataset(dataset)
    doc = json.loads((dataset / DEFAULT_DIR / f"{DEFAULT_STEM}.tether.json").read_text("utf-8"))
    by_sop = {inst["elements"]["00080018"]["Value"][0]: inst for inst in doc["instances"]}
    for ds in series.datasets:
        original = {t for t in ds.to_json_dict() if t != "7FE00010"}
        inst = by_sop[ds.SOPInstanceUID]
        assert set(doc["shared"]) | set(inst["elements"]) == original
        rebuilt = pydicom.Dataset.from_json({**doc["shared"], **inst["elements"]})
        assert rebuilt.ImagePositionPatient == ds.ImagePositionPatient
        assert rebuilt.PatientName == ds.PatientName


# --- sidecar, manifest, layout -----------------------------------------------------------------


def test_sidecar(tmp_path):
    dataset = _dataset(tmp_path, make_series(SeriesSpec(rescale=(2.0, -5.0))))
    convert_dataset(dataset)
    sidecar = json.loads((dataset / DEFAULT_DIR / f"{DEFAULT_STEM}.json").read_text("utf-8"))
    assert sidecar["EchoTime"] == 0.015  # seconds, BIDS convention
    assert sidecar["RepetitionTime"] == 0.5
    assert sidecar["SeriesNumber"] == 1
    assert sidecar["ImageType"] == ["ORIGINAL", "PRIMARY", "OTHER"]
    assert sidecar["RescaleSlope"] == 2.0
    assert sidecar["DcmTether"]["SuffixRule"] == "parameters:TR/TE"
    for key in ("PatientName", "PatientID", "PatientBirthDate", "StudyDate"):
        assert key not in sidecar


def test_manifest_records_outputs(tmp_path):
    dataset = _dataset(
        tmp_path,
        make_series(),
        make_localizer(),  # same patient as make_series(): a different PatientID would clash
        make_4d(SeriesSpec(series_number=7), kind="none"),
        names=["a", "loc", "none"],
    )
    convert_dataset(dataset)
    rows = {r["source_dir"]: r for r in _manifest_rows(dataset)}
    converted, split, unsupported = rows["a"], rows["loc"], rows["none"]
    assert converted["conversion_status"] == "converted"
    assert converted["subject"] == "SynthPhant"
    assert converted["session"] == "20260101"
    assert split["conversion_status"] == "converted"
    assert unsupported["conversion_status"] is None  # never attempted
    outputs = output_rows(dataset)
    assert [(o["series_id"], o["chunk"]) for o in outputs] == [
        (converted["id"], 1),
        (split["id"], 1),
        (split["id"], 2),
        (split["id"], 3),
    ]
    first = outputs[0]
    assert first["nifti_path"] == (DEFAULT_DIR / f"{DEFAULT_STEM}.nii.gz").as_posix()
    assert first["tether_path"].endswith(".tether.json")
    assert first["sidecar_path"].endswith("_T1w.json")
    assert first["n_volumes"] == 1
    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        assert db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("4",)


def test_dataset_readme_warns_and_is_not_overwritten(tmp_path):
    dataset = _dataset(tmp_path, make_series())
    convert_dataset(dataset)
    readme = dataset / "README.md"
    assert "identifies patients" in readme.read_text("utf-8")
    readme.write_text("my notes")
    convert_dataset(dataset)
    assert readme.read_text() == "my notes"


def test_rerun_converts_nothing_new(tmp_path):
    dataset = _dataset(tmp_path, make_series())
    assert len(convert_dataset(dataset)) == 1
    assert convert_dataset(dataset) == []


def test_existing_output_is_never_overwritten(tmp_path):
    dataset = _dataset(tmp_path, make_series())
    target = dataset / DEFAULT_DIR / f"{DEFAULT_STEM}.json"
    target.parent.mkdir(parents=True)
    target.write_text("mine")
    [outcome] = convert_dataset(dataset)
    assert not outcome.ok
    assert "already exists" in outcome.message
    assert target.read_text() == "mine"
    assert not (dataset / DEFAULT_DIR / f"{DEFAULT_STEM}.nii.gz").exists()
    # After the user removes the clash, the failed series is retried.
    target.unlink()
    assert [o.ok for o in convert_dataset(dataset)] == [True]


def test_changed_source_is_detected(tmp_path):
    series = make_series()
    dataset = _dataset(tmp_path, series)
    sorted(Path(tmp_path / "src" / "s0").iterdir())[2].unlink()  # a middle slice disappears
    [outcome] = convert_dataset(dataset)
    assert not outcome.ok
    assert outcome.message.startswith("source file no longer readable")


# --- labels: clashes and sessions --------------------------------------------------------------


def _patient(seed, name, pid, date="20260101", time="120000", study=None, number=1, desc="t1"):
    spec = SeriesSpec(
        seed=seed,
        series_number=number,
        series_description=desc,
        extra={"PatientName": name, "PatientID": pid, "StudyDate": date, "StudyTime": time},
    )
    s = make_series(spec)
    if study is not None:
        for ds in s.datasets:
            ds.StudyInstanceUID = study
    return s


# --- acq- source: SeriesDescription, else ProtocolName -----------------------------------------


def _named_series(protocol: str | None, description: str) -> SyntheticSeries:
    extra = {} if protocol is None else {"ProtocolName": protocol}
    return make_series(SeriesSpec(series_description=description, extra=extra))


def _converted_stem_and_sidecar(tmp_path, series) -> tuple[str, dict]:
    dataset = _dataset(tmp_path, series)
    assert all(o.ok for o in convert_dataset(dataset))
    [row] = output_rows(dataset)
    sidecar = json.loads((dataset / row["sidecar_path"]).read_text("utf-8"))
    return Path(row["nifti_path"]).name.removesuffix(".nii.gz"), sidecar


def test_acq_uses_series_description(tmp_path):
    stem, sidecar = _converted_stem_and_sidecar(
        tmp_path, _named_series("t1 something", "sag PD fs")
    )
    assert stem == "sub-SynthPhant_ses-20260101_acq-sagPDFs_ser-01_PDw"
    assert sidecar["DcmTether"]["AcqSource"] == "SeriesDescription"
    assert sidecar["DcmTether"]["SuffixRule"] == "keyword:pdw"


def test_suffix_keywords_come_from_the_series_description(tmp_path):
    stem, _ = _converted_stem_and_sidecar(tmp_path, _named_series("t1 something", "t2_tse"))
    assert stem.endswith("_acq-t2Tse_ser-01_T2w")


def test_acq_falls_back_to_protocol_name(tmp_path):
    stem, sidecar = _converted_stem_and_sidecar(tmp_path, _named_series("sag PD fs", ""))
    assert stem == "sub-SynthPhant_ses-20260101_acq-sagPDFs_ser-01_PDw"
    assert sidecar["DcmTether"]["AcqSource"] == "ProtocolName"


def test_no_acq_without_protocol_and_description(tmp_path):
    stem, sidecar = _converted_stem_and_sidecar(tmp_path, _named_series(None, ""))
    assert stem == "sub-SynthPhant_ses-20260101_ser-01_T1w"
    assert "AcqSource" not in sidecar["DcmTether"]


def _subjects(dataset: Path) -> dict[int, str]:
    return {r["id"]: r["subject"] for r in _manifest_rows(dataset)}


def test_subject_label_clash_is_numbered_without_naming_them(tmp_path):
    a = _patient(1, "MUELLER^THOMAS", "P1")
    b = _patient(2, "MUELLERS^THOMASINA", "P2")
    c = _patient(3, "SMITH^ANNA", "P3")
    dataset = _dataset(tmp_path, a, b, c)
    outcomes = {o.series_id: o for o in convert_dataset(dataset)}
    assert all(o.ok for o in outcomes.values())
    assert _subjects(dataset) == {1: "MuellThoma01", 2: "MuellThoma02", 3: "SmithAnna"}
    for sid in (1, 2):
        warning = outcomes[sid].warning
        assert warning.startswith("subject label clash: 2 different PatientIDs")
        assert "series #1, #2" in warning
        assert "Muell" not in warning and "MUELLER" not in warning
    assert outcomes[3].warning is None


def test_subject_numbers_follow_the_first_series_of_each_patient(tmp_path):
    a1 = _patient(1, "PHANTOM^QA", "QA-2")
    b = _patient(2, "PHANTOM^QA", "QA-1")
    a2 = _patient(3, "PHANTOM^QA", "QA-2", date="20260202", number=2)
    dataset = _dataset(tmp_path, a1, b, a2)
    assert all(o.ok for o in convert_dataset(dataset))
    assert _subjects(dataset) == {1: "PhantQa01", 2: "PhantQa02", 3: "PhantQa01"}


def test_converted_patient_keeps_its_subject_label(tmp_path):
    write_series(_patient(1, "PHANTOM^QA", "QA-1"), tmp_path / "src" / "s0")
    dataset = tmp_path / "dataset"
    write_manifest(dataset, scan(tmp_path / "src"))
    assert all(o.ok for o in convert_dataset(dataset))
    # A second phantom (other PatientID) and a new session of the first one arrive later.
    write_series(_patient(2, "PHANTOM^QA", "QA-2"), tmp_path / "src" / "s1")
    write_series(_patient(3, "PHANTOM^QA", "QA-1", date="20260202"), tmp_path / "src" / "s2")
    update_manifest(dataset, scan(tmp_path / "src"))
    outcomes = convert_dataset(dataset)
    assert all(o.ok for o in outcomes)
    assert _subjects(dataset) == {1: "PhantQa", 2: "PhantQa02", 3: "PhantQa"}


def _subject_names(dataset: Path) -> list[tuple]:
    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        return db.execute("SELECT value FROM meta WHERE key = 'subject_names'").fetchall()


def test_full_name_subject_labels_are_kept_for_the_dataset(tmp_path):
    write_series(_patient(1, "MÜLLER^THOMAS", "P1"), tmp_path / "src" / "s0")
    dataset = tmp_path / "dataset"
    write_manifest(dataset, scan(tmp_path / "src"))
    assert all(o.ok for o in convert_dataset(dataset, full_name=True))
    assert (dataset / "sub-MuellerThomas").is_dir()
    assert _subject_names(dataset) == [("full",)]
    # Later runs need no flag: the dataset remembers.
    write_series(_patient(2, "MÜLLER^THOMAS", "P1", date="20260202"), tmp_path / "src" / "s1")
    update_manifest(dataset, scan(tmp_path / "src"))
    assert all(o.ok for o in convert_dataset(dataset))
    assert _subjects(dataset) == {1: "MuellerThomas", 2: "MuellerThomas"}


def test_full_names_are_refused_for_a_dataset_converted_with_short_labels(tmp_path):
    dataset = _dataset(tmp_path, _patient(1, "MUELLER^THOMAS", "P1"))
    assert all(o.ok for o in convert_dataset(dataset))
    assert _subject_names(dataset) == [("brief",)]
    with pytest.raises(ManifestError, match="short subject labels") as exc:
        convert_dataset(dataset, full_name=True)
    assert "uell" not in str(exc.value)
    assert _subjects(dataset) == {1: "MuellThoma"}


def test_full_names_can_be_chosen_while_nothing_is_converted(tmp_path):
    series = _patient(1, "MUELLER^THOMAS", "P1")
    for ds in series.datasets:
        ds.StudyDate = ""  # fails: nothing gets converted
    dataset = _dataset(tmp_path, series)
    assert not any(o.ok for o in convert_dataset(dataset))
    convert_dataset(dataset, full_name=True)
    assert _subject_names(dataset) == [("full",)]


@pytest.mark.parametrize(
    ("first", "second", "subjects"),
    [
        ("PHANTOM1^QA", "PHANTOM2^QA", {1: "Phantom1Qa", 2: "Phantom2Qa"}),  # digits
        ("DOE^JOHN^A", "DOE^JOHN^B", {1: "DoeJohnA", 2: "DoeJohnB"}),  # middle name
        ("MUELLER^THOMAS", "MUELLERS^THOMASINA", {1: "MuellerThomas", 2: "MuellersThomasina"}),
    ],
)
def test_full_names_that_differ_are_no_clash(tmp_path, first, second, subjects):
    dataset = _dataset(tmp_path, _patient(1, first, "P1"), _patient(2, second, "P2"))
    outcomes = convert_dataset(dataset, full_name=True)
    assert all(o.ok and o.warning is None for o in outcomes)
    assert _subjects(dataset) == subjects


def test_clash_numbers_never_give_another_patients_label(tmp_path):
    dataset = _dataset(
        tmp_path,
        _patient(1, "PHANTOM1^", "P1"),
        _patient(2, "PHANTOM1^", "P2"),
        _patient(3, "PHANTOM101^", "P3"),
    )
    assert all(o.ok for o in convert_dataset(dataset, full_name=True))
    assert _subjects(dataset) == {1: "Phantom102", 2: "Phantom103", 3: "Phantom101"}


def test_full_name_clash_is_numbered(tmp_path):
    dataset = _dataset(
        tmp_path, _patient(1, "PHANTOM^QA", "QA-1"), _patient(2, "PHANTOM^QA", "QA-2")
    )
    assert all(o.ok for o in convert_dataset(dataset, full_name=True))
    assert _subjects(dataset) == {1: "PhantomQa01", 2: "PhantomQa02"}


def test_same_patient_two_studies_same_day_get_time_suffix(tmp_path):
    a = _patient(1, "DOE^JANE", "P1", time="093000", study="2.25.11")
    b = _patient(2, "DOE^JANE", "P1", time="141500", study="2.25.12", number=2)
    c = _patient(3, "DOE^JANE", "P1", date="20260202", study="2.25.13", number=3)
    dataset = _dataset(tmp_path, a, b, c)
    assert all(o.ok for o in convert_dataset(dataset))
    sessions = sorted(r["session"] for r in _manifest_rows(dataset))
    assert sessions == ["20260101T0930", "20260101T1415", "20260202"]


def test_same_day_same_time_studies_clash(tmp_path):
    a = _patient(1, "DOE^JANE", "P1", study="2.25.11")
    b = _patient(2, "DOE^JANE", "P1", study="2.25.12", number=2)
    outcomes = convert_dataset(_dataset(tmp_path, a, b))
    assert [o.message for o in outcomes] == [
        "session label clash: several studies with the same StudyDate/StudyTime"
    ] * 2


def test_repeated_description_and_series_number_are_numbered(tmp_path):
    a = _patient(1, "DOE^JANE", "P1", study="2.25.11", desc="t1 a")
    b = _patient(2, "DOE^JANE", "P1", study="2.25.11", desc="t1 a")
    dataset = _dataset(tmp_path, a, b)
    outcomes = convert_dataset(dataset)
    assert all(o.ok for o in outcomes)
    assert [o.warning for o in outcomes] == [
        "duplicate name: series #1, #2 map to the same name, numbered _01.._02"
    ] * 2
    rows = sorted(output_rows(dataset), key=lambda r: r["nifti_path"])
    stems = [Path(r["nifti_path"]).name.removesuffix(".nii.gz") for r in rows]
    assert stems == [
        "sub-DoeJane_ses-20260101_acq-t1A_ser-01_01_T1w",
        "sub-DoeJane_ses-20260101_acq-t1A_ser-01_02_T1w",
    ]
    for n, row in enumerate(rows, start=1):
        sidecar = json.loads((dataset / row["sidecar_path"]).read_text("utf-8"))
        assert sidecar["DcmTether"]["Duplicate"] == n
    # A re-run skips both and keeps the numbering.
    assert convert_dataset(dataset) == []


def test_missing_patient_name_fails_cleanly(tmp_path):
    outcomes = convert_dataset(_dataset(tmp_path, _patient(1, "", "P1")))
    assert [o.message for o in outcomes] == ["PatientName is missing"]


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_to_nii(tmp_path):
    a = _patient(1, "MUELLER^THOMAS", "P1", study="2.25.11")
    b = _patient(2, "MUELLER^THOMAS", "P1", study="2.25.12", number=2)  # same date and time
    c = _patient(3, "MUELLERS^THOMASINA", "P2")
    ok = _patient(4, "SMITH^ANNA", "P3")
    dataset = _dataset(tmp_path, a, b, c, ok)
    result = CliRunner().invoke(main, ["to-nii", str(dataset)])
    assert result.exit_code == 1
    assert "Converted 2 series, 2 failed." in result.output
    assert "session label clash" in result.output
    assert "warning: subject label clash" in result.output
    for secret in ("MUELLER", "Muell", "SMITH", "Smith", "2026", "P1", "2.25."):
        assert secret not in result.output


def test_cli_to_nii_reports_duplicate_names_as_warning(tmp_path):
    a = _patient(1, "DOE^JANE", "P1", study="2.25.11")
    b = _patient(2, "DOE^JANE", "P1", study="2.25.11")
    dataset = _dataset(tmp_path, a, b)
    result = CliRunner().invoke(main, ["to-nii", str(dataset)])
    assert result.exit_code == 0
    assert "Converted 2 series, 0 failed." in result.output
    assert "series #1 warning: duplicate name: series #1, #2" in result.output
    [log_file] = (dataset / "logs").glob("dcmtether_*.jsonl")
    log = log_file.read_text("utf-8")
    assert '"series_warning"' in log


def test_cli_to_nii_without_manifest(tmp_path):
    result = CliRunner().invoke(main, ["to-nii", str(tmp_path)])
    assert result.exit_code == 1
    assert "no manifest" in result.output


# --- dcm2niix oracle ---------------------------------------------------------------------------


def _dcm2niix(source: Path, out: Path) -> Path:
    exe = shutil.which("dcm2niix")
    if exe is None:
        pytest.skip("dcm2niix not installed")
    out.mkdir()
    subprocess.run(
        [exe, "-z", "n", "-b", "n", "-f", "oracle", "-o", str(out), str(source)],
        check=True,
        capture_output=True,
    )
    [nifti] = list(out.glob("oracle*.nii"))
    return nifti


def _canonical_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Unscaled voxels and affine, reoriented to closest RAS (dcm2niix does not reorient)."""
    img = nib.load(path)
    raw = np.asarray(img.dataobj.get_unscaled(), dtype=np.int64)
    data, affine, _ = orientation.to_closest_ras(raw, img.affine)
    return data, affine


def _assert_same_image(ours: Path, theirs: Path) -> None:
    (a, a_affine), (b, b_affine) = _canonical_raw(ours), _canonical_raw(theirs)
    assert a.shape == b.shape
    np.testing.assert_allclose(a_affine, b_affine, atol=1e-3)
    np.testing.assert_array_equal(a, b)


ORACLE_SPECS = {k: v for k, v in SPECS.items() if k != "single-slice"}


@pytest.mark.parametrize("spec", ORACLE_SPECS.values(), ids=ORACLE_SPECS.keys())
def test_matches_dcm2niix_on_synthetic_series(tmp_path, spec):
    dataset = _dataset(tmp_path, make_series(spec))
    convert_dataset(dataset)
    theirs = _dcm2niix(tmp_path / "src", tmp_path / "oracle")
    _assert_same_image(dataset / DEFAULT_DIR / f"{DEFAULT_STEM}.nii.gz", theirs)


@pytest.mark.network
def test_matches_dcm2niix_on_real_ge_adc_series(tmp_path, test_file):
    src = tmp_path / "src" / "adc"
    src.mkdir(parents=True)
    for i in range(20):  # the public copy is anonymized; naming needs a PatientName
        ds = pydicom.dcmread(test_file(f"dcmqi/paramaps/701-ADCb500/{i:06d}.dcm"))
        ds.PatientName = "Public^Adc"
        ds.save_as(src / f"{i:06d}.dcm")
    dataset = tmp_path / "dataset"
    write_manifest(dataset, scan(tmp_path / "src"))
    outcomes = convert_dataset(dataset)
    assert [o.ok for o in outcomes] == [True], [o.message for o in outcomes]
    [row] = output_rows(dataset)
    theirs = _dcm2niix(src, tmp_path / "oracle")
    _assert_same_image(dataset / row["nifti_path"], theirs)


@pytest.mark.parametrize("kind", ["dynamic", "diffusion"])
def test_4d_matches_dcm2niix(tmp_path, kind):
    dataset = _dataset(tmp_path, make_4d(SPECS["oblique"], n_volumes=3, kind=kind))
    convert_dataset(dataset)
    theirs = _dcm2niix(tmp_path / "src", tmp_path / "oracle")
    _assert_same_image(_single_output(dataset), theirs)


def test_multi_echo_matches_dcm2niix_per_echo(tmp_path):
    # dcm2niix writes one file per echo (oracle_e1 ...); we write one 4D file.
    dataset = _dataset(tmp_path, make_4d(SPECS["oblique"], n_volumes=3, kind="echo"))
    convert_dataset(dataset)
    exe = shutil.which("dcm2niix")
    if exe is None:
        pytest.skip("dcm2niix not installed")
    out = tmp_path / "oracle"
    out.mkdir()
    subprocess.run(
        [exe, "-z", "n", "-b", "n", "-f", "oracle", "-o", str(out), str(tmp_path / "src")],
        check=True,
        capture_output=True,
    )
    ours, our_affine = _canonical_raw(_single_output(dataset))
    for v in range(3):
        theirs, their_affine = _canonical_raw(out / f"oracle_e{v + 1}.nii")
        np.testing.assert_allclose(our_affine, their_affine, atol=1e-3)
        np.testing.assert_array_equal(ours[..., v], theirs)


# --- compressed sources ------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("codec", ["RLE", "JPLL", "JLSL", "J2KR"])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_lossless_compressed_gives_the_same_nifti_as_uncompressed(tmp_path, test_file, codec, n):
    """WG04 ships each image uncompressed (REF) and in every codec: the NIfTIs must agree."""
    images = {}
    for variant in (f"REF/MR{n}_UNC", f"{codec}/MR{n}_{codec}"):
        source = tmp_path / variant.replace("/", "_") / "src" / "s"
        source.mkdir(parents=True)
        shutil.copy(test_file(f"WG04/{variant}"), source / "file.dcm")
        dataset = source.parent.parent / "dataset"
        write_manifest(dataset, scan(source.parent))
        assert all(o.ok for o in convert_dataset(dataset))
        [row] = output_rows(dataset)
        img = nib.load(dataset / row["nifti_path"])
        images[variant] = (np.asarray(img.dataobj.get_unscaled()), img.affine)
    (ref, ref_affine), (other, other_affine) = images.values()
    np.testing.assert_array_equal(ref, other)
    np.testing.assert_array_equal(ref_affine, other_affine)


# --- enhanced multi-frame ----------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["axial", "oblique-40", "descending", "single-slice"])
def test_enhanced_3d_voxels_match_ground_truth(tmp_path, spec):
    series = make_enhanced(make_series(SPECS[spec]))
    dataset = _dataset(tmp_path, series)
    outcomes = convert_dataset(dataset)
    assert [o.ok for o in outcomes] == [True], [o.message for o in outcomes]
    _world_space_check(_single_output(dataset), series)


@pytest.mark.parametrize("kind", ["echo", "dynamic", "diffusion"])
@pytest.mark.parametrize("n_files", [1, 3])
def test_enhanced_4d_voxels_match_ground_truth(tmp_path, kind, n_files):
    series = make_enhanced(make_4d(SPECS["oblique"], kind=kind), n_files=n_files)
    dataset = _dataset(tmp_path, series)
    outcomes = convert_dataset(dataset)
    assert [o.ok for o in outcomes] == [True], [o.message for o in outcomes]
    _world_space_check(_single_output(dataset), series)


def test_enhanced_tether_lists_frames(tmp_path):
    dataset = _dataset(tmp_path, make_enhanced(make_4d(n_volumes=2), n_files=2))
    convert_dataset(dataset)
    nifti = _single_output(dataset)
    doc = json.loads(nifti.with_suffix("").with_suffix(".tether.json").read_text("utf-8"))
    assert doc["pixel"]["shape_dicom"] == [6, 8, 5, 2]
    assert [i["frames"] for i in doc["instances"]] == [[[k, v] for k in range(5)] for v in range(2)]
    assert all("index" not in i for i in doc["instances"])


def test_enhanced_files_with_one_plane_each_become_chunks(tmp_path):
    series = make_enhanced(make_two_planes(sagittal_slices=4), n_files=2)
    dataset = _dataset(tmp_path, series)
    outcomes = convert_dataset(dataset)
    assert [(o.ok, o.outputs) for o in outcomes] == [(True, 2)], [o.message for o in outcomes]


def test_enhanced_file_with_several_planes_is_not_converted(tmp_path):
    dataset = _dataset(tmp_path, make_enhanced(make_two_planes(sagittal_rows=6)))
    assert convert_dataset(dataset) == []  # unsupported at scan: nothing to attempt
    assert [r["status"] for r in _manifest_rows(dataset)] == ["unsupported"]


def test_enhanced_sidecar_uses_functional_groups(tmp_path):
    dataset = _dataset(tmp_path, make_enhanced(make_4d(kind="echo")))
    convert_dataset(dataset)
    nifti = _single_output(dataset)
    sidecar = json.loads(nifti.with_suffix("").with_suffix(".json").read_text("utf-8"))
    assert sidecar["DcmTether"]["VolumeKey"] == ["EffectiveEchoTime"]
    # under the sidecar key, in seconds; the DICOM keyword stays in VolumeKey only
    assert sidecar["DcmTether"]["PerVolume"] == {"EchoTime": [0.015, 0.03, 0.045]}
    assert sidecar["SliceThickness"] == 3.0  # from PixelMeasuresSequence


ADC_IMAGE_TYPE = ["DERIVED", "PRIMARY", "DIFFUSION", "ADC"]


@pytest.mark.parametrize(
    ("description", "image_type", "suffix", "rule"),
    [
        ("ep2d_diff", ADC_IMAGE_TYPE, "ADC", "imagetype:adc"),
        ("ep2d_diff_ADC", None, "ADC", "keyword:adc"),
        ("DTI_FA", None, "FA", "keyword:fa"),
        # single-b-value diffusion series: no differing b-values to find
        (
            "abdomen_b800",
            ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE"],
            "dwi",
            "imagetype:diffusion",
        ),
        ("IVIM_b50", None, "dwi", "keyword:ivim"),
        ("ep2d_diff_b1000", None, "dwi", "keyword:diff"),
    ],
)
def test_diffusion_maps_go_to_dwi_with_their_own_suffix(
    tmp_path, description, image_type, suffix, rule
):
    extra = {} if image_type is None else {"ImageType": image_type}
    series = make_series(SeriesSpec(series_description=description, extra=extra))
    dataset = _dataset(tmp_path, series)
    assert all(o.ok for o in convert_dataset(dataset))
    [row] = output_rows(dataset)
    nifti = Path(row["nifti_path"])
    assert nifti.parts[:3] == ("sub-SynthPhant", "ses-20260101", "dwi")
    assert nifti.name.endswith(f"_ser-01_{suffix}.nii.gz")
    sidecar = json.loads((dataset / row["sidecar_path"]).read_text("utf-8"))
    assert sidecar["DcmTether"]["SuffixRule"] == rule


def test_flip_angle_in_the_description_is_not_an_fa_map(tmp_path):
    stem, _ = _converted_stem_and_sidecar(
        tmp_path, make_series(SeriesSpec(series_description="t1_fa15"))
    )
    assert stem.endswith("_T1w")
    [row] = output_rows(tmp_path / "dataset")
    assert Path(row["nifti_path"]).parts[2] == "anat"
