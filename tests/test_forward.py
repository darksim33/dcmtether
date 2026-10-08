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
from roundtrip import SPECS
from synthetic import (
    SAGITTAL,
    SeriesSpec,
    SyntheticSeries,
    make_localizer,
    make_series,
    write_series,
)

from dcmtether import orientation
from dcmtether.cli import main
from dcmtether.forward import convert_dataset
from dcmtether.manifest import write_manifest
from dcmtether.scan import scan

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
    """Every NIfTI voxel must hold the ground-truth value at the same LPS position."""
    img = nib.load(nifti_path)
    data = np.asarray(img.dataobj.get_unscaled())
    ijk = np.indices(data.shape).reshape(3, -1)
    ras = img.affine @ np.vstack([ijk, np.ones(ijk.shape[1])])
    lps = orientation.LPS_TO_RAS @ ras
    truth_ijk = np.linalg.solve(series.affine_lps, lps)[:3]
    np.testing.assert_allclose(truth_ijk, np.rint(truth_ijk), atol=1e-3)
    idx = tuple(np.rint(truth_ijk).astype(int))
    np.testing.assert_array_equal(data.reshape(-1), series.volume[idx])


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

    assert doc["format_version"] == 2
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
    dataset = _dataset(tmp_path, make_series(), make_localizer(seed=5), names=["a", "loc"])
    convert_dataset(dataset)
    rows = {r["source_dir"]: r for r in _manifest_rows(dataset)}
    converted, split = rows["a"], rows["loc"]
    assert converted["conversion_status"] == "converted"
    assert converted["subject"] == "SynthPhant"
    assert converted["session"] == "20260101"
    assert converted["nifti_path"] == (DEFAULT_DIR / f"{DEFAULT_STEM}.nii.gz").as_posix()
    assert converted["tether_path"].endswith(".tether.json")
    assert converted["sidecar_path"].endswith("_T1w.json")
    assert split["conversion_status"] is None  # only supported series are converted
    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        assert db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("2",)


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


def test_subject_label_clash_fails_both_without_naming_them(tmp_path):
    a = _patient(1, "MUELLER^THOMAS", "P1")
    b = _patient(2, "MUELLERS^THOMASINA", "P2")
    c = _patient(3, "SMITH^ANNA", "P3")
    dataset = _dataset(tmp_path, a, b, c)
    outcomes = {o.series_id: o for o in convert_dataset(dataset)}
    failed = [o for o in outcomes.values() if not o.ok]
    assert len(failed) == 2
    for o in failed:
        assert o.message.startswith("subject label clash: 2 different PatientIDs")
        assert "Muell" not in o.message and "MUELLER" not in o.message
    assert sum(o.ok for o in outcomes.values()) == 1


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


def test_repeated_series_number_in_one_study_clashes(tmp_path):
    a = _patient(1, "DOE^JANE", "P1", study="2.25.11", desc="t1 a")
    b = _patient(2, "DOE^JANE", "P1", study="2.25.11", desc="t1 a")
    outcomes = convert_dataset(_dataset(tmp_path, a, b))
    assert all(o.message.startswith("file name clash: series #1, #2") for o in outcomes)


def test_missing_patient_name_fails_cleanly(tmp_path):
    outcomes = convert_dataset(_dataset(tmp_path, _patient(1, "", "P1")))
    assert [o.message for o in outcomes] == ["PatientName is missing"]


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_to_nii(tmp_path):
    a = _patient(1, "MUELLER^THOMAS", "P1")
    b = _patient(2, "MUELLERS^THOMASINA", "P2")
    ok = _patient(3, "SMITH^ANNA", "P3")
    dataset = _dataset(tmp_path, a, b, ok)
    result = CliRunner().invoke(main, ["to-nii", str(dataset)])
    assert result.exit_code == 1
    assert "Converted 1 series, 2 failed." in result.output
    assert "subject label clash" in result.output
    for secret in ("MUELLER", "Muell", "SMITH", "Smith", "2026", "P1", "2.25."):
        assert secret not in result.output


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
    [row] = _manifest_rows(dataset)
    theirs = _dcm2niix(src, tmp_path / "oracle")
    _assert_same_image(dataset / row["nifti_path"], theirs)
