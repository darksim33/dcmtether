"""Phase 2: supported DICOM series → NIfTI + ``*.tether.json`` + human sidecar.

Error messages produced here never contain element values (names, IDs, dates, UIDs); series
are referred to by manifest id. Unexpected exceptions are stored in full in the (local,
PHI-grade) manifest only.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from nibabel.openers import ImageOpener
from nibabel.volumeutils import array_to_file

from dcmtether import __version__, manifest, naming, orientation, tether
from dcmtether._quiet import quiet_pydicom
from dcmtether.scan import Series, Status, classify, read_instance

DATATYPE_DIR = "anat"
NIFTI_SUFFIX = ".nii.gz"
_NIFTI1_DATA_OFFSET = 352
SIDECAR_SUFFIX = ".json"


class ConversionError(RuntimeError):
    """Expected failure with a safe (value-free) message."""


# --- labels ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelInput:
    series_id: int
    patient_id: str | None
    patient_name: str | None
    study_instance_uid: str | None
    study_date: str | None
    study_time: str | None
    series_number: int | None
    series_description: str | None
    repetition_time: float | None
    echo_time: float | None
    inversion_time: float | None


@dataclass(frozen=True)
class Planned:
    series_id: int
    subject: str | None = None
    session: str | None = None
    stem: str | None = None
    suffix: naming.Suffix | None = None
    error: str | None = None


def _float(ds: pydicom.Dataset, keyword: str) -> float | None:
    try:
        value = ds.get(keyword)
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def read_label_input(series_id: int, path: Path) -> LabelInput:
    with quiet_pydicom():
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        series_number = ds.get("SeriesNumber")
        return LabelInput(
            series_id=series_id,
            patient_id=str(ds.get("PatientID", "")) or None,
            patient_name=str(ds.get("PatientName", "")) or None,
            study_instance_uid=str(ds.get("StudyInstanceUID", "")) or None,
            study_date=str(ds.get("StudyDate", "")) or None,
            study_time=str(ds.get("StudyTime", "")) or None,
            series_number=int(series_number) if series_number not in (None, "") else None,
            series_description=str(ds.get("SeriesDescription", "")) or None,
            repetition_time=_float(ds, "RepetitionTime"),
            echo_time=_float(ds, "EchoTime"),
            inversion_time=_float(ds, "InversionTime"),
        )


def _ids(series_ids) -> str:
    return ", ".join(f"#{i}" for i in sorted(series_ids))


def plan_labels(inputs: list[LabelInput]) -> dict[int, Planned]:
    """Subject/session labels and file stems for all series, with clash detection."""
    errors: dict[int, str] = {}
    subjects: dict[int, str] = {}
    for item in inputs:
        try:
            subjects[item.series_id] = naming.subject_label(item.patient_name)
        except naming.LabelError as exc:
            errors[item.series_id] = str(exc)

    # Different patients (PatientID) must never share a subject label.
    patients_by_label: dict[str, set] = defaultdict(set)
    series_by_label: dict[str, list[int]] = defaultdict(list)
    for item in inputs:
        if item.series_id in subjects:
            patients_by_label[subjects[item.series_id]].add(item.patient_id)
            series_by_label[subjects[item.series_id]].append(item.series_id)
    for label, patients in patients_by_label.items():
        if len(patients) > 1:
            ids = series_by_label[label]
            for sid in ids:
                errors[sid] = (
                    f"subject label clash: {len(patients)} different PatientIDs give the same "
                    f"label (series {_ids(ids)})"
                )

    # Sessions: the date, plus StudyTime only when one subject has several studies that day.
    studies_per_day: dict[tuple[str, str], set] = defaultdict(set)
    for item in inputs:
        if item.series_id in subjects and item.series_id not in errors:
            studies_per_day[(subjects[item.series_id], item.study_date or "")].add(
                item.study_instance_uid
            )
    sessions: dict[int, str] = {}
    for item in inputs:
        sid = item.series_id
        if sid not in subjects or sid in errors:
            continue
        same_day = len(studies_per_day[(subjects[sid], item.study_date or "")]) > 1
        try:
            sessions[sid] = naming.session_label(item.study_date, item.study_time, same_day)
        except naming.LabelError as exc:
            errors[sid] = str(exc)

    study_by_session: dict[tuple[str, str], set] = defaultdict(set)
    for item in inputs:
        if item.series_id in sessions:
            study_by_session[(subjects[item.series_id], sessions[item.series_id])].add(
                item.study_instance_uid
            )
    for item in inputs:
        sid = item.series_id
        if sid in sessions and len(study_by_session[(subjects[sid], sessions[sid])]) > 1:
            errors[sid] = "session label clash: several studies with the same StudyDate/StudyTime"

    planned: dict[int, Planned] = {}
    stems: dict[str, list[int]] = defaultdict(list)
    for item in inputs:
        sid = item.series_id
        if sid in errors:
            continue
        try:
            ser = naming.series_label(item.series_number)
        except naming.LabelError as exc:
            errors[sid] = str(exc)
            continue
        suffix = naming.contrast_suffix(
            item.series_description, item.repetition_time, item.echo_time, item.inversion_time
        )
        stem = naming.file_stem(
            subjects[sid],
            sessions[sid],
            ser,
            suffix.value,
            naming.acq_label(item.series_description),
        )
        planned[sid] = Planned(sid, subjects[sid], sessions[sid], stem, suffix)
        stems[stem].append(sid)
    for ids in stems.values():
        if len(ids) > 1:
            for sid in ids:
                errors[sid] = f"file name clash: series {_ids(ids)} map to the same name"
                planned.pop(sid, None)

    for sid, error in errors.items():
        planned[sid] = Planned(sid, error=error)
    return planned


# --- volume ------------------------------------------------------------------------------------


@dataclass
class Volume:
    datasets: list[pydicom.Dataset]  # sorted along the slice normal
    data: np.ndarray  # DICOM order (rows, columns, slices), raw stored values
    affine_lps: np.ndarray


def _revalidate(paths: list[Path]) -> None:
    """The files must still form the series that was classified as supported."""
    instances = []
    for path in paths:
        result = read_instance(path)
        if not hasattr(result, "series_instance_uid"):
            raise ConversionError(f"source file no longer readable: {result.reason}")
        instances.append(result)
    series = Series(instances[0].series_instance_uid, instances)
    classify(series)
    if series.status is not Status.SUPPORTED or len({i.series_instance_uid for i in instances}) > 1:
        raise ConversionError("series changed since scan: " + "; ".join(series.reasons))


def assemble(paths: list[Path]) -> Volume:
    _revalidate(paths)
    with quiet_pydicom():
        datasets = [pydicom.dcmread(p) for p in paths]
        normal = orientation.slice_normal(datasets[0].ImageOrientationPatient)
        datasets.sort(
            key=lambda ds: float(np.dot(np.asarray(ds.ImagePositionPatient, float), normal))
        )
        data = np.stack([ds.pixel_array for ds in datasets], axis=-1)

        positions = np.asarray([ds.ImagePositionPatient for ds in datasets], dtype=float) @ normal
        if len(datasets) > 1:
            spacing = float(np.mean(np.diff(positions)))
        else:
            ds = datasets[0]
            spacing = float(ds.get("SpacingBetweenSlices") or ds.get("SliceThickness") or 1.0)
        affine = orientation.affine_lps(
            datasets[0].ImageOrientationPatient,
            datasets[0].ImagePositionPatient,
            datasets[0].PixelSpacing,
            spacing,
        )
    return Volume(datasets, data, affine)


def rescale(datasets: list[pydicom.Dataset]) -> tuple[float, float] | None:
    """Uniform (slope, intercept) of the series, ``None`` if absent (scan rejects non-uniform)."""
    first = datasets[0]
    if "RescaleSlope" not in first and "RescaleIntercept" not in first:
        return None
    return float(first.get("RescaleSlope", 1.0)), float(first.get("RescaleIntercept", 0.0))


def write_nifti(path: Path, data: np.ndarray, affine: np.ndarray, scaling=None) -> None:
    """Write raw values unchanged; uniform DICOM rescale goes into scl_slope/scl_inter only.

    nibabel's normal save recomputes scl_slope/scl_inter from the array, so the header is
    written directly.
    """
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    header = img.header
    header.set_data_dtype(data.dtype)
    header.set_xyzt_units("mm")
    if scaling is None:
        header.set_slope_inter(None, None)
    else:
        header.set_slope_inter(*scaling)
    header.set_data_offset(_NIFTI1_DATA_OFFSET)
    with ImageOpener(path, "wb") as f:
        header.write_to(f)
        if f.tell() != _NIFTI1_DATA_OFFSET:  # header + 4-byte extension flag, no extensions
            raise ConversionError("unexpected NIfTI header size")
        array_to_file(data, f, data.dtype, offset=_NIFTI1_DATA_OFFSET, order="F")


# --- sidecar -----------------------------------------------------------------------------------

_SIDECAR_KEYWORDS = [  # (DICOM keyword, BIDS key, scale)
    ("Modality", "Modality", None),
    ("Manufacturer", "Manufacturer", None),
    ("ManufacturerModelName", "ManufacturersModelName", None),
    ("SoftwareVersions", "SoftwareVersions", None),
    ("MagneticFieldStrength", "MagneticFieldStrength", None),
    ("ReceiveCoilName", "ReceiveCoilName", None),
    ("BodyPartExamined", "BodyPartExamined", None),
    ("SeriesDescription", "SeriesDescription", None),
    ("ProtocolName", "ProtocolName", None),
    ("SeriesNumber", "SeriesNumber", None),
    ("ImageType", "ImageType", None),
    ("ScanningSequence", "ScanningSequence", None),
    ("SequenceVariant", "SequenceVariant", None),
    ("ScanOptions", "ScanOptions", None),
    ("SequenceName", "SequenceName", None),
    ("MRAcquisitionType", "MRAcquisitionType", None),
    ("SliceThickness", "SliceThickness", None),
    ("SpacingBetweenSlices", "SpacingBetweenSlices", None),
    ("EchoTime", "EchoTime", 1e-3),  # BIDS: seconds
    ("RepetitionTime", "RepetitionTime", 1e-3),
    ("InversionTime", "InversionTime", 1e-3),
    ("FlipAngle", "FlipAngle", None),
    ("EchoTrainLength", "EchoTrainLength", None),
    ("PixelBandwidth", "PixelBandwidth", None),
    ("ImagingFrequency", "ImagingFrequency", None),
    ("RescaleSlope", "RescaleSlope", None),
    ("RescaleIntercept", "RescaleIntercept", None),
]


def _plain(value):
    if isinstance(value, pydicom.multival.MultiValue | list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, int | float | str):
        number = value
        if isinstance(value, pydicom.valuerep.DSfloat | pydicom.valuerep.DSdecimal):
            number = float(value)
        if isinstance(value, pydicom.valuerep.IS):
            number = int(value)
        return number
    return str(value)


def sidecar(first: pydicom.Dataset, suffix: naming.Suffix) -> dict:
    """Readable metadata (keyword: value). Not used for recovery; no patient identifiers."""
    out = {}
    with quiet_pydicom():
        for keyword, key, scale in _SIDECAR_KEYWORDS:
            value = first.get(keyword)
            if value is None or value == "":
                continue
            value = _plain(value)
            if scale is not None:
                value = round(float(value) * scale, 9)
            out[key] = value
    out["DcmTether"] = {"Version": __version__, "SuffixRule": suffix.rule}
    return out


# --- orchestration -----------------------------------------------------------------------------


@dataclass
class Outcome:
    series_id: int
    ok: bool
    message: str = ""


def _convert_one(
    paths: list[Path], source_dir: Path, plan: Planned, dataset: Path
) -> tuple[Path, Path, Path]:
    out_dir = dataset / f"sub-{plan.subject}" / f"ses-{plan.session}" / DATATYPE_DIR
    nifti_path = out_dir / f"{plan.stem}{NIFTI_SUFFIX}"
    tether_path = out_dir / f"{plan.stem}{tether.TETHER_SUFFIX}"
    sidecar_path = out_dir / f"{plan.stem}{SIDECAR_SUFFIX}"
    targets = (nifti_path, tether_path, sidecar_path)
    if any(t.exists() for t in targets):
        raise ConversionError("output file already exists (collisions are never overwritten)")

    volume = assemble(paths)
    affine_ras = orientation.LPS_TO_RAS @ volume.affine_lps
    data_ras, affine_ras, ornt = orientation.to_closest_ras(volume.data, affine_ras)
    file_names = [Path(ds.filename).relative_to(source_dir).as_posix() for ds in volume.datasets]
    document = tether.build(
        volume.datasets,
        file_names,
        [[k] for k in range(len(volume.datasets))],
        volume.data.dtype,
        volume.data.shape,
        ornt,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        write_nifti(nifti_path, data_ras, affine_ras, rescale(volume.datasets))
        tether.write(document, tether_path)
        sidecar_path.write_text(
            json.dumps(sidecar(volume.datasets[0], plan.suffix), indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        for t in targets:
            t.unlink(missing_ok=True)
        raise
    return targets


DATASET_README = """\
# dcmtether dataset

**This dataset identifies patients.** Subject labels are built from PatientName and session
labels from StudyDate, so file and folder names alone are personal health data. Do not share
it as is. Research use only.

## Layout

    sub-<subject>/ses-<session>/anat/<stem>.nii.gz       voxels (raw stored values, closest RAS)
    sub-<subject>/ses-<session>/anat/<stem>.json         human-readable sidecar (BIDS-style keys)
    sub-<subject>/ses-<session>/anat/<stem>.tether.json  everything needed to rebuild the DICOM
    manifest.sqlite                                      scan and conversion records

## File name entities (BIDS-inspired, not BIDS-compliant)

- `sub-`: first 5 letters of the family name + first 5 of the given name (ASCII, CamelCase).
- `ses-`: StudyDate `YYYYMMDD`; `THHMM` from StudyTime is appended only when one subject has
  several studies on the same day.
- `acq-` (custom use): SeriesDescription reduced to letters/digits, camelCase, max 24 chars.
- `ser-` (custom): SeriesNumber, at least two digits.
- suffix: `T1w`, `T2w`, `PDw`, `FLAIR` from description keywords or TR/TE/TI, else `MR`; the
  rule used is recorded in the sidecar under `DcmTether.SuffixRule`.
"""


def _write_readme(dataset: Path) -> None:
    readme = dataset / "README.md"
    if not readme.exists():
        readme.write_text(DATASET_README, encoding="utf-8")


def convert_dataset(dataset: Path) -> list[Outcome]:
    """Convert every supported, not yet converted series in the dataset's manifest."""
    db = manifest.open_manifest(dataset)
    _write_readme(dataset)
    try:
        root = manifest.source_root(db)
        rows = db.execute(
            "SELECT id, source_dir, conversion_status FROM series WHERE status = 'supported' "
            "ORDER BY id"
        ).fetchall()
        inputs, outcomes = [], []
        for row in rows:
            first = root / manifest.instance_paths(db, row["id"])[0]
            try:
                inputs.append(read_label_input(row["id"], first))
            except Exception as exc:
                manifest.record_failed(db, row["id"], f"cannot read header: {exc!r}")
                outcomes.append(
                    Outcome(row["id"], False, f"cannot read header ({type(exc).__name__})")
                )
        plans = plan_labels(inputs)

        for row in rows:
            sid = row["id"]
            if sid not in plans or row["conversion_status"] == "converted":
                continue
            plan = plans[sid]
            if plan.error:
                manifest.record_failed(db, sid, plan.error)
                outcomes.append(Outcome(sid, False, plan.error))
                continue
            paths = [root / p for p in manifest.instance_paths(db, sid)]
            try:
                targets = _convert_one(paths, root / row["source_dir"], plan, dataset)
            except (ConversionError, tether.TetherError) as exc:
                manifest.record_failed(db, sid, str(exc))
                outcomes.append(Outcome(sid, False, str(exc)))
                continue
            except Exception as exc:
                manifest.record_failed(db, sid, f"unexpected error: {exc!r}")
                outcomes.append(
                    Outcome(sid, False, f"unexpected error ({type(exc).__name__}), see manifest")
                )
                continue
            rel = [t.relative_to(dataset).as_posix() for t in targets]
            manifest.record_converted(db, sid, plan.subject, plan.session, *rel)
            outcomes.append(Outcome(sid, True))
        return outcomes
    finally:
        db.close()
