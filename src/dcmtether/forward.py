"""Supported DICOM series → NIfTI + ``*.tether.json`` + human sidecar.

Error messages produced here never contain element values (names, IDs, dates, UIDs); series
are referred to by manifest id. Unexpected exceptions are stored in full in the (local,
PHI-grade) manifest only.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import chain, count
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from nibabel.openers import ImageOpener
from nibabel.volumeutils import array_to_file

from dcmtether import __version__, diffusion, frames, manifest, naming, orientation, tether
from dcmtether._quiet import quiet_pydicom
from dcmtether.errors import DcmtetherError
from dcmtether.progress import Progress, no_progress
from dcmtether.scanning import Part, Series, Status, classify, read_instances

DATATYPE_DIR = "anat"
DWI_DIR = "dwi"
NIFTI_SUFFIX = ".nii.gz"
_NIFTI1_DATA_OFFSET = 352
SIDECAR_SUFFIX = ".json"


class ConversionError(DcmtetherError, RuntimeError):
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
    protocol_name: str | None
    repetition_time: float | None
    echo_time: float | None
    inversion_time: float | None
    image_type: tuple[str, ...] = ()
    #: The scan found diffusion volumes (from the manifest geometry, not the header).
    diffusion: bool = False


@dataclass(frozen=True)
class Converted:
    """A series converted earlier, with the names it has on disk (from the manifest). They are
    fixed: never rebuilt from the source or from today's naming rules."""

    series_id: int
    names: manifest.Names
    study_instance_uid: str | None


DWI_SUFFIX = naming.Suffix("dwi", "diffusion volumes")


@dataclass(frozen=True)
class Planned:
    series_id: int
    subject: str | None = None
    session: str | None = None
    #: File stem without chunk; also what name clashes are checked on.
    stem: str | None = None
    suffix: naming.Suffix | None = None
    error: str | None = None
    ser: str | None = None
    acq: str | None = None
    #: Keyword the acq- entity and suffix keywords came from (SeriesDescription/ProtocolName).
    acq_source: str | None = None
    #: Counter for series that would otherwise share a name (``ser-01_02``).
    dup: int | None = None
    #: Non-fatal problem reported with the outcome (value-free, like errors).
    warning: str | None = None
    patient_id: str = ""
    #: Subject label before clash numbering.
    subject_base: str | None = None

    @property
    def datatype(self) -> str:
        return DWI_DIR if _is_diffusion_suffix(self.suffix) else DATATYPE_DIR

    @property
    def names(self) -> manifest.Names:
        return manifest.Names(
            self.subject, self.session, self.stem, self.dup, self.patient_id, self.subject_base
        )

    def stem_for(self, chunk: int | None) -> str:
        return naming.file_stem(
            self.subject, self.session, self.ser, self.suffix.value, self.acq, chunk, self.dup
        )


def _float(ds: pydicom.Dataset, keyword: str) -> float | None:
    try:
        value = ds.get(keyword)
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def read_label_input(series_id: int, path: Path, diffusion: bool = False) -> LabelInput:
    with quiet_pydicom():
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        series_number = ds.get("SeriesNumber")
        image_type = ds.get("ImageType") or ()
        if isinstance(image_type, str):
            image_type = (image_type,)
        return LabelInput(
            series_id=series_id,
            patient_id=str(ds.get("PatientID", "")) or None,
            patient_name=str(ds.get("PatientName", "")) or None,
            study_instance_uid=str(ds.get("StudyInstanceUID", "")) or None,
            study_date=str(ds.get("StudyDate", "")) or None,
            study_time=str(ds.get("StudyTime", "")) or None,
            series_number=int(series_number) if series_number not in (None, "") else None,
            series_description=str(ds.get("SeriesDescription", "")) or None,
            protocol_name=str(ds.get("ProtocolName", "")) or None,
            repetition_time=_float(ds, "RepetitionTime"),
            echo_time=_float(ds, "EchoTime"),
            inversion_time=_float(ds, "InversionTime"),
            image_type=tuple(str(v) for v in image_type),
            diffusion=diffusion,
        )


def _ids(series_ids) -> str:
    return ", ".join(f"#{i}" for i in sorted(series_ids))


@dataclass
class _Patient:
    ids: list[int] = field(default_factory=list)  # all series, converted ones included
    new: list[int] = field(default_factory=list)  # series still to convert
    subject: str | None = None  # the label the patient was converted under, if any


def _number_subjects(
    inputs: list[LabelInput], bases: dict[int, str], converted: list[Converted]
) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    """Subject labels for the series in ``bases`` (series id → label built from the name).

    Different PatientIDs with the same label (e.g. phantoms, study codes as names) are numbered
    ``<label>01``, ``<label>02``, ... in manifest order of each patient's first series. Stable
    across runs: manifest ids never change, and a patient already converted keeps the label it
    was converted under (its number is then not reused). A number that would give another
    patient's label is skipped.

    Returns ``(subjects, warnings, errors)`` by series id.
    """
    patients: dict[str, dict[str, _Patient]] = defaultdict(dict)  # label -> PatientID -> ...
    unknown: dict[str, list[int]] = defaultdict(list)  # converted, PatientID not known
    for series in converted:
        names = series.names
        if names.patient_id is None:
            unknown[names.subject_base].append(series.series_id)
            continue
        patient = patients[names.subject_base].setdefault(names.patient_id, _Patient())
        patient.ids.append(series.series_id)
        patient.subject = names.subject
    for item in inputs:
        if item.series_id in bases:
            base = bases[item.series_id]
            patient = patients[base].setdefault(item.patient_id or "", _Patient())
            patient.ids.append(item.series_id)
            patient.new.append(item.series_id)

    taken = {series.names.subject for series in converted}
    subjects: dict[int, str] = {}
    warnings: dict[int, str] = {}
    errors: dict[int, str] = {}
    for base, group in patients.items():
        order = sorted(group.values(), key=lambda patient: min(patient.ids))
        clash = len(order) > 1
        for n, patient in enumerate(order, start=1):
            if patient.subject is None and base in unknown:
                # Whether this is the patient of those series cannot be told: no guessing.
                error = (
                    f"cannot check subject label against series {_ids(unknown[base])} "
                    "(source not readable)"
                )
                errors.update(dict.fromkeys(patient.new, error))
                continue
            if patient.subject is None:
                candidates = chain([] if clash else [base], (f"{base}{k:02d}" for k in count(n)))
                # Not another patient's label, converted ("PhantQa02") or own ("Phantom101").
                patient.subject = next(
                    label
                    for label in candidates
                    if label not in taken and (label == base or label not in patients)
                )
                taken.add(patient.subject)
            subjects.update(dict.fromkeys(patient.new, patient.subject))
        if clash:
            everyone = [sid for patient in order for sid in patient.ids]
            warning = (
                f"subject label clash: {len(order)} different PatientIDs give the same label "
                f"(series {_ids(everyone)}), numbered to keep them apart"
            )
            warnings.update(
                {sid: warning for patient in order for sid in patient.new if sid in subjects}
            )
    return subjects, warnings, errors


def plan_labels(
    inputs: list[LabelInput], converted: list[Converted] = (), full_name: bool = False
) -> dict[int, Planned]:
    """Subject/session labels and file stems for the series still to convert (``inputs``),
    with clash detection against each other and against the ``converted`` series.

    ``full_name``: subject labels from the full patient name instead of 5+5 letters.
    """
    errors: dict[int, str] = {}
    bases: dict[int, str] = {}
    for item in inputs:
        try:
            bases[item.series_id] = naming.subject_label(item.patient_name, full_name)
        except naming.LabelError as exc:
            errors[item.series_id] = str(exc)

    # Different patients (PatientID) must never share a subject label: number them.
    warnings: dict[int, list[str]] = defaultdict(list)
    subjects, subject_warnings, subject_errors = _number_subjects(inputs, bases, converted)
    errors.update(subject_errors)
    for sid, warning in subject_warnings.items():
        warnings[sid].append(warning)

    # Sessions: the date, plus StudyTime only when one subject has several studies that day.
    # A study with converted series keeps the session label those were converted under.
    converted_session = {
        (series.names.subject, series.study_instance_uid): series.names.session
        for series in converted
    }
    studies_per_day: dict[tuple[str, str], set] = defaultdict(set)
    study_by_session: dict[tuple[str, str], set] = defaultdict(set)
    for series in converted:
        names = series.names
        studies_per_day[(names.subject, names.session[:8])].add(series.study_instance_uid)
        study_by_session[(names.subject, names.session)].add(series.study_instance_uid)
    for item in inputs:
        if item.series_id in subjects:
            studies_per_day[(subjects[item.series_id], item.study_date or "")].add(
                item.study_instance_uid
            )
    sessions: dict[int, str] = {}
    for item in inputs:
        sid = item.series_id
        if sid not in subjects:
            continue
        kept = converted_session.get((subjects[sid], item.study_instance_uid))
        if kept is not None:
            sessions[sid] = kept
            continue
        same_day = len(studies_per_day[(subjects[sid], item.study_date or "")]) > 1
        try:
            sessions[sid] = naming.session_label(item.study_date, item.study_time, same_day)
        except naming.LabelError as exc:
            errors[sid] = str(exc)

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
    # Stems are compared without case: Windows and macOS file systems do not tell them apart.
    sharing: dict[str, list[int]] = defaultdict(list)  # stem -> all series, converted included
    counters: dict[str, set[int]] = defaultdict(set)  # stem -> counters in use
    for series in converted:
        key = series.names.stem.casefold()
        sharing[key].append(series.series_id)
        counters[key].add(series.names.dup or 1)  # a bare name counts as the first
    for item in inputs:
        sid = item.series_id
        if sid in errors:
            continue
        try:
            ser = naming.series_label(item.series_number)
        except naming.LabelError as exc:
            errors[sid] = str(exc)
            continue
        text, acq_source = naming.naming_text(item.protocol_name, item.series_description)
        suffix = (
            DWI_SUFFIX
            if item.diffusion
            else naming.diffusion_map_suffix(item.image_type, text)
            or naming.dwi_suffix(item.image_type, text)
            or naming.contrast_suffix(
                text, item.repetition_time, item.echo_time, item.inversion_time
            )
        )
        acq = naming.acq_label(text)
        stem = naming.file_stem(subjects[sid], sessions[sid], ser, suffix.value, acq)
        planned[sid] = Planned(
            sid,
            subjects[sid],
            sessions[sid],
            stem,
            suffix,
            ser=ser,
            acq=acq,
            acq_source=acq_source,
            patient_id=item.patient_id or "",
            subject_base=bases[sid],
        )
        sharing[stem.casefold()].append(sid)
    # Same name (e.g. a repeated SeriesDescription + SeriesNumber): number the new ones in
    # manifest order; converted series keep the counter (or bare name) they have.
    for key, ids in sharing.items():
        new = sorted(sid for sid in ids if sid in planned)
        if len(ids) > 1 and new:
            for sid in new:
                dup = next(n for n in count(1) if n not in counters[key])
                counters[key].add(dup)
                planned[sid] = replace(planned[sid], dup=dup)
            warning = (
                f"duplicate name: series {_ids(ids)} map to the same name, "
                f"numbered _01.._{max(counters[key]):02d}"
            )
            for sid in new:
                warnings[sid].append(warning)

    for sid, messages in warnings.items():
        if sid in planned:
            planned[sid] = replace(planned[sid], warning="; ".join(messages))
    for sid, error in errors.items():
        planned[sid] = Planned(sid, error=error)
    return planned


# --- volume ------------------------------------------------------------------------------------


@dataclass
class Volume:
    datasets: list[pydicom.Dataset]  # one per file, in order of their first image
    strings: list[tether.Strings]  # original strings, per file
    #: Per file: {"index": [...]} (classic) or {"frames": [[...], ...]} (multi-frame).
    layouts: list[dict]
    views: list[pydicom.Dataset]  # one per image (frame), part order
    indices: list[tuple[int, ...]]  # (slice,) or (slice, volume), per image
    data: np.ndarray  # DICOM order (rows, columns, slices[, volumes]), raw stored values
    affine_lps: np.ndarray


def _revalidate(paths: list[Path]) -> Series:
    """The files must still form the series that was classified as convertible."""
    instances = []
    for path in paths:
        result = read_instances(path)
        if not isinstance(result, list):
            raise ConversionError(f"source file no longer readable: {result.reason}")
        instances.extend(result)
    series = Series(instances[0].series_instance_uid, instances)
    classify(series)
    if series.status is Status.UNSUPPORTED or len({i.series_instance_uid for i in instances}) > 1:
        raise ConversionError("series changed since scan: " + "; ".join(series.reasons))
    return series


def volume_affine(datasets: list[pydicom.Dataset]) -> np.ndarray:
    """LPS affine of a stack sorted along the slice normal (DICOM order)."""
    first = datasets[0]
    normal = orientation.slice_normal(first.ImageOrientationPatient)
    positions = np.asarray([ds.ImagePositionPatient for ds in datasets], dtype=float) @ normal
    if len(datasets) > 1:
        spacing = float(np.mean(np.diff(positions)))
    else:
        spacing = float(first.get("SpacingBetweenSlices") or first.get("SliceThickness") or 1.0)
    return orientation.affine_lps(
        first.ImageOrientationPatient, first.ImagePositionPatient, first.PixelSpacing, spacing
    )


def first_volume(datasets: list, indices: list) -> list:
    """The images (datasets or frame views) of volume 0 (all of them for 3D), in slice order."""
    chosen = [
        (index[0], ds)
        for ds, index in zip(datasets, indices, strict=True)
        if index[1:] in ((), (0,))
    ]
    return [ds for _, ds in sorted(chosen, key=lambda item: item[0])]


def assemble(part: Part) -> Volume:
    with quiet_pydicom():
        paths = list(dict.fromkeys(i.path for i in part.instances))
        datasets = [pydicom.dcmread(path) for path in paths]
        # Before anything touches the elements: conversion loses their original strings.
        strings = [tether.capture_strings(ds) for ds in datasets]
        by_path = dict(zip(paths, datasets, strict=True))
        pixels = {path: ds.pixel_array for path, ds in by_path.items()}
        all_views = {path: frames.frame_views(ds) for path, ds in by_path.items()}

        planes, views = [], []
        frame_indices: dict[Path, dict[int, list[int]]] = {path: {} for path in paths}
        for inst, index in zip(part.instances, part.indices, strict=True):
            array = pixels[inst.path]
            frame = inst.frame or 0
            planes.append(array[frame] if array.ndim == 3 else array)
            views.append(all_views[inst.path][frame])
            frame_indices[inst.path][frame] = list(index)
        multi_frame = {i.path: i.frame is not None for i in part.instances}
        layouts = []
        for path in paths:
            by_frame = frame_indices[path]
            if not multi_frame[path]:
                layouts.append({"index": by_frame[0]})
                continue
            if sorted(by_frame) != list(range(len(all_views[path]))):
                raise ConversionError("not every frame of a multi-frame instance is used")
            layouts.append({"frames": [by_frame[n] for n in sorted(by_frame)]})

        n_slices = 1 + max(index[0] for index in part.indices)
        shape = (*planes[0].shape, n_slices) + ((part.n_volumes,) if part.volume_key else ())
        # Native byte order: pixel_array of big-endian files is big-endian (e.g. ">i2").
        data = np.empty(shape, dtype=planes[0].dtype.newbyteorder("="))
        for plane, index in zip(planes, part.indices, strict=True):
            data[(slice(None), slice(None), *index)] = plane
        affine = volume_affine(first_volume(views, part.indices))
    return Volume(datasets, strings, layouts, views, list(part.indices), data, affine)


def nifti_scaling(datasets: list[pydicom.Dataset]) -> tuple[float, float] | None:
    """(slope, intercept) for scl_slope/scl_inter: the series' uniform DICOM rescale.

    ``None`` if there is none, or if it differs between volumes (NIfTI has one global scaling;
    the values then live in tether.json and the sidecar only).
    """
    pairs = {(_float(ds, "RescaleSlope"), _float(ds, "RescaleIntercept")) for ds in datasets}
    if len(pairs) != 1 or pairs == {(None, None)}:
        return None
    slope, intercept = pairs.pop()
    return (1.0 if slope is None else slope, 0.0 if intercept is None else intercept)


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
    ("EffectiveEchoTime", "EchoTime", 1e-3),  # Enhanced MR (MREchoSequence); used if no EchoTime
    ("RepetitionTime", "RepetitionTime", 1e-3),
    ("InversionTime", "InversionTime", 1e-3),
    ("FlipAngle", "FlipAngle", None),
    ("EchoTrainLength", "EchoTrainLength", None),
    ("PixelBandwidth", "PixelBandwidth", None),
    ("ImagingFrequency", "ImagingFrequency", None),
    ("RescaleSlope", "RescaleSlope", None),
    ("RescaleIntercept", "RescaleIntercept", None),
]
_SIDECAR_KEYS = {keyword: (key, scale) for keyword, key, scale in _SIDECAR_KEYWORDS}


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


def _sidecar_value(ds: pydicom.Dataset, keyword: str, scale: float | None):
    value = ds.get(keyword)
    if value is None or value == "":
        return None
    value = _plain(value)
    if scale is not None:
        value = round(float(value) * scale, 9)
    return value


def _is_diffusion_suffix(suffix: naming.Suffix) -> bool:
    return suffix.value == DWI_SUFFIX.value or suffix.value in naming.DIFFUSION_MAP_SUFFIXES


def _diffusion(volumes: list[pydicom.Dataset], per_volume: dict) -> str | None:
    """Fill b-values/gradients (standard elements, else vendor private tags) into the sidecar.

    Always in ``per_volume``, one entry per volume, also for a series with a single volume
    (e.g. one series per b-value), so they are always found in the same place. Returns where
    they came from
    (``DcmTether.DiffusionSource``), or ``None`` if there are none or the standard elements
    were used. Gradients not in patient LPS (GE, Philips) are listed unconverted under
    ``DiffusionGradientVendor`` instead of ``DiffusionGradientOrientation``. A b-matrix (xx, xy,
    xz, yy, yz, zz) is listed under ``DiffusionBMatrix`` if any volume has one.
    """
    values = [diffusion.diffusion_values(ds) for ds in volumes]
    sources = {v.source for v in values if v.source is not None}
    if not sources:
        return None
    lps = all(v.gradient_lps for v in values if v.source is not None)
    gradient_key = "DiffusionGradientOrientation" if lps else "DiffusionGradientVendor"
    # replaces what the volume key filled in from the standard elements (empty if private)
    per_volume.pop("DiffusionGradientOrientation", None)
    per_volume["DiffusionBValue"] = [v.b_value for v in values]
    per_volume[gradient_key] = [None if v.gradient is None else list(v.gradient) for v in values]
    if any(v.b_matrix is not None for v in values):
        per_volume["DiffusionBMatrix"] = [
            None if v.b_matrix is None else list(v.b_matrix) for v in values
        ]
    standard = sources <= {diffusion.STANDARD}
    return None if standard else "; ".join(diffusion.SOURCE_TAGS[s] for s in sorted(sources))


def sidecar(
    datasets: list[pydicom.Dataset],
    indices: list[tuple[int, ...]],
    suffix: naming.Suffix,
    volume_key: tuple[str, ...] = (),
    chunk: int | None = None,
    acq_source: str | None = None,
    dup: int | None = None,
) -> dict:
    """Readable metadata (keyword: value). Not used for recovery; no patient identifiers.

    ``datasets`` are the images (classic datasets or enhanced frame views, see
    :mod:`dcmtether.frames`). Values come from the first image of each volume; values that
    differ between volumes go to ``DcmTether.PerVolume`` (one entry per volume) instead of the
    top level. ``PerVolume`` follows the same key and unit rules as the top level (times in
    seconds), also for the volume-key attributes; ``VolumeKey`` names the DICOM keywords.
    """
    firsts: dict[int, pydicom.Dataset] = {}
    for ds, index in zip(datasets, indices, strict=True):
        firsts.setdefault(index[1] if len(index) > 1 else 0, ds)
    volumes = [firsts[v] for v in sorted(firsts)]
    out: dict = {}
    per_volume: dict = {}
    with quiet_pydicom():
        for keyword in volume_key:
            key, scale = _SIDECAR_KEYS.get(keyword, (keyword, None))
            per_volume.setdefault(key, [_sidecar_value(ds, keyword, scale) for ds in volumes])
        diffusion_source = None
        # Only diffusion series: Philips writes a zero b-value into other images too.
        if "DiffusionBValue" in volume_key or _is_diffusion_suffix(suffix):
            diffusion_source = _diffusion(volumes, per_volume)
        for keyword, key, scale in _SIDECAR_KEYWORDS:
            if key in out or key in per_volume:
                continue  # an earlier keyword already filled it
            values = [_sidecar_value(ds, keyword, scale) for ds in volumes]
            if all(v is None for v in values):
                continue
            if all(v == values[0] for v in values):
                if values[0] is not None:
                    out[key] = values[0]
            else:
                per_volume[key] = values
    info: dict = {"Version": __version__, "SuffixRule": suffix.rule}
    if acq_source is not None:
        info["AcqSource"] = acq_source
    if diffusion_source is not None:
        info["DiffusionSource"] = diffusion_source
    if dup is not None:
        info["Duplicate"] = dup
    if chunk is not None:
        info["Chunk"] = chunk
    if volume_key:
        info["VolumeKey"] = list(volume_key)
    if volume_key or per_volume:  # a 3D series: only diffusion values
        info["PerVolume"] = per_volume
    out["DcmTether"] = info
    return out


# --- orchestration -----------------------------------------------------------------------------


@dataclass
class Outcome:
    series_id: int
    ok: bool
    message: str = ""
    #: Number of NIfTI files written (chunks), if converted.
    outputs: int = 0
    #: Non-fatal problem, e.g. a numbered duplicate name.
    warning: str | None = None


def _convert_part(
    part: Part, source_dir: Path, plan: Planned, dataset: Path, chunk: int | None
) -> tuple[Path, ...]:
    stem = plan.stem_for(chunk)
    out_dir = dataset / f"sub-{plan.subject}" / f"ses-{plan.session}" / plan.datatype
    nifti_path = out_dir / f"{stem}{NIFTI_SUFFIX}"
    tether_path = out_dir / f"{stem}{tether.TETHER_SUFFIX}"
    sidecar_path = out_dir / f"{stem}{SIDECAR_SUFFIX}"
    targets = (nifti_path, tether_path, sidecar_path)
    if any(t.exists() for t in targets):
        raise ConversionError("output file already exists (collisions are never overwritten)")

    volume = assemble(part)
    affine_ras = orientation.LPS_TO_RAS @ volume.affine_lps
    data_ras, affine_ras, ornt = orientation.to_closest_ras(volume.data, affine_ras)
    file_names = [Path(ds.filename).relative_to(source_dir).as_posix() for ds in volume.datasets]
    document = tether.build(
        volume.datasets,
        file_names,
        volume.layouts,
        volume.data.dtype,
        volume.data.shape,
        ornt,
        volume.strings,
    )
    info = sidecar(
        volume.views,
        volume.indices,
        plan.suffix,
        part.volume_key,
        chunk,
        plan.acq_source,
        plan.dup,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        write_nifti(nifti_path, data_ras, affine_ras, nifti_scaling(volume.views))
        tether.write(document, tether_path)
        sidecar_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        for t in targets:
            t.unlink(missing_ok=True)
        raise
    return targets


def _convert_series(
    paths: list[Path], source_dir: Path, plan: Planned, dataset: Path
) -> list[manifest.Output]:
    """All parts of a series (one NIfTI each), or nothing: a failure removes earlier parts."""
    parts = _revalidate(paths).parts
    outputs: list[manifest.Output] = []
    written: list[Path] = []
    try:
        for n, part in enumerate(parts, start=1):
            chunk = n if len(parts) > 1 else None
            targets = _convert_part(part, source_dir, plan, dataset, chunk)
            written += targets
            rel = [t.relative_to(dataset).as_posix() for t in targets]
            outputs.append(manifest.Output(chunk or 1, *rel, n_volumes=part.n_volumes))
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return outputs


DATASET_README = """\
# dcmtether dataset

**This dataset identifies patients.** Subject labels are built from PatientName and session
labels from StudyDate, so file and folder names alone are personal health data. Do not share
it as is. Research use only.

## Layout

    sub-<subject>/ses-<session>/anat/<stem>.nii.gz       voxels (raw stored values, closest RAS)
    sub-<subject>/ses-<session>/anat/<stem>.json         human-readable sidecar (BIDS-style keys)
    sub-<subject>/ses-<session>/anat/<stem>.tether.json  everything needed to rebuild the DICOM
    sub-<subject>/ses-<session>/dwi/<stem>.*             the same for diffusion series and maps
    manifest.sqlite                                      scan and conversion records
    logs/dcmtether_<time>.jsonl                          structured log (ids and counts only)

4D series (multi-echo, diffusion, dynamics) are one 4D NIfTI; values that differ between
volumes are listed in the sidecar under `DcmTether.PerVolume`. Diffusion b-values and
gradients come from the standard DICOM elements, else from vendor private tags (Siemens, GE,
Philips; named in `DcmTether.DiffusionSource`). GE/Philips gradients are not in patient
coordinates and are listed unconverted as `DiffusionGradientVendor`. They are always under
`PerVolume`, also for a series with a single volume (e.g. one series per b-value). A b-matrix
(standard enhanced-MR elements or Siemens' private tag) is listed as `DiffusionBMatrix`.

## File name entities (BIDS-inspired, not BIDS-compliant)

- `sub-`: first 5 letters of the family name + first 5 of the given name (ASCII, CamelCase), or
  the full family + given + middle name (digits kept) if the dataset was started with
  `--full-name`.
  Different PatientIDs with the same label (e.g. phantoms, study codes as names) are numbered
  `01`, `02`, ... in manifest order (`sub-PhantQa01`); a converted patient keeps its label.
- `ses-`: StudyDate `YYYYMMDD`; `THHMM` from StudyTime is appended only when one subject has
  several studies on the same day.
- `acq-` (custom use): SeriesDescription (else ProtocolName) reduced to letters/digits,
  camelCase, max 24 chars; the source is recorded in the sidecar under `DcmTether.AcqSource`.
- `ser-` (custom): SeriesNumber, at least two digits; series that would share a name (e.g.
  repeated SeriesDescription + SeriesNumber) get a counter `_01`, `_02`, ... after it, in
  manifest order, recorded in the sidecar under `DcmTether.Duplicate`.
- `chunk-`: one per image plane when a series holds several (e.g. localizers), numbered in
  InstanceNumber order.
- suffix: `dwi` for diffusion series (b-values that differ between volumes, else ImageType
  `DIFFUSION`, else keywords `dwi`, `dti`, `ivim`, `diff`); `ADC`, `eADC`, `FA`, `TRACE` for
  diffusion maps calculated by the scanner (from ImageType, else keywords; stored in `dwi/`);
  else `T1w`, `T2w`, `PDw`, `FLAIR` from keywords in the same text as `acq-`, or from TR/TE/TI,
  else `MR`; the rule used is recorded in the sidecar under `DcmTether.SuffixRule`.
"""


def _write_readme(dataset: Path) -> None:
    readme = dataset / "README.md"
    if not readme.exists():
        readme.write_text(DATASET_README, encoding="utf-8")


def _is_diffusion(geometry_json: str) -> bool:
    geometry = json.loads(geometry_json)
    keys = [geometry.get("volume_key", [])] + [
        p.get("volume_key", []) for p in geometry.get("parts", [])
    ]
    return any("DiffusionBValue" in key for key in keys)


def _subject_names(db, full_name: bool | None) -> bool:
    """Settle how this dataset builds subject labels (stored in the manifest): one dataset
    never mixes both forms, or a patient would end up in two subject folders."""
    stored = manifest.subject_names(db)
    current = stored or manifest.BRIEF_NAMES
    wanted = manifest.FULL_NAMES if full_name else current
    if wanted != current:
        converted = db.execute(
            "SELECT 1 FROM series WHERE conversion_status = 'converted' LIMIT 1"
        ).fetchone()
        if converted:
            raise manifest.ManifestError(
                "this dataset uses short subject labels; --full-name cannot be added later"
            )
    if wanted != stored:
        manifest.set_subject_names(db, wanted)
    return wanted == manifest.FULL_NAMES


_DUP_IN_NAME = re.compile(r"(_ser-\d+)_(\d{2,})(?=_)")


def _stem_and_dup(nifti_path: str) -> tuple[str, int | None]:
    """File stem (without counter and chunk) and ``_XX`` counter from an output path."""
    name = re.sub(r"_chunk-\d+(?=_)", "", Path(nifti_path).name.removesuffix(NIFTI_SUFFIX))
    match = _DUP_IN_NAME.search(name)
    if match is None:
        return name, None
    return _DUP_IN_NAME.sub(r"\1", name, count=1), int(match[2])


def _converted(db, root: Path, row, full_name: bool) -> Converted:
    """The names of a converted series, from the manifest alone.

    Series converted before the manifest stored all of them are completed once: stem and
    counter from the output path, PatientID from the source header if that is still readable
    (else it stays unknown and is tried again next time).
    """
    names = manifest.Names(
        *(row[key] for key in ("subject", "session", "stem", "dup", "patient_id", "subject_base"))
    )
    if names.stem is not None and names.patient_id is not None:
        return Converted(row["id"], names, row["study_instance_uid"])

    stem, dup = names.stem, names.dup
    if stem is None:
        output = db.execute(
            "SELECT nifti_path FROM outputs WHERE series_id = ? ORDER BY chunk", (row["id"],)
        ).fetchone()
        stem, dup = _stem_and_dup(output["nifti_path"])
    # Without the header: the subject label minus a clash number (short labels have no digits).
    base = names.subject_base or re.sub(r"\d+$", "", names.subject)
    patient_id = names.patient_id
    try:
        header = read_label_input(row["id"], root / manifest.instance_paths(db, row["id"])[0])
        patient_id = header.patient_id or ""
        from_name = naming.subject_label(header.patient_name, full_name)
        if re.fullmatch(re.escape(from_name) + r"\d*", names.subject):
            base = from_name
    except Exception:
        pass  # source gone or unreadable: the series stays converted as it is
    names = replace(names, stem=stem, dup=dup, patient_id=patient_id, subject_base=base)
    with db:
        manifest.set_names(db, row["id"], names)
    return Converted(row["id"], names, row["study_instance_uid"])


def convert_dataset(
    dataset: Path, progress: Progress = no_progress, full_name: bool | None = None
) -> list[Outcome]:
    """Convert every supported or split, not yet converted series in the dataset's manifest.

    Converted series are never touched and their source is not needed again: the names they
    were converted under come from the manifest.

    ``full_name=True`` builds subject labels from the full patient name; ``None`` keeps what
    the dataset uses already (5+5 letters unless it was started with full names).
    """
    db = manifest.open_manifest(dataset)
    try:
        full = _subject_names(db, full_name)
        _write_readme(dataset)
        root = manifest.source_root(db)
        rows = db.execute(
            "SELECT id, source_dir, conversion_status, geometry, study_instance_uid, subject, "
            "session, stem, dup, patient_id, subject_base FROM series "
            "WHERE status IN ('supported', 'split') ORDER BY id"
        ).fetchall()
        inputs, converted, outcomes = [], [], []
        for row in rows:
            if row["conversion_status"] == "converted":
                converted.append(_converted(db, root, row, full))
                continue
            paths = manifest.instance_paths(db, row["id"])
            if not paths:  # every file of it belongs to another series by now
                manifest.record_failed(db, row["id"], "source files no longer found")
                outcomes.append(Outcome(row["id"], False, "source files no longer found"))
                continue
            try:
                inputs.append(
                    read_label_input(row["id"], root / paths[0], _is_diffusion(row["geometry"]))
                )
            except Exception as exc:
                manifest.record_failed(db, row["id"], f"cannot read header: {exc!r}")
                outcomes.append(
                    Outcome(row["id"], False, f"cannot read header ({type(exc).__name__})")
                )
        plans = plan_labels(inputs, converted, full)

        todo = [r for r in rows if r["id"] in plans]
        for row in progress(todo, total=len(todo), desc="Converting", unit="series"):
            sid = row["id"]
            plan = plans[sid]
            if plan.error:
                manifest.record_failed(db, sid, plan.error)
                outcomes.append(Outcome(sid, False, plan.error))
                continue
            paths = [root / p for p in manifest.instance_paths(db, sid)]
            try:
                outputs = _convert_series(paths, root / row["source_dir"], plan, dataset)
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
            manifest.record_converted(db, sid, plan.names, outputs)
            outcomes.append(Outcome(sid, True, outputs=len(outputs), warning=plan.warning))
        return sorted(outcomes, key=lambda o: o.series_id)
    finally:
        db.close()
