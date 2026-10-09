"""Phase 2: supported DICOM series → NIfTI + ``*.tether.json`` + human sidecar.

Error messages produced here never contain element values (names, IDs, dates, UIDs); series
are referred to by manifest id. Unexpected exceptions are stored in full in the (local,
PHI-grade) manifest only.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from nibabel.openers import ImageOpener
from nibabel.volumeutils import array_to_file

from dcmtether import __version__, diffusion, frames, manifest, naming, orientation, tether
from dcmtether._quiet import quiet_pydicom
from dcmtether.scan import Part, Series, Status, classify, read_instances

DATATYPE_DIR = "anat"
DWI_DIR = "dwi"
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
    protocol_name: str | None
    repetition_time: float | None
    echo_time: float | None
    inversion_time: float | None
    #: The scan found diffusion volumes (from the manifest geometry, not the header).
    diffusion: bool = False


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

    @property
    def datatype(self) -> str:
        return DWI_DIR if self.suffix == DWI_SUFFIX else DATATYPE_DIR

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
            diffusion=diffusion,
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
        text, acq_source = naming.naming_text(item.protocol_name, item.series_description)
        suffix = (
            DWI_SUFFIX
            if item.diffusion
            else naming.contrast_suffix(
                text, item.repetition_time, item.echo_time, item.inversion_time
            )
        )
        acq = naming.acq_label(text)
        stem = naming.file_stem(subjects[sid], sessions[sid], ser, suffix.value, acq)
        planned[sid] = Planned(
            sid, subjects[sid], sessions[sid], stem, suffix, ser=ser, acq=acq, acq_source=acq_source
        )
        stems[stem].append(sid)
    # Same name (e.g. a repeated SeriesDescription + SeriesNumber): number them in manifest order.
    for ids in stems.values():
        if len(ids) > 1:
            ids = sorted(ids)
            warning = (
                f"duplicate name: series {_ids(ids)} map to the same name, "
                f"numbered _01.._{len(ids):02d}"
            )
            for n, sid in enumerate(ids, start=1):
                planned[sid] = replace(planned[sid], dup=n, warning=warning)

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


def _private_diffusion(volumes: list[pydicom.Dataset], per_volume: dict) -> str | None:
    """Fill b-values/gradients read from vendor private tags into ``per_volume``.

    Returns where they came from (``DcmTether.DiffusionSource``), or ``None`` if the standard
    elements were used. Gradients not in patient LPS (GE, Philips) are listed unconverted
    under ``DiffusionGradientVendor`` instead of ``DiffusionGradientOrientation``.
    """
    values = [diffusion.diffusion_values(ds) for ds in volumes]
    sources = {v.source for v in values if v.source is not None}
    if sources <= {diffusion.STANDARD}:
        return None
    gradients = [None if v.gradient is None else list(v.gradient) for v in values]
    per_volume["DiffusionBValue"] = [v.b_value for v in values]
    if all(v.gradient_lps for v in values if v.source is not None):
        per_volume["DiffusionGradientOrientation"] = gradients
    else:
        per_volume.pop("DiffusionGradientOrientation", None)
        per_volume["DiffusionGradientVendor"] = gradients
    return "; ".join(diffusion.SOURCE_TAGS[s] for s in sorted(sources))


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
    top level.
    """
    firsts: dict[int, pydicom.Dataset] = {}
    for ds, index in zip(datasets, indices, strict=True):
        firsts.setdefault(index[1] if len(index) > 1 else 0, ds)
    volumes = [firsts[v] for v in sorted(firsts)]
    out: dict = {}
    per_volume: dict = {}
    diffusion_source = None
    with quiet_pydicom():
        for keyword in volume_key:
            per_volume[keyword] = [_sidecar_value(ds, keyword, None) for ds in volumes]
        if "DiffusionBValue" in volume_key:
            diffusion_source = _private_diffusion(volumes, per_volume)
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
    sub-<subject>/ses-<session>/dwi/<stem>.*             the same for diffusion series
    manifest.sqlite                                      scan and conversion records
    logs/dcmtether.jsonl                                 structured log (ids and counts only)

4D series (multi-echo, diffusion, dynamics) are one 4D NIfTI; values that differ between
volumes are listed in the sidecar under `DcmTether.PerVolume`. Diffusion b-values and
gradients come from the standard DICOM elements, else from vendor private tags (Siemens, GE,
Philips; named in `DcmTether.DiffusionSource`). GE/Philips gradients are not in patient
coordinates and are listed unconverted as `DiffusionGradientVendor`.

## File name entities (BIDS-inspired, not BIDS-compliant)

- `sub-`: first 5 letters of the family name + first 5 of the given name (ASCII, CamelCase).
- `ses-`: StudyDate `YYYYMMDD`; `THHMM` from StudyTime is appended only when one subject has
  several studies on the same day.
- `acq-` (custom use): SeriesDescription (else ProtocolName) reduced to letters/digits,
  camelCase, max 24 chars; the source is recorded in the sidecar under `DcmTether.AcqSource`.
- `ser-` (custom): SeriesNumber, at least two digits; series that would share a name (e.g.
  repeated SeriesDescription + SeriesNumber) get a counter `_01`, `_02`, ... after it, in
  manifest order, recorded in the sidecar under `DcmTether.Duplicate`.
- `chunk-`: one per image plane when a series holds several (e.g. localizers), numbered in
  InstanceNumber order.
- suffix: `dwi` for diffusion series; else `T1w`, `T2w`, `PDw`, `FLAIR` from keywords in the
  same text as `acq-`, or from TR/TE/TI, else `MR`; the rule used is recorded in the sidecar
  under `DcmTether.SuffixRule`.
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


def convert_dataset(dataset: Path) -> list[Outcome]:
    """Convert every supported or split, not yet converted series in the dataset's manifest."""
    db = manifest.open_manifest(dataset)
    _write_readme(dataset)
    try:
        root = manifest.source_root(db)
        rows = db.execute(
            "SELECT id, source_dir, conversion_status, geometry FROM series "
            "WHERE status IN ('supported', 'split') ORDER BY id"
        ).fetchall()
        inputs, outcomes = [], []
        for row in rows:
            first = root / manifest.instance_paths(db, row["id"])[0]
            try:
                inputs.append(read_label_input(row["id"], first, _is_diffusion(row["geometry"])))
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
            manifest.record_converted(db, sid, plan.subject, plan.session, outputs)
            outcomes.append(Outcome(sid, True, outputs=len(outputs), warning=plan.warning))
        return outcomes
    finally:
        db.close()
