"""Phase 1: walk a directory, read DICOM headers only, group by series and classify.

Classification (PLAN.md §2):

- ``supported``: one regular 3D stack (or a single slice), or a 4D series of such stacks whose
  volumes one attribute tells apart (:data:`VOLUME_KEYS`).
- ``split``: several image planes in one series (e.g. localizers), each of them a supported
  stack; converted as one NIfTI per plane (``chunk-<n>``).
- ``unsupported``: cannot be converted without guessing or resampling (incl. lossy
  compression: lossless compressed transfer syntaxes are supported).

Reasons never contain element values that could identify a patient (no UIDs, names, dates,
descriptions or paths) so they are safe to print and to include in failure reports.
"""

import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import (
    JPEG2000,
    UID,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEGLossless,
    JPEGLosslessSV1,
    JPEGLSLossless,
    JPEGLSNearLossless,
    MRImageStorage,
    RLELossless,
)

from dcmtether._quiet import quiet_pydicom

SUPPORTED_SOP_CLASSES = frozenset({MRImageStorage})
UNCOMPRESSED_TRANSFER_SYNTAXES = frozenset(
    {ImplicitVRLittleEndian, ExplicitVRLittleEndian, ExplicitVRBigEndian}
)
#: Decoded for the NIfTI; to-dcm writes Explicit VR Little Endian instead (PLAN.md §2).
LOSSLESS_COMPRESSED_TRANSFER_SYNTAXES = frozenset(
    {RLELossless, JPEGLossless, JPEGLosslessSV1, JPEGLSLossless, JPEG2000Lossless}
)
SUPPORTED_TRANSFER_SYNTAXES = UNCOMPRESSED_TRANSFER_SYNTAXES | LOSSLESS_COMPRESSED_TRANSFER_SYNTAXES
#: Never supported: decoders may differ, so the voxels are not reproducible.
LOSSY_TRANSFER_SYNTAXES = frozenset(
    {JPEGBaseline8Bit, JPEGExtended12Bit, JPEGLSNearLossless, JPEG2000}
)

#: Direction cosines closer than this are the same orientation.
ORIENTATION_TOL = 1e-4
#: Positions (mm) closer than this are the same; also the allowed slice-spacing jitter.
POSITION_TOL_MM = 0.01

#: Attributes that may tell the volumes of a 4D series apart, in order of preference. The first
#: whose values are distinct at every slice position, with the same set of values at every
#: position, defines the volumes; if none does, the series is unsupported (no guessing).
VOLUME_KEYS: tuple[tuple[str, ...], ...] = (
    ("EchoNumbers",),
    ("TemporalPositionIdentifier",),
    ("DiffusionBValue", "DiffusionGradientOrientation"),
    ("AcquisitionNumber",),
    ("EchoTime",),
    ("InversionTime",),
    ("TriggerTime",),
)
_VOLUME_ATTRS = {
    "EchoNumbers": "echo_number",
    "TemporalPositionIdentifier": "temporal_position",
    "DiffusionBValue": "diffusion_b_value",
    "DiffusionGradientOrientation": "diffusion_gradient",
    "AcquisitionNumber": "acquisition_number",
    "EchoTime": "echo_time",
    "InversionTime": "inversion_time",
    "TriggerTime": "trigger_time",
}


class Status(StrEnum):
    SUPPORTED = "supported"
    SPLIT = "split"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Instance:
    """Header fields of one DICOM file that scanning and later phases need."""

    path: Path
    series_instance_uid: str
    study_instance_uid: str | None
    sop_instance_uid: str | None
    sop_class_uid: str | None
    transfer_syntax_uid: str | None
    implementation_class_uid: str | None
    implementation_version_name: str | None
    modality: str | None
    manufacturer: str | None
    manufacturer_model_name: str | None
    software_versions: str | None
    body_part_examined: str | None
    series_number: int | None
    series_description: str | None
    frame_of_reference_uid: str | None
    instance_number: int | None
    image_type: tuple[str, ...]
    number_of_frames: int
    samples_per_pixel: int | None
    photometric_interpretation: str | None
    bits_allocated: int | None
    bits_stored: int | None
    pixel_representation: int | None
    rows: int | None
    columns: int | None
    image_position: tuple[float, ...] | None
    image_orientation: tuple[float, ...] | None
    pixel_spacing: tuple[float, ...] | None
    slice_thickness: float | None
    spacing_between_slices: float | None
    rescale_slope: float | None
    rescale_intercept: float | None
    # Candidates for telling the volumes of a 4D series apart (VOLUME_KEYS).
    echo_number: int | None = None
    echo_time: float | None = None
    inversion_time: float | None = None
    temporal_position: int | None = None
    trigger_time: float | None = None
    acquisition_number: int | None = None
    diffusion_b_value: float | None = None
    diffusion_gradient: tuple[float, ...] | None = None


@dataclass
class Part:
    """The instances of one NIfTI: a 3D stack, or a 4D series of stacks at the same positions."""

    #: Volume by volume; within a volume, slices ascending along the slice normal.
    instances: list[Instance]
    #: Per instance: ``(slice,)`` for 3D, ``(slice, volume)`` for 4D.
    indices: list[tuple[int, ...]]
    #: Keywords that tell the volumes apart (4D only).
    volume_key: tuple[str, ...] = ()

    @property
    def n_volumes(self) -> int:
        return 1 + max((index[1] for index in self.indices if len(index) > 1), default=0)


@dataclass
class Series:
    series_instance_uid: str
    instances: list[Instance]
    status: Status = Status.SUPPORTED
    reasons: list[str] = field(default_factory=list)
    geometry: dict = field(default_factory=dict)
    #: What to convert, one entry per NIfTI (chunk); empty if unsupported.
    parts: list[Part] = field(default_factory=list)

    @property
    def first(self) -> Instance:
        return self.instances[0]


@dataclass
class SkippedFile:
    path: Path
    reason: str


@dataclass
class ScanResult:
    root: Path
    series: list[Series]
    skipped: list[SkippedFile]

    @property
    def n_files(self) -> int:
        return sum(len(s.instances) for s in self.series) + len(self.skipped)


# --- reading ---------------------------------------------------------------------------------


def _str(ds: pydicom.Dataset, keyword: str) -> str | None:
    value = ds.get(keyword)
    if value is None or value == "":
        return None
    if isinstance(value, pydicom.multival.MultiValue):
        return "\\".join(str(v) for v in value)
    return str(value)


def _int(ds: pydicom.Dataset, keyword: str) -> int | None:
    try:
        value = ds.get(keyword)
        return None if value is None or value == "" else int(value)
    except (TypeError, ValueError):
        return None


def _float(ds: pydicom.Dataset, keyword: str) -> float | None:
    try:
        value = ds.get(keyword)
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _floats(ds: pydicom.Dataset, keyword: str, n: int) -> tuple[float, ...] | None:
    """A multi-valued numeric element with exactly ``n`` values, else ``None``."""
    try:
        values = tuple(float(v) for v in ds[keyword].value)
    except (KeyError, TypeError, ValueError):
        return None
    return values if len(values) == n and all(np.isfinite(values)) else None


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def read_instance(path: Path) -> Instance | SkippedFile:
    """Read the header of one file; never reads pixel data.

    pydicom value-validation warnings are suppressed: they quote element values (names, IDs,
    UIDs), which must not reach the console or logs. Invalid values are kept as read.
    """
    with quiet_pydicom():
        return _read_instance(path)


def _classify_non_part10(path: Path) -> str:
    """Tell bare DICOM datasets (no preamble/file meta) apart from non-DICOM files."""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, defer_size="4 KB", force=True)
        if "SOPClassUID" in ds and "SOPInstanceUID" in ds:
            return "DICOM dataset without preamble/file meta (not supported)"
    except Exception:
        pass
    return "not a DICOM file"


def _read_instance(path: Path) -> Instance | SkippedFile:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, defer_size="4 KB")
    except InvalidDicomError:
        return SkippedFile(path, _classify_non_part10(path))
    except Exception as exc:
        return SkippedFile(path, f"unreadable DICOM header ({type(exc).__name__})")

    series_uid = _str(ds, "SeriesInstanceUID")
    if series_uid is None:
        return SkippedFile(path, "no SeriesInstanceUID (not an image instance)")

    meta = getattr(ds, "file_meta", pydicom.Dataset())
    image_type = ds.get("ImageType") or ()
    if isinstance(image_type, str):
        image_type = (image_type,)
    return Instance(
        path=path,
        series_instance_uid=series_uid,
        study_instance_uid=_str(ds, "StudyInstanceUID"),
        sop_instance_uid=_str(ds, "SOPInstanceUID"),
        sop_class_uid=_str(ds, "SOPClassUID"),
        transfer_syntax_uid=_str(meta, "TransferSyntaxUID"),
        implementation_class_uid=_str(meta, "ImplementationClassUID"),
        implementation_version_name=_str(meta, "ImplementationVersionName"),
        modality=_str(ds, "Modality"),
        manufacturer=_str(ds, "Manufacturer"),
        manufacturer_model_name=_str(ds, "ManufacturerModelName"),
        software_versions=_str(ds, "SoftwareVersions"),
        body_part_examined=_str(ds, "BodyPartExamined"),
        series_number=_int(ds, "SeriesNumber"),
        series_description=_str(ds, "SeriesDescription"),
        frame_of_reference_uid=_str(ds, "FrameOfReferenceUID"),
        instance_number=_int(ds, "InstanceNumber"),
        image_type=tuple(str(v).upper() for v in image_type),
        number_of_frames=_int(ds, "NumberOfFrames") or 1,
        samples_per_pixel=_int(ds, "SamplesPerPixel"),
        photometric_interpretation=_str(ds, "PhotometricInterpretation"),
        bits_allocated=_int(ds, "BitsAllocated"),
        bits_stored=_int(ds, "BitsStored"),
        pixel_representation=_int(ds, "PixelRepresentation"),
        rows=_int(ds, "Rows"),
        columns=_int(ds, "Columns"),
        image_position=_floats(ds, "ImagePositionPatient", 3),
        image_orientation=_floats(ds, "ImageOrientationPatient", 6),
        pixel_spacing=_floats(ds, "PixelSpacing", 2),
        slice_thickness=_float(ds, "SliceThickness"),
        spacing_between_slices=_float(ds, "SpacingBetweenSlices"),
        rescale_slope=_float(ds, "RescaleSlope"),
        rescale_intercept=_float(ds, "RescaleIntercept"),
        echo_number=_int(ds, "EchoNumbers"),
        echo_time=_float(ds, "EchoTime"),
        inversion_time=_float(ds, "InversionTime"),
        temporal_position=_int(ds, "TemporalPositionIdentifier"),
        trigger_time=_float(ds, "TriggerTime"),
        acquisition_number=_int(ds, "AcquisitionNumber"),
        diffusion_b_value=_float(ds, "DiffusionBValue"),
        diffusion_gradient=_floats(ds, "DiffusionGradientOrientation", 3),
    )


# --- classification --------------------------------------------------------------------------


class _Findings:
    def __init__(self) -> None:
        self.items: list[tuple[Status, str]] = []

    def add(self, status: Status, reason: str) -> None:
        self.items.append((status, reason))

    @property
    def status(self) -> Status:
        statuses = {s for s, _ in self.items}
        for status in (Status.UNSUPPORTED, Status.SPLIT):
            if status in statuses:
                return status
        return Status.SUPPORTED


def _uid_name(uid: str | None) -> str:
    """Name of a standard UID; a placeholder for private ones (which may be site-specific)."""
    if uid is None:
        return "missing"
    name = UID(uid).name
    return name if name != uid else "private/unknown UID"


def _fmt(values: Iterable[float]) -> str:
    return ", ".join(f"{v:g}" for v in values)


def _check_consistency(instances: list[Instance], findings: _Findings) -> None:
    for attr, label in [
        ("sop_class_uid", "SOP classes"),
        ("transfer_syntax_uid", "transfer syntaxes"),
        ("modality", "modalities"),
        ("frame_of_reference_uid", "frames of reference"),
        ("study_instance_uid", "studies"),
    ]:
        n = len({getattr(i, attr) for i in instances})
        if n > 1:
            findings.add(Status.UNSUPPORTED, f"mixed {label} ({n} distinct)")

    sop_uids = Counter(i.sop_instance_uid for i in instances)
    if None in sop_uids:
        findings.add(Status.UNSUPPORTED, "instances without SOPInstanceUID")
    duplicates = sum(c - 1 for uid, c in sop_uids.items() if uid is not None and c > 1)
    if duplicates:
        findings.add(
            Status.UNSUPPORTED, f"duplicate SOPInstanceUID ({plural(duplicates, 'extra file')})"
        )


def _check_content(instances: list[Instance], findings: _Findings) -> None:
    first = instances[0]
    sop_classes = {i.sop_class_uid for i in instances}
    if not sop_classes <= SUPPORTED_SOP_CLASSES:
        names = sorted(_uid_name(u) for u in sop_classes - SUPPORTED_SOP_CLASSES)
        findings.add(Status.UNSUPPORTED, f"SOP class not supported: {', '.join(names)}")
    syntaxes = {i.transfer_syntax_uid for i in instances}
    lossy = syntaxes & LOSSY_TRANSFER_SYNTAXES
    if lossy:
        names = sorted(_uid_name(u) for u in lossy)
        findings.add(Status.UNSUPPORTED, f"lossy transfer syntax not supported: {', '.join(names)}")
    other = syntaxes - SUPPORTED_TRANSFER_SYNTAXES - LOSSY_TRANSFER_SYNTAXES
    if other:
        names = sorted(_uid_name(u) for u in other)
        findings.add(Status.UNSUPPORTED, f"transfer syntax not supported: {', '.join(names)}")
    if any(i.number_of_frames > 1 for i in instances):
        findings.add(Status.UNSUPPORTED, "multi-frame instances")
    if any("MOSAIC" in i.image_type for i in instances):
        findings.add(Status.UNSUPPORTED, "Siemens mosaic images")
    if first.samples_per_pixel != 1 or first.photometric_interpretation not in {
        "MONOCHROME1",
        "MONOCHROME2",
    }:
        findings.add(Status.UNSUPPORTED, "not a single-channel grayscale image")
    pixel_formats = {
        (i.bits_allocated, i.bits_stored, i.pixel_representation, i.photometric_interpretation)
        for i in instances
    }
    if len(pixel_formats) > 1:
        findings.add(Status.UNSUPPORTED, f"mixed pixel formats ({len(pixel_formats)} distinct)")


def _plane_key_matches(a: Instance, b: Instance) -> bool:
    return (
        a.rows == b.rows
        and a.columns == b.columns
        and np.allclose(a.image_orientation, b.image_orientation, atol=ORIENTATION_TOL)
        and np.allclose(a.pixel_spacing, b.pixel_spacing, atol=ORIENTATION_TOL)
    )


def _group_planes(instances: list[Instance]) -> list[list[Instance]]:
    groups: list[list[Instance]] = []
    for inst in instances:
        for group in groups:
            if _plane_key_matches(group[0], inst):
                group.append(inst)
                break
        else:
            groups.append([inst])
    return groups


def _cluster_positions(distances: np.ndarray) -> list[tuple[float, int]]:
    """Sorted distinct positions along the normal with the number of instances at each."""
    clusters: list[list[float]] = []
    for d in np.sort(distances):
        if clusters and d - clusters[-1][-1] <= POSITION_TOL_MM:
            clusters[-1].append(float(d))
        else:
            clusters.append([float(d)])
    return [(float(np.mean(c)), len(c)) for c in clusters]


def _sortable(value: tuple) -> tuple:
    return tuple((v is None, 0 if v is None else v) for v in value)


def _assign_volumes(
    instances: list[Instance], slice_of: list[int]
) -> tuple[tuple[str, ...], list[int]] | None:
    """(volume key, volume index per instance) from the first key in VOLUME_KEYS that fits."""
    by_position: dict[int, list[int]] = defaultdict(list)
    for n, k in enumerate(slice_of):
        by_position[k].append(n)
    for key in VOLUME_KEYS:
        values = [tuple(getattr(i, _VOLUME_ATTRS[kw]) for kw in key) for i in instances]
        sets = [[values[n] for n in members] for members in by_position.values()]
        if any(len(set(v)) != len(v) for v in sets) or any(set(v) != set(sets[0]) for v in sets):
            continue
        # Volumes in acquisition order (lowest InstanceNumber first), then by key value.
        first_number: dict[tuple, float] = defaultdict(lambda: float("inf"))
        for value, inst in zip(values, instances, strict=True):
            if inst.instance_number is not None:
                first_number[value] = min(first_number[value], inst.instance_number)
        order = sorted(set(values), key=lambda v: (first_number[v], _sortable(v)))
        volume = {value: n for n, value in enumerate(order)}
        return key, [volume[v] for v in values]
    return None


def _check_spacing(centres: np.ndarray, geometry: dict, findings: _Findings) -> None:
    if len(centres) < 2:
        geometry["slice_spacing"] = None
        return
    steps = np.diff(centres)
    median = float(np.median(steps))
    if np.max(np.abs(steps - median)) <= POSITION_TOL_MM:
        geometry["slice_spacing"] = round(float(np.mean(steps)), 6)
        return

    geometry["slice_spacing"] = None
    geometry["slice_spacing_range"] = [round(float(steps.min()), 6), round(float(steps.max()), 6)]
    base = float(steps.min())
    multiples = np.rint(steps / base)
    if np.all(np.abs(steps - multiples * base) <= POSITION_TOL_MM * multiples):
        missing = int(np.sum(multiples - 1))
        geometry["missing_slices"] = missing
        findings.add(Status.UNSUPPORTED, f"missing slices ({missing} missing in the stack)")
    else:
        findings.add(
            Status.UNSUPPORTED,
            f"non-uniform slice spacing ({steps.min():.3g} to {steps.max():.3g} mm)",
        )


def _check_volume_content(
    instances: list[Instance], volume_of: list[int], findings: _Findings
) -> None:
    """Rescale must be uniform within a volume (it may differ between volumes); one thickness."""
    rescale: dict[int, set] = defaultdict(set)
    for inst, v in zip(instances, volume_of, strict=True):
        rescale[v].add((inst.rescale_slope, inst.rescale_intercept))
    worst = max(len(pairs) for pairs in rescale.values())
    if worst > 1:
        findings.add(
            Status.UNSUPPORTED,
            f"rescale slope/intercept differ within a volume ({worst} distinct)",
        )
    thickness = sorted({i.slice_thickness for i in instances if i.slice_thickness is not None})
    if len(thickness) > 1:
        findings.add(Status.UNSUPPORTED, f"mixed slice thickness ({_fmt(thickness)} mm)")


def _analyze_stack(instances: list[Instance], findings: _Findings) -> tuple[dict, Part | None]:
    """Geometry of one image plane group: positions, spacing, gaps, 4D layout."""
    iop = np.asarray(instances[0].image_orientation)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    geometry: dict = {
        "orientation": [round(float(v), 6) for v in iop],
        "rows": instances[0].rows,
        "columns": instances[0].columns,
        "pixel_spacing": list(instances[0].pixel_spacing),
    }
    if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-3) or not np.isclose(
        row_dir.dot(col_dir), 0.0, atol=1e-3
    ):
        findings.add(Status.UNSUPPORTED, "ImageOrientationPatient is not orthonormal")
        return geometry, None

    positions = np.asarray([i.image_position for i in instances])
    distances = positions @ normal
    in_plane = positions - np.outer(distances, normal)
    if np.max(np.linalg.norm(in_plane - in_plane[0], axis=1)) > POSITION_TOL_MM:
        findings.add(
            Status.UNSUPPORTED, "slices are not stacked along the slice normal (sheared stack)"
        )

    clusters = _cluster_positions(distances)
    centres = np.asarray([p for p, _ in clusters])
    slice_of = [int(np.argmin(np.abs(centres - d))) for d in distances]
    counts = {c for _, c in clusters}
    geometry["n_positions"] = len(clusters)
    volume_key: tuple[str, ...] = ()
    volume_of = [0] * len(instances)
    if counts != {1}:
        if len(counts) == 1:
            n_volumes = counts.pop()
            geometry["n_volumes"] = n_volumes
            assigned = _assign_volumes(instances, slice_of)
            if assigned is None:
                findings.add(
                    Status.UNSUPPORTED,
                    f"{n_volumes} volumes at the same {len(clusters)} slice positions, "
                    "but no attribute tells them apart",
                )
            else:
                volume_key, volume_of = assigned
                geometry["volume_key"] = list(volume_key)
        else:
            findings.add(Status.UNSUPPORTED, "uneven number of instances per slice position")

    _check_volume_content(instances, volume_of, findings)
    _check_spacing(centres, geometry, findings)

    order = sorted(range(len(instances)), key=lambda n: (volume_of[n], slice_of[n]))
    part = Part(
        instances=[instances[n] for n in order],
        indices=[(slice_of[n], volume_of[n]) if volume_key else (slice_of[n],) for n in order],
        volume_key=volume_key,
    )
    return geometry, part


def _plane_order(plane: list[Instance]) -> tuple:
    numbers = [i.instance_number for i in plane if i.instance_number is not None]
    return (min(numbers) if numbers else float("inf"), min(i.path for i in plane))


def classify(series: Series) -> None:
    """Set ``status``, ``reasons``, ``geometry`` and ``parts`` on ``series``."""
    findings = _Findings()
    instances = series.instances
    _check_consistency(instances, findings)
    _check_content(instances, findings)

    geometry: dict = {"n_instances": len(instances)}
    missing = [
        label
        for attr, label in [
            ("image_position", "ImagePositionPatient"),
            ("image_orientation", "ImageOrientationPatient"),
            ("pixel_spacing", "PixelSpacing"),
            ("rows", "Rows"),
            ("columns", "Columns"),
        ]
        if any(getattr(i, attr) is None for i in instances)
    ]
    image_storage = {i.sop_class_uid for i in instances} <= SUPPORTED_SOP_CLASSES and all(
        i.number_of_frames == 1 for i in instances
    )
    parts: list[Part] = []
    if not image_storage:
        pass  # geometry lives elsewhere (functional groups) or does not exist; already rejected
    elif missing:
        findings.add(Status.UNSUPPORTED, f"missing or invalid {', '.join(missing)}")
    else:
        planes = sorted(_group_planes(instances), key=_plane_order)
        geometry["n_planes"] = len(planes)
        if len(planes) > 1:
            findings.add(
                Status.SPLIT,
                f"{len(planes)} distinct image planes (orientation, matrix or pixel spacing)",
            )
        part_geometry = []
        for n, plane in enumerate(planes, start=1):
            plane_findings = _Findings()
            plane_geometry, part = _analyze_stack(plane, plane_findings)
            for status, reason in plane_findings.items:
                findings.add(status, reason if len(planes) == 1 else f"plane {n}: {reason}")
            part_geometry.append(plane_geometry)
            if part is not None:
                parts.append(part)
        if len(planes) == 1:
            geometry.update(part_geometry[0])
        else:
            geometry["parts"] = part_geometry
    for attr in ("slice_thickness", "spacing_between_slices"):
        geometry[attr] = sorted({v for i in instances if (v := getattr(i, attr)) is not None})

    series.status = findings.status
    series.reasons = [reason for _, reason in findings.items]
    series.geometry = geometry
    series.parts = parts if series.status is not Status.UNSUPPORTED else []


# --- scanning --------------------------------------------------------------------------------


def _walk(root: Path, exclude: Path | None) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if exclude is not None:
            dirnames[:] = [d for d in dirnames if (here / d).resolve() != exclude]
        dirnames.sort()
        files.extend(here / name for name in sorted(filenames))
    return files


def scan(root: Path, exclude: Path | None = None) -> ScanResult:
    """Scan ``root`` recursively; ``exclude`` (e.g. the output dataset) is skipped."""
    root = root.resolve()
    exclude = exclude.resolve() if exclude is not None else None
    by_series: dict[str, list[Instance]] = defaultdict(list)
    skipped: list[SkippedFile] = []
    for path in _walk(root, exclude):
        result = read_instance(path)
        if isinstance(result, SkippedFile):
            skipped.append(result)
        else:
            by_series[result.series_instance_uid].append(result)

    series = []
    for uid, instances in by_series.items():
        s = Series(uid, sorted(instances, key=lambda i: i.path))
        classify(s)
        series.append(s)
    series.sort(key=lambda s: (s.first.path.parent, s.first.series_number or 0, s.first.path))
    return ScanResult(root, series, skipped)
