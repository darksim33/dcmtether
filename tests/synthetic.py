"""Synthetic, PHI-free MR series with known geometry.

Every series comes with its ground truth: the voxel volume in DICOM spatial order
(rows, columns, slices sorted along the slice normal) and, for evenly spaced series,
the LPS affine. Voxel values are unique per voxel (for volumes up to 4096 voxels),
so any permutation or flip bug shows up as a value mismatch, not just a geometry one.

Conventions (DICOM PS3.3 C.7.6.2):
- ``row_dir`` = first three ImageOrientationPatient values (direction of increasing column index).
- ``col_dir`` = last three values (direction of increasing row index).
- PixelSpacing = [spacing between rows, spacing between columns].
- Slice normal = ``row_dir x col_dir``; slice offsets are measured along it.
"""

import copy
import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from pydicom import Dataset, FileMetaDataset
from pydicom.sequence import Sequence as PydicomSequence
from pydicom.uid import (
    PYDICOM_IMPLEMENTATION_UID,
    EnhancedMRImageStorage,
    ExplicitVRLittleEndian,
    MRImageStorage,
)
from pydicom.valuerep import DS

UID_NAMESPACE = uuid.UUID("5f0c6f4e-1d2b-4c43-9a35-7a0d6b8f2e11")

AXIAL = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
CORONAL = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
SAGITTAL = ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))

Vec = tuple[float, float, float]


def synthetic_uid(*parts: object) -> str:
    """Deterministic ``2.25.<uuid5-as-integer>`` UID (same scheme as PLAN.md §2)."""
    return f"2.25.{uuid.uuid5(UID_NAMESPACE, '|'.join(map(str, parts))).int}"


def rotated(orientation: tuple[Vec, Vec], axis: Vec, degrees: float) -> tuple[Vec, Vec]:
    """Rotate both orientation vectors about ``axis`` (Rodrigues) for oblique series."""
    k = np.asarray(axis, dtype=float)
    k /= np.linalg.norm(k)
    theta = np.deg2rad(degrees)

    def rot(v: Vec) -> Vec:
        v = np.asarray(v, dtype=float)
        r = v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * k.dot(v) * (1 - np.cos(theta))
        return tuple(float(x) for x in r)

    return rot(orientation[0]), rot(orientation[1])


def flipped(
    orientation: tuple[Vec, Vec], rows: bool = False, cols: bool = False
) -> tuple[Vec, Vec]:
    """Negate the row and/or column direction (i.e. mirror the in-plane axes)."""
    row_dir, col_dir = orientation
    if rows:
        row_dir = tuple(-x for x in row_dir)
    if cols:
        col_dir = tuple(-x for x in col_dir)
    return row_dir, col_dir


@dataclass(frozen=True)
class SeriesSpec:
    rows: int = 6
    columns: int = 8
    n_slices: int = 5
    pixel_spacing: tuple[float, float] = (0.8, 0.9)
    slice_spacing: float = 3.0
    #: Explicit slice offsets along the normal (overrides ``slice_spacing``): use for gaps.
    slice_offsets: Sequence[float] | None = None
    #: One value for all slices, or one per slice (in spatial order).
    slice_thickness: float | Sequence[float] = 3.0
    orientation: tuple[Vec, Vec] = AXIAL
    origin: Vec = (-10.0, -20.0, 30.0)
    #: Acquisition (InstanceNumber) order relative to the slice normal.
    descending: bool = False
    #: Return datasets in random order (tests that sorting never relies on file order).
    shuffle: bool = False
    #: ``None`` = no Modality LUT elements; otherwise one pair or one pair per slice.
    rescale: tuple[float, float] | Sequence[tuple[float, float]] | None = None
    signed: bool = False
    series_number: int = 1
    series_description: str = "SYNTHETIC"
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class SyntheticSeries:
    spec: SeriesSpec
    datasets: list[Dataset]
    #: Stored values, shape (rows, columns, n_slices), slices sorted along the normal.
    volume: np.ndarray
    #: Voxel index (row, column, slice) → LPS mm; ``None`` if slice spacing is uneven.
    affine_lps: np.ndarray | None
    series_instance_uid: str
    #: Multi-plane series: one entry per image plane, in InstanceNumber order.
    parts: list["SyntheticSeries"] = field(default_factory=list)


def _per_slice(value, n: int, name: str, scalar: bool) -> list:
    """Broadcast a single value to ``n`` slices, or check a per-slice sequence."""
    if scalar:
        return [value] * n
    if len(value) != n:
        raise ValueError(f"{name}: expected {n} values, got {len(value)}")
    return list(value)


def _ds(values: Sequence[float]) -> list[DS]:
    return [DS(float(v), auto_format=True) for v in values]


def make_series(spec: SeriesSpec | None = None) -> SyntheticSeries:
    spec = spec or SeriesSpec()
    row_dir = np.asarray(spec.orientation[0], dtype=float)
    col_dir = np.asarray(spec.orientation[1], dtype=float)
    normal = np.cross(row_dir, col_dir)
    origin = np.asarray(spec.origin, dtype=float)

    offsets = (
        list(spec.slice_offsets)
        if spec.slice_offsets is not None
        else [k * spec.slice_spacing for k in range(spec.n_slices)]
    )
    n = len(offsets)
    if sorted(offsets) != offsets:
        raise ValueError(
            "slice_offsets must be ascending (use descending=True to reverse acquisition)"
        )
    thickness = _per_slice(
        spec.slice_thickness, n, "slice_thickness", isinstance(spec.slice_thickness, int | float)
    )
    rescale = (
        _per_slice(spec.rescale, n, "rescale", isinstance(spec.rescale[0], int | float))
        if spec.rescale is not None
        else [None] * n
    )

    dtype = np.int16 if spec.signed else np.uint16
    index = np.arange(spec.rows * spec.columns * n).reshape(n, spec.rows, spec.columns)
    volume = np.moveaxis(index % 4096 - (2048 if spec.signed else 0), 0, -1).astype(dtype)

    study_uid = synthetic_uid("study", spec.seed)
    series_uid = synthetic_uid("series", spec.seed, spec.series_number, spec)
    for_uid = synthetic_uid("for", spec.seed)

    acquisition = list(range(n))[::-1] if spec.descending else list(range(n))
    datasets: list[Dataset] = []
    for instance_number, k in enumerate(acquisition, start=1):
        ipp = origin + offsets[k] * normal
        sop_uid = synthetic_uid("sop", series_uid, k)

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        ds = Dataset()
        ds.file_meta = meta
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "OTHER"]
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.StudyDate = "20260101"
        ds.StudyTime = "120000"
        ds.AccessionNumber = ""
        ds.Modality = "MR"
        ds.Manufacturer = "dcmtether-synthetic"
        ds.ReferringPhysicianName = ""
        ds.SeriesDescription = spec.series_description
        ds.PatientName = "Synthetic^Phantom"
        ds.PatientID = f"SYNTH-{spec.seed:04d}"
        ds.PatientBirthDate = ""
        ds.PatientSex = "O"
        ds.ScanningSequence = "SE"
        ds.SequenceVariant = "NONE"
        ds.ScanOptions = ""
        ds.MRAcquisitionType = "2D"
        ds.PatientPosition = "HFS"
        ds.SliceThickness = DS(float(thickness[k]), auto_format=True)
        ds.RepetitionTime = DS("500")
        ds.EchoTime = DS("15")
        ds.EchoTrainLength = 1
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.StudyID = "1"
        ds.SeriesNumber = spec.series_number
        ds.InstanceNumber = instance_number
        ds.ImagePositionPatient = _ds(ipp)
        ds.ImageOrientationPatient = _ds([*row_dir, *col_dir])
        ds.FrameOfReferenceUID = for_uid
        ds.PositionReferenceIndicator = ""
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = spec.rows
        ds.Columns = spec.columns
        ds.PixelSpacing = _ds(spec.pixel_spacing)
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1 if spec.signed else 0
        if rescale[k] is not None:
            ds.RescaleIntercept = DS(float(rescale[k][1]), auto_format=True)
            ds.RescaleSlope = DS(float(rescale[k][0]), auto_format=True)
            ds.RescaleType = "US"
        for keyword, value in spec.extra.items():
            setattr(ds, keyword, value)
        ds.PixelData = np.ascontiguousarray(volume[:, :, k]).tobytes()
        datasets.append(ds)

    if spec.shuffle:
        random.Random(spec.seed).shuffle(datasets)

    affine = None
    steps = np.diff(offsets)
    if n == 1 or np.allclose(steps, steps[0]):
        step = steps[0] if n > 1 else spec.slice_spacing
        affine = np.eye(4)
        affine[:3, 0] = col_dir * spec.pixel_spacing[0]
        affine[:3, 1] = row_dir * spec.pixel_spacing[1]
        affine[:3, 2] = normal * step
        affine[:3, 3] = origin + offsets[0] * normal

    return SyntheticSeries(spec, datasets, volume, affine, series_uid)


def make_localizer(seed: int = 0) -> SyntheticSeries:
    """Three-plane localizer: one axial, one coronal, one sagittal slice in one series."""
    base = SeriesSpec(n_slices=1, series_description="SYNTHETIC LOCALIZER", seed=seed)
    parts = [make_series(replace(base, orientation=o)) for o in (AXIAL, CORONAL, SAGITTAL)]
    series_uid = synthetic_uid("localizer", seed)
    datasets = []
    for number, part in enumerate(parts, start=1):
        ds = part.datasets[0]
        ds.SeriesInstanceUID = series_uid
        ds.InstanceNumber = number
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = synthetic_uid(
            "sop", series_uid, number
        )
        datasets.append(ds)
    volume = np.concatenate([p.volume for p in parts], axis=-1)
    return SyntheticSeries(base, datasets, volume, None, series_uid, parts)


#: Volume kinds for :func:`make_4d`: which attributes tell the volumes apart.
VOLUME_KINDS = (
    "echo",
    "dynamic",
    "diffusion",
    "diffusion_siemens",
    "diffusion_ge",
    "diffusion_philips",
    "none",
)

_GRADIENTS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.6, 0.8, 0.0)]


def _private_diffusion(ds: Dataset, vendor: str, v: int) -> None:
    """b-value/gradient only in the vendor's private tags (older classic exports)."""
    b_value = 0 if v == 0 else 1000
    gradient = (0.0, 0.0, 0.0) if v == 0 else _GRADIENTS[(v - 1) % 4]
    if vendor == "siemens":
        ds.Manufacturer = "SIEMENS"
        block = ds.private_block(0x0019, "SIEMENS MR HEADER", create=True)
        block.add_new(0x0C, "IS", b_value)
        if v:
            block.add_new(0x0E, "FD", list(gradient))
    elif vendor == "ge":
        ds.Manufacturer = "GE MEDICAL SYSTEMS"
        parm = ds.private_block(0x0043, "GEMS_PARM_01", create=True)
        parm.add_new(0x39, "IS", [1_000_000_000 + b_value, 0, 0, 0])
        acqu = ds.private_block(0x0019, "GEMS_ACQU_01", create=True)
        for offset, value in zip((0xBB, 0xBC, 0xBD), gradient, strict=True):
            acqu.add_new(offset, "DS", value)
    elif vendor == "philips":
        ds.Manufacturer = "Philips Medical Systems"
        ds.private_block(0x2001, "Philips Imaging DD 001", create=True).add_new(
            0x03, "FL", float(b_value)
        )
        block = ds.private_block(0x2005, "Philips MR Imaging DD 001", create=True)
        for offset, value in zip((0xB0, 0xB1, 0xB2), gradient, strict=True):
            block.add_new(offset, "FL", value)
    else:
        raise ValueError(f"unknown vendor: {vendor}")


def _volume_attributes(ds: Dataset, kind: str, v: int) -> None:
    if kind.startswith("diffusion_"):
        _private_diffusion(ds, kind.removeprefix("diffusion_"), v)
    elif kind == "echo":
        ds.EchoNumbers = v + 1
        ds.EchoTime = DS(15 * (v + 1))
    elif kind == "dynamic":
        ds.TemporalPositionIdentifier = v + 1
        ds.TriggerTime = DS(1000 * v)
    elif kind == "diffusion":
        ds.add_new(0x00189087, "FD", 0.0 if v == 0 else 1000.0)  # DiffusionBValue
        if v:  # b0 has no gradient direction
            ds.add_new(
                0x00189089, "FD", list(_GRADIENTS[(v - 1) % 4])
            )  # DiffusionGradientOrientation
    elif kind != "none":
        raise ValueError(f"unknown volume kind: {kind}")


def make_4d(
    spec: SeriesSpec | None = None,
    n_volumes: int = 3,
    kind: str = "echo",
    volume_rescale: Sequence[tuple[float, float]] | None = None,
) -> SyntheticSeries:
    """``n_volumes`` volumes at identical slice positions in one series.

    ``kind`` sets the attributes that distinguish the volumes (see :data:`VOLUME_KINDS`);
    ``"none"`` leaves them indistinguishable. ``volume_rescale`` gives one (slope, intercept)
    per volume. Volumes differ by pixel values; ``volume`` is 4D (rows, columns, slices,
    volumes), volumes in the order of ``kind``'s attribute.
    """
    spec = spec or SeriesSpec()
    series_uid = synthetic_uid("4d", spec.seed, spec, kind)
    parts = [
        make_series(
            replace(
                spec,
                series_number=spec.series_number + v,
                shuffle=False,
                rescale=volume_rescale[v] if volume_rescale else spec.rescale,
            )
        )
        for v in range(n_volumes)
    ]
    datasets = []
    for v, part in enumerate(parts):
        for k, ds in enumerate(part.datasets):
            ds.SeriesInstanceUID = series_uid
            ds.SeriesNumber = spec.series_number
            _volume_attributes(ds, kind, v)
            if kind == "dynamic":
                ds.NumberOfTemporalPositions = n_volumes
            ds.InstanceNumber = v * len(part.datasets) + k + 1
            ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = synthetic_uid(
                "sop", series_uid, v, k
            )
            pixels = np.frombuffer(ds.PixelData, dtype=part.volume.dtype) + v * 100
            ds.PixelData = pixels.astype(part.volume.dtype).tobytes()
            datasets.append(ds)
    if spec.shuffle:
        random.Random(spec.seed).shuffle(datasets)
    volume = np.stack([p.volume + v * 100 for v, p in enumerate(parts)], axis=-1)
    affine = parts[0].affine_lps
    return SyntheticSeries(spec, datasets, volume.astype(parts[0].volume.dtype), affine, series_uid)


def make_two_planes(seed: int = 0) -> SyntheticSeries:
    """An axial stack (4 slices) and a sagittal stack (3 slices) in one series."""
    axial = make_series(SeriesSpec(n_slices=4, seed=seed))
    sagittal = make_series(SeriesSpec(n_slices=3, orientation=SAGITTAL, rows=7, seed=seed))
    series_uid = synthetic_uid("two-planes", seed)
    datasets = []
    for number, ds in enumerate([*axial.datasets, *sagittal.datasets], start=1):
        ds.SeriesInstanceUID = series_uid
        ds.InstanceNumber = number
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = synthetic_uid(
            "sop", series_uid, number
        )
        datasets.append(ds)
    return SyntheticSeries(axial.spec, datasets, axial.volume, None, series_uid, [axial, sagittal])


def compressed(series: SyntheticSeries, transfer_syntax: str) -> SyntheticSeries:
    """Encode every instance's pixel data with ``transfer_syntax`` (in place; returns it).

    Uses pydicom's encoders: RLE natively, JPEG-LS / JPEG 2000 via the pylibjpeg plugins.
    JPEG 2000 needs at least 32 x 32 pixels (openjpeg's default resolution levels).
    """
    for ds in series.datasets:
        ds.compress(transfer_syntax, generate_instance_uid=False)  # keep the SOPInstanceUID
    return series


#: Per-image attributes that Enhanced MR moves into functional groups.
_PER_FRAME = (
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "SliceThickness",
    "RescaleSlope",
    "RescaleIntercept",
    "RescaleType",
    "EchoTime",
    "EchoNumbers",
    "TemporalPositionIdentifier",
    "TriggerTime",
    "InstanceNumber",
    "SOPInstanceUID",
    "PixelData",
    "DiffusionBValue",
    "DiffusionGradientOrientation",
)


def _item(**values) -> Sequence:
    item = Dataset()
    for keyword, value in values.items():
        if value is not None:
            setattr(item, keyword, value)
    return PydicomSequence([item])


def _frame_groups(ds: Dataset) -> Dataset:
    """Per-frame functional groups for one classic image."""
    groups = Dataset()
    groups.PlanePositionSequence = _item(ImagePositionPatient=ds.ImagePositionPatient)
    groups.FrameContentSequence = _item(
        TemporalPositionIndex=ds.get("TemporalPositionIdentifier"),
        FrameAcquisitionNumber=ds.InstanceNumber,
    )
    groups.MREchoSequence = _item(EffectiveEchoTime=float(ds.EchoTime))
    if "DiffusionBValue" in ds:
        gradient = ds.get("DiffusionGradientOrientation")
        diffusion = _item(
            DiffusionBValue=ds.DiffusionBValue,
            DiffusionDirectionality="DIRECTIONAL" if gradient is not None else "NONE",
        )
        if gradient is not None:
            diffusion[0].DiffusionGradientDirectionSequence = _item(
                DiffusionGradientOrientation=list(gradient)
            )
        groups.MRDiffusionSequence = diffusion
    if "RescaleSlope" in ds:
        groups.PixelValueTransformationSequence = _item(
            RescaleSlope=ds.RescaleSlope, RescaleIntercept=ds.RescaleIntercept, RescaleType="US"
        )
    return groups


def make_enhanced(classic: SyntheticSeries, n_files: int = 1) -> SyntheticSeries:
    """The same images as Enhanced MR: ``n_files`` multi-frame instances.

    Frames are in InstanceNumber order of ``classic`` (volume by volume), split evenly over the
    files. Geometry, rescale, echo time, temporal position and diffusion move into functional
    groups (pixel measures and orientation shared, the rest per frame). ``volume`` and
    ``affine_lps`` are those of ``classic``.
    """
    images = sorted(classic.datasets, key=lambda ds: ds.InstanceNumber)
    if len(images) % n_files:
        raise ValueError("frames must split evenly over the files")
    per_file = len(images) // n_files
    datasets = []
    for f in range(n_files):
        frames = images[f * per_file : (f + 1) * per_file]
        first = frames[0]
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.update(first.file_meta)
        for elem in first:
            if elem.keyword not in _PER_FRAME:
                ds.add(copy.deepcopy(elem))
        sop_uid = synthetic_uid("enhanced", classic.series_instance_uid, f)
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = EnhancedMRImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = sop_uid
        ds.InstanceNumber = f + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "NONE"]
        ds.NumberOfFrames = len(frames)
        shared = Dataset()
        shared.PixelMeasuresSequence = _item(
            PixelSpacing=first.PixelSpacing, SliceThickness=first.SliceThickness
        )
        shared.PlaneOrientationSequence = _item(
            ImageOrientationPatient=first.ImageOrientationPatient
        )
        ds.SharedFunctionalGroupsSequence = PydicomSequence([shared])
        ds.PerFrameFunctionalGroupsSequence = PydicomSequence([_frame_groups(fr) for fr in frames])
        ds.PixelData = b"".join(fr.PixelData for fr in frames)
        datasets.append(ds)
    return replace(classic, datasets=datasets)


def write_series(series: SyntheticSeries, directory: Path) -> list[Path]:
    """Write datasets as ``IM_0001``… in list order; return the paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, ds in enumerate(series.datasets, start=1):
        path = directory / f"IM_{i:04d}"
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths
