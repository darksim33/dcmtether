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

import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from pydicom import Dataset, FileMetaDataset
from pydicom.uid import PYDICOM_IMPLEMENTATION_UID, ExplicitVRLittleEndian, MRImageStorage
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
    return SyntheticSeries(base, datasets, volume, None, series_uid)


def write_series(series: SyntheticSeries, directory: Path) -> list[Path]:
    """Write datasets as ``IM_0001``… in list order; return the paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, ds in enumerate(series.datasets, start=1):
        path = directory / f"IM_{i:04d}"
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths
