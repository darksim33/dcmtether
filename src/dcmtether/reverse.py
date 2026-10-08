"""Phase 3: NIfTI + ``*.tether.json`` → DICOM files identical to the originals.

Error messages never contain element values or file names (PHI rules).
"""

import base64
from pathlib import Path, PurePosixPath, PureWindowsPath

import nibabel as nib
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRBigEndian

from dcmtether import forward, orientation, tether
from dcmtether._quiet import quiet_pydicom
from dcmtether.scan import POSITION_TOL_MM

PARTIAL_SUFFIX = ".partial"


class ReverseError(RuntimeError):
    """Expected failure with a safe (value-free) message."""


def tether_path_for(nifti_path: Path) -> Path:
    name = nifti_path.name
    for suffix in (forward.NIFTI_SUFFIX, ".nii"):
        if name.endswith(suffix):
            return nifti_path.with_name(name.removesuffix(suffix) + tether.TETHER_SUFFIX)
    raise ReverseError("not a NIfTI file name (expected .nii.gz or .nii)")


def _safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or PureWindowsPath(name).drive
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ReverseError("unsafe file name in tether.json")
    return path


def _targets(document: dict, output_dir: Path) -> list[Path]:
    names = [_safe_relative(i["file_name"]) for i in document["instances"]]
    if len(set(names)) != len(names):
        raise ReverseError("duplicate file name in tether.json")
    targets = [output_dir.joinpath(*name.parts) for name in names]
    if any(t.exists() for t in targets):
        raise ReverseError("output file already exists (existing files are never overwritten)")
    return targets


def _dicom_order_volume(nifti_path: Path, document: dict, datasets: list) -> np.ndarray:
    """Raw voxels in DICOM order, after checking the NIfTI belongs to this tether.json."""
    pixel = document["pixel"]
    ornt = np.asarray(pixel["orientation"]["ornt_transform"])
    shape_dicom = tuple(pixel["shape_dicom"])
    img = nib.load(nifti_path)
    data = np.asarray(img.dataobj.get_unscaled())

    if data.dtype.name != pixel["dtype"]:
        raise ReverseError("NIfTI data type does not match the tether.json")
    expected_shape = list(shape_dicom)  # axes after the third (volumes) are not reoriented
    for axis, (target, _) in enumerate(ornt):
        expected_shape[int(target)] = shape_dicom[axis]
    if data.shape != tuple(expected_shape):
        raise ReverseError("NIfTI shape does not match the tether.json")

    with quiet_pydicom():
        stack = forward.first_volume(datasets, _indices(document, len(shape_dicom)))
        affine_ras = orientation.LPS_TO_RAS @ forward.volume_affine(stack)
        expected_scaling = forward.nifti_scaling(datasets) or (1.0, 0.0)
    expected = orientation.reoriented_affine(affine_ras, ornt, shape_dicom)
    if not np.allclose(img.affine, expected, rtol=0, atol=POSITION_TOL_MM):
        raise ReverseError("NIfTI affine does not match the geometry in the tether.json")
    scaling = (float(img.dataobj.slope), float(img.dataobj.inter))
    if not np.allclose(scaling, expected_scaling, rtol=1e-6, atol=1e-6):
        raise ReverseError("NIfTI scaling (scl_slope/scl_inter) does not match the tether.json")

    return orientation.from_closest_ras(data, ornt)


def _indices(document: dict, ndim: int) -> list[tuple[int, ...]]:
    """Instance indices into the DICOM-order volume: (slice,) for 3D, (slice, volume) for 4D."""
    indices = [tuple(i["index"]) for i in document["instances"]]
    if ndim not in (3, 4) or any(len(index) != ndim - 2 for index in indices):
        raise ReverseError("instance indices do not match the volume shape in the tether.json")
    return indices


def _set_pixel_data(ds: pydicom.Dataset, plane: np.ndarray, vr: str, trailing: str | None) -> None:
    big_endian = ds.file_meta.TransferSyntaxUID == ExplicitVRBigEndian
    dtype = plane.dtype.newbyteorder(">" if big_endian else "<")
    data = np.ascontiguousarray(plane, dtype=dtype).tobytes()
    if trailing is not None:
        data += base64.b64decode(trailing)
    if len(data) % 2:
        data += b"\0"
    ds.PixelData = data
    ds["PixelData"].VR = vr


def rebuild(nifti_path: Path) -> tuple[dict, list[pydicom.Dataset]]:
    """The tether document and the complete datasets (with pixel data), in instance order."""
    tether_path = tether_path_for(nifti_path)
    if not tether_path.is_file():
        raise ReverseError("no *.tether.json next to the NIfTI")
    document = tether.load(tether_path)
    datasets = tether.rebuild_datasets(document)
    volume = _dicom_order_volume(nifti_path, document, datasets)
    vr = document["pixel"]["pixel_data_vr"]
    indices = _indices(document, volume.ndim)
    for index, ds, instance in zip(indices, datasets, document["instances"], strict=True):
        plane = volume[(slice(None), slice(None), *index)]
        _set_pixel_data(ds, plane, vr, instance.get("pixel_data_trailing"))
    return document, datasets


def to_dicom(nifti_path: Path, output_dir: Path) -> list[Path]:
    """Write the DICOM files of one NIfTI + tether.json pair under ``output_dir``.

    File names come from the tether.json (relative to the series' source directory). Existing
    files are never overwritten; on failure nothing is left behind.
    """
    document, datasets = rebuild(nifti_path)
    targets = _targets(document, output_dir)
    partials = [t.with_name(t.name + PARTIAL_SUFFIX) for t in targets]
    try:
        with quiet_pydicom():
            for ds, partial in zip(datasets, partials, strict=True):
                partial.parent.mkdir(parents=True, exist_ok=True)
                ds.save_as(partial, enforce_file_format=True)
        for partial, target in zip(partials, targets, strict=True):
            partial.rename(target)
    except BaseException:
        for path in (*partials, *targets):  # targets did not exist before (checked above)
            path.unlink(missing_ok=True)
        raise
    return targets
