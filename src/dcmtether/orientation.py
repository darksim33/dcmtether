"""Geometry: DICOM-order volume ↔ closest-RAS NIfTI (axis permutation/flip only, PLAN.md §2).

"DICOM order" means array axes (row index i, column index j, slice k), with slices sorted by
ascending position along the slice normal ``cross(row_dir, col_dir)``. This is the order of
``tether.json`` ``pixel.shape_dicom`` and ``instances[].index``.
"""

from collections.abc import Sequence

import numpy as np
from nibabel import orientations

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def slice_normal(image_orientation: Sequence[float]) -> np.ndarray:
    iop = np.asarray(image_orientation, dtype=float)
    return np.cross(iop[:3], iop[3:])


def affine_lps(
    image_orientation: Sequence[float],
    first_position: Sequence[float],
    pixel_spacing: Sequence[float],
    slice_spacing: float,
) -> np.ndarray:
    """Voxel (i, j, k) in DICOM order → LPS mm."""
    iop = np.asarray(image_orientation, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    affine = np.eye(4)
    affine[:3, 0] = col_dir * float(pixel_spacing[0])  # i walks down rows → along col_dir
    affine[:3, 1] = row_dir * float(pixel_spacing[1])  # j walks along a row → along row_dir
    affine[:3, 2] = slice_normal(iop) * float(slice_spacing)
    affine[:3, 3] = np.asarray(first_position, dtype=float)
    return affine


def to_closest_ras(
    data: np.ndarray, affine_ras: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute/flip the first three axes so they point closest to R, A, S.

    Returns ``(data_ras, affine_ras_reoriented, ornt)`` where ``ornt`` is nibabel's orientation
    array (input axis → [output axis, flip]); store it as ``ornt_transform``.
    """
    ornt = orientations.io_orientation(affine_ras)
    data_ras = orientations.apply_orientation(data, ornt)
    return data_ras, reoriented_affine(affine_ras, ornt, data.shape[:3]), ornt


def reoriented_affine(affine_ras: np.ndarray, ornt: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Affine of the array after applying ``ornt`` to an array of ``shape`` (DICOM order)."""
    return affine_ras @ orientations.inv_ornt_aff(ornt, tuple(shape[:3]))


def invert_ornt(ornt: np.ndarray) -> np.ndarray:
    """Orientation that undoes ``ornt`` (for RAS → DICOM order)."""
    ornt = np.asarray(ornt)
    inverse = np.empty_like(ornt)
    for axis, (target, flip) in enumerate(ornt):
        inverse[int(target)] = [axis, flip]
    return inverse


def from_closest_ras(data_ras: np.ndarray, ornt: np.ndarray) -> np.ndarray:
    """Undo :func:`to_closest_ras` on the voxel array."""
    return orientations.apply_orientation(data_ras, invert_ornt(ornt))
