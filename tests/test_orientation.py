import itertools

import nibabel as nib
import numpy as np
import pytest

from dcmtether import orientation


def _all_axis_permutations_and_flips():
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product([1, -1], repeat=3):
            affine = np.eye(4)
            affine[:3, :3] = 0
            for axis, (target, flip) in enumerate(zip(perm, flips, strict=True)):
                affine[target, axis] = flip * (axis + 1.5)  # distinct spacing per axis
            yield affine


@pytest.mark.parametrize("affine", list(_all_axis_permutations_and_flips()))
def test_round_trip_over_all_48_orientations(affine):
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    data_ras, affine_ras, ornt = orientation.to_closest_ras(data, affine)
    assert nib.aff2axcodes(affine_ras) == ("R", "A", "S")
    np.testing.assert_array_equal(orientation.from_closest_ras(data_ras, ornt), data)
    # Same voxel, same world position before and after.
    for ijk in itertools.product(*(range(n) for n in data.shape)):
        world = affine @ [*ijk, 1]
        ras_ijk = np.rint(np.linalg.solve(affine_ras, world)[:3]).astype(int)
        assert data_ras[tuple(ras_ijk)] == data[ijk]


def test_invert_ornt_is_an_involution():
    ornt = np.array([[1, 1], [2, -1], [0, -1]])
    np.testing.assert_array_equal(orientation.invert_ornt(orientation.invert_ornt(ornt)), ornt)


def test_affine_lps_axes():
    iop = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]  # sagittal: row_dir +y (posterior), col_dir -z
    affine = orientation.affine_lps(iop, [10.0, -20.0, 30.0], [0.5, 0.7], 2.0)
    np.testing.assert_allclose(affine[:3, 0], [0, 0, -0.5])  # i (down the rows) → col_dir
    np.testing.assert_allclose(affine[:3, 1], [0, 0.7, 0])  # j (along a row) → row_dir
    np.testing.assert_allclose(affine[:3, 2], [-2.0, 0, 0])  # k → cross(row_dir, col_dir)
    np.testing.assert_allclose(affine[:3, 3], [10, -20, 30])
