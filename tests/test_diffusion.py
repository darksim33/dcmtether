import pytest
from pydicom.dataset import Dataset
from synthetic import _private_diffusion

from dcmtether.diffusion import GE, PHILIPS, SIEMENS, STANDARD, Diffusion, diffusion_values
from dcmtether.frames import frame_views


def _mr(**elements) -> Dataset:
    ds = Dataset()
    ds.Modality = "MR"
    for keyword, value in elements.items():
        setattr(ds, keyword, value)
    return ds


@pytest.mark.parametrize(
    ("vendor", "source", "lps"),
    [("siemens", SIEMENS, True), ("ge", GE, False), ("philips", PHILIPS, False)],
)
def test_private_tags_per_vendor(vendor, source, lps):
    ds = _mr()
    _private_diffusion(ds, vendor, 2)
    values = diffusion_values(ds)
    assert values == Diffusion(1000.0, (0.0, 1.0, 0.0), source)
    assert values.gradient_lps is lps


@pytest.mark.parametrize("vendor", ["siemens", "ge", "philips"])
def test_private_b0(vendor):
    ds = _mr()
    _private_diffusion(ds, vendor, 0)
    assert diffusion_values(ds).b_value == 0.0


def test_standard_wins_over_private():
    ds = _mr(DiffusionBValue=500.0, DiffusionGradientOrientation=[0.0, 0.0, 1.0])
    _private_diffusion(ds, "siemens", 1)
    assert diffusion_values(ds) == Diffusion(500.0, (0.0, 0.0, 1.0), STANDARD)


def test_private_tags_need_the_matching_manufacturer():
    ds = _mr()
    _private_diffusion(ds, "siemens", 1)
    ds.Manufacturer = "Philips Medical Systems"
    assert diffusion_values(ds) == Diffusion()


def test_private_tags_only_for_mr():
    ds = _mr()
    _private_diffusion(ds, "siemens", 1)
    ds.Modality = "CT"
    assert diffusion_values(ds) == Diffusion()


def test_private_creator_is_looked_up_not_assumed():
    ds = _mr(Manufacturer="SIEMENS")
    ds.private_block(0x0019, "SOME OTHER CREATOR", create=True)  # takes block 0x10
    block = ds.private_block(0x0019, "SIEMENS MR HEADER", create=True)  # block 0x11
    block.add_new(0x0C, "IS", 800)
    assert (0x0019, 0x110C) in ds
    assert diffusion_values(ds).b_value == 800.0
    # A Siemens-looking element under another creator is not a b-value.
    other = _mr(Manufacturer="SIEMENS")
    other.private_block(0x0019, "SOME OTHER CREATOR", create=True).add_new(0x0C, "IS", 800)
    assert diffusion_values(other) == Diffusion()


def test_ge_needs_the_gradient_tags():
    ds = _mr()
    _private_diffusion(ds, "ge", 1)
    del ds[ds.private_block(0x0019, "GEMS_ACQU_01").get_tag(0xBB)]
    assert diffusion_values(ds) == Diffusion()


def test_unparsable_private_value_is_ignored():
    ds = _mr(Manufacturer="SIEMENS")
    ds.private_block(0x0019, "SIEMENS MR HEADER", create=True).add_new(0x0C, "LO", "n/a")
    assert diffusion_values(ds) == Diffusion()


_B_MATRIX = (1.0, 0.5, 0.0, 800.0, 0.25, 100.0)


def _siemens_with_matrix(values) -> Dataset:
    ds = _mr()
    _private_diffusion(ds, "siemens", 1)
    ds.private_block(0x0019, "SIEMENS MR HEADER").add_new(0x27, "FD", list(values))
    return ds


def test_siemens_b_matrix():
    values = diffusion_values(_siemens_with_matrix(_B_MATRIX))
    assert values.b_matrix == _B_MATRIX
    assert values.source == SIEMENS


def test_siemens_b_matrix_needs_six_values():
    assert diffusion_values(_siemens_with_matrix(_B_MATRIX[:5])).b_matrix is None


def test_standard_b_matrix():
    ds = _mr(DiffusionBValue=800.0)
    for axes, value in zip(("XX", "XY", "XZ", "YY", "YZ", "ZZ"), _B_MATRIX, strict=True):
        setattr(ds, f"DiffusionBValue{axes}", value)
    assert diffusion_values(ds).b_matrix == _B_MATRIX


def test_standard_b_matrix_needs_all_elements():
    ds = _mr(DiffusionBValue=800.0, DiffusionBValueXX=1.0)
    assert diffusion_values(ds).b_matrix is None


@pytest.mark.parametrize("vendor", ["ge", "philips"])
def test_vendors_without_b_matrix(vendor):
    ds = _mr()
    _private_diffusion(ds, vendor, 1)
    assert diffusion_values(ds).b_matrix is None


def test_no_diffusion():
    assert diffusion_values(_mr(Manufacturer="SIEMENS")) == Diffusion()


def test_b_matrix_of_an_enhanced_frame():
    matrix = Dataset()
    for axes, value in zip(("XX", "XY", "XZ", "YY", "YZ", "ZZ"), _B_MATRIX, strict=True):
        setattr(matrix, f"DiffusionBValue{axes}", value)
    item = Dataset()
    item.DiffusionBValue = 800.0
    item.DiffusionDirectionality = "BMATRIX"
    item.DiffusionBMatrixSequence = [matrix]
    group = Dataset()
    group.MRDiffusionSequence = [item]
    ds = _mr(NumberOfFrames=1)
    ds.PerFrameFunctionalGroupsSequence = [group]
    (view,) = frame_views(ds)
    assert diffusion_values(view) == Diffusion(800.0, None, STANDARD, _B_MATRIX)
