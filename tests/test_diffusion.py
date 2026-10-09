import pytest
from pydicom.dataset import Dataset
from synthetic import _private_diffusion

from dcmtether.diffusion import GE, PHILIPS, SIEMENS, STANDARD, Diffusion, diffusion_values


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


def test_no_diffusion():
    assert diffusion_values(_mr(Manufacturer="SIEMENS")) == Diffusion()
