import pytest

from dcmtether import naming
from dcmtether.naming import LabelError, contrast_suffix


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("MÜLLER^THOMAS", "MuellThoma"),
        ("Müller^Thomas", "MuellThoma"),
        ("STRAUSS^JÜRGEN", "StrauJuerg"),
        ("Strauß^Jürgen", "StrauJuerg"),
        ("Pérez^José", "PerezJose"),
        ("O'BRIEN^ANN", "ObrieAnn"),
        ("Li^Wei", "LiWei"),
        ("van der Berg^Anna-Lena", "VandeAnnal"),  # letters only, first 5
        ("DOE", "Doe"),
        ("DOE^JOHN^MIDDLE^DR^JR", "DoeJohn"),
    ],
)
def test_subject_label(name, label):
    assert naming.subject_label(name) == label


@pytest.mark.parametrize("name", [None, "", "^", "123^456", "李^伟"])
def test_subject_label_errors_never_echo_the_name(name):
    with pytest.raises(LabelError) as exc:
        naming.subject_label(name)
    if name:
        for part in name.split("^"):
            assert not part or part not in str(exc.value)


def test_session_label():
    assert naming.session_label("20260314") == "20260314"
    assert naming.session_label("20260314", "093012.5", with_time=True) == "20260314T0930"
    for bad in [None, "", "2026-03-14", "202603"]:
        with pytest.raises(LabelError):
            naming.session_label(bad)
    with pytest.raises(LabelError, match="StudyTime"):
        naming.session_label("20260314", None, with_time=True)


@pytest.mark.parametrize(
    ("description", "label"),
    [
        ("sag PD fs", "sagPDFs"),
        ("t2_tse_cor", "t2TseCor"),
        ("T1 MPRAGE 1mm iso", "T1MPRAGE1mmIso"),
        ("Diffusion ÄDC", "DiffusionAeDC"),
        ("   ", None),
        (None, None),
        ("a" * 40, "a" * 24),
    ],
)
def test_acq_label(description, label):
    assert naming.acq_label(description) == label


@pytest.mark.parametrize(
    ("protocol", "description", "expected"),
    [
        ("sag PD fs", "sag PD fs_ND", ("sag PD fs_ND", "SeriesDescription")),
        (None, "t1 axial", ("t1 axial", "SeriesDescription")),
        ("sag PD fs", None, ("sag PD fs", "ProtocolName")),
        ("sag PD fs", "", ("sag PD fs", "ProtocolName")),
        ("sag PD fs", "   ", ("sag PD fs", "ProtocolName")),
        ("sag PD fs", "***", ("sag PD fs", "ProtocolName")),  # nothing usable
        ("  ", None, (None, None)),
        (None, None, (None, None)),
    ],
)
def test_naming_text_prefers_series_description(protocol, description, expected):
    assert naming.naming_text(protocol, description) == expected


@pytest.mark.parametrize(
    ("description", "suffix", "rule"),
    [
        ("ax_t1_tse", "T1w", "keyword:t1w"),
        ("T1 MPRAGE", "T1w", "keyword:t1w"),
        ("t2_tse_tra", "T2w", "keyword:t2w"),
        ("cor PD fs", "PDw", "keyword:pdw"),
        ("sag_pdfs", "PDw", "keyword:pdw"),
        ("T2 FLAIR", "FLAIR", "keyword:flair"),
        ("tirm_dark-fluid", "FLAIR", "keyword:flair"),
        ("T2* GRE", "MR", "fallback"),
        ("t2star_me", "MR", "fallback"),
        ("T1 T2 map", "MR", "fallback"),
        ("localizer", "MR", "fallback"),
        ("T10 something", "MR", "fallback"),
    ],
)
def test_contrast_suffix_keywords(description, suffix, rule):
    assert contrast_suffix(description, None, None, None) == naming.Suffix(suffix, rule)


@pytest.mark.parametrize(
    ("tr", "te", "ti", "suffix"),
    [
        (500, 15, None, "T1w"),
        (4000, 100, None, "T2w"),
        (3000, 30, None, "PDw"),
        (9000, 120, 2500, "FLAIR"),
        (1200, 50, None, "MR"),
        (None, 15, None, "MR"),
    ],
)
def test_contrast_suffix_parameters(tr, te, ti, suffix):
    result = contrast_suffix("series", tr, te, ti)
    assert result.value == suffix
    assert result.rule == ("fallback" if suffix == "MR" else result.rule)


def test_keyword_beats_parameters():
    assert contrast_suffix("t2_tse", 500, 15, None).value == "T2w"


def test_file_stem():
    assert (
        naming.file_stem("MuellThoma", "20260314", "05", "PDw", "sagPDFs")
        == "sub-MuellThoma_ses-20260314_acq-sagPDFs_ser-05_PDw"
    )
    assert naming.file_stem("Doe", "20260314", "12", "MR") == "sub-Doe_ses-20260314_ser-12_MR"
    assert (
        naming.file_stem("Doe", "20260314", "01", "T1w", dup=2)
        == "sub-Doe_ses-20260314_ser-01_02_T1w"
    )
    assert (
        naming.file_stem("Doe", "20260314", "01", "MR", chunk=1, dup=2)
        == "sub-Doe_ses-20260314_ser-01_02_chunk-1_MR"
    )
    assert naming.series_label(5) == "05"
    assert naming.series_label(123) == "123"
    with pytest.raises(LabelError):
        naming.series_label(None)


@pytest.mark.parametrize(
    ("image_type", "text", "suffix", "rule"),
    [
        (("DERIVED", "PRIMARY", "DIFFUSION", "ADC", "ND"), "ep2d_diff", "ADC", "imagetype:adc"),
        (("DERIVED", "PRIMARY", "DIFFUSION", "FA"), "ep2d_diff", "FA", "imagetype:fa"),
        (("DERIVED", "PRIMARY", "DIFFUSION", "TRACEW"), "ep2d_diff", "TRACE", "imagetype:tracew"),
        (("derived", "primary", "eadc"), None, "eADC", "imagetype:eadc"),
        (("DERIVED", "PRIMARY", "ADC"), "DTI_FA", "ADC", "imagetype:adc"),  # ImageType wins
        (("ORIGINAL", "PRIMARY", "M", "ND"), "ep2d_diff_ADC", "ADC", "keyword:adc"),
        ((), "Apparent Diffusion Coefficient", "ADC", "keyword:adc"),
        ((), "ep2d_diff_TRACEW", "TRACE", "keyword:trace"),
        ((), "DTI_FA", "FA", "keyword:fa"),
        ((), "Fractional Anisotropy", "FA", "keyword:fa"),
    ],
)
def test_diffusion_map_suffix(image_type, text, suffix, rule):
    assert naming.diffusion_map_suffix(image_type, text) == naming.Suffix(suffix, rule)
    assert suffix in naming.DIFFUSION_MAP_SUFFIXES


@pytest.mark.parametrize(
    ("image_type", "text"),
    [
        (("ORIGINAL", "PRIMARY", "M", "ND"), "T2_tse"),
        ((), "t1_fa15"),  # flip angle, no diffusion word
        ((), "vibe_fa_2"),
        ((), "fast_spin_echo"),
        ((), "adcx"),
        ((), None),
        ((), "DTI_ADC_FA"),  # two maps named: ambiguous
        (("DERIVED", "PRIMARY", "ADC", "FA"), "ep2d_diff"),
    ],
)
def test_diffusion_map_suffix_none(image_type, text):
    assert naming.diffusion_map_suffix(image_type, text) is None


@pytest.mark.parametrize(
    ("image_type", "text", "rule"),
    [
        (("ORIGINAL", "PRIMARY", "DIFFUSION", "NONE"), "T2_tse", "imagetype:diffusion"),
        (("original", "primary", "diffusion"), None, "imagetype:diffusion"),
        (("ORIGINAL", "PRIMARY", "M", "ND"), "IVIM_abdomen", "keyword:ivim"),
        ((), "abd_ivim_10b", "keyword:ivim"),
        ((), "ep2d_diff_b800", "keyword:diff"),
        ((), "Diffusion weighted", "keyword:diff"),
        ((), "DWI_bh", "keyword:dwi"),
        ((), "dti_30dir", "keyword:dti"),
    ],
)
def test_dwi_suffix(image_type, text, rule):
    assert naming.dwi_suffix(image_type, text) == naming.Suffix("dwi", rule)


@pytest.mark.parametrize(
    ("image_type", "text"),
    [
        (("ORIGINAL", "PRIMARY", "M", "ND"), "T2_tse"),
        ((), "subtraction difference"),
        ((), "privim_t1"),  # not at a word start
        ((), None),
    ],
)
def test_dwi_suffix_none(image_type, text):
    assert naming.dwi_suffix(image_type, text) is None


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("MÜLLER^THOMAS", "MuellerThomas"),
        ("Strauß^Jürgen", "StraussJuergen"),
        ("van der Berg^Anna-Lena", "VanderbergAnnalena"),
        ("DOE", "Doe"),
        ("DOE^JOHN^MIDDLE^DR^JR", "DoeJohnMiddle"),
        ("Li^Wei", "LiWei"),
        ("PHANTOM 2^QA", "Phantom2Qa"),
        ("Study01^Sub-007", "Study01Sub007"),
        ("123^456", "123456"),
    ],
)
def test_subject_label_full(name, label):
    assert naming.subject_label(name, full=True) == label


@pytest.mark.parametrize("name", [None, "", "^", "李^伟"])
def test_subject_label_full_errors(name):
    with pytest.raises(LabelError):
        naming.subject_label(name, full=True)


# --- short names (the default): <FirstName>_<NN>_<ProtocolName> --------------------------------


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("MÜLLER^THOMAS", "Thomas"),
        ("Strauß^Jürgen", "Juergen"),
        ("van der Berg^Anna-Lena", "Annalena"),
        ("DOE^JOHN^MIDDLE^DR^JR", "John"),  # given name only
        ("Study01^Sub-007", "Sub007"),  # digits kept
    ],
)
def test_given_name_label(name, label):
    assert naming.given_name_label(name) == label


@pytest.mark.parametrize("name", [None, "", "PHANTOM_QA", "DOE^", "DOE^李", "DOE^-"])
def test_no_given_name_label(name):
    """No usable given name: the caller falls back to the subject label."""
    assert naming.given_name_label(name) is None


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("t2_tse_tra_fs_384", "t2_tse_tra_fs_384"),  # kept as is, underscores included
        ("sag PD fs", "sag_PD_fs"),
        ("t2*_gre / me", "t2_gre_me"),  # unsafe characters: one "_" per run
        ("ep2d-diff  b50", "ep2d-diff_b50"),
        ("  _localizer_ ", "localizer"),
        ("Diffusion ÄDC", "Diffusion_AeDC"),
        ("a" * 80, "a" * 80),  # never cut
        ("*** ", None),
        (None, None),
    ],
)
def test_sequence_label(text, label):
    assert naming.sequence_label(text) == label


@pytest.mark.parametrize(
    ("protocol", "description", "expected"),
    [
        ("t1_vibe", "T1 VIBE post", ("t1_vibe", "ProtocolName")),
        ("", "T1 VIBE post", ("T1 VIBE post", "SeriesDescription")),
        ("***", "T1 VIBE post", ("T1 VIBE post", "SeriesDescription")),
        (None, None, (None, None)),
    ],
)
def test_protocol_text_prefers_protocol_name(protocol, description, expected):
    assert naming.protocol_text(protocol, description) == expected


def test_short_file_stem():
    assert naming.short_file_stem("Thomas", "05", "t2_tse") == "Thomas_05_t2_tse"
    assert naming.short_file_stem("Thomas", "05") == "Thomas_05"
    assert naming.short_file_stem("Thomas", "05", "t2_tse", dup=2) == "Thomas_05_t2_tse_02"
    assert (
        naming.short_file_stem("Thomas", "01", "loc", chunk=1, dup=2) == "Thomas_01_loc_02_chunk-1"
    )
