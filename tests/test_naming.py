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
        ("sag PD fs", "sag PD fs_ND", ("sag PD fs", "ProtocolName")),
        ("sag PD fs", None, ("sag PD fs", "ProtocolName")),
        ("", "t1 axial", ("t1 axial", "SeriesDescription")),
        ("   ", "t1 axial", ("t1 axial", "SeriesDescription")),
        ("***", "t1 axial", ("t1 axial", "SeriesDescription")),  # nothing usable
        (None, "t1 axial", ("t1 axial", "SeriesDescription")),
        (None, "  ", (None, None)),
        (None, None, (None, None)),
    ],
)
def test_naming_text_prefers_protocol_name(protocol, description, expected):
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
    assert naming.series_label(5) == "05"
    assert naming.series_label(123) == "123"
    with pytest.raises(LabelError):
        naming.series_label(None)
