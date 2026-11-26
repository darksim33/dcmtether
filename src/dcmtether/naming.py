"""File names, in one of two forms (fixed per dataset, see :mod:`dcmtether.forward`).

Short (the default), in ``sub-<FamilyGivenMiddle>/ses-<YYYYMMDD[THHMM]>/<datatype>/``::

    <GivenName | subject label>_<NN>[_<ProtocolName>][_<XX>][_chunk-<n>]

BIDS-inspired (``--BIDS``)::

    sub-<Family5><Given5 | FamilyGiven>_ses-<YYYYMMDD[THHMM]>[_acq-<desc>]_ser-<NN>[_<XX>]_<suffix>

(No body-part entity: BodyPartExamined stays in manifest and sidecar.)

Subject and session labels and the given name are built from PatientName and StudyDate, so
**file names identify the patient** (deliberately: they are readable for the people working on
the data). Code must still never print them: callers report problems by manifest series id.
"""

import re
import unicodedata
from dataclasses import dataclass

from pydicom.valuerep import PersonName

NAME_PART_LENGTH = 5
ACQ_MAX_LENGTH = 24

# German transliteration first (Ü → Ue, not U); everything else via Unicode decomposition.
_TRANSLITERATION = str.maketrans(
    {"Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "ẞ": "Ss"}
)


class LabelError(ValueError):
    """A label cannot be built. Messages never contain the offending values."""


def _fold(text: str) -> str:
    """Transliterate, strip accents, and turn any remaining non-ASCII character into a space."""
    text = unicodedata.normalize("NFKD", text.translate(_TRANSLITERATION))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "".join(c if c.isascii() else " " for c in text)


def _ascii_letters(text: str) -> str:
    return "".join(c for c in _fold(text) if c.isalpha())


def _name_part(text: str) -> str:
    letters = _ascii_letters(text)[:NAME_PART_LENGTH]
    return letters[:1].upper() + letters[1:].lower()


def _full_name_part(text: str) -> str:
    """Letters and digits, uncut: digits often are what tells phantoms or study codes apart."""
    kept = "".join(c for c in _fold(text) if c.isalnum())
    return kept[:1].upper() + kept[1:].lower()


def subject_label(patient_name: str | PersonName | None, full: bool = False) -> str:
    """``MÜLLER^THOMAS`` → ``MuellThoma`` (first 5 letters of family and given name).

    With ``full``: the whole family, given and middle name, digits included
    (``MuellerThomas``, ``Phantom2Qa``).
    """
    if patient_name is None:
        raise LabelError("PatientName is missing")
    name = patient_name if isinstance(patient_name, PersonName) else PersonName(patient_name)
    if full:
        parts = (name.family_name, name.given_name, name.middle_name)
        label = "".join(_full_name_part(part) for part in parts)
    else:
        label = _name_part(name.family_name) + _name_part(name.given_name)
    if not label:
        raise LabelError("PatientName has no usable letters")
    return label


def given_name_label(patient_name: str | PersonName | None) -> str | None:
    """``MÜLLER^THOMAS`` → ``Thomas``: the whole given name, digits kept (short file names).

    ``None`` if there is no usable given name (e.g. ``PHANTOM_QA``); callers then use the
    subject label instead.
    """
    if not patient_name:
        return None
    name = patient_name if isinstance(patient_name, PersonName) else PersonName(patient_name)
    return _full_name_part(name.given_name) or None


def session_label(
    study_date: str | None, study_time: str | None = None, with_time: bool = False
) -> str:
    """``20260314``, or ``20260314T0930`` when several studies share the date."""
    if not study_date or not re.fullmatch(r"\d{8}", study_date.strip()):
        raise LabelError("StudyDate is missing or not YYYYMMDD")
    label = study_date.strip()
    if with_time:
        if not study_time or not re.match(r"\d{4}", study_time.strip()):
            raise LabelError("StudyTime is missing but needed to separate same-day studies")
        label += "T" + study_time.strip()[:4]
    return label


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^A-Za-z0-9]+", _fold(text)) if w]


def acq_label(text: str | None) -> str | None:
    """``sag PD fs`` → ``sagPDFs``; ``None`` if nothing usable is left."""
    words = _words(text or "")
    if not words:
        return None
    label = words[0] + "".join(w[:1].upper() + w[1:] for w in words[1:])
    return label[:ACQ_MAX_LENGTH]


def naming_text(
    protocol_name: str | None, series_description: str | None
) -> tuple[str | None, str | None]:
    """The text ``acq-`` and the suffix keywords are taken from, and its source keyword.

    SeriesDescription first (the name users see and refer to series by), else ProtocolName; a
    value with nothing usable for :func:`acq_label` counts as missing. ``(None, None)`` if
    neither has anything.
    """
    for text, source in (
        (series_description, "SeriesDescription"),
        (protocol_name, "ProtocolName"),
    ):
        if acq_label(text) is not None:
            return text, source
    return None, None


def sequence_label(text: str | None) -> str | None:
    """``t2_tse_tra fs*`` → ``t2_tse_tra_fs``: the text in full, only made safe for file names
    (ASCII; every run of other characters than letters, digits and ``-`` becomes one ``_``).
    ``None`` if nothing is left."""
    label = re.sub(r"[^A-Za-z0-9-]+", "_", _fold(text or "")).strip("_-")
    return label or None


def protocol_text(
    protocol_name: str | None, series_description: str | None
) -> tuple[str | None, str | None]:
    """The text the sequence name of short file names is taken from, and its source keyword.

    ProtocolName first, else SeriesDescription; a value with nothing usable for
    :func:`sequence_label` counts as missing. ``(None, None)`` if neither has anything.
    """
    for text, source in (
        (protocol_name, "ProtocolName"),
        (series_description, "SeriesDescription"),
    ):
        if sequence_label(text) is not None:
            return text, source
    return None, None


@dataclass(frozen=True)
class Suffix:
    value: str
    #: How it was chosen, recorded in the sidecar (e.g. ``keyword:flair``, ``parameters``).
    rule: str


# A token starts after a non-alphanumeric character (``_`` counts as a separator, unlike ``\b``).
_START = r"(?<![a-z0-9])"
_KEYWORDS = [
    ("FLAIR", re.compile(rf"flair|{_START}tirm|dark[\s_-]?fluid", re.I)),
    (
        "T1w",
        re.compile(rf"{_START}t1(?![0-9])|mprage|{_START}spgr|{_START}vibe|{_START}bravo", re.I),
    ),
    # T2* / T2star is gradient-echo susceptibility weighting, not a T2w image.
    ("T2w", re.compile(rf"{_START}t2(?![0-9*]|[\s_-]?star)", re.I)),
    # "pd", "pdw", "pd_fs", "pdfs" -- but not words like "pdx"/"pdg".
    ("PDw", re.compile(rf"{_START}pd(?![a-eg-vx-z])|proton[\s_-]?density", re.I)),
]


def contrast_suffix(
    text: str | None,
    repetition_time: float | None,
    echo_time: float | None,
    inversion_time: float | None,
) -> Suffix:
    """T1w / T2w / PDw / FLAIR from keywords in ``text`` (see :func:`naming_text`), else from
    TR/TE/TI, else ``MR``.

    Keywords win over parameters. Several contrasts in one text (e.g. "T1 T2 map") are
    ambiguous and fall through to the parameters.
    """
    hits = [suffix for suffix, pattern in _KEYWORDS if pattern.search(text or "")]
    if hits == ["FLAIR", "T2w"] or hits == ["FLAIR", "T1w"]:
        hits = ["FLAIR"]  # "T2 FLAIR" / "T1 FLAIR" are FLAIR images
    if len(hits) == 1:
        return Suffix(hits[0], f"keyword:{hits[0].lower()}")

    tr, te, ti = repetition_time, echo_time, inversion_time
    if tr is not None and te is not None:
        if ti is not None and ti >= 1500 and tr >= 4000:
            return Suffix("FLAIR", "parameters:TR/TE/TI")
        if tr <= 800 and te <= 30:
            return Suffix("T1w", "parameters:TR/TE")
        if tr >= 2000 and te >= 80:
            return Suffix("T2w", "parameters:TR/TE")
        if tr >= 2000 and te <= 40:
            return Suffix("PDw", "parameters:TR/TE")
    return Suffix("MR", "fallback")


#: Suffixes of scanner-calculated diffusion maps; they are stored next to the diffusion series.
DIFFUSION_MAP_SUFFIXES = frozenset({"ADC", "eADC", "FA", "TRACE"})

_MAP_IMAGE_TYPES = {"ADC": "ADC", "EADC": "eADC", "FA": "FA", "TRACEW": "TRACE", "TRACE": "TRACE"}
_END = r"(?![a-z0-9])"
_MAP_KEYWORDS = [
    ("ADC", re.compile(rf"{_START}adc{_END}|apparent[\s_-]?diffusion", re.I)),
    ("TRACE", re.compile(rf"{_START}tracew?{_END}", re.I)),
    ("FA", re.compile(r"fractional[\s_-]?anisotropy", re.I)),
]
# A bare "fa" is also a flip angle ("t1_fa15", "vibe_fa_2"): it needs a diffusion word next to it.
_FA_TOKEN = re.compile(rf"{_START}fa{_END}", re.I)
_DIFFUSION_WORD = re.compile(r"dti|dwi|diff|ivim|tensor", re.I)
# Words that name a diffusion series; "diff" but not "difference".
_DWI_KEYWORD = re.compile(rf"{_START}(dwi|dti|ivim|diff(?!eren))", re.I)


def diffusion_map_suffix(image_type: tuple[str, ...], text: str | None) -> Suffix | None:
    """ADC / eADC / FA / TRACE for a scanner-calculated diffusion map, else ``None``.

    ImageType values win over keywords in ``text`` (see :func:`naming_text`). Several different
    maps named at once are ambiguous and give ``None``.
    """
    values = [v.upper() for v in image_type]
    hits = {v: _MAP_IMAGE_TYPES[v] for v in values if v in _MAP_IMAGE_TYPES}
    if hits:
        if len(set(hits.values())) > 1:
            return None
        value, suffix = next(iter(hits.items()))
        return Suffix(suffix, f"imagetype:{value.lower()}")

    text = text or ""
    found = [suffix for suffix, pattern in _MAP_KEYWORDS if pattern.search(text)]
    if "FA" not in found and _FA_TOKEN.search(text) and _DIFFUSION_WORD.search(text):
        found.append("FA")
    if len(found) == 1:
        return Suffix(found[0], f"keyword:{found[0].lower()}")
    return None


def dwi_suffix(image_type: tuple[str, ...], text: str | None) -> Suffix | None:
    """``dwi`` for a diffusion series the scan could not tell from its volumes (e.g. a single
    b-value), else ``None``: ImageType ``DIFFUSION``, else keywords in ``text`` (``dwi``,
    ``dti``, ``ivim``, ``diff``). Check :func:`diffusion_map_suffix` first: maps carry
    ``DIFFUSION`` too.
    """
    if "DIFFUSION" in (v.upper() for v in image_type):
        return Suffix("dwi", "imagetype:diffusion")
    match = _DWI_KEYWORD.search(text or "")
    if match:
        return Suffix("dwi", f"keyword:{match.group(1).lower()}")
    return None


def series_label(series_number: int | None) -> str:
    if series_number is None or series_number < 0:
        raise LabelError("SeriesNumber is missing")
    return f"{series_number:02d}"


def file_stem(
    subject: str,
    session: str,
    series: str,
    suffix: str,
    acq: str | None = None,
    chunk: int | None = None,
    dup: int | None = None,
) -> str:
    """``dup`` numbers series that would otherwise share a name: ``ser-01_02``."""
    parts = [f"sub-{subject}", f"ses-{session}"]
    if acq:
        parts.append(f"acq-{acq}")
    parts.append(f"ser-{series}")
    if dup is not None:
        parts.append(f"{dup:02d}")
    if chunk is not None:
        parts.append(f"chunk-{chunk}")
    parts.append(suffix)
    return "_".join(parts)


def short_file_stem(
    first: str,
    series: str,
    sequence: str | None = None,
    chunk: int | None = None,
    dup: int | None = None,
) -> str:
    """``Thomas_05_t2_tse``; ``dup`` and ``chunk`` go at the end: ``…_t2_tse_02_chunk-1``."""
    parts = [first, series]
    if sequence:
        parts.append(sequence)
    if dup is not None:
        parts.append(f"{dup:02d}")
    if chunk is not None:
        parts.append(f"chunk-{chunk}")
    return "_".join(parts)
