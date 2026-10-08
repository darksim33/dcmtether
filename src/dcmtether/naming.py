"""BIDS-inspired file names (PLAN.md §2, "Naming").

    sub-<Family5><Given5>_ses-<YYYYMMDD[THHMM]>[_acq-<desc>]_ser-<NN>_<suffix>

(No body-part entity, by decision of 2026-10-07; BodyPartExamined stays in manifest/sidecar.)

Subject and session labels are built from PatientName and StudyDate, so **file names identify
the patient** (decision of 2026-10-07, see PLAN.md §2). Code must still never print them: callers
report problems by manifest series id.
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


def subject_label(patient_name: str | PersonName | None) -> str:
    """``MÜLLER^THOMAS`` → ``MuellThoma`` (first 5 letters of family and given name)."""
    if patient_name is None:
        raise LabelError("PatientName is missing")
    name = patient_name if isinstance(patient_name, PersonName) else PersonName(patient_name)
    label = _name_part(name.family_name) + _name_part(name.given_name)
    if not label:
        raise LabelError("PatientName has no usable letters")
    return label


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


def acq_label(series_description: str | None) -> str | None:
    """``sag PD fs`` → ``sagPDFs``; ``None`` if nothing usable is left."""
    words = _words(series_description or "")
    if not words:
        return None
    label = words[0] + "".join(w[:1].upper() + w[1:] for w in words[1:])
    return label[:ACQ_MAX_LENGTH]


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
    series_description: str | None,
    repetition_time: float | None,
    echo_time: float | None,
    inversion_time: float | None,
) -> Suffix:
    """T1w / T2w / PDw / FLAIR from description keywords, else from TR/TE/TI, else ``MR``.

    Keywords win over parameters. Several contrasts in one description (e.g. "T1 T2 map")
    are ambiguous and fall through to the parameters.
    """
    text = series_description or ""
    hits = [suffix for suffix, pattern in _KEYWORDS if pattern.search(text)]
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
) -> str:
    parts = [f"sub-{subject}", f"ses-{session}"]
    if acq:
        parts.append(f"acq-{acq}")
    parts.append(f"ser-{series}")
    if chunk is not None:
        parts.append(f"chunk-{chunk}")
    parts.append(suffix)
    return "_".join(parts)
