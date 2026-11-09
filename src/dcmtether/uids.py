"""Deterministic UIDs for derived objects.

Every new UID is ``<prefix>.<digits>``. The digits come from a UUID (``uuid5``) of a name that
holds the source UID, the kind of derivation, a hash of the derived content and the role
(series or instance), so identical reruns reproduce the UIDs while different content always
gets different UIDs (reusing a UID for changed data violates DICOM uniqueness; a PACS may
silently keep only one copy). No date, time or counter goes in.

The prefix is chosen per call (``uid_prefix``):

- **synthetic** (default): ``2.25.<uuid as integer>``, needs no organisation root;
- **org root**: a root the user's organisation owns, used as given;
- **source stem**: the source series UID up to its last dot. That root belongs to whoever
  issued the source UID, so this mode is not strictly DICOM-conformant; a stem that is not
  usable falls back to synthetic UIDs.

A prefix is at most ``MAX_ROOT`` characters, so at least ``MIN_DIGITS`` digits (about 80 bits)
remain within the 64-character UID limit.
"""

import re
import uuid

from dcmtether.errors import DcmtetherError

#: Fixed namespace for every UID dcmtether derives.
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/darksim33/dcmtether/derived-uids")
#: Longest UID DICOM allows.
UID_MAX = 64
#: Fewest hash digits after an org root or source stem (about 80 bits).
MIN_DIGITS = 24
#: Longest org root or source stem: the dot and ``MIN_DIGITS`` digits must still fit.
MAX_ROOT = UID_MAX - 1 - MIN_DIGITS

_COMPONENT = r"(0|[1-9][0-9]*)"
_ROOT = re.compile(rf"{_COMPONENT}(\.{_COMPONENT})*")


class UidError(DcmtetherError, ValueError):
    """Invalid UID options (value-free message)."""


def _is_root(value: str) -> bool:
    return len(value) <= MAX_ROOT and _ROOT.fullmatch(value) is not None


def check_root(root: str) -> str:
    """``root`` without surrounding blanks, if it can prefix derived UIDs; else ``UidError``."""
    root = root.strip()
    if _ROOT.fullmatch(root) is None:
        raise UidError("UID root must be digits separated by dots, without leading zeros")
    if len(root) > MAX_ROOT:
        raise UidError(f"UID root is longer than {MAX_ROOT} characters")
    return root


def stem(uid: str) -> str | None:
    """``uid`` up to its last dot, if that can prefix derived UIDs; else ``None``."""
    head, dot, _ = uid.strip().rpartition(".")
    return head if dot and _is_root(head) else None


def uid_prefix(
    source_series_uid: str, uid_root: str | None = None, uid_source: bool = False
) -> tuple[str | None, bool]:
    """(prefix for ``derived_uid``, whether a source stem fell back to synthetic UIDs).

    ``None`` as prefix means synthetic ``2.25.`` UIDs.
    """
    if uid_root is not None and uid_source:
        raise UidError("--uid-root and --uid-source cannot be combined")
    if uid_root is not None:
        return check_root(uid_root), False
    if uid_source:
        prefix = stem(source_series_uid)
        return prefix, prefix is None
    return None, False


def derived_uid(
    source_uid: str, derivation: str, content_hash: str, role: str, root: str | None = None
) -> str:
    """UID for ``role`` (e.g. ``"series"``, ``"instance"``) of a derivation of ``source_uid``.

    ``root`` (from ``uid_prefix``) replaces ``2.25``; the digits are then cut to fit 64
    characters (at least ``MIN_DIGITS`` remain).
    """
    name = "|".join((source_uid, derivation, content_hash, role))
    number = uuid.uuid5(NAMESPACE, name).int
    if root is None:
        return f"2.25.{number}"
    if not _is_root(root):
        raise UidError("UID root must be digits separated by dots, without leading zeros")
    return f"{root}.{str(number)[: UID_MAX - len(root) - 1]}"
