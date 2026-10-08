"""Deterministic UIDs for derived objects (PLAN.md §2).

``2.25.<uuid5 as integer>`` needs no organisation root. The name hashed into the UUID includes
the source UID, the kind of derivation and a hash of the derived content, so identical reruns
reproduce the UIDs while different content always gets different UIDs (reusing a UID for
changed data violates DICOM uniqueness; a PACS may silently keep only one copy).
"""

import uuid

#: Fixed namespace for every UID dcmtether derives.
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/darksim33/dcmtether/derived-uids")


def derived_uid(source_uid: str, derivation: str, content_hash: str, role: str) -> str:
    """UID for ``role`` (e.g. ``"series"``, ``"instance"``) of a derivation of ``source_uid``."""
    name = "|".join((source_uid, derivation, content_hash, role))
    return f"2.25.{uuid.uuid5(NAMESPACE, name).int}"
