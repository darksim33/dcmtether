"""Structured JSON-lines log in the dataset: ``DATASET/logs/dcmtether.jsonl`` (PLAN.md Phase 6).

One record per event: time, tool version, command, event and a few fields from a fixed
allowlist. Records never contain element values, names, UIDs or paths (PHI rules): series are
named by manifest id and failures by their safe (value-free) message. Unknown field names are
refused, so new code cannot start logging values by accident.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from dcmtether import __version__
from dcmtether.manifest import MANIFEST_NAME

LOG_DIR = "logs"
LOG_NAME = "dcmtether.jsonl"

#: Every field a record may have. Values must be numbers, safe strings or lists of ids.
ALLOWED_FIELDS = frozenset(
    {
        "time",
        "version",
        "command",
        "event",
        "series_id",
        "status",
        "reason",
        "count",
        "outputs",
        "n_files",
        "n_series",
        "n_skipped",
        "n_supported",
        "n_split",
        "n_unsupported",
        "n_converted",
        "n_failed",
        "new",
        "rescanned",
        "unchanged",
        "changed_after_conversion",
        "missing",
        "kind",
    }
)


def find_dataset(path: Path) -> Path | None:
    """The dataset directory ``path`` lies in (the nearest parent with a manifest), if any."""
    for parent in Path(path).resolve().parents:
        if (parent / MANIFEST_NAME).is_file():
            return parent
    return None


class DatasetLog:
    """Appends records for one command; a no-op when there is no dataset."""

    def __init__(self, dataset: Path | None, command: str) -> None:
        self.path = dataset / LOG_DIR / LOG_NAME if dataset is not None else None
        self.command = command

    def event(self, event: str, **fields) -> None:
        unknown = set(fields) - ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"log fields not allowed: {', '.join(sorted(unknown))}")
        if self.path is None:
            return
        record = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "version": __version__,
            "command": self.command,
            "event": event,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=False) + "\n")
