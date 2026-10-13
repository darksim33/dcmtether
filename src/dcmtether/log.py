"""Structured JSON-lines log in the dataset: ``DATASET/logs/dcmtether_<time>.jsonl``.

``<time>`` is the UTC start of the command (``YYYYMMDDTHHMMSS``). ``run``, ``scan`` and
``to-nii`` start a file each; commands on a single series append to the newest one.
One record per event: time, tool version, command, event and a few fields from a fixed
allowlist. Records never contain element values, names, UIDs or paths (PHI rules): series are
named by manifest id and failures by their safe (value-free) message. Unknown field names are
refused, so new code cannot start logging values by accident.
"""

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from dcmtether import __version__
from dcmtether.errors import DcmtetherError
from dcmtether.manifest import MANIFEST_NAME
from dcmtether.scanning import LOG_DIR

#: Log files of any version: ``dcmtether.jsonl`` (before the names had a timestamp),
#: ``dcmtether_<time>.jsonl`` and ``dcmtether_<time>_<n>.jsonl`` (same second).
LOG_FILE = re.compile(r"dcmtether(?:_(\d{8}T\d{6})(?:_(\d+))?)?\.jsonl")

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


def log_files(dataset: Path) -> list[Path]:
    """The dataset's log files, oldest first."""
    found = []
    for path in (dataset / LOG_DIR).glob("dcmtether*.jsonl"):
        match = LOG_FILE.fullmatch(path.name)
        if match and path.is_file():
            found.append(((match[1] or "", int(match[2] or 1)), path))
    return [path for _, path in sorted(found)]


def _new_log_file(dataset: Path) -> Path:
    stem = f"dcmtether_{datetime.now(UTC):%Y%m%dT%H%M%S}"
    path, n = dataset / LOG_DIR / f"{stem}.jsonl", 1
    while path.exists():  # another command in the same second
        n += 1
        path = dataset / LOG_DIR / f"{stem}_{n}.jsonl"
    return path


class DatasetLog:
    """Appends records for one command; a no-op when there is no dataset.

    ``new_file=True`` starts a log file for the command; otherwise it goes on in the newest
    timestamped file, if there is one.
    """

    def __init__(self, dataset: Path | None, command: str, new_file: bool = False) -> None:
        self.path = None
        if dataset is not None:
            current = [p for p in log_files(dataset) if LOG_FILE.fullmatch(p.name)[1]]
            self.path = current[-1] if current and not new_file else _new_log_file(dataset)
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


@contextmanager
def logged(dataset: Path | None, command: str, new_file: bool = False) -> Iterator[DatasetLog]:
    """start/end records around a command; a :class:`DcmtetherError` is recorded as ``error``.

    ``new_file``: the command starts its own log file (``run``, ``scan``, ``to-nii``).
    ``logger.end_fields`` are added to the end record.
    """
    logger = DatasetLog(dataset, command, new_file)
    logger.end_fields = {}
    logger.event("start")
    try:
        yield logger
    except DcmtetherError as exc:
        logger.event("error", reason=str(exc))
        raise
    finally:
        logger.event("end", **logger.end_fields)
