"""Progress reporting hook for long loops, so library code stays free of UI.

Callers pass a tqdm-compatible wrapper (the CLI passes tqdm); the default does nothing.
Descriptions are fixed labels: progress output must never name files, patients or UIDs.
"""

from collections.abc import Iterable
from typing import Protocol


class Progress(Protocol):
    def __call__[T](
        self, iterable: Iterable[T], *, total: int | None = None, desc: str = "", unit: str = "it"
    ) -> Iterable[T]: ...


def no_progress[T](iterable: Iterable[T], **_) -> Iterable[T]:
    return iterable
