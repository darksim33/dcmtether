"""Runs one command at a time off the Tk thread and hands its messages back through a queue.

Tk widgets may only be touched from the thread that runs the event loop, so the worker never
does; the window polls :meth:`Worker.messages` instead.
"""

import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from dcmtether.errors import DcmtetherError


@dataclass(frozen=True)
class Progress:
    desc: str
    n: int
    total: int | None


@dataclass(frozen=True)
class Lines:
    """Text for the output pane, shown while the command still runs."""

    lines: list[str]


@dataclass(frozen=True)
class Done:
    lines: list[str]
    #: False if something did not work (a series failed, a comparison found differences).
    ok: bool


@dataclass(frozen=True)
class Failed:
    message: str


type Message = Progress | Lines | Done | Failed
#: A command: gets a progress callable and a way to show lines early; returns (lines, ok).
type Task = Callable[[Callable, Callable[[list[str]], None]], tuple[list[str], bool]]


def safe_message(exc: BaseException) -> str:
    """What may be shown for ``exc``: never a path, a name or a traceback (PHI rules)."""
    if isinstance(exc, DcmtetherError):
        return str(exc)
    if isinstance(exc, OSError):
        if exc.filename is None and exc.args and isinstance(exc.args[0], str):
            return exc.args[0]  # raised by dcmtether with a path-free message
        return exc.strerror or type(exc).__name__  # strerror leaves the file name out
    return f"Unexpected error ({type(exc).__name__})"


class Worker:
    def __init__(self) -> None:
        self._queue: queue.Queue[Message] = queue.Queue()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def progress(
        self, iterable: Iterable, total: int | None = None, desc: str = "", **_
    ) -> Iterator:
        """A ``progress=`` callable for the API (same call signature as tqdm)."""
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self._queue.put(Progress(desc, 0, total))
        for n, item in enumerate(iterable, start=1):
            yield item
            self._queue.put(Progress(desc, n, total))

    def start(self, task: Task) -> None:
        if self.busy:
            raise RuntimeError("a command is already running")

        def run() -> None:
            try:
                lines, ok = task(self.progress, lambda lines: self._queue.put(Lines(lines)))
            except BaseException as exc:
                self._queue.put(Failed(safe_message(exc)))
            else:
                self._queue.put(Done(lines, ok))

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def messages(self) -> list[Message]:
        """Everything posted since the last call (never blocks)."""
        found = []
        while True:
            try:
                found.append(self._queue.get_nowait())
            except queue.Empty:
                return found
