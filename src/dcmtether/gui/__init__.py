"""Desktop window for dcmtether (tkinter): the CLI's commands with folder pickers.

Start with ``dcmtether-gui`` (or the portable Windows app). ``--self-test`` checks that a
frozen build has everything it needs and exits.
"""

import sys
from importlib import metadata, resources
from pathlib import Path


def _dpi_aware() -> None:
    """Sharp text on scaled Windows displays (otherwise Windows stretches a blurry bitmap)."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def self_test() -> list[str]:
    """Problems that would break the app; empty if it is complete."""
    problems = []
    try:
        metadata.version("dcmtether")
    except metadata.PackageNotFoundError:
        problems.append("dcmtether package metadata missing")
    if not (resources.files("highdicom") / "_standard" / "iod_module_map.json").is_file():
        problems.append("highdicom data files missing")
    if not all(
        (resources.files("dcmtether.gui") / f"icon-{size}.png").is_file() for size in (32, 64, 256)
    ):
        problems.append("window icons missing")
    from pydicom.pixels.decoders.base import get_decoder

    from dcmtether.scanning import LOSSLESS_COMPRESSED_TRANSFER_SYNTAXES

    for uid in sorted(LOSSLESS_COMPRESSED_TRANSFER_SYNTAXES):
        if "pylibjpeg" not in get_decoder(uid).available_plugins:
            problems.append(f"no pylibjpeg decoder for {uid.name}")
    try:
        import tkinter

        tkinter.Tcl()
    except Exception as exc:
        problems.append(f"Tcl/Tk not usable ({type(exc).__name__})")
    return problems


def _report_self_test() -> int:
    problems = self_test()
    text = "\n".join(problems) if problems else "ok"
    if sys.stdout is None:  # windowed build: no console, so leave a file next to the exe
        Path(sys.executable).with_name("self-test.txt").write_text(text + "\n", "utf-8")
    else:
        print(text)
    return 1 if problems else 0


def main() -> None:
    if "--self-test" in sys.argv[1:]:
        sys.exit(_report_self_test())
    import tkinter as tk

    from dcmtether.gui.app import App

    _dpi_aware()
    root = tk.Tk()
    App(root)
    root.mainloop()
