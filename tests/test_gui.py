"""The desktop window: commands run off the Tk thread and report like the CLI."""

import json
import time
import tkinter as tk
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from roundtrip import output_rows
from synthetic import SeriesSpec, make_series, write_series
from test_pmap import META as PMAP_META
from test_seg import META as SEG_META

from dcmtether.errors import DcmtetherError
from dcmtether.gui import self_test
from dcmtether.gui.worker import Done, Failed, Lines, Progress, Worker, safe_message


def _messages(worker: Worker, timeout: float = 60) -> list:
    found, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        found += worker.messages()
        if found and isinstance(found[-1], Done | Failed):
            return found
        time.sleep(0.01)
    raise TimeoutError


# --- worker (no window needed) -----------------------------------------------------------------


def test_worker_reports_progress_lines_and_result():
    worker = Worker()

    def task(progress, show):
        items = list(progress(["a", "b", "c"], desc="Reading headers"))
        show(["early"])
        return [f"{len(items)} items"], True

    worker.start(task)
    messages = _messages(worker)
    assert [m for m in messages if isinstance(m, Progress)] == [
        Progress("Reading headers", n, 3) for n in range(4)
    ]
    assert Lines(["early"]) in messages
    assert messages[-1] == Done(["3 items"], True)
    assert not worker.busy


@pytest.mark.parametrize(
    ("error", "shown"),
    [
        (
            DcmtetherError("manifest already exists in DATASET"),
            "manifest already exists in DATASET",
        ),
        (FileNotFoundError("SOURCE does not exist"), "SOURCE does not exist"),
        (PermissionError(13, "Permission denied", "C:/Mueller/IM_0001"), "Permission denied"),
        (KeyError("Mueller^Thomas"), "Unexpected error (KeyError)"),
    ],
)
def test_errors_are_shown_without_values(error, shown):
    assert safe_message(error) == shown
    worker = Worker()

    def task(progress, show):
        raise error

    worker.start(task)
    assert _messages(worker)[-1] == Failed(shown)


def test_self_test_passes_in_the_development_environment():
    assert self_test() == []


# --- window ------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tk_root():
    # One Tcl interpreter per test session: creating and destroying several in one process
    # makes Tcl's own initialisation fail now and then on Windows.
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for Tk")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(tk_root, monkeypatch):
    from dcmtether.gui import app as app_module

    errors = []
    monkeypatch.setattr(app_module.App, "_show_error", lambda self, text: errors.append(text))
    window = tk.Toplevel(tk_root)
    window.withdraw()
    gui = app_module.App(window)
    gui.errors = errors
    yield gui
    window.destroy()


def _press(app, command, timeout: float = 120) -> str:
    command()
    deadline = time.monotonic() + timeout
    while (
        app.worker.busy
        or app.status.get() in ("Working…",)
        or not app.status.get().endswith(("Done.", "(see above).", "Failed."))
    ):
        app.root.update()
        if time.monotonic() > deadline:
            raise TimeoutError(app.text())
        time.sleep(0.01)
    app.root.update()
    return app.text()


def _no_secrets(text: str, tmp_path: Path) -> None:
    for secret in (tmp_path.name, "Synthetic", "Phantom", "2.25.", "sub-"):
        assert secret not in text, secret


def test_window_has_the_logo_as_icon(app):
    assert [icon.width() for icon in app._icons] == [32, 64, 256]


def test_convert_rebuild_and_compare(app, tmp_path):
    src = tmp_path / "src"
    write_series(make_series(SeriesSpec(series_number=1)), src / "a")
    app.fields["source"].set(str(src))
    app.fields["dataset"].set(str(tmp_path / "dataset"))

    text = _press(app, app.convert)
    assert "Scanned 5 files: 5 DICOM instances in 1 series, 0 files skipped." in text
    assert "Manifest: 1 new series, 0 rescanned, 0 unchanged." in text
    assert "Converted 1 series, 0 failed." in text
    assert app.status.get() == "Done."
    _no_secrets(text, tmp_path)

    nifti = tmp_path / "dataset" / output_rows(tmp_path / "dataset")[0]["nifti_path"]
    app.fields["rebuild_nifti"].set(str(nifti))
    app.fields["rebuild_output"].set(str(tmp_path / "rebuilt"))
    app.fields["rebuild_original"].set(str(src / "a"))
    text = _press(app, app.rebuild)
    assert "Wrote 5 DICOM files." in text and "Identical: 5 instances compared." in text
    _no_secrets(text, tmp_path)

    app.fields["original"].set(str(src / "a"))
    app.fields["rebuilt"].set(str(tmp_path / "rebuilt"))
    assert "Identical: 5 instances compared." in _press(app, app.compare)


def test_failures_and_errors(app, tmp_path):
    src = tmp_path / "src"
    series = make_series()
    for ds in series.datasets:
        ds.PatientName = ""
    write_series(series, src / "a")
    app.fields["source"].set(str(src))
    app.fields["dataset"].set(str(tmp_path / "dataset"))
    text = _press(app, app.convert)
    assert "series #1 failed: PatientName is missing" in text
    assert app.status.get() == "Done, with problems (see above)."

    app.fields["source"].set(str(tmp_path / "missing"))
    text = _press(app, app.convert)
    assert text.rstrip().endswith("Error: SOURCE does not exist")
    assert app.errors == ["SOURCE does not exist"]

    app.fields["source"].set("")
    app.convert()
    assert app.errors[-1] == "Please fill in every field."


@pytest.mark.parametrize(
    ("kind", "meta", "expected"),
    [
        ("seg", SEG_META, "Wrote DICOM SEG with 2 segments on 5 source images."),
        ("pmap", PMAP_META, "Wrote DICOM Parametric Map (float32) on 5 source images."),
        ("mr", PMAP_META, "Wrote 5 derived MR images (map)."),
    ],
)
def test_derivatives(app, tmp_path, kind, meta, expected):
    src = tmp_path / "src"
    write_series(make_series(), src / "a")
    app.fields["source"].set(str(src))
    app.fields["dataset"].set(str(tmp_path / "dataset"))
    _press(app, app.convert)
    nifti = tmp_path / "dataset" / output_rows(tmp_path / "dataset")[0]["nifti_path"]
    img = nib.load(nifti)
    if kind == "seg":
        data = np.zeros(img.shape, np.uint8)
        data[1, 1, 1], data[2, 2, 2] = 1, 2
    else:
        data = np.linspace(0, 1, np.prod(img.shape), dtype=np.float32).reshape(img.shape)
    nib.save(nib.Nifti1Image(data, img.affine), tmp_path / "derivative.nii.gz")
    (tmp_path / "template.json").write_text(json.dumps(meta), "utf-8")

    app.derivative.set(kind)
    app.fields["source_nifti"].set(str(nifti))
    app.fields["derivative"].set(str(tmp_path / "derivative.nii.gz"))
    app.fields["template"].set(str(tmp_path / "template.json"))
    output = tmp_path / ("out" if kind == "mr" else "out.dcm")
    app.fields["derived_output"].set(str(output))
    text = _press(app, app.derive)
    assert expected in text, text
    assert output.exists()
    _no_secrets(text, tmp_path)
