"""The desktop window: commands run off the Tk thread and report like the CLI."""

import json
import time
import tkinter as tk
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from roundtrip import output_rows
from synthetic import SeriesSpec, make_series, write_series
from test_merge import _b_series
from test_pmap import META as PMAP_META
from test_seg import META as SEG_META

from dcmtether import api, report
from dcmtether.errors import DcmtetherError
from dcmtether.gui import self_test
from dcmtether.gui.worker import Done, Failed, Lines, Progress, Worker, safe_message
from dcmtether.seg import SegResult


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
        or not app.status.get().endswith(("Done.", "(see log).", "Failed."))
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
    assert float(app.bar["value"]) == float(app.bar["maximum"])  # a finished run leaves it full
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
    assert app.status.get() == "Done, with problems (see log)."

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
    app.uid_mode.set("root")
    app.fields["uid_root"].set("1.2.3")
    text = _press(app, app.derive)
    assert expected in text, text
    assert output.exists()
    _no_secrets(text, tmp_path)
    written = sorted(output.iterdir()) if kind == "mr" else [output]
    for path in written:  # assert only, never print the UIDs
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        assert ds.SeriesInstanceUID.startswith("1.2.3.") and len(ds.SOPInstanceUID) <= 64
        assert ds.SOPInstanceUID.startswith("1.2.3.")


class _Calls(list):
    fallback: bool


@pytest.fixture
def fake_seg(app, monkeypatch):
    """``api.to_seg`` replaced: records its keywords, returns ``fake_seg.fallback``."""
    calls = _Calls()

    def to_seg(*args, **options):
        calls.append(options)
        return SegResult(Path(args[3]), {1: 1}, 5, calls.fallback)

    calls.fallback = False
    monkeypatch.setattr(api, "to_seg", to_seg)
    app.derivative.set("seg")
    for key in ("source_nifti", "derivative", "template", "derived_output"):
        app.fields[key].set(key)
    return calls


@pytest.mark.parametrize(
    ("mode", "root", "expected"),
    [
        ("synthetic", "", {"uid_root": None, "uid_source": False}),
        ("root", " 1.2.3 ", {"uid_root": "1.2.3", "uid_source": False}),
        ("source", "1.2.3", {"uid_root": None, "uid_source": True}),  # root ignored
    ],
)
def test_derivative_uid_modes_are_passed_through(app, fake_seg, mode, root, expected):
    app.uid_mode.set(mode)
    app.fields["uid_root"].set(root)
    _press(app, app.derive)
    assert fake_seg == [expected]
    assert app.status.get() == "Done."


def test_uid_root_entry_is_only_enabled_for_org_root(app):
    for mode, enabled in (("synthetic", False), ("root", True), ("source", False)):
        app.uid_mode.set(mode)
        app._uid_mode_changed()
        assert app.uid_root_entry.instate(["!disabled"]) is enabled


@pytest.mark.parametrize(
    ("root", "error"),
    [
        ("", "Please fill in every field."),
        ("1..2", "UID root must be digits separated by dots, without leading zeros"),
        ("1." * 25 + "1", "UID root is longer than 39 characters"),
    ],
)
def test_bad_org_root_is_refused_before_running(app, fake_seg, root, error):
    app.uid_mode.set("root")
    app.fields["uid_root"].set(root)
    app.derive()
    assert app.errors == [error]
    assert fake_seg == [] and not app.worker.busy


def test_uid_fallback_note_is_shown(app, fake_seg):
    fake_seg.fallback = True
    app.uid_mode.set("source")
    assert report.UID_FALLBACK in _press(app, app.derive)
    assert app.status.get() == "Done, with problems (see log)."
    assert app.show_log.get()


def test_progress_block_shows_counts_and_stays_full_after_a_run(app, tmp_path):
    src = tmp_path / "src"
    write_series(make_series(SeriesSpec(series_number=1)), src / "a")
    app.fields["source"].set(str(src))
    app.fields["dataset"].set(str(tmp_path / "dataset"))
    seen = []
    original = app.count.set
    app.count.set = lambda value: (seen.append(value), original(value))[1]
    _press(app, app.convert)
    assert any(" / " in value for value in seen)  # "n / total" while counting
    assert str(app.bar["mode"]) == "determinate"
    # a command without steps to count (Compare) still ends with a full bar
    app.fields["original"].set(str(src / "a"))
    app.fields["rebuilt"].set(str(src / "a"))
    _press(app, app.compare)
    assert float(app.bar["value"]) == float(app.bar["maximum"]) == 1


def test_log_is_hidden_by_default_but_filled(app, tmp_path):
    src = tmp_path / "src"
    write_series(make_series(SeriesSpec(series_number=1)), src / "a")
    app.fields["source"].set(str(src))
    app.fields["dataset"].set(str(tmp_path / "dataset"))
    assert not app.show_log.get() and not app._log.winfo_manager()
    text = _press(app, app.convert)
    assert "Converted 1 series, 0 failed." in text  # collected while hidden
    assert app.status.get() == "Done." and not app._log.winfo_manager()
    app.show_log.set(True)
    app._toggle_log()
    assert app._log.winfo_manager() == "pack"
    app.show_log.set(False)
    app._toggle_log()
    assert not app._log.winfo_manager()


def test_log_opens_when_a_run_fails(app, tmp_path):
    app.fields["source"].set(str(tmp_path / "missing"))
    app.fields["dataset"].set(str(tmp_path / "dataset"))
    _press(app, app.convert)
    assert app.show_log.get() and app._log.winfo_manager() == "pack"
    assert app.status.get() == "Failed."


def _ivim_dataset(app, tmp_path):
    src = tmp_path / "src"
    for number, b_value in ((5, 0), (6, 50), (7, 800)):
        write_series(_b_series(b_value, number), src / f"s{number}")
    app.fields["source"].set(str(src))
    app.fields["dataset"].set(str(tmp_path / "dataset"))
    _press(app, app.convert)
    app.fields["merge_dataset"].set(str(tmp_path / "dataset"))


def test_merge_finds_the_groups(app, tmp_path):
    _ivim_dataset(app, tmp_path)
    text = _press(app, app.merge)
    assert "(3 volumes)." in text and text.count("Merged series") == 1
    assert app.status.get() == "Done." and not app.show_log.get()
    assert list((tmp_path / "dataset" / "derivatives" / "merged").rglob("*_desc-merged_dwi.nii.gz"))
    assert float(app.bar["value"]) == float(app.bar["maximum"])
    _no_secrets(text, tmp_path)


def test_merge_ids_are_checked(app, tmp_path):
    _ivim_dataset(app, tmp_path)
    app.fields["merge_ids"].set("4,x")
    app.merge()
    assert app.errors[-1] == "expected manifest ids like 4,5,6"
    app.fields["merge_ids"].set("1")
    text = _press(app, app.merge)
    assert app.status.get() == "Failed." and "name at least two different series" in text
    assert app.errors[-1] == "name at least two different series"


def test_merge_replace_asks_first(app, tmp_path, monkeypatch):
    _ivim_dataset(app, tmp_path)
    asked = []
    monkeypatch.setattr(app, "_confirm", lambda text: asked.append(text) or False)
    app.placement.set("replace")
    app.merge()
    assert len(asked) == 1 and not app.worker.busy
    assert not list((tmp_path / "dataset").rglob("*_desc-merged_dwi.nii.gz"))
    monkeypatch.setattr(app, "_confirm", lambda text: True)
    text = _press(app, app.merge)
    assert "replaced the single series" in text


# --- theme -------------------------------------------------------------------------------------


@pytest.fixture
def light_theme(tk_root):
    from dcmtether.gui import theme

    theme.apply("light")
    yield theme
    theme.apply("light")


def test_dark_mode_switch_changes_theme_and_log_colors(light_theme, app):
    assert not app.dark_mode.get()
    assert str(app.output.cget("background")) == light_theme.COLORS["light"]["base"]

    app.dark_mode.set(True)
    app._toggle_theme()

    assert light_theme.current() == "dark"
    assert str(app.output.cget("background")) == light_theme.COLORS["dark"]["base"]
    assert str(app.output.cget("foreground")) == light_theme.COLORS["dark"]["text"]

    app.dark_mode.set(False)
    app._toggle_theme()

    assert light_theme.current() == "light"
    assert str(app.output.cget("background")) == light_theme.COLORS["light"]["base"]


def test_hint_labels_follow_the_theme(light_theme, app):
    from tkinter import ttk

    app.dark_mode.set(True)
    app._toggle_theme()

    assert ttk.Style().lookup("Hint.TLabel", "foreground") == light_theme.COLORS["dark"]["hint"]


def test_system_theme_is_light_or_dark():
    from dcmtether.gui import theme

    assert theme.system_theme() in {"light", "dark"}
