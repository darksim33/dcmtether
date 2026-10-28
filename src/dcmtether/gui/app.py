"""The window: one tab per kind of command, a shared output pane and a progress bar.

Paths entered here stay in the input fields: they are never shown in the output, the title or
any file, because folder and file names can identify patients.
"""

import base64
import tkinter as tk
from collections.abc import Callable
from importlib import resources
from tkinter import filedialog, messagebox, ttk

from dcmtether import __version__, api, report
from dcmtether.gui.worker import Done, Failed, Lines, Progress, Task, Worker

POLL_MS = 100
TITLE = "dcmtether"
NOTICE = "Research use only — not for clinical or diagnostic use."

#: Browse dialogs by kind of path field.
_DIRECTORY, _FILE, _NIFTI, _JSON, _SAVE_DCM = "directory", "file", "nifti", "json", "save-dcm"
_FILETYPES = {
    _NIFTI: [("NIfTI", "*.nii.gz *.nii"), ("All files", "*.*")],
    _JSON: [("JSON template", "*.json"), ("All files", "*.*")],
    _SAVE_DCM: [("DICOM", "*.dcm"), ("All files", "*.*")],
}
#: Window and taskbar icon (the logo of docs/assets), small to large.
ICON_SIZES = (32, 64, 256)
DERIVATIVES = {
    "seg": "Segmentation (SEG)",
    "pmap": "Parametric Map",
    "mr": "Derived MR images (any PACS)",
}


def _icons() -> list[tk.PhotoImage]:
    folder = resources.files("dcmtether.gui")
    return [
        tk.PhotoImage(data=base64.b64encode((folder / f"icon-{size}.png").read_bytes()))
        for size in ICON_SIZES
    ]


class App:
    def __init__(self, root: tk.Tk | tk.Toplevel) -> None:
        self.root = root
        self.worker = Worker()
        self.fields: dict[str, tk.StringVar] = {}
        self.full_name = tk.BooleanVar(value=False)
        self.compare_after = tk.BooleanVar(value=True)
        self.derivative = tk.StringVar(value="seg")
        self._buttons: list[ttk.Button] = []

        root.title(TITLE)
        self._icons = _icons()  # keep references: Tk drops images that Python lets go of
        root.iconphoto(True, *self._icons)
        root.minsize(640, 480)
        root.protocol("WM_DELETE_WINDOW", self.close)
        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)

        tabs = ttk.Notebook(outer)
        tabs.pack(fill="x")
        self.tabs = tabs
        self._convert_tab(tabs)
        self._rebuild_tab(tabs)
        self._compare_tab(tabs)
        self._derivatives_tab(tabs)

        output = ttk.Frame(outer)
        output.pack(fill="both", expand=True, pady=(8, 0))
        self.output = tk.Text(output, height=14, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(output, command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        self.output.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        status = ttk.Frame(outer)
        status.pack(fill="x", pady=(8, 0))
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(status, textvariable=self.status, width=24).pack(side="left")
        self.bar = ttk.Progressbar(status, mode="determinate")
        self.bar.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Label(status, text=f"{NOTICE}  v{__version__}", foreground="gray").pack(side="right")

    # --- layout helpers --------------------------------------------------------------------

    def _path_row(self, frame: ttk.Frame, row: int, key: str, label: str, kind: str) -> None:
        var = self.fields.setdefault(key, tk.StringVar())
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=var, width=60).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(frame, text="Browse…", command=lambda: self._browse(var, kind)).grid(
            row=row, column=2
        )

    def _run_button(self, frame: ttk.Frame, row: int, text: str, command: Callable) -> None:
        button = ttk.Button(frame, text=text, command=command)
        button.grid(row=row, column=1, sticky="e", pady=(8, 0))
        self._buttons.append(button)

    def _tab(self, tabs: ttk.Notebook, title: str, hint: str) -> ttk.Frame:
        frame = ttk.Frame(tabs, padding=8)
        frame.columnconfigure(1, weight=1)
        tabs.add(frame, text=title)
        ttk.Label(frame, text=hint, foreground="gray").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 6)
        )
        return frame

    def _browse(self, var: tk.StringVar, kind: str) -> None:
        if kind == "output":  # derivatives: a file for SEG / Parametric Map, a folder for MR
            kind = _DIRECTORY if self.derivative.get() == "mr" else _SAVE_DCM
        if kind == _DIRECTORY:
            path = filedialog.askdirectory(parent=self.root, mustexist=False)
        elif kind == _SAVE_DCM:
            path = filedialog.asksaveasfilename(
                parent=self.root, defaultextension=".dcm", filetypes=_FILETYPES[_SAVE_DCM]
            )
        else:
            path = filedialog.askopenfilename(
                parent=self.root, filetypes=_FILETYPES.get(kind, [("All files", "*.*")])
            )
        if path:
            var.set(path)

    # --- tabs ------------------------------------------------------------------------------

    def _convert_tab(self, tabs: ttk.Notebook) -> None:
        frame = self._tab(
            tabs,
            "Convert",
            "DICOM folder → NIfTI dataset. Safe to run again: only new or failed series are done.",
        )
        self._path_row(frame, 1, "source", "DICOM folder", _DIRECTORY)
        self._path_row(frame, 2, "dataset", "Dataset folder", _DIRECTORY)
        ttk.Label(frame, text="Empty: written into the DICOM folder.", foreground="gray").grid(
            row=3, column=1, sticky="w", padx=4
        )
        ttk.Checkbutton(
            frame, text="Full patient name in subject labels", variable=self.full_name
        ).grid(row=4, column=1, sticky="w", padx=4)
        self._run_button(frame, 5, "Convert", self.convert)

    def _rebuild_tab(self, tabs: ttk.Notebook) -> None:
        frame = self._tab(tabs, "Rebuild", "Converted NIfTI → the original DICOM files.")
        self._path_row(frame, 1, "rebuild_nifti", "NIfTI file", _NIFTI)
        self._path_row(frame, 2, "rebuild_output", "Output folder", _DIRECTORY)
        ttk.Checkbutton(
            frame, text="Compare with the original folder:", variable=self.compare_after
        ).grid(row=3, column=1, sticky="w", padx=4)
        self._path_row(frame, 4, "rebuild_original", "Original folder", _DIRECTORY)
        self._run_button(frame, 5, "Rebuild", self.rebuild)

    def _compare_tab(self, tabs: ttk.Notebook) -> None:
        frame = self._tab(tabs, "Compare", "Two DICOM series, element by element.")
        self._path_row(frame, 1, "original", "Original folder", _DIRECTORY)
        self._path_row(frame, 2, "rebuilt", "Rebuilt folder", _DIRECTORY)
        self._run_button(frame, 3, "Compare", self.compare)

    def _derivatives_tab(self, tabs: ttk.Notebook) -> None:
        frame = self._tab(
            tabs,
            "Derivatives",
            "Label map or map on the grid of a converted NIfTI → DICOM. Templates: "
            "docs/examples/ (dcmqi format).",
        )
        kinds = ttk.Frame(frame)
        kinds.grid(row=1, column=1, sticky="w", padx=4)
        for value, text in DERIVATIVES.items():
            ttk.Radiobutton(kinds, text=text, value=value, variable=self.derivative).pack(
                side="left", padx=(0, 12)
            )
        self._path_row(frame, 2, "source_nifti", "Source NIfTI", _NIFTI)
        self._path_row(frame, 3, "derivative", "Label map / map", _NIFTI)
        self._path_row(frame, 4, "template", "Template (JSON)", _JSON)
        self._path_row(frame, 5, "derived_output", "Output", "output")
        self._run_button(frame, 6, "Write DICOM", self.derive)

    # --- commands --------------------------------------------------------------------------

    def value(self, key: str) -> str:
        return self.fields[key].get().strip()

    def _missing(self, *keys: str) -> bool:
        empty = [k for k in keys if not self.value(k)]
        if empty:
            self._show_error("Please fill in every field.")
        return bool(empty)

    def convert(self) -> None:
        if self._missing("source"):
            return
        source, dataset, full_name = (
            self.value("source"),
            self.value("dataset"),
            self.full_name.get(),
        )

        def task(progress, show) -> tuple[list[str], bool]:
            result = api.run(
                source,
                dataset or None,
                full_name=full_name,
                progress=progress,
                _scanned=lambda summary, update: show(report.update_lines(summary, update)),
            )
            return report.conversion_lines(result.conversion), result.ok

        self._start("Convert", task)

    def rebuild(self) -> None:
        compare = self.compare_after.get()
        keys = ("rebuild_nifti", "rebuild_output") + (("rebuild_original",) if compare else ())
        if self._missing(*keys):
            return
        nifti, output = self.value("rebuild_nifti"), self.value("rebuild_output")
        original = self.value("rebuild_original")

        def task(progress, show) -> tuple[list[str], bool]:
            lines = report.dcm_lines(api.to_dcm(nifti, output))
            if not compare:
                return lines, True
            show(lines)
            result = api.compare(original, output)
            return report.compare_lines(result), result.identical

        self._start("Rebuild", task)

    def compare(self) -> None:
        if self._missing("original", "rebuilt"):
            return
        original, rebuilt = self.value("original"), self.value("rebuilt")

        def task(progress, show) -> tuple[list[str], bool]:
            result = api.compare(original, rebuilt)
            return report.compare_lines(result), result.identical

        self._start("Compare", task)

    def derive(self) -> None:
        if self._missing("source_nifti", "derivative", "template", "derived_output"):
            return
        kind = self.derivative.get()
        args = [self.value(k) for k in ("source_nifti", "derivative", "template", "derived_output")]

        def task(progress, show) -> tuple[list[str], bool]:
            if kind == "seg":
                return report.seg_lines(api.to_seg(*args)), True
            if kind == "pmap":
                return report.pmap_lines(api.to_pmap(*args)), True
            return report.mr_lines(api.to_mr(*args)), True

        self._start(DERIVATIVES[kind], task)

    # --- running ---------------------------------------------------------------------------

    def _start(self, title: str, task: Task) -> None:
        self._clear()
        self._write([f"{title}…"])
        self._set_busy(True)
        self.status.set("Working…")
        self.bar.configure(mode="indeterminate")
        self.bar.start(15)
        self.worker.start(task)
        self.root.after(POLL_MS, self._poll)

    def _poll(self) -> None:
        for message in self.worker.messages():
            match message:
                case Progress(desc, n, total):
                    self.status.set(desc or "Working…")
                    if total:
                        self.bar.stop()
                        self.bar.configure(mode="determinate", maximum=total, value=n)
                case Lines(lines):
                    self._write(lines)
                case Done(lines, ok):
                    self._write(lines)
                    self._finish("Done." if ok else "Done, with problems (see above).")
                    return
                case Failed(text):
                    self._write([f"Error: {text}"])
                    self._finish("Failed.")
                    self._show_error(text)
                    return
        self.root.after(POLL_MS, self._poll)

    def _finish(self, status: str) -> None:
        self.bar.stop()
        self.bar.configure(mode="determinate", value=0)
        self.status.set(status)
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        for button in self._buttons:
            button.state(["disabled"] if busy else ["!disabled"])

    def _clear(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

    def _write(self, lines: list[str]) -> None:
        self.output.configure(state="normal")
        for line in lines:
            self.output.insert("end", line + "\n")
        self.output.configure(state="disabled")
        self.output.see("end")

    def text(self) -> str:
        """The output pane's content."""
        return self.output.get("1.0", "end-1c")

    def _show_error(self, text: str) -> None:
        messagebox.showerror(TITLE, text, parent=self.root)

    def close(self) -> None:
        if self.worker.busy and not messagebox.askyesno(
            TITLE,
            "A command is still running. Quit anyway? The series it is working on may be "
            "left unfinished.",
            parent=self.root,
        ):
            return
        self.root.destroy()
