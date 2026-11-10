# Development install and Windows build

> **TL;DR** — Set up a development checkout with `uv`, run the checks and start the CLI or the
> desktop app from source (**Setup**). Build the portable Windows app with one PowerShell
> script, which runs a self-test and writes a zip to `dist/` (**Build**). **Check the build**
> covers the manual test before a release; **Troubleshooting** covers a failing self-test and
> missing files in the frozen app.

---

## Requirements

| Tool | Notes |
|---|---|
| Windows 10 or 11, 64-bit | Only needed for the Windows build; development works on any OS. |
| [uv](https://docs.astral.sh/uv/) | Installs Python 3.12+ and every dependency. |
| Git | For the clone. |
| PowerShell | Windows PowerShell 5.1 or PowerShell 7; the build script runs in either. |

## Setup

```bash
git clone https://github.com/darksim33/dcmtether
cd dcmtether
uv sync
```

`uv sync` creates `.venv/` with the runtime and the `dev` dependency group (pytest, ruff,
dcm2niix, dicom-validator).

Run the checks:

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
```

The first test run downloads public test data and the DICOM standard into `.cache/`
([test data](test-data.md)). Tests only use synthetic or public data.

Start from source:

```bash
uv run dcmtether --help
uv run dcmtether-gui
```

## Build

From the repository root:

```bash
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

Output: `dist/dcmtether-gui-<version>-win64.zip`, containing the folder `dcmtether-gui/` with
`dcmtether-gui.exe`. The version comes from `pyproject.toml`.

What the script does:

1. `uv sync --no-dev`: drops the dev tools, so they cannot end up in the app.
2. Runs PyInstaller (`uv run --no-dev --with pyinstaller`, not a project dependency) on
   `packaging/dcmtether-gui.spec`. The spec builds the icon from `docs/assets/logo.png` and
   collects the package metadata and data files that the import analysis cannot see.
3. Starts `dist/dcmtether-gui/dcmtether-gui.exe --self-test`. The app has no console, so the
   result goes to `self-test.txt` next to the exe; the script prints it and stops if it failed.
4. Zips `dist/dcmtether-gui/`.
5. `uv sync`: restores the development environment.

`build/` and `dist/` are git-ignored.

| File | Role |
|---|---|
| `scripts/build_windows.ps1` | The build: sync, PyInstaller, self-test, zip. |
| `packaging/dcmtether-gui.spec` | What goes into the app: metadata, data files, hidden imports, excludes. |
| `packaging/launch_gui.py` | Entry script (PyInstaller needs a file, not an entry point). |
| `src/dcmtether/gui/__init__.py` | `self_test()`: package metadata, highdicom data, window icons, theme, JPEG decoders, Tcl/Tk. |

## Check the build

Before a release, on a machine without Python if possible:

1. Unzip the archive anywhere and start `dcmtether-gui.exe`. The app is unsigned: SmartScreen
   may warn on the first start ("More info" → "Run anyway").
2. Run Convert → Rebuild with "Compare" ticked on a synthetic or public series; the result must
   be "Identical".
3. Write one derivative per kind on the Derivatives tab.

Never test the build with patient data you are not allowed to process on that machine; the
app writes its datasets with PatientName and StudyDate in the file names.

Attach the zip to the GitHub release; the [desktop app guide](../gui.md) points users there.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Script stops before running: "running scripts is disabled" | Call it with `-ExecutionPolicy Bypass`, as shown above. |
| "self-test failed" with "… metadata missing" | A package found through its metadata (entry points, `importlib.metadata`) is not collected. Add it to the `copy_metadata` list in the spec. |
| "self-test failed" with "… data files missing" | Add `collect_data_files("<package>")` to `datas` in the spec. |
| "no pylibjpeg decoder for …" | A pylibjpeg plugin is missing: check `hiddenimports` and the `pylibjpeg*` metadata in the spec. |
| "Tcl/Tk not usable" | PyInstaller did not bundle Tcl/Tk: build with a uv-managed or python.org Python, which ship it. |
| The exe starts and closes at once | Run the frozen app's code from source (`uv run dcmtether-gui`) to see the error; for the frozen app, temporarily set `console=True` in the spec's `EXE(...)`. |
| Development tools missing after a failed build | The final `uv sync` did not run; run `uv sync`. |

For a new runtime dependency, add any metadata or data files it loads at run time to the spec,
and a check to `self_test()` if missing files would only show up when the feature is used.
