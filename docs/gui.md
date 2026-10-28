# Desktop app

A small window for the same commands as the CLI: convert a DICOM folder, rebuild the DICOM
files, compare two series, and write derivatives back as DICOM. Research use only.

## Start

- **Windows, without Python:** download `dcmtether-gui-<version>-win64.zip` from the release,
  unzip it anywhere (a USB stick works too) and start `dcmtether-gui.exe` in the unzipped
  folder. Nothing is installed and no admin rights are needed. To remove it, delete the folder.
  The app is not signed: on the first start Windows SmartScreen may say "Windows protected your
  PC"; choose "More info" → "Run anyway".
- **With Python:** `uv run dcmtether-gui` in a clone of the repository, or
  `uv tool install <wheel>` and then `dcmtether-gui`.

## Tabs

- **Convert** — DICOM folder → dataset (the CLI's `run`). Leave the dataset folder empty to
  write the dataset into the DICOM folder itself. "Full patient name in subject labels" is the
  `--full-name` option; it is fixed for a dataset on its first conversion. Safe to run again:
  new series are added, failed ones retried, converted ones left alone.
- **Rebuild** — a converted NIfTI (with its `*.tether.json` next to it) → the original DICOM
  files in the output folder (`to-dcm`), then, if ticked, a comparison with the original folder
  (`compare`).
- **Compare** — two DICOM folders, element by element.
- **Derivatives** — a label map or map on the grid of a converted NIfTI → DICOM SEG,
  Parametric Map or derived MR images (`to-seg`, `to-pmap`, `to-mr`). The template is a dcmqi
  JSON file; examples are in [`docs/examples/`](examples/). SEG and Parametric Map write one
  file, derived MR images a folder.

The output pane shows what the CLI would print: counts, and series by manifest id with the
reason when something was not converted. The status bar shows progress.

## Privacy

The window never shows patient names, UIDs or paths, and the app stores nothing: no settings,
no recent folders. The datasets it writes do identify patients (file names are built from
PatientName and StudyDate), exactly as with the CLI — never share them as they are.

## Limits

- A running command cannot be cancelled. Closing the window asks first; the series being
  converted at that moment may be left unfinished.
- One command at a time.
