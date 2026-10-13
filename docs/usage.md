# Usage

All commands: `uv run dcmtether --help`, `uv run dcmtether COMMAND --help`. Output never shows
patient names, UIDs or file paths; series are named by their **manifest id** (`series #3`).
Existing files are never overwritten. Research use only.

Every command is also a Python function with the same behaviour: see [Python API](api.md).

## Workflow

```
DICOM folder ──run──▶ dataset (NIfTI + *.tether.json + sidecar, manifest.sqlite)
                         │
                         ├── to-dcm ──▶ DICOM identical to the original ──compare──▶ check
                         │
your tool ──▶ label map / ADC map on the NIfTI grid
                         ├── to-seg  ──▶ DICOM SEG
                         ├── to-pmap ──▶ DICOM Parametric Map
                         └── to-mr   ──▶ derived MR images (for PACS without SEG/PM support)
```

## Converting

### `run SOURCE [--output/-o DATASET] [--full-name/-f] [--verbose/-v]`

Scans SOURCE (headers only) and converts every convertible series into DATASET, which is SOURCE
itself unless `--output` names another directory. In-place, the dataset's own entries
(`manifest.sqlite`, `README.md`, `sub-*/` and its log files `logs/dcmtether*.jsonl`) are left
out of later scans. A SOURCE that already has `sub-*` folders of its own is therefore refused
on the first in-place run ("SOURCE already has sub-* folders (N); use --output"); a source
folder called `logs` is read as usual.

Safe to run again on the same SOURCE and DATASET: new series are added (with new ids), series that are not converted
yet are rescanned, failed ones are retried. Converted series are never touched; if their source
files changed, that is reported (delete their outputs and manifest entry to redo them). A
different SOURCE for an existing dataset is refused. `--verbose` lists every series.

`--full-name` builds subject labels from the full family, given and middle name, digits
included (`sub-MuellerThomas`, `sub-Phantom2Qa`), instead of the first 5 letters of family and
given name (`sub-MuellThoma`, `sub-PhantQa`). Use it when names differ only after the fifth
letter, in a number or in the middle name; with short labels such patients are numbered as a
label clash. The choice is fixed for a dataset on
its first conversion: later runs keep it without the option, and it is refused for a dataset
already converted with short labels. Full names make paths longer (mind the Windows path
limit). Short options combine: `run SOURCE -fv`.

Exit code 1 if any series failed.

### `scan SOURCE DATASET` and `to-nii DATASET [--full-name/-f] [--verbose/-v]`

The two halves of `run`: `scan` creates the manifest (and refuses an existing one), `to-nii`
converts what the manifest marks `supported` or `split`.

Each series ends up in one of three states (see [supported inputs](supported-inputs.md)):
`supported` (one NIfTI), `split` (one NIfTI per image plane, `chunk-<n>`) or `unsupported`
(with a reason; never converted "best effort").

### Progress

In a terminal, `run`, `scan` and `to-nii` show progress bars on stderr (reading headers,
classifying, converting). They show counts only, never file or patient names, and disappear
when done. They are not shown when the output is redirected; set `TQDM_DISABLE=1` to turn them
off in a terminal as well.

## Back to DICOM

### `to-dcm NIFTI OUTPUT`

Rebuilds the DICOM files of one NIfTI from its `*.tether.json` (next to it), element- and
pixel-identical to the original. Files keep their original names relative to the series'
source folder, so use **one OUTPUT folder per series**. Lossless compressed originals are
written uncompressed (Explicit VR Little Endian); the command says so.

### `compare ORIGINAL REBUILT`

Compares two folders instance by instance (paired by SOPInstanceUID): every element's VR and
encoded value, sequences item by item, and the decoded pixels. Exit code 0 if identical.
Notes (not differences) are listed once with a count, e.g. a lossless→uncompressed transfer
syntax change.

## Derivatives

A derivative must lie **on the grid of the source NIfTI** (same shape and affine); other
grids are refused. The source DICOM is rebuilt from the `*.tether.json`, so the original
files are not needed. Derived objects get deterministic UIDs (same input → same UIDs, any
change → new UIDs), reference their source images, and are labelled "RESEARCH USE ONLY".
Only 3D sources are supported so far.

Metadata uses the [dcmqi](https://qiicr.org/dcmqi/) JSON templates; examples:
[`seg-template.json`](examples/seg-template.json), [`pmap-template.json`](examples/pmap-template.json).

### `to-seg NIFTI LABELS METADATA OUTPUT`

Label map → DICOM SEG (binary segmentation). Label values must be integers and described in
`segmentAttributes`; non-consecutive labelIDs are renumbered 1..n (printed). Optional
template extensions: `SegmentAlgorithmVersion`, `SegmentAlgorithmFamilyCodeSequence`.

### `to-pmap NIFTI MAP METADATA OUTPUT`

Map (ADC, QSM, …) → DICOM Parametric Map with quantity, units and method. Values are stored
exactly: float32/float64 as floats; integer maps as 16-bit values whose real-world value
mapping reproduces the NIfTI values (including its scaling). NaN/Inf are refused.

### `to-mr NIFTI DERIVATIVE METADATA OUTPUT`

Fallback for PACS that cannot show SEG or Parametric Maps: one classic MR image per source
slice, built from the source image's header (private tags dropped), `ImageType
DERIVED\SECONDARY`. METADATA decides the kind: a SEG template writes label values as pixels
(exact), a parametric map template writes the map with RescaleSlope/Intercept. Integer maps
keep their stored values; if the scaling does not fit the 16 characters of a DICOM decimal
string, "Rescale rounded to DS precision: max error …" is printed. Float maps are
**quantized to 16 bits** ("Quantized to 16 bits: max error …"). `to-pmap` is the exact
form.

## Logs

Commands that work on a dataset (`run`, `scan`, `to-nii`, and `to-dcm`/`to-seg`/`to-pmap`/
`to-mr` on a dataset NIfTI) write JSON lines to `DATASET/logs/dcmtether_<time>.jsonl` (a new
file per `run`, `scan` and `to-nii`; the other commands append to the newest): time,
version, command, event, manifest series id, safe reasons and counts — never element values,
names, UIDs or paths. See [formats](formats.md#log).

## Privacy

The dataset identifies patients: folder and file names contain the subject label (from
PatientName) and the study date, and `manifest.sqlite`/`*.tether.json` hold full DICOM
headers. Treat it like the source DICOM. Console output and logs are designed to be safe to
share (ids, keywords and counts only).
