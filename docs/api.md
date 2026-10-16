# Python API

Every command of the CLI is also a function. The functions do exactly what the commands do (the
CLI calls them) and return what the CLI prints as objects.

```python
import dcmtether as dt

result = dt.run("/path/to/dicom", "/path/to/dataset")
result.converted  # manifest ids, e.g. [1, 2, 4]
result.failed  # e.g. {3: "PatientName is missing"}

rebuilt = dt.to_dcm("dataset/sub-…/….nii.gz", "rebuilt/")
dt.compare("/path/to/dicom/series", "rebuilt/").identical  # True
```

Only the names listed on this page (`dcmtether.__all__`) are the API. Modules such as
`dcmtether.forward` or `dcmtether.manifest` are internal and may change.

## Rules that hold for every function

- **Paths** may be `str` or `pathlib.Path`.
- **A series that cannot be converted does not raise.** It is listed in the result with its
  reason, as on the console. Errors that stop a command raise a `DcmtetherError`.
- **Messages are safe.** Exception messages and reasons never contain patient names, UIDs,
  dates or paths; series are named by manifest id. Result objects do hold the paths that were
  written.
- **Logging.** Functions that work on a dataset write the same records to
  `DATASET/logs/dcmtether_<time>.jsonl` as the CLI ([formats](formats.md#log)). Pass
  `log=False` to write none.
- **Progress.** `scan`, `to_nii`, `run` and `merge_series` take `progress=`, a callable used like `tqdm`
  (`progress(iterable, total=…, desc=…, unit=…)`), e.g. `progress=tqdm.tqdm`. Default: none.
- Existing files are never overwritten. Research use only.

## Converting

### `run(source, output=None, *, full_name=False, log=True, progress=None)` → `RunResult`

Scan `source` and convert it into a dataset; safe to call again ([usage](usage.md#converting)).
The dataset is `output`, or `source` itself.

```python
result = dt.run("dicom/", "study/", full_name=True)
if not result.ok:
    for series_id, reason in result.failed.items():
        print(series_id, reason)
```

`RunResult`:

| Field | |
|---|---|
| `scan` | `ScanSummary` of this scan |
| `new`, `rescanned`, `unchanged` | series ids, by what the scan meant for the manifest |
| `changed_after_conversion` | converted series whose source files differ now (not reconverted) |
| `missing` | series in the manifest that the scan no longer found (kept) |
| `conversion` | `ConversionResult` |
| `converted`, `failed`, `ok` | shortcuts to the same fields of `conversion` |

### `scan(source, dataset, *, log=True, progress=None)` → `ScanSummary`

Index and classify the series under `source` (headers only) and write the manifest to
`dataset`. Raises `ManifestError` if `dataset` already has one.

`ScanSummary`: `n_files`, `n_instances` (DICOM files), `n_skipped`, `skipped` (reason → number
of files), `series` and `count(status)`. Each entry of `series` is a `SeriesInfo`: `id`,
`status` (a `Status`: `SUPPORTED`, `SPLIT` or `UNSUPPORTED`), `reasons`, `n_instances` and
`body_part` (BodyPartExamined, or `None`).

```python
summary = dt.scan("dicom/", "study/")
for s in summary.series:
    if s.status is dt.Status.UNSUPPORTED:
        print(s.id, "; ".join(s.reasons))
```

### `to_nii(dataset, *, full_name=False, log=True, progress=None)` → `ConversionResult`

Convert every scanned series of `dataset` that is not converted yet; failed ones are retried.

`ConversionResult`: `outcomes` (one `Outcome` per attempted series: `series_id`, `ok`,
`message`, `outputs`, `warning`), and from them `converted` (ids), `failed` (id → reason),
`warnings` (id → text, e.g. a numbered duplicate name) and `ok`.

### `merge_series(dataset, ids=None, *, log=True, progress=None)` → `MergeResult`

Merge series of one measurement (e.g. one series per IVIM b-value) into one 4D NIfTI under
`dataset/derivatives/merged/` (experimental; see [usage](usage.md)). Without `ids`, groups are
found automatically. `MergeResult`: `groups`, `merged` and `skipped` (lists of `MergedGroup`:
`series_ids`, `n_volumes`, `path`, `skipped` (the reason a group was not written),
`tether_path`, `no_tether` (the reason no `*.tether.json` was written) and `rule`
(`"siemens-xa-split"`, `"diffusion-protocol"` or `"ids"`)).
Raises `MergeError` if the series named in `ids` cannot be merged.

## Back to DICOM

### `to_dcm(nifti, output, *, log=True)` → `DcmResult`

Rebuild the DICOM files of a converted NIfTI from it and the `*.tether.json` next to it, into
the folder `output`. `DcmResult`: `paths` and `transfer_syntax_change` (`(source, written)`
names for lossless compressed sources, else `None`).

### `compare(original, rebuilt)` → `Report`

Compare two DICOM series (folders) element by element, paired by SOPInstanceUID. `Report`:
`identical`, `n_compared`, `differences` and `notes` (lists of `Difference` with `instance`,
`where`, `kind`), and `lines()` for printing.

## Derivatives

`metadata` is a dcmqi template: a path to the JSON file or the loaded `dict`
([examples](examples/)). The derivative must lie on the grid of `nifti`, a converted source.

### `to_seg(nifti, labels, metadata, output, *, log=True)` → `SegResult`

Write the label map `labels` as a DICOM SEG file `output`. `SegResult`: `path`,
`segment_numbers` (label value → segment number) and `n_sources`.

### `to_pmap(nifti, map, metadata, output, *, log=True)` → `PmapResult`

Write `map` (e.g. ADC, QSM) as a DICOM Parametric Map file `output`. `PmapResult`: `path`,
`dtype` and `n_sources`.

### `to_mr(nifti, derivative, metadata, output, *, log=True)` → `DerivedMRResult`

Write a label map (SEG template) or map (parametric map template) as classic derived MR
images into the folder `output`, for PACS without SEG / Parametric Map support.
`DerivedMRResult`: `paths`, `kind` (`"labels"` or `"map"`), `max_error` and `rounded`
(the error comes only from rounding the rescale values, not from quantizing a float map).

```python
template = {"SeriesNumber": "900", "QuantityValueCode": {...}, "MeasurementUnitsCode": {...}}
dt.to_pmap(nifti, "adc.nii.gz", template, "adc.dcm")
```

## Errors

| Exception | Raised when |
|---|---|
| `DcmtetherError` | base class of all of the following |
| `ManifestError` | no manifest, a manifest that already exists, another SOURCE, a SOURCE with `sub-*` folders on the first in-place run, `full_name` on a short-label dataset |
| `ConversionError` | a series cannot be converted (normally reported per series, not raised) |
| `ReverseError` | `to_dcm`: the NIfTI does not match its tether, or output files exist |
| `MergeError` | `merge_series`: named series that are not converted, split, or not on one grid |
| `TetherError` | a `*.tether.json` is unreadable, foreign, too old or damaged |
| `DerivedError` | `to_seg`, `to_pmap`, `to_mr`: bad template, derivative off the grid, unsupported source, output exists |

A path that does not exist raises `FileNotFoundError` (`NotADirectoryError` for a file where a
folder is needed). `__version__` holds the installed version.
