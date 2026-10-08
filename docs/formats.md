# Formats

## Dataset layout

```
dataset/
├── manifest.sqlite                 scan and conversion records
├── README.md                       identifying-data warning, file name entities
├── logs/dcmtether.jsonl            structured log (no PHI)
└── sub-<subject>/ses-<session>/
    ├── anat/<stem>.nii.gz          voxels (raw stored values, closest RAS)
    ├── anat/<stem>.json            human-readable sidecar (BIDS-style keys)
    ├── anat/<stem>.tether.json     everything needed to rebuild the DICOM
    └── dwi/<stem>.*                the same for diffusion series
```

## File names

BIDS-inspired, not BIDS-compliant:
`sub-<subject>_ses-<session>[_acq-<acq>]_ser-<NN>[_chunk-<n>]_<suffix>`.

| Entity | Content |
|---|---|
| `sub-` | first 5 letters of the family name + first 5 of the given name (ASCII, CamelCase) — **identifies the patient** |
| `ses-` | StudyDate `YYYYMMDD`; `THHMM` from StudyTime only when one subject has several studies that day |
| `acq-` | ProtocolName (else SeriesDescription) reduced to letters/digits, camelCase, ≤24 chars |
| `ser-` | SeriesNumber, at least two digits |
| `chunk-` | one per image plane when a series holds several (e.g. localizers), InstanceNumber order |
| suffix | `dwi` for diffusion; else `T1w`/`T2w`/`PDw`/`FLAIR` from keywords in the `acq-` text or from TR/TE/TI; else `MR` |

Name clashes are never resolved silently: the affected series fail with a reason.

## NIfTI

gzip, raw stored values in the source data type, reoriented to the closest RAS orientation by
axis permutation/flip only (never resampled; oblique rotation stays in the affine), sform =
qform (scanner). A uniform DICOM rescale is written to `scl_slope`/`scl_inter`; if it differs
between volumes of a 4D series, the header stays unscaled and the values are listed in the
sidecar. 4D series (multi-echo, diffusion, dynamics) are one 4D NIfTI.

## `*.tether.json` (format version 3)

The DICOM headers as [DICOM JSON](https://dicom.nema.org/medical/dicom/current/output/chtml/part18/chapter_F.html),
without pixel data:

| Key | Content |
|---|---|
| `format_version`, `tool` | 3; tool name and version |
| `source` | series/SOP class/transfer syntax UIDs, number of instances |
| `file_meta`, `shared` | elements equal in every instance |
| `original_strings` | original text of elements a JSON round trip would change (DS/IS formatting, padding) |
| `instances[]` | per file: `file_name` (relative to the series folder), `index` (`[slice]` or `[slice, volume]`) or for multi-frame files `frames` (one index per frame), its own `file_meta`, `elements`, `original_strings`, optional `pixel_data_trailing` |
| `pixel` | `dtype`, `shape_dicom`, `pixel_data_vr`, `orientation.ornt_transform` (DICOM order → NIfTI) |

Indices always refer to DICOM order (rows, columns, slices along the slice normal[, volumes]),
never to NIfTI order. Version 2 files are still read.

## Sidecar (`<stem>.json`)

Readable BIDS-style keys (times in seconds) — not used to rebuild the DICOM, no patient
identifiers. `DcmTether` holds tool information: `Version`, `SuffixRule`, `AcqSource`
(ProtocolName/SeriesDescription), `Chunk`, and for 4D series `VolumeKey` (the attribute that
tells the volumes apart) and `PerVolume` (values that differ between volumes).

## Manifest (`manifest.sqlite`, schema 3)

| Table | Content |
|---|---|
| `meta` | schema/tool version, source root, scan time |
| `series` | one row per SeriesInstanceUID: id, status (`supported`/`split`/`unsupported`), reasons, equipment, geometry summary, subject/session labels, conversion status and error |
| `instances` | one row per file (path relative to the source root) |
| `outputs` | one row per NIfTI: series id, chunk, NIfTI/tether/sidecar paths, number of volumes |
| `skipped_files` | files that are not DICOM images, with the reason |

The manifest holds UIDs and source paths: it is as sensitive as the DICOM headers.

## Log

`logs/dcmtether.jsonl`: one JSON object per line, appended by every command that works on the
dataset. Fields come from a fixed allowlist (`dcmtether.log.ALLOWED_FIELDS`): `time`,
`version`, `command`, `event`, `series_id`, `status`, `reason`, counts (`count`, `outputs`,
`n_files`, `n_series`, `n_converted`, `n_failed`, …), lists of series ids (`new`,
`rescanned`, `unchanged`, `changed_after_conversion`, `missing`) and `kind`. Events: `start`,
`end`, `error`, `scanned`, `manifest_updated`, `series_classified`, `series_converted`,
`series_failed`, `written`.
