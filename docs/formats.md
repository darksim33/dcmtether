# Formats

## Dataset layout

```
dataset/
├── manifest.sqlite                 scan and conversion records
├── README.md                       identifying-data warning, file name entities
├── logs/dcmtether_<time>.jsonl     structured log (no PHI)
├── derivatives/merged/sub-…/…      merged series (`merge`; NIfTI + sidecar + tether;
│                                   with --in-place/--replace beside the series instead)
└── sub-<subject>/ses-<session>/
    ├── anat/<stem>.nii.gz          voxels (raw stored values, closest RAS)
    ├── anat/<stem>.json            human-readable sidecar (BIDS-style keys)
    ├── anat/<stem>.tether.json     everything needed to rebuild the DICOM
    └── dwi/<stem>.*                the same for diffusion series and maps
```

## File names

BIDS-inspired, not BIDS-compliant:
`sub-<subject>_ses-<session>[_acq-<acq>]_ser-<NN>[_<XX>][_chunk-<n>]_<suffix>`.

| Entity | Content |
|---|---|
| `sub-` | first 5 letters of the family name + first 5 of the given name (ASCII, CamelCase), or family + given + middle name in full, digits included, with `--full-name` (one form per dataset) — **identifies the patient** |
| `ses-` | StudyDate `YYYYMMDD`; `THHMM` from StudyTime only when one subject has several studies that day |
| `acq-` | SeriesDescription (else ProtocolName) reduced to letters/digits, camelCase, ≤24 chars |
| `ser-` | SeriesNumber, at least two digits |
| `_XX` | counter `01`, `02`, … only for series that would otherwise share a name (e.g. same SeriesDescription and SeriesNumber), in manifest order |
| `chunk-` | one per image plane when a series holds several (e.g. localizers), InstanceNumber order |
| suffix | `dwi` for diffusion series; `ADC`/`eADC`/`FA`/`TRACE` for diffusion maps calculated by the scanner; else `T1w`/`T2w`/`PDw`/`FLAIR` from keywords in the `acq-` text or from TR/TE/TI; else `MR` |

Diffusion maps are recognised by an ImageType value (`ADC`, `EADC`, `FA`, `TRACEW`, `TRACE`),
else by keywords in the `acq-` text (`adc`, `apparent diffusion`, `trace`, `tracew`,
`fractional anisotropy`; a bare `fa` only next to `dti`, `dwi`, `diff`, `ivim` or `tensor`,
because it also means flip angle). They are stored in `dwi/`. A text naming several maps is not
used.

Diffusion series are recognised by b-values that differ between volumes, else (e.g. a series
with a single b-value) by the ImageType value `DIFFUSION`, else by the keywords `dwi`, `dti`,
`ivim` or `diff` (not `difference`) in the `acq-` text. Map rules are checked first, since
scanner maps carry `DIFFUSION` too.

Name clashes are never resolved silently:

- Different PatientIDs giving the same `sub-` label (e.g. phantoms or study codes in
  PatientName) get a two-digit number appended, `sub-PhantQa01`, `sub-PhantQa02`, …, in the
  manifest order of each patient's first series, reported as a warning. A patient converted
  before the clash appeared keeps its label; its number is not reused.
- Series sharing a file name are numbered (`_XX`) and reported as a warning. Names that
  differ only in upper/lower case count as the same name.
- Session label clashes (several studies of one subject with the same StudyDate and StudyTime)
  make the affected series fail with a reason.

A converted series keeps its names for good. They are stored in the manifest, so later runs
do not need its source files again and do not rename it when naming rules change. Series
that join a name later get the next free `_XX`; a series converted without a counter keeps
its bare name and counts as the first.

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

A merged tether (`merge`) has the same format. It holds the instances of several series:
`source` describes the first and adds `n_series`; `file_name` starts with the series' folder
relative to a common one; series *k*'s volumes follow those of the series before it.

## Sidecar (`<stem>.json`)

Readable BIDS-style keys (times in seconds) — not used to rebuild the DICOM, no patient
identifiers. `DcmTether` holds tool information: `Version`, `SuffixRule`, `AcqSource`
(SeriesDescription/ProtocolName), `Duplicate` (the `_XX` counter), `DiffusionSource`, `Chunk`,
for 4D series `VolumeKey` (the DICOM attribute that tells the volumes apart), and
`PerVolume` (values that differ between volumes, and diffusion values; see below).

`PerVolume` uses the same keys and units as the top level: times are in seconds, and the
echo time of an enhanced series (`EffectiveEchoTime`) is listed as `EchoTime`. Attributes
without a sidecar key (`EchoNumbers`, `TriggerTime`, diffusion values, …) are listed under
their DICOM keyword, unchanged. Sidecars written before this rule may list `EchoTime` or
`InversionTime` in milliseconds when that attribute was the volume key; they are not
rewritten.

Diffusion b-values and gradients come from `DiffusionBValue (0018,9087)` /
`DiffusionGradientOrientation (0018,9089)`; when those are missing (older classic exports), from
the vendor private tags, found by private creator:

| Vendor | b-value | Gradient |
|---|---|---|
| Siemens | `SIEMENS MR HEADER` (0019,xx0C) | (0019,xx0E), patient LPS |
| GE | `GEMS_PARM_01` (0043,xx39) first value, 10⁹ offset removed | `GEMS_ACQU_01` (0019,xxBB–xxBD), scanner frame |
| Philips | `Philips Imaging DD 001` (2001,xx03) | `Philips MR Imaging DD 001` (2005,xxB0–xxB2), RL/AP/FH |

`DiffusionSource` then names the tags. GE and Philips gradients are not in patient LPS and are
listed unconverted under `DiffusionGradientVendor` instead of
`DiffusionGradientOrientation`. GE values are used only when the gradient tags are present too.
Diffusion values are always under `PerVolume`, one entry per volume, also when they are the
same for every volume or the series has a single volume: one series per IVIM b-value gets
`"PerVolume": {"DiffusionBValue": [800.0], "DiffusionGradientOrientation": [[1.0, 0.0, 0.0]]}`.
Images without a gradient (b0, trace) have `null`. If the DICOM stores a b-matrix, it is listed
per volume as `DiffusionBMatrix`, the upper triangle `[Bxx, Bxy, Bxz, Byy, Byz, Bzz]` (`null` where
a volume has none, absent if no volume has one). It comes from `DiffusionBValueXX`–`ZZ`
(0018,9602–9607) in enhanced MR, else from Siemens `SIEMENS MR HEADER` (0019,xx27); GE and Philips
store none. Values are as stored, not rotated. Only diffusion series (suffix `dwi` or a
diffusion map) get them: Philips writes a zero b-value into other images too.

## Manifest (`manifest.sqlite`, schema 5)

| Table | Content |
|---|---|
| `meta` | schema/tool version, source root, scan time, `subject_names` (`brief`/`full`, set on the first conversion) |
| `series` | one row per SeriesInstanceUID: id, status (`supported`/`split`/`unsupported`), reasons, equipment, geometry summary, conversion status and error, and the names it was converted under (subject and session labels, file stem, `_XX` counter, PatientID, subject label before clash numbering), `merged_into` (`merge --replace`: the series whose output holds this one's volumes) |
| `instances` | one row per file (path relative to the source root) |
| `outputs` | one row per NIfTI: series id, chunk, NIfTI/tether/sidecar paths, number of volumes |
| `skipped_files` | files that are not DICOM images, with the reason |

Older manifests are upgraded in place when opened. The manifest holds UIDs, PatientIDs and source paths: it is as sensitive as the DICOM headers.

## Log

`logs/dcmtether_<time>.jsonl`: one JSON object per line, written by every command that works
on the dataset. `<time>` is the UTC start of the command (`YYYYMMDDTHHMMSS`; `_2`, `_3`, …
are added within the same second). `run`, `scan` and `to-nii` start a file each; `to-dcm`,
`to-seg`, `to-pmap`, `to-mr` and `merge` append to the newest one, or start one if there is none. A
`logs/dcmtether.jsonl` from before the names had a timestamp is left as it is. Fields come from a fixed allowlist (`dcmtether.log.ALLOWED_FIELDS`): `time`,
`version`, `command`, `event`, `series_id`, `series_ids`, `status`, `reason`, counts (`count`, `outputs`,
`n_files`, `n_series`, `n_converted`, `n_failed`, …), lists of series ids (`new`,
`rescanned`, `unchanged`, `changed_after_conversion`, `missing`) and `kind`. Events: `start`,
`end`, `error`, `scanned`, `manifest_updated`, `series_classified`, `series_converted`,
`series_failed`, `written`, `merged`, `not_merged`, `replaced`, `not_replaced`.
