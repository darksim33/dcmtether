# Changelog

All notable changes to dcmtether are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break the CLI or
file formats).

## [0.2.0] - 2026-10-23

### Added

- `merge DATASET [--ids …]` / `merge_series` (experimental): merge series of one measurement
  (e.g. one series per IVIM b-value) into one 4D NIfTI under `derivatives/merged/`; groups are
  found automatically (Siemens XA split series: `MFSPLIT` + same SeriesTime, any type;
  single-b-value diffusion series of one protocol) or named by manifest id. The converted
  series stay as they are; a combined `*.tether.json` lets `to-dcm` write all source series
  back from the merged file. `--in-place` writes beside the single series, `--replace` also
  removes them (manifest schema 5: `merged_into`). Volumes are sorted by b-value, then gradient
  direction (in acquisition order), else by SeriesNumber. Shows a progress bar in a terminal
  (`progress=` in the API). Exit code 1 if a named group was not merged or not replaced. The
  sidecar records the grouping rule (`DcmTether.MergeRule`). The API adds `MergeResult`,
  `MergedGroup` and `MergeError`.

- `--body-part/-b` for `scan` and `run`: list every scanned series with its BodyPartExamined;
  `SeriesInfo.body_part` in the Python API.

### Changed

- Licensed under GPL-3.0-or-later (`LICENSE`); `dcmtether --version` shows the copyright
  notice.

- `scan` and `run` ignore `derivatives/merged/` when the dataset lies inside the scanned
  folder. Existing manifests upgrade to schema 5 automatically.

- Diffusion series with a single b-value are now named `dwi` too: from ImageType `DIFFUSION`,
  else the keywords `dwi`, `dti`, `ivim` or `diff` in SeriesDescription/ProtocolName (before:
  only series whose b-values differ between volumes). `ivim` also counts as a diffusion word
  for bare `fa` (FA maps).

### Fixed

- Diffusion b-values and gradients are now written to the sidecar for every diffusion series
  (`dwi` or a diffusion map), always under `DcmTether.PerVolume` (one entry per volume), also
  for a series with a single b-value (e.g. one series per IVIM b-value). Before, they only
  appeared when they told the volumes of a 4D series apart.

## [0.1.0] - 2026-10-12

First release. Research use only — not for clinical or diagnostic use.

### Added

- `run SOURCE [-o DATASET]`: scan and convert in one step, safe to run again. Manifest ids
  stay stable, new series are added, series not converted yet are rescanned and failed ones
  retried; converted series are never touched. The dataset defaults to SOURCE itself.
- `scan SOURCE DATASET` and `to-nii DATASET`: the two halves of `run`.
- Python API: every command as a function (`dcmtether.run`, `scan`, `to_nii`, `to_dcm`,
  `compare`, `to_seg`, `to_pmap`, `to_mr`) returning result objects, with one base class
  for errors (`DcmtetherError`) and `log=False` to skip the dataset log. See `docs/api.md`.
- Header-only scan that classifies every series as `supported`, `split` or `unsupported`,
  always with a reason; nothing is converted "best effort".
- DICOM → NIfTI conversion: raw stored values, closest-RAS reorientation without resampling,
  uniform rescale in `scl_slope`/`scl_inter`, a BIDS-style sidecar and a `*.tether.json`
  (format v3) holding everything needed to rebuild the DICOM.
- Sidecar times are in seconds throughout, including the per-volume values of 4D series
  (`DcmTether.PerVolume`).
- 4D series (multi-echo, diffusion, dynamics) as one 4D NIfTI; series with several image
  planes (localizers) as one NIfTI per plane (`chunk-<n>`).
- Diffusion b-values and gradients from the standard elements, or from Siemens, GE and
  Philips private tags in classic exports.
- Enhanced MR (multi-frame) input, and lossless RLE, JPEG Lossless, JPEG-LS Lossless and
  JPEG 2000 Lossless transfer syntaxes.
- `to-dcm NIFTI OUTPUT`: rebuild the original DICOM from the tether, element- and
  pixel-identical (lossless compressed sources are written uncompressed).
- `compare ORIGINAL REBUILT`: instance-by-instance check of elements and decoded pixels.
- Derivatives on the source NIfTI grid, with deterministic UIDs and references to the source
  images:
  - `to-seg`: label map → DICOM SEG.
  - `to-pmap`: map → DICOM Parametric Map, values stored exactly.
  - `to-mr`: label map or map → classic derived MR images, for PACS without SEG or
    Parametric Map support.
- BIDS-inspired file names:
  `sub-<subject>_ses-<session>[_acq-<acq>]_ser-<NN>[_<XX>][_chunk-<n>]_<suffix>`, in
  `anat/` and `dwi/` folders. `acq-` comes from SeriesDescription (ProtocolName as fallback).
  The suffix (`dwi`, `T1w`, `T2w`, `PDw`, `FLAIR`, `MR`) comes from keywords or TR/TE/TI.
- Diffusion maps calculated by the scanner are stored in `dwi/` as `ADC`, `eADC`, `FA` or
  `TRACE`, recognised by ImageType or by keywords.
- `--full-name`/`-f` for `run` and `to-nii`: subject labels from the full patient name,
  digits included (`sub-MuellerThomas`) instead of 5+5 letters (`sub-MuellThoma`); fixed per dataset.
- Name clashes are numbered and reported as warnings, never resolved silently:
  - Different PatientIDs with the same subject label get `sub-<label>01`, `02`, ….
  - Series that would share a file name get `_01`, `_02`, … after `ser-`. Names that differ
    only in upper/lower case count as the same.
- Converted series are fixed: their names are stored in the manifest, so later runs neither
  need their source files again nor rename them.
- Manifest (`manifest.sqlite`), a dataset README, and a structured JSON-lines log
  (`logs/dcmtether_<time>.jsonl`, one file per `run`, `scan` and `to-nii`) holding ids,
  keywords and counts only.
- Console output never names patients, UIDs or paths; series are named by manifest id.
- `run` without `--output` refuses a SOURCE that already has `sub-*` folders.
- Progress bars on stderr in a terminal (`TQDM_DISABLE=1` turns them off).

### Known limitations

- Not converted (a reason is always given):
  - Siemens mosaic.
  - Lossy compression.
  - Missing slices or uneven spacing.
  - Volumes no attribute tells apart (e.g. GE fMRI).
  - Legacy Converted Enhanced MR.
  - Enhanced MR files whose frames lie in several image planes.
  - Non-MR images.
- Derivatives need a 3D source.
- Only `anat/` and `dwi/` folders so far.
- The dataset identifies patients, through file names, the manifest and the tethers. Do not
  share it as is.

[Unreleased]: https://github.com/darksim33/dcmtether/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/darksim33/dcmtether/releases/tag/v0.1.0
