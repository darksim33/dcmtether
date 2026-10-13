# Changelog

All notable changes to dcmtether are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break the CLI or
file formats).

## [Unreleased]

## [0.1.0] - 2026-10-12

First release. Research use only — not for clinical or diagnostic use.

### Added

- `run SOURCE [-o DATASET]`: scan and convert in one step, safe to run again. Manifest ids
  stay stable, new series are added, series not converted yet are rescanned and failed ones
  retried; converted series are never touched. The dataset defaults to SOURCE itself.
- `scan SOURCE DATASET` and `to-nii DATASET`: the two halves of `run`.
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
  (`logs/dcmtether.jsonl`) holding ids, keywords and counts only.
- Console output never names patients, UIDs or paths; series are named by manifest id.
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
