# Roadmap

Ideas for later versions. Nothing here is scheduled or promised; the order is not a priority.
What works today is described in [supported inputs](supported-inputs.md).

## More inputs

- **Siemens mosaic**: unpack the tiles into slices, and pack them again for `to-dcm`.
- **Volumes that only InstanceNumber tells apart** (e.g. GE fMRI): use the instance order as a
  last-resort volume key.
- **Enhanced MR files with several image planes**: one multi-frame file whose frames lie in
  several planes, written as one NIfTI per plane. Needs a tether format that can assign single
  frames of a file to different outputs.

## More of the dataset layout

- **More datatype folders and suffixes.** Today there are `anat/` and `dwi/`. Candidates,
  following BIDS names where they fit:
  - `func/` (fMRI, `bold`), `fmap/` (field maps), `perf/` (ASL);
  - QSM/SWI: multi-echo GRE as `MEGRE` with `part-mag` / `part-phase`, scanner-reconstructed
    maps as `Chimap` / `swi`.
- **`.bval` / `.bvec` files** next to diffusion series and merged diffusion files, with the
  gradients rotated into the NIfTI frame (GE and Philips gradients are listed unconverted in
  the sidecar today).
- **Siemens B-matrix**: the upper triangle (Bxx, Bxy, Bxz, Byy, Byz, Bzz) from SIEMENS MR
  HEADER `(0019,xx27)`, per volume in the sidecar (`DcmTether.PerVolume`) and, like the
  `.bvec`, in a file of its own next to the NIfTI (one row per volume); also for merged files.

## Derivatives

- Derivatives of **4D sources**.
- Derivatives on a **grid other than the source's** (e.g. a cropped or resampled label map).
- **Batch derivatives**: convert everything under `dataset/derivatives/<tool>/` to SEG,
  Parametric Map or derived MR in one go.

## Working with whole datasets

- **Dataset-wide `to-dcm`**: rebuild every converted series into `OUTPUT/<series id>/`,
  optionally followed by `compare`.
- **BIDS exporter** for subsets that need strict BIDS compliance.
- **Anonymized export**: a de-identified copy of a dataset that can be shared. Pseudonymous
  `sub-` / `ses-` labels (the mapping stays local), DICOM PS3.15 Annex E Basic Profile applied
  to tethers, sidecars and derivatives, consistent UID remapping and date shifting per
  subject, no manifest or source paths. The original dataset stays untouched; an anonymized
  export can by definition not be rebuilt into the original DICOM.

## Packaging

- **Docker image** that also carries dcm2niix and dciodvfy for checks.
