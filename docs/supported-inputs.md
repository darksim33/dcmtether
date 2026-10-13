# Supported inputs

dcmtether never converts "best effort": every series is classified with a reason, and only
series that round-trip without guessing are converted. Reasons are printed by `scan`/`run` and
stored in the manifest.

## Converted

| Input | Notes |
|---|---|
| MR Image Storage, one regular stack | equal orientation, matrix, pixel spacing; even slice spacing (0.01 mm tolerance) |
| 4D series (multi-echo, diffusion, dynamics) | volumes at identical positions, told apart by EchoNumbers, TemporalPositionIdentifier/Index, DiffusionBValue + gradient, AcquisitionNumber, EffectiveEchoTime, ComplexImageComponent, EchoTime, InversionTime or TriggerTime (or a combination) |
| Diffusion maps calculated by the scanner (ADC, eADC, FA, trace-weighted) | stored in `dwi/` with their own suffix; colour FA maps are not converted |
| Classic diffusion with b-values only in private tags | Siemens, GE, Philips vendor tags (see [formats](formats.md#sidecar-stemjson)) |
| Series with several image planes (localizers) | one NIfTI per plane (`chunk-<n>`) if every plane is a regular stack |
| Enhanced MR (multi-frame) | geometry and per-frame values from the functional groups; one series may span several files |
| Transfer syntaxes | Implicit/Explicit VR Little Endian, Explicit VR Big Endian; lossless RLE, JPEG Lossless, JPEG-LS Lossless, JPEG 2000 Lossless (rebuilt uncompressed) |
| Rescale | uniform within each volume (may differ between volumes) |

## Not converted (reason given)

| Input | Why |
|---|---|
| Lossy JPEG / JPEG-LS near-lossless / lossy JPEG 2000 | decoders may differ: voxels are not reproducible |
| Siemens mosaic | not unpacked yet |
| Missing slices, uneven spacing, sheared stacks, mixed slice thickness | would need resampling or guessing |
| Volumes no attribute tells apart (e.g. GE fMRI) | the volume of each image would be a guess |
| Rescale differing within one volume | NIfTI has one scaling per volume at most |
| Mixed SOP classes, transfer syntaxes, frames of reference or studies in one series | not one image series |
| Non-MR SOP classes, colour images, Legacy Converted Enhanced MR, multi-frame classic MR | out of scope |
| Missing PatientName or StudyDate | file names are built from them |
| Header values NaN/Inf | not storable in JSON |

## Derivatives

`to-seg`, `to-pmap` and `to-mr` need a 3D source (no 4D sources; SEG/Parametric Map: no
enhanced source split over several files; `to-mr`: no enhanced sources) and a derivative on
the source NIfTI grid.
