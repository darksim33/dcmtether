# Test data inventory

Sources:
- `robyoung/dicom-test-files`, pinned in `tests/data/dicom_test_files.json` (below).
- Chris Rorden's **dcm_qa** repositories (dcm2niix validation data), pinned in
  `tests/data/dcm_qa.json` (see [dcm_qa](#dcm_qa)).

Source details for `robyoung/dicom-test-files`:
Table generated with `uv run python scripts/inventory_test_files.py` (MR files only, headers only).

## Summary (Phase 0)

- **Only one real multi-slice MR series:** `dcmqi/paramaps/701-ADCb500/000000–000019.dcm`
  (20 slices, GE, Explicit VR LE, 256 x 256). It is a scanner-derived **ADC map** (`ImageType DERIVED\SECONDARY`), not raw DWI; dcmqi uses it as the Parametric Map source.
  `pm-example-slice.dcm` shares that series. This is the main real-data round-trip case for Phases 2–3.
- **Single-slice MR, many codecs:** WG04 `MR1`–`MR4` (Toshiba, Philips 1024 x 1024, GE, Philips) in
  uncompressed, RLE, JPEG lossless/lossy, JPEG-LS and JPEG 2000 variants; pydicom `MR_small*`
  (byte-order / VR / padding / truncation variants). Useful for transfer-syntax and element round trips (Phase 4).
- **Siemens:** only `pydicom/MR-SIEMENS-DICOM-WithOverlays.dcm` (single slice, overlays, Siemens private group 0029).
- **Enhanced MR:** `pydicom/emri_small*` (10 frames, 64 x 64) has **no functional group sequences**,
  so it is not a realistic enhanced multi-frame case. Phase 4 needs other data (synthetic or TCIA).
- **No** multi-echo, 4D DWI with gradient tables, dynamics, localizers or mosaic series.
- Derivative references for Phase 5: `dcmqi/segmentations/*` (CT source images + SEG),
  `dcmqi/paramaps/pm-example-slice.dcm`, `dcmqi/sr-example/*`.

Geometry edge cases (orientation permutations/flips, oblique, gaps, mixed thickness, per-slice
rescale, localizers) are therefore covered by the synthetic generator in `tests/synthetic.py`.

## MR files

| Files | SOP class | Distinct SOPInstanceUIDs | Layout | Rows x Cols | Manufacturer | Transfer syntaxes |
|---|---|---|---|---|---|---|
| `WG04/J2KI/MR1_J2KI`<br>`WG04/J2KR/MR1_J2KR`<br>`WG04/JLSL/MR1_JLSL`<br>`WG04/JLSN/MR1_JLSN`<br>`WG04/JPLL/MR1_JPLL`<br>`WG04/JPLY/MR1_JPLY`<br>`WG04/REF/MR1_UNC`<br>`pydicom/MR_small.dcm`<br>`pydicom/MR_small_RLE.dcm`<br>`pydicom/MR_small_bigendian.dcm`<br>`pydicom/MR_small_expb.dcm`<br>`pydicom/MR_small_implicit.dcm`<br>`pydicom/MR_small_jp2klossless.dcm`<br>`pydicom/MR_small_jpeg_ls_lossless.dcm`<br>`pydicom/MR_small_padded.dcm`<br>`pydicom/MR_truncated.dcm` | MR Image Storage | 7 | single-frame | 512 x 512 | TOSHIBA_MEC | Explicit VR Big Endian<br>Explicit VR Little Endian<br>Implicit VR Little Endian<br>JPEG 2000 Image Compression<br>JPEG 2000 Image Compression (Lossless Only)<br>JPEG Extended (Process 2 and 4)<br>JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])<br>JPEG-LS Lossless Image Compression<br>JPEG-LS Lossy (Near-Lossless) Image Compression<br>RLE Lossless |
| `WG04/J2KI/MR2_J2KI`<br>`WG04/J2KR/MR2_J2KR`<br>`WG04/JLSL/MR2_JLSL`<br>`WG04/JLSN/MR2_JLSN`<br>`WG04/JPLL/MR2_JPLL`<br>`WG04/JPLY/MR2_JPLY`<br>`WG04/REF/MR2_UNC`<br>`pydicom/MR2_J2KI.dcm`<br>`pydicom/MR2_J2KR.dcm`<br>`pydicom/MR2_UNCI.dcm`<br>`pydicom/MR2_UNCR.dcm` | MR Image Storage | 7 | single-frame | 1024 x 1024 | Philips | Explicit VR Little Endian<br>JPEG 2000 Image Compression<br>JPEG 2000 Image Compression (Lossless Only)<br>JPEG Extended (Process 2 and 4)<br>JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])<br>JPEG-LS Lossless Image Compression<br>JPEG-LS Lossy (Near-Lossless) Image Compression |
| `WG04/J2KI/MR3_J2KI`<br>`WG04/J2KR/MR3_J2KR`<br>`WG04/JLSL/MR3_JLSL`<br>`WG04/JLSN/MR3_JLSN`<br>`WG04/JPLL/MR3_JPLL`<br>`WG04/JPLY/MR3_JPLY`<br>`WG04/REF/MR3_UNC` | MR Image Storage | 7 | single-frame | 512 x 512 | GE MEDICAL SYSTEMS | Explicit VR Little Endian<br>JPEG 2000 Image Compression<br>JPEG 2000 Image Compression (Lossless Only)<br>JPEG Extended (Process 2 and 4)<br>JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])<br>JPEG-LS Lossless Image Compression<br>JPEG-LS Lossy (Near-Lossless) Image Compression |
| `WG04/J2KI/MR4_J2KI`<br>`WG04/J2KR/MR4_J2KR`<br>`WG04/JLSL/MR4_JLSL`<br>`WG04/JLSN/MR4_JLSN`<br>`WG04/JPLL/MR4_JPLL`<br>`WG04/JPLY/MR4_JPLY`<br>`WG04/REF/MR4_UNC` | MR Image Storage | 7 | single-frame | 512 x 512 | Philips | Explicit VR Little Endian<br>JPEG 2000 Image Compression<br>JPEG 2000 Image Compression (Lossless Only)<br>JPEG Extended (Process 2 and 4)<br>JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])<br>JPEG-LS Lossless Image Compression<br>JPEG-LS Lossy (Near-Lossless) Image Compression |
| `WG04/RLE/MR1_RLE` | MR Image Storage | 1 | single-frame | 512 x 512 | TOSHIBA_MEC | RLE Lossless |
| `WG04/RLE/MR2_RLE` | MR Image Storage | 1 | single-frame | 1024 x 1024 | Philips | RLE Lossless |
| `WG04/RLE/MR3_RLE` | MR Image Storage | 1 | single-frame | 512 x 512 | GE MEDICAL SYSTEMS | RLE Lossless |
| `WG04/RLE/MR4_RLE` | MR Image Storage | 1 | single-frame | 512 x 512 | Philips | RLE Lossless |
| `dcmqi/paramaps/701-ADCb500/000000.dcm`<br>`dcmqi/paramaps/701-ADCb500/000001.dcm`<br>`dcmqi/paramaps/701-ADCb500/000002.dcm`<br>`dcmqi/paramaps/701-ADCb500/000003.dcm`<br>`dcmqi/paramaps/701-ADCb500/000004.dcm`<br>`dcmqi/paramaps/701-ADCb500/000005.dcm`<br>`dcmqi/paramaps/701-ADCb500/000006.dcm`<br>`dcmqi/paramaps/701-ADCb500/000007.dcm`<br>`dcmqi/paramaps/701-ADCb500/000008.dcm`<br>`dcmqi/paramaps/701-ADCb500/000009.dcm`<br>`dcmqi/paramaps/701-ADCb500/000010.dcm`<br>`dcmqi/paramaps/701-ADCb500/000011.dcm`<br>`dcmqi/paramaps/701-ADCb500/000012.dcm`<br>`dcmqi/paramaps/701-ADCb500/000013.dcm`<br>`dcmqi/paramaps/701-ADCb500/000014.dcm`<br>`dcmqi/paramaps/701-ADCb500/000015.dcm`<br>`dcmqi/paramaps/701-ADCb500/000016.dcm`<br>`dcmqi/paramaps/701-ADCb500/000017.dcm`<br>`dcmqi/paramaps/701-ADCb500/000018.dcm`<br>`dcmqi/paramaps/701-ADCb500/000019.dcm`<br>`dcmqi/paramaps/pm-example-slice.dcm` | MR Image Storage | 20 | single-frame | 256 x 256 | GE MEDICAL SYSTEMS | Explicit VR Little Endian |
| `pydicom/MR-SIEMENS-DICOM-WithOverlays.dcm` | MR Image Storage | 1 | single-frame | 484 x 484 | SIEMENS | Explicit VR Little Endian |
| `pydicom/emri_small.dcm`<br>`pydicom/emri_small_RLE.dcm`<br>`pydicom/emri_small_big_endian.dcm`<br>`pydicom/emri_small_jpeg_2k_lossless.dcm`<br>`pydicom/emri_small_jpeg_2k_lossless_too_short.dcm`<br>`pydicom/emri_small_jpeg_ls_lossless.dcm` | Enhanced MR Image Storage | 1 | multi-frame (10 frames) | 64 x 64 | — | Explicit VR Big Endian<br>Explicit VR Little Endian<br>JPEG 2000 Image Compression (Lossless Only)<br>JPEG-LS Lossless Image Compression<br>RLE Lossless |

## dcm_qa

Chris Rorden's dcm2niix validation repositories (`github.com/neurolabusc/dcm_qa*`, BSD-2-Clause).
Only their `In/` folders (the DICOM input) are pinned, per commit, with every file's git blob
SHA-1; `tests/dcmqa.py` downloads on demand into `.cache/dcm_qa/` (about 113 MB on the first run)
and unpacks `.gz` files. Re-pin or add repositories with `uv run python scripts/pin_dcm_qa.py`.

Classification on 2026-10-08 (Phase 4a; enhanced row updated in 4c); `tests/test_dcmqa.py` round-trips every converted
series and pins the converted count per repository:

| Repository | Content | Series | Converted | Not converted because |
|---|---|---|---|---|
| `dcm_qa_12bit` | Siemens 12-bit fMRI | 1 | 0 | mosaic |
| `dcm_qa_enh` | Canon/Philips/Siemens enhanced MR | 5 | 5 (Phase 4c: DTI, magnitude/phase, 2 echoes, fMRI time × echo, multi-file) | |
| `dcm_qa_me` | Siemens multi-echo GRE fieldmaps, fMRI | 3 | 2 (one 4D by EchoNumbers) | mosaic |
| `dcm_qa_nih` | GE + Siemens fMRI | 8 | 0 | GE: volumes differ only by InstanceNumber; Siemens: mosaic |
| `dcm_qa_pdt2` | Siemens dual-echo PD/T2 | 1 | 1 (4D by EchoNumbers) | |
| `dcm_qa_philips_dwi` | Philips DWI, classic + enhanced | 3 | 0 | both exports share one series UID; presentation states, raw data mixed in |
| `dcm_qa_ts` | one image in 7 transfer syntaxes | 1 | 0 | all variants share one series UID; compressed (Phase 4b) |

All converted series round-trip identically (`compare`: no differences, no notes).

Not pinned yet (large): `dcm_qa_ge` (142 MB, GE DTI/MPRAGE), `dcm_qa_toshiba` (214 MB, DWI),
`dcm_qa_dti` (51 MB, Siemens DTI), `dcm_qa_canon_enh` (72 MB), `dcm_qa_philips_enh` (~370 MB).
