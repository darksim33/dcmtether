# PLAN — DCMtether: DICOM ⇄ NIfTI round-trip tool

> Project name: **DCMtether** — package, import and CLI: `dcmtether`. Recovery files: `*.tether.json`.
> Status: Phases 0–3 done (gate passed); next: Phase 4. This file records goals, settled decisions, the phase plan and open questions.
> Update it whenever a decision is made so later sessions don't re-litigate it.

---

## 1. Goal

Convert MRI DICOM series (whole-body: knee, abdomen, etc.) into **NIfTI + metadata pairs** that can be
converted back into DICOM that is **element- and pixel-identical** to the original.

The pairs are the basis for **derivatives** (segmentations, ADC maps, QSM maps) that are later written back
as properly referenced DICOM objects (DICOM SEG, Parametric Map, or derived MR images).

### Non-goals
- Byte-identical DICOM files (see §6 — possible stretch goal for uncompressed originals only).
- Strict BIDS compliance (layout is BIDS-*inspired*; a BIDS exporter can be added later if needed).
- Resampling, registration or any image processing — the tool converts and references, it does not process.
- Clinical/diagnostic use (research use only, see §6).

---

## 2. Settled decisions

| Topic | Decision |
|---|---|
| Language | Python. Core libs: `pydicom`, `nibabel`, `numpy`, `highdicom`; CLI: `click`. Tooling: `uv`, `pytest`, `ruff`. A Rust port is optional later, using the Python tests as reference. |
| Test data | `dicom-test-files` only. **No real patient data in Claude Code sessions** — not as files, not in logs, not in error messages. Real data is run locally, outside sessions; only anonymized failure reports come back. |
| NIfTI orientation | Reoriented to **closest RAS** (axis permutation/flip only, **never resampled**). The applied transform is stored in `tether.json`; a helper applies/inverts it. Oblique rotation stays in the affine. |
| NIfTI voxel values | **Raw stored values** in the original dtype. RescaleSlope/Intercept (and vendor private scaling, e.g. Philips) stay in the metadata, not baked into voxels. If slope/intercept are uniform across the series they are **also** written to NIfTI `scl_slope`/`scl_inter` so viewers show real values; per-slice varying rescale → Phase 1 classifies as `split`/`unsupported` (NIfTI has only one global scaling). |
| Recovery metadata | **DICOM JSON Model** (PS3.18 Annex F) via `Dataset.to_json_dict()` / `Dataset.from_json()`. One `*.tether.json` **per NIfTI** (= per series), structured as file meta + shared + per-instance elements. PixelData excluded. |
| Shared vs per-instance | An element goes into `shared` only if it is **present in every instance with an equal value**; otherwise it is stored per instance (absence is represented by omission in that instance). |
| Instance index order | `instances[].index` refers to positions in the **DICOM-order volume** (`pixel.shape_dicom`, before reorientation), never to NIfTI/RAS order. Mapping to NIfTI goes through `ornt_transform` only. |
| Original strings | Store the **original string** of every text-VR element (DS/IS, but also PN/LO/CS/UI/…) whose raw bytes a DICOM JSON round trip would not reproduce, in an `original_strings` side map in `tether.json`, so round trips stay byte-identical per element rather than merely equal in value. Extended from DS/IS-only on 2026-10-08 after whitespace-only PN values (GE) came back empty. |
| Identity (compare) | Instances paired by SOPInstanceUID. Non-sequence elements: same VR **and same encoded value bytes**; sequences compared item by item; PixelData: equal **decoded arrays** (raw byte differences are reported as notes only). Not compared: defined/undefined lengths, element order, preamble, file meta group length. Reports never contain values, UIDs or file names. |
| Human metadata | Separate BIDS-style `*.json` sidecar (keyword: value) — readable, not used for recovery. |
| Naming | BIDS-inspired key-value filenames with custom entities, always including the series number: `sub-MuellThoma_ses-20260314_acq-sagPDFs_ser-05_PDw.nii.gz`, folder `sub-…/ses-…/anat/`. **No body-part entity** (decided 2026-10-07; BodyPartExamined stays in manifest and sidecar). `acq-` = SeriesDescription reduced to letters/digits, camelCase, ≤24 chars. Suffix `T1w`/`T2w`/`PDw`/`FLAIR` from description keywords (T2* excluded), else TR/TE/TI rules, else `MR`; rule recorded in the sidecar. |
| Collisions | Never silent. Existing target → error (or explicit suffix + log). Manifest enforces uniqueness on SeriesInstanceUID. |
| Manifest | **SQLite** (stdlib, supports in-place status updates for resumable runs; Parquet would add `pyarrow` for little gain). One table per dataset: one row per NIfTI with paths, subject, session, body part, modality, vendor, SOPClassUID, TransferSyntaxUID, SeriesInstanceUID, StudyInstanceUID, source directory, status. |
| New UIDs | `2.25.<uuid-as-integer>` (no org root needed). Deterministic via `uuid.uuid5(namespace, source_uid + derivation + content_hash)` so identical reruns reproduce UIDs, but **different content always gets a different UID** (reusing a UID for changed data violates DICOM uniqueness and PACS may silently keep only one copy). Note: pydicom's `generate_uid()` uses pydicom's root by default — don't use it as-is. |
| Subject/session labels | **Decided 2026-10-07 (replaces the earlier pseudonymous mapping-file plan):** `sub-` = first 5 letters of family + first 5 of given name (umlauts transliterated, ASCII letters only, CamelCase); `ses-` = StudyDate, plus `THHMM` from StudyTime only when one subject has several studies that day. **The dataset itself therefore identifies patients and must not be shared as is.** Clashes are never silent: different PatientIDs with the same subject label, same-day studies with equal times, or two series mapping to one file name → those series fail with an error naming manifest ids only. Missing PatientName or StudyDate → the series fails (anonymized data cannot be converted under this scheme). |
| Validation | `dicom-validator` (pure Python, `validate_iods`) in pytest. `dciodvfy` (dicom3tools, C++) as an extra check later via subprocess/Docker. |
| Derivatives | `highdicom`: `seg.Segmentation` for label maps, `pm.ParametricMap` for ADC/QSM. Fallback: derived MR images (`ImageType = DERIVED\SECONDARY`) for PACS that can't display SEG/PM. Source datasets are rebuilt in memory from `tether.json` — no original files needed. |
| Research labeling | SeriesDescription suffix "RESEARCH USE ONLY", ContentLabel/ContentDescription (SEG/PM), ImageComments, Manufacturer/SoftwareVersions = this tool + version. No burned-in watermark by default (if ever added: set `BurnedInAnnotation = YES`). |
| dcm2niix | Not used for the recovery path. Used as a **test oracle** and optionally to produce the human sidecar. dcm2niix does not reorient to RAS, so oracle comparisons are done **in world space** (via the affines / index permutation), never on raw arrays. |
| Synthetic test data | PHI-free series generated with pydicom from a template: all axis permutations/flips, oblique angles, gaps, mixed thickness, localizers, per-slice rescale. Primary fixture source for Phases 1–3; complements `dicom-test-files`. |

---

## 3. Output layout

```
dataset/
├── manifest.sqlite
├── README.md                       # identifying-data warning + custom entities (acq-, ser-, …)
└── sub-MuellThoma/
    └── ses-20260314/
        └── anat/
            ├── sub-MuellThoma_ses-20260314_acq-sagPDFs_ser-05_PDw.nii.gz
            ├── sub-MuellThoma_ses-20260314_acq-sagPDFs_ser-05_PDw.json         # human sidecar
            └── sub-MuellThoma_ses-20260314_acq-sagPDFs_ser-05_PDw.tether.json   # recovery metadata
```

Derivatives later go under `dataset/derivatives/<tool-name>/…` mirroring the same structure.

---

## 4. `*.tether.json` format (v1 draft)

```json
{
  "format_version": 2,
  "tool": { "name": "dcmtether", "version": "0.1.0" },
  "source": {
    "series_instance_uid": "…",
    "sop_class_uid": "1.2.840.10008.5.1.4.1.1.4",
    "transfer_syntax_uid": "1.2.840.10008.1.2.1",
    "n_instances": 30
  },
  "file_meta": { "/* DICOM JSON: group 0002 */": {} },
  "shared":    { "/* DICOM JSON: elements identical across all instances */": {} },
  "original_strings": { "00180080": ["500 "], "00081140/0/00180050": ["3.00"] },
  "instances": [
    {
      "index": [0],
      "file_name": "IM_0001",
      "file_meta": { "/* only elements that differ from shared file_meta, e.g. MediaStorageSOPInstanceUID */": {} },
      "elements":  { "/* DICOM JSON: elements that differ per instance */": {} },
      "original_strings": { "/* per-instance original strings */": [] },
      "pixel_data_trailing": "/* optional, base64: bytes after the last pixel */"
    }
  ],
  "pixel": {
    "dtype": "int16",
    "shape_dicom": [512, 512, 30],
    "pixel_data_vr": "OW",
    "orientation": {
      "from": "dicom_rows_columns_slices",
      "to": "closest_RAS",
      "ornt_transform": [[1, 1], [0, -1], [2, 1]]
    }
  }
}
```

Notes:
- `pixel.orientation.from = dicom_rows_columns_slices`: array axes (row index, column index, slice), slices sorted by ascending position along `cross(row_dir, col_dir)` (not acquisition order).
- `file_name` is the source path relative to the series' source directory (POSIX).
- `index` is in DICOM volume order (`shape_dicom`), not NIfTI order. It is a list so 4D series fit: `[slice]` for 3D, `[slice, volume]` for multi-echo / DWI / dynamics.
- Enhanced multi-frame: one instance with N frames; per-frame functional groups stay in `elements` (format may get a `frames` block in Phase 4).
- PixelData (7FE0,0010) is never stored here.
- An element appears in `shared` only if every instance has it with an equal value.
- `original_strings` (top level = shared, and per instance): element path → original strings, one
  per value, exactly as encoded incl. the even-length pad. Path = `TAG`, or `TAG/item/TAG/…` inside
  sequences. Only elements whose JSON round trip would change the bytes are listed. An element is
  `shared` only if its JSON value **and** its original strings are equal in every instance.
- `pixel_data_vr`: PixelData is not in the JSON, so its VR (OB/OW) is recorded here.
- `pixel_data_trailing`: bytes stored after the last pixel (e.g. `MR_small_padded.dcm` has 128),
  other than the normal single pad byte.
- Format version 2 since Phase 3; `to-dcm` refuses other versions ("re-run to-nii").

---

## 5. Phases

Each phase: tests first, one branch/session per phase, update this file with decisions at the end.

### Phase 0 — Setup
- Repo skeleton: `uv` project, `src/` layout, `pytest`, `ruff`, pre-commit (optional). Default branch `main`.
- Confirm Python 3.13 wheels exist for `highdicom`, `pylibjpeg` + plugins and `dicom-validator`; lower `requires-python` if not.
- `CLAUDE.md` with conventions, PHI rules, and the one check command.
- `.gitignore` PHI guards: `*.dcm`, `*.nii`, `*.nii.gz`, `*.tether.json`, `dataset/`, subject-mapping file; approved fixtures re-included via explicit `!` rules only.
- Test data fixtures from `dicom-test-files`; inventory which MR series are available (single-frame, enhanced, compressed).
- Synthetic series generator (see §2) for orientation/geometry edge cases.
- CLI skeleton (`dcmtether scan | to-nii | to-dcm | compare`).

**Done when:** `uv run pytest` passes on an empty test suite in a fresh clone; CLI prints help.

**Status: ✅ done (2026-10-07, branch `phase-0-setup`).** Fresh-copy check: format + lint clean, 43 tests pass, none skipped.
Outcome and decisions:
- `dicom-test-files` = GitHub `robyoung/dicom-test-files` (not on PyPI), pinned to commit `6279c96a`. `tests/data/dicom_test_files.json` lists names + SHA-256; `tests/testdata.py` downloads on demand into `.cache/` (ignored), verifies the hash and decompresses `.zst` (`zstandard` dev dep). No DICOM files are committed.
- Inventory in `docs/dev/test-data.md`: the only real multi-slice MR series is a GE ADC map (20 slices); `emri_small` has no functional groups; no multi-echo/DWI/dynamics/localizer/mosaic. Synthetic data must carry Phases 1–3; Phase 4 will need more data.
- Synthetic generator `tests/synthetic.py`: orientations, flips, oblique, ascending/descending/shuffled, gaps, per-slice thickness and rescale, signed, localizer; returns ground-truth `volume` + `affine_lps`; passes `dicom-validator` (rescale is the one deliberate, Philips-like IOD deviation).
- `dicom-validator` pinned to DICOM edition `2026d`, cached in `.cache/dicom-validator`.
- Python 3.13 confirmed: wheels exist for highdicom, dicom-validator, pylibjpeg + libjpeg/openjpeg/rle plugins (Windows/Linux/macOS, also 3.14).
- CLI: `click` (switched from argparse); input paths must exist (usage error, exit 2), stubs exit with code 3 ("not implemented"); output is ASCII only (cp1252 consoles crashed on `⇄`).
- Pre-commit: skipped for now (optional); the check command lives in `CLAUDE.md`.

### Phase 1 — Scan & index
- Walk a directory, read headers only (`stop_before_pixels`), group by SeriesInstanceUID.
- Record per series: SOPClassUID, TransferSyntaxUID, Implementation UID/Version, Manufacturer, Model, SoftwareVersions, orientation(s), slice positions, spacing, thickness, instance count, gaps.
- Classify: `supported` / `split` / `unsupported` with a reason. Cases: localizer with mixed orientations, non-uniform spacing, missing slices, mixed thickness, per-slice varying RescaleSlope/Intercept, Siemens mosaic, enhanced multi-frame, compressed transfer syntax.
- Write the manifest.

**Done when:** every test series gets a correct classification with a human-readable reason; scan output contains no pixel data reads.

**Status: ✅ done (2026-10-07, branch `phase-1-scan`).** Fresh-copy check: format + lint clean, 94 tests pass, none skipped. Code: `src/dcmtether/scan.py`, `src/dcmtether/manifest.py`, `dcmtether scan SOURCE DATASET`.
Outcome and decisions:
- **Status rules.** `supported` = one regular stack or a single slice. `split` = several stacks in one series: distinct image planes (orientation/matrix/pixel spacing, e.g. localizers) or equal-count volumes at the same positions (multi-echo/DWI/dynamics). `unsupported` = anything else needing a guess: missing slices, non-uniform spacing, sheared stack, mixed thickness, per-instance rescale, duplicate SOPInstanceUID, uneven instances per position, mosaic, multi-frame, compressed or deflated transfer syntax, non-MR SOP class, missing geometry, mixed SOP class/TS/modality/frame of reference/study, non-grayscale or mixed pixel format. Unsupported wins over split. **All** findings are reported, not just the first; geometry checks are skipped for rejected SOP classes and multi-frame (noise).
- **Supported SOP classes:** MR Image Storage only. **Transfer syntaxes:** Implicit VR LE, Explicit VR LE, Explicit VR BE.
- **Tolerances:** orientation cosines and pixel spacing 1e-4; positions and slice-spacing jitter 0.01 mm (`ORIENTATION_TOL`, `POSITION_TOL_MM`). Jitter inside the tolerance is lossless anyway, since `tether.json` keeps the exact IPP.
- **Privacy.** Reasons and CLI output never contain UIDs, names, descriptions, dates or paths; series are named by manifest id. pydicom value-validation warnings are suppressed during the scan (they quote element values); invalid values are kept as read.
- **Skipped files** are recorded with a reason: not DICOM, no SeriesInstanceUID, unreadable header, or *bare dataset without preamble/file meta* (detected separately, not supported).
- **Manifest schema v1** (`manifest.sqlite`): `meta` (schema/tool version, source root, scan time), `series` (one row per SeriesInstanceUID, UNIQUE; ids 1..n in scan order; reasons and geometry as JSON), `instances` (relative POSIX paths), `skipped_files`. Subject/session/NIfTI path columns are added in Phase 2 with a schema bump. An existing manifest is never overwritten; writes go to a `.partial` file first.
- The target dataset directory is excluded from the walk when it lies inside the source.
- On the public test data, all MR codec variants share series UIDs, so only the GE ADC series (when scanned on its own) and single Siemens/pydicom MR files are `supported`; everything else is correctly `unsupported`.
- Note for Phase 4: per-volume rescale in 4D series (Philips) is currently `unsupported`; the splitting policy should allow rescale to vary between volumes but not within one.

### Phase 2 — Forward: DICOM → NIfTI + metadata
- Volume assembly from pydicom (own writer, sort by ImagePositionPatient projected on slice normal).
- Affine from IOP/IPP/PixelSpacing; LPS → RAS; reorient to closest RAS, record `ornt_transform`.
- Raw stored values, original dtype; record rescale/vendor scaling; uniform slope/intercept also into NIfTI `scl_slope`/`scl_inter`.
- `tether.json`: split shared vs per-instance elements (rule in §2); file meta included.
- Human sidecar; naming; collision checks; manifest update.
- Oracle test: compare shape/affine/voxels against dcm2niix output in world space (where dcm2niix is available).

**Done when:** all `supported` test series convert; oracle comparisons agree or differences are documented as deliberate.

**Status: ✅ done (2026-10-07, branch `phase-2-forward`).** Format + lint clean, 225 tests pass, none skipped. Code: `src/dcmtether/{forward,naming,orientation,tether,_quiet}.py`, `dcmtether to-nii DATASET`.
Outcome and decisions:
- **Oracle:** dcm2niix (PyPI `dcm2niix`, dev dependency) agrees exactly with our output on 10 synthetic orientation variants (axial/coronal/sagittal, flips, two obliques, descending, shuffled, signed) and on the real oblique GE ADC series: same shape, affine within 1e-3 mm, identical raw voxels after reorienting both to closest RAS. No deliberate differences. Single-slice series are only checked against ground truth.
- **Ground truth:** every NIfTI voxel is checked in world space against the synthetic volume (NIfTI RAS → LPS → synthetic index), independent of our reorientation code; all 48 axis permutation/flip cases round-trip through `to_closest_ras`/`from_closest_ras`.
- **NIfTI:** gzip, raw stored values in the source dtype, sform = qform (code 1, scanner), units mm. The header is written directly because nibabel's normal save recomputes `scl_slope`/`scl_inter`; uniform DICOM rescale is stored there, otherwise slope/inter stay unset. (nibabel moves scaling into `img.dataobj` on load, so check `dataobj.slope`, not `header.get_slope_inter()`.) Single slice: slice spacing from SpacingBetweenSlices, else SliceThickness, else 1 mm.
- **Re-validation:** before converting, each series' headers are re-read and re-classified; a series that is no longer `supported` (files changed or missing since the scan) fails instead of converting.
- **tether.json:** built with `to_json_dict()` (InlineBinary for bulk data), pixel elements (7FE0,0010/0008/0009) removed, shared/per-instance split also for file meta. A header that cannot be expressed as DICOM JSON fails the series (no silent loss). DS/IS original strings are **not** captured yet (Phase 3).
- **Sidecar:** curated BIDS-style keys (times in seconds), `DcmTether.{Version, SuffixRule}`; no patient identifiers or dates.
- **Manifest schema v2:** adds `subject`, `session`, `nifti_path`, `tether_path`, `sidecar_path`, `conversion_status` (`converted`/`failed`), `conversion_error`. New manifests are created as v1 and upgraded, so the upgrade path is always exercised. Re-running `to-nii` skips converted series and retries failed ones. Unexpected exceptions are stored in full in the manifest only; the CLI prints the exception type.
- **Collisions:** existing output files are never overwritten (series fails, nothing partial is left behind).
- pydicom value warnings are suppressed during conversion too (`_quiet.quiet_pydicom`).

### Phase 3 — Reverse + round-trip verification ⛔ gate
- Rebuild per-instance datasets from `tether.json` (`Dataset.from_json`), apply inverse orientation, slice voxels back into PixelData, restore file meta, write files.
- `compare` command: element-by-element (incl. sequences and private tags) + decoded pixel arrays; clear diff report.
- Implement DS/IS original-string preservation (decided in §2; see §6, caveat 3).
- Targeted tests for known JSON-model losses (§6, caveats 3–5): DS/IS formatting, character-set re-encoding, NaN/Inf in FL/FD, ambiguous `US or SS` VRs, private-tag VRs.
- `dicom-validator` on reconstructed files.

**Done when:** round trip is element- and pixel-identical for every `supported` test series. **No phase 4+ work before this passes.**

**Status: ✅ done (2026-10-08, branch `phase-3-reverse`).** Format + lint clean, 318 tests pass, none skipped. Code: `src/dcmtether/{reverse,compare}.py`, `tether.py` (format v2), `dcmtether to-dcm NIFTI OUTPUT`, `dcmtether compare ORIGINAL REBUILT`.
Outcome and decisions:
- **Gate result:** byte-identical per element, with no notes, for all 11 synthetic geometry variants, uniform rescale, 12-bit and 8-bit odd-length data, the GE ADC series (20 slices) and every uncompressed public MR file (`MR_small` incl. implicit/big-endian/padded variants, Siemens with overlays + CSA, `MR2_UNCI/UNCR`, WG04 `MR1–4_UNC`). Rebuilt files give the same `dicom-validator` errors as their originals.
- **JSON-model edge cases** (§6, 3–5), each with a round-trip test: DS/IS formatting (`"500"`, `" 1.0"`, `"0005"`), whitespace-only/trailing-space text, Latin-1, UTF-8 and ISO 2022 (Japanese PN), `US or SS` (signed and unsigned), private tags in explicit and implicit VR, nested/empty sequences and items, empty vs absent. All are lossless.
- **Original strings** are captured from the raw bytes **right after `dcmread`**, before anything accesses the elements (pydicom strips spaces on conversion). Exception: SpecificCharacterSet, which pydicom converts while reading, so it is not captured (compare still checks it). Capture checks that restoring gives the exact bytes and fails the series otherwise.
- **NaN/Inf** in FL/FD: the series fails in `to-nii` ("header contains NaN or infinite values"); the JSON is written with `allow_nan=False`.
- **to-dcm** checks that the NIfTI belongs to the tether.json (dtype, shape after reorientation, affine within 0.01 mm, scl_slope/inter). It rejects unsafe or duplicate file names, never overwrites (`.partial` files, all-or-nothing) and writes the original transfer syntax (big endian byte-swapped). Output files keep their names relative to the series' source directory.
- **Unused high bits** (above BitsStored) are not kept: the NIfTI holds decoded values. compare reports this as a note, not a difference (test `test_unused_high_bits_are_not_kept`).
- `MR_truncated.dcm` is classified `supported` by the header-only scan but fails cleanly in `to-nii` (pixel data too short).
- Test infrastructure: dicom-validator changes pydicom's global validation mode; the `iod_errors` fixture now restores it. `SPECS` moved to `tests/roundtrip.py` (shared round-trip helpers).

### Phase 4 — Harder inputs
- Enhanced multi-frame MR (per-frame functional groups).
- 4D: multi-echo, DWI incl. b-values/gradient directions (needed for ADC; Siemens classic stores these partly in the CSA private header), dynamics.
- Compressed transfer syntaxes (`pylibjpeg` + plugins): decode to NIfTI; on the way back write an uncompressed TS (record that the TS changed) — compressed originals cannot be byte-identical anyway.
- Splitting policy for `split` series from Phase 1.

**Done when:** each new input type has its own round-trip tests and passes the Phase 3 compare.

### Phase 5 — Derivatives
- Label-map metadata JSON (segment number → label, SNOMED CT category/type codes, color, algorithm info).
- NIfTI label map + source pair → DICOM SEG (`highdicom.seg.Segmentation`).
- NIfTI float map + source pair → Parametric Map (`highdicom.pm.ParametricMap`) with units/real-world value mapping (ADC in mm²/s, QSM in ppm).
- Derived MR fallback via the template approach.
- Grid check: same affine + shape → per-frame source references; different grid → explicit `plane_positions`/`pixel_measures`, frame-of-reference-only linking.
- Research labeling (§2), deterministic content-hashed UIDs.
- `dicom-validator` (+ optionally `dciodvfy`) on all outputs.

**Done when:** SEG and Parametric Map outputs validate and load in 3D Slicer / OHIF with correct geometry.

### Phase 6 — Packaging & pipeline
- Batch mode, resumable runs via the manifest, structured logging (no PHI in logs by default).
- Docker image (incl. dcm2niix and dciodvfy for checks).
- README for users; document custom BIDS entities.
- Optional: BIDS exporter for subsets that need strict compliance.

---

## 6. Caveats & risks

1. **PHI.** DICOM headers and therefore `*.tether.json` files are personal health data (GDPR). Keep real data out of Claude Code sessions entirely; treat output datasets with the same care as the source DICOMs. **File and folder names contain patient names and study dates** (§2), and the manifest holds UIDs and source paths, so the whole dataset is identifying. Code still never prints labels, names or output paths; problems are reported by manifest series id.
2. **Identity definition.** Target = every data element equal + decoded pixels equal. Byte identity additionally depends on transfer syntax, defined vs undefined sequence lengths, padding and compression — stretch goal only for uncompressed originals.
3. **DS/IS formatting.** DICOM JSON stores Decimal String / Integer String values as JSON numbers, so `"0.500000"` comes back as `"0.5"`. Decided: keep original strings where they differ (§2).
4. **Character sets.** Respect SpecificCharacterSet (PN, LO, etc. with non-ASCII names) on both sides. JSON holds decoded UTF-8; re-encoding may not reproduce the original bytes (notably ISO 2022 escape sequences) — test explicitly.
5. **Private tags.** Need correct VR to survive; large vendor blobs (Siemens CSA) end up as base64 `InlineBinary` — fine, but files get bigger.
   - **Other JSON-model edge cases:** NaN/Inf in FL/FD cannot be JSON numbers; ambiguous `US or SS` / `OB or OW` VRs must resolve the same way on both sides; empty vs absent values must be preserved.
6. **Non-volume series.** Localizers, gaps, variable spacing/thickness and mosaics are common in clinical data; the Phase 1 classification is the safety net — never "best effort" convert silently.
7. **Orientation bugs** are the most likely silent failure. Always verify with the oracle test (world space), the synthetic permutation/flip series, and visual spot checks of a few slices.
8. **Viewer support.** SEG/Parametric Map work well in 3D Slicer and OHIF; clinical PACS support varies — check the target viewer early.
9. **Regulatory.** If derivatives ever influence clinical diagnosis, the tool may fall under EU MDR as medical device software. Intended use = research; labeling supports but does not by itself settle this.
10. **Test data coverage.** `dicom-test-files` has limited multi-slice MRI. Synthetic series (§2) cover geometry; publicly available, de-identified data (e.g. TCIA) may still be needed later for vendor/sequence coverage.
11. **UID reuse.** Deterministic UIDs without a content hash would give changed derivatives the same UID; mitigated by the §2 rule.
12. **to-nii does not check its own output** (postponed, decided 2026-10-08). A JSON-model loss nobody anticipated only shows up when `to-dcm` + `compare` are run. Possible later step: rebuild each series in memory during `to-nii` and fail it on any difference (roughly doubles conversion time).
13. **Pixel raw bytes vs decoded values.** Bits above BitsStored (old overlay-in-pixel-data, scanner junk) are lost, because the NIfTI holds decoded values. Signed data with BitsStored < BitsAllocated may also be re-encoded with full sign extension. The decoded pixels stay equal; compare reports it as a note.
14. **File meta strings** are not captured into `original_strings` (pydicom converts group 0002 while reading). Every test file round-trips anyway; a real-data failure here would show up in compare.

---

## 7. Open questions

- [ ] Vendor mix (Siemens / Philips / GE) and share of classic vs enhanced multi-frame in the real data → answered by running Phase 1 locally.
- [ ] Target viewer(s) for derivatives (3D Slicer / OHIF / clinical PACS)?
- [x] DS/IS handling → store original strings where they differ (§2).
- [x] Manifest format → SQLite (§2).
- [x] Subject mapping → none: labels come from PatientName/StudyDate (§2, decided 2026-10-07).
- [ ] Anonymized inputs (empty PatientName) currently fail naming. Fallback wanted (e.g. sanitized PatientID)?
- [ ] Label-map metadata schema for SEG (reuse dcmqi's JSON template format?).
- [ ] Is additional public test data (TCIA) acceptable, and which collections?

---

## 8. Working with Claude Code

- One phase per session/branch; start each session by reading `docs/dev/PLAN.md` and `CLAUDE.md`.
- Tests first; `uv run pytest` must pass before a phase is called done.
- Never commit or paste real patient data; fixtures come only from approved test data.
- Record every design decision in §2 of this file (and conventions in `CLAUDE.md`) at the end of the session.
- Keep changes small and reviewable; the Phase 3 gate is non-negotiable.
