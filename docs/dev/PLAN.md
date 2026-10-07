# PLAN — DCMtether: DICOM ⇄ NIfTI round-trip tool

> Project name: **DCMtether** — package, import and CLI: `dcmtether`. Recovery files: `*.tether.json`.
> Status: Phase 0 done; next: Phase 1. This file records goals, settled decisions, the phase plan and open questions.
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
| DS/IS strings | Store the **original string** for DS/IS values whose canonical re-formatting differs (side map in `tether.json`), so round trips stay string-identical rather than merely numerically equal. |
| Human metadata | Separate BIDS-style `*.json` sidecar (keyword: value) — readable, not used for recovery. |
| Naming | BIDS-inspired key-value filenames with custom entities, always including the series number: `sub-01_ses-01_bodypart-knee_acq-sagPD_ser-05_T2w.nii.gz`. |
| Collisions | Never silent. Existing target → error (or explicit suffix + log). Manifest enforces uniqueness on SeriesInstanceUID. |
| Manifest | **SQLite** (stdlib, supports in-place status updates for resumable runs; Parquet would add `pyarrow` for little gain). One table per dataset: one row per NIfTI with paths, subject, session, body part, modality, vendor, SOPClassUID, TransferSyntaxUID, SeriesInstanceUID, StudyInstanceUID, source directory, status. |
| New UIDs | `2.25.<uuid-as-integer>` (no org root needed). Deterministic via `uuid.uuid5(namespace, source_uid + derivation + content_hash)` so identical reruns reproduce UIDs, but **different content always gets a different UID** (reusing a UID for changed data violates DICOM uniqueness and PACS may silently keep only one copy). Note: pydicom's `generate_uid()` uses pydicom's root by default — don't use it as-is. |
| Subject mapping | The PatientID/study → `sub-XX`/`ses-XX` map is a re-identification key (PHI). Stored **outside** the shareable dataset folder, never committed, never logged. |
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
├── README.md                       # documents custom entities (bodypart-, ser-, …)
└── sub-01/
    └── ses-01/
        └── anat/
            ├── sub-01_ses-01_bodypart-knee_acq-sagPD_ser-05_T2w.nii.gz
            ├── sub-01_ses-01_bodypart-knee_acq-sagPD_ser-05_T2w.json         # human sidecar
            └── sub-01_ses-01_bodypart-knee_acq-sagPD_ser-05_T2w.tether.json   # recovery metadata
```

Derivatives later go under `dataset/derivatives/<tool-name>/…` mirroring the same structure.

---

## 4. `*.tether.json` format (v1 draft)

```json
{
  "format_version": 1,
  "tool": { "name": "dcmtether", "version": "0.1.0" },
  "source": {
    "series_instance_uid": "…",
    "sop_class_uid": "1.2.840.10008.5.1.4.1.1.4",
    "transfer_syntax_uid": "1.2.840.10008.1.2.1",
    "n_instances": 30
  },
  "file_meta": { "/* DICOM JSON: group 0002 */": {} },
  "shared":    { "/* DICOM JSON: elements identical across all instances */": {} },
  "instances": [
    {
      "index": [0],
      "file_name": "IM_0001",
      "file_meta": { "/* only elements that differ from shared file_meta, e.g. MediaStorageSOPInstanceUID */": {} },
      "elements":  { "/* DICOM JSON: elements that differ per instance */": {} }
    }
  ],
  "pixel": {
    "dtype": "int16",
    "shape_dicom": [512, 512, 30],
    "orientation": {
      "from": "dicom_acquisition_order",
      "to": "closest_RAS",
      "ornt_transform": [[1, 1], [0, -1], [2, 1]]
    }
  }
}
```

Notes:
- `index` is in DICOM volume order (`shape_dicom`), not NIfTI order. It is a list so 4D series fit: `[slice]` for 3D, `[slice, volume]` for multi-echo / DWI / dynamics.
- Enhanced multi-frame: one instance with N frames; per-frame functional groups stay in `elements` (format may get a `frames` block in Phase 4).
- PixelData (7FE0,0010) is never stored here.
- An element appears in `shared` only if every instance has it with an equal value.
- Planned addition (Phase 3): `ds_is_strings` — original DS/IS strings where re-formatting would change them (see §2).

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

### Phase 2 — Forward: DICOM → NIfTI + metadata
- Volume assembly from pydicom (own writer, sort by ImagePositionPatient projected on slice normal).
- Affine from IOP/IPP/PixelSpacing; LPS → RAS; reorient to closest RAS, record `ornt_transform`.
- Raw stored values, original dtype; record rescale/vendor scaling; uniform slope/intercept also into NIfTI `scl_slope`/`scl_inter`.
- `tether.json`: split shared vs per-instance elements (rule in §2); file meta included.
- Human sidecar; naming; collision checks; manifest update.
- Oracle test: compare shape/affine/voxels against dcm2niix output in world space (where dcm2niix is available).

**Done when:** all `supported` test series convert; oracle comparisons agree or differences are documented as deliberate.

### Phase 3 — Reverse + round-trip verification ⛔ gate
- Rebuild per-instance datasets from `tether.json` (`Dataset.from_json`), apply inverse orientation, slice voxels back into PixelData, restore file meta, write files.
- `compare` command: element-by-element (incl. sequences and private tags) + decoded pixel arrays; clear diff report.
- Implement DS/IS original-string preservation (decided in §2; see §6, caveat 3).
- Targeted tests for known JSON-model losses (§6, caveats 3–5): DS/IS formatting, character-set re-encoding, NaN/Inf in FL/FD, ambiguous `US or SS` VRs, private-tag VRs.
- `dicom-validator` on reconstructed files.

**Done when:** round trip is element- and pixel-identical for every `supported` test series. **No phase 4+ work before this passes.**

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

1. **PHI.** DICOM headers and therefore `*.tether.json` files are personal health data (GDPR). Keep real data out of Claude Code sessions entirely; treat output datasets with the same care as the source DICOMs. The subject mapping and the manifest (Study/Series UIDs) are identifying too.
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

---

## 7. Open questions

- [ ] Vendor mix (Siemens / Philips / GE) and share of classic vs enhanced multi-frame in the real data → answered by running Phase 1 locally.
- [ ] Target viewer(s) for derivatives (3D Slicer / OHIF / clinical PACS)?
- [x] DS/IS handling → store original strings where they differ (§2).
- [x] Manifest format → SQLite (§2).
- [ ] Where does the subject mapping file live (path/storage outside the dataset), and who may access it?
- [ ] Label-map metadata schema for SEG (reuse dcmqi's JSON template format?).
- [ ] Is additional public test data (TCIA) acceptable, and which collections?

---

## 8. Working with Claude Code

- One phase per session/branch; start each session by reading `docs/dev/PLAN.md` and `CLAUDE.md`.
- Tests first; `uv run pytest` must pass before a phase is called done.
- Never commit or paste real patient data; fixtures come only from approved test data.
- Record every design decision in §2 of this file (and conventions in `CLAUDE.md`) at the end of the session.
- Keep changes small and reviewable; the Phase 3 gate is non-negotiable.
