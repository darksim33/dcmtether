# Segmentation templates

> **TL;DR** — `to-seg` (and `to-mr` for label maps) needs a JSON template that says what each
> label value in your label map is. **Structure** explains the fields, **Codes** shows where to
> find the SNOMED codes that name each segment, **Common codes** lists frequent ones, and
> **Checklist** covers the mistakes that are easy to make and hard to see later.

---

## Structure

A template is a [dcmqi](https://qiicr.org/dcmqi/#/seg) segmentation template: one JSON object
for the whole SEG, with one entry per label value. A minimal template for a label map with
kidney = 1 and tumor = 2:

```json
{
  "SeriesDescription": "Kidney segmentation",
  "SeriesNumber": "300",
  "ContentCreatorName": "Doe^Jane",
  "segmentAttributes": [
    [
      {
        "labelID": 1,
        "SegmentLabel": "kidney",
        "SegmentAlgorithmType": "AUTOMATIC",
        "SegmentAlgorithmName": "nnU-Net",
        "SegmentedPropertyCategoryCodeSequence": {
          "CodeValue": "123037004", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Anatomical Structure"
        },
        "SegmentedPropertyTypeCodeSequence": {
          "CodeValue": "64033007", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Kidney"
        },
        "recommendedDisplayRGBValue": [185, 102, 83]
      },
      {
        "labelID": 2,
        "SegmentLabel": "tumor",
        "SegmentAlgorithmType": "MANUAL",
        "SegmentedPropertyCategoryCodeSequence": {
          "CodeValue": "49755003", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Morphologically Abnormal Structure"
        },
        "SegmentedPropertyTypeCodeSequence": {
          "CodeValue": "108369006", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Neoplasm"
        },
        "recommendedDisplayRGBValue": [255, 0, 0]
      }
    ]
  ]
}
```

Ready to copy: [`examples/seg-template-abdomen.json`](examples/seg-template-abdomen.json)
(kidney = 1, liver = 2) and [`examples/seg-template.json`](examples/seg-template.json) (liver
and liver tumor).

### Series fields

| Field | Required | Meaning |
|---|---|---|
| `SeriesNumber` | yes | series number of the SEG in the PACS (an integer, may be a string) |
| `SeriesDescription` | no | shown in the PACS; "RESEARCH USE ONLY" is appended. Default: "Segmentation" |
| `ContentCreatorName` | no | who made it, as `Family^Given` |
| `InstanceNumber` | no | default `1` |
| `ContentLabel` | no | short label, upper case, at most 16 characters. Default: `SEGMENTATION` |
| `segmentAttributes` | yes | a list with exactly **one** list of segments (one label map per SEG) |

### Segment fields

| Field | Required | Meaning |
|---|---|---|
| `labelID` | yes | the voxel value in the label map (1, 2, …). Every value in the map needs an entry |
| `SegmentLabel` / `SegmentDescription` | one of them | short name / readable name |
| `SegmentAlgorithmType` | yes | `MANUAL`, `SEMIAUTOMATIC` or `AUTOMATIC` |
| `SegmentAlgorithmName` | for (semi)automatic | e.g. `nnU-Net`, `TotalSegmentator` |
| `SegmentAlgorithmVersion` | no | default `unknown` |
| `SegmentAlgorithmFamilyCodeSequence` | no | default: Artificial Intelligence (`123110`, `DCM`) |
| `SegmentedPropertyCategoryCodeSequence` | yes | what kind of thing: anatomy, abnormality, tissue, … |
| `SegmentedPropertyTypeCodeSequence` | yes | what exactly: kidney, neoplasm, … |
| `SegmentedPropertyTypeModifierCodeSequence` | no | e.g. left / right |
| `AnatomicRegionSequence` | no | where an abnormality lies, e.g. a neoplasm in the kidney |
| `recommendedDisplayRGBValue` | no | display color: a list of three integers 0–255 |

Label values that are not 1, 2, 3, … are renumbered to segment numbers in the SEG; `to-seg`
prints the mapping.

---

## Codes

Every `…CodeSequence` is a triple:

```json
{ "CodeValue": "64033007", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Kidney" }
```

`SCT` is SNOMED CT. Viewers and PACS go by `CodeValue` (+ scheme); `CodeMeaning` is only the
text shown to people. A wrong `CodeValue` with the right meaning is a wrong segmentation.

Three ways to find a code:

1. **dcmqi metadata creator** — [qiicr.org/dcmqi/#/seg](https://qiicr.org/dcmqi/#/seg): pick
   category and type from lists, add segments, download the finished template. Easiest for a
   new template.
2. **pydicom** (installed with dcmtether) knows the DICOM code lists:

   ```bash
   uv run python -c "from pydicom.sr.codedict import codes; print(codes.SCT.Kidney)"
   # Code(value='64033007', scheme_designator='SCT', meaning='Kidney', scheme_version=None)
   uv run python -c "from pydicom.sr.codedict import codes; print(codes.cid7151.dir('kidney'))"
   # ['ArcuateArteryOfTheKidney', ..., 'Kidney', 'KidneyStent', ...]
   ```

   `cid7150` holds the categories, `cid7151` the types, `cid7152` the modifiers and `cid4031`
   the anatomic regions.
3. **DICOM standard, Part 16** — the same lists online:
   [CID 7150](https://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_7150.html)
   (categories) and
   [CID 7151](https://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_7151.html)
   (types). 3D Slicer uses the same codes, so a SEG exported from Slicer shows them too.

---

## Common codes

All `SCT`. Pair each type with the category in the same row group.

| Category | CodeValue | Types (CodeValue) |
|---|---|---|
| Anatomical Structure | `123037004` | Liver `10200004`, Kidney `64033007`, Spleen `78961009`, Pancreas `15776009`, Prostate `41216001`, Brain `12738006`, Lung `39607008`, Heart `80891009`, Bladder `89837001` |
| Morphologically Abnormal Structure | `49755003` | Neoplasm `108369006`, Lesion `52988006`, Mass `4147007`, Cyst `367643001`, Necrosis `6574001`, Edema `79654002` |
| Tissue | `85756007` | Fat `55603005`, Muscle `71616004` |

Modifiers: Left `7771000`, Right `24028007`, Bilateral `51440002`.

> [!NOTE]
> "Anatomical Structure" exists under two codes: `123037004` (used by dcmqi, 3D Slicer and
> older DICOM editions) and `91723000` (the current CID 7150). Viewers accept both; use one
> consistently.

A tumor in the left kidney, as one segment:

```json
{
  "labelID": 3,
  "SegmentLabel": "tumor",
  "SegmentAlgorithmType": "MANUAL",
  "SegmentedPropertyCategoryCodeSequence": { "CodeValue": "49755003", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Morphologically Abnormal Structure" },
  "SegmentedPropertyTypeCodeSequence": { "CodeValue": "108369006", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Neoplasm" },
  "AnatomicRegionSequence": { "CodeValue": "64033007", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Kidney" },
  "recommendedDisplayRGBValue": [255, 0, 0]
}
```

For a left kidney (anatomy, not a lesion), put the modifier on the type instead:
`"SegmentedPropertyTypeModifierCodeSequence": { "CodeValue": "7771000", "CodingSchemeDesignator": "SCT", "CodeMeaning": "Left" }`.

---

## Checklist

- **Copy codes, never edit them by hand.** dcmtether checks that every code has all three
  parts, but it cannot know whether `CodeValue` and `CodeMeaning` belong together. A code
  copied from another segment and only renamed is the most common mistake.
- **Category and type must fit.** Organs go under Anatomical Structure, tumors and lesions
  under Morphologically Abnormal Structure.
- **One entry per label value**, each `labelID` once. A value in the map without an entry
  stops `to-seg` with "label values not described in the metadata".
- **Distinct colors** for segments that touch, or a viewer shows them as one.
- **Test it.** `to-seg` checks the template before it writes anything; a problem is reported
  as `segment N: …` (N counts entries from 1) or `metadata: …`.

```bash
uv run dcmtether to-seg DATASET/sub-…/….nii.gz labels.nii.gz my-template.json seg.dcm
```

The same template works for `to-mr` (label values become the pixel values of the derived MR
images).
