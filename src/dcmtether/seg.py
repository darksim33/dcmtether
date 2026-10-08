"""Phase 5a: label map NIfTI + source NIfTI/tether pair -> DICOM Segmentation (SEG).

The source images are rebuilt in memory from the tether.json (no original files needed). The
label map must lie on the source NIfTI grid (same shape and affine): every SEG frame then
references its source image exactly. Segment metadata uses the dcmqi JSON template
(https://qiicr.org/dcmqi/#/seg). Error messages never contain element values or file names.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import highdicom as hd
import numpy as np
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

from dcmtether import __version__
from dcmtether._quiet import quiet_pydicom
from dcmtether.derived import (
    CONTENT_DESCRIPTION,
    EQUIPMENT,
    DerivedError,
    Source,
    check_output,
    load_on_grid,
    load_source,
    series_description,
    template_code,
    uids_for,
    write,
)

_ALGORITHM_TYPES = {v.name for v in hd.seg.SegmentAlgorithmTypeValues}


#: Failures of SEG creation (all derived-object failures share one type).
SegError = DerivedError


# --- metadata (dcmqi template) -----------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    label_id: int
    label: str
    category: Code
    type: Code
    algorithm_type: str
    algorithm_name: str | None = None
    algorithm_version: str = "unknown"
    algorithm_family: Code = codes.cid7162.ArtificialIntelligence
    type_modifier: Code | None = None
    anatomic_region: Code | None = None
    rgb: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class SegMetadata:
    series_number: int
    instance_number: int
    series_description: str
    content_creator_name: str | None
    content_label: str
    segments: list[Segment] = field(default_factory=list)
    #: Canonical JSON of the template, part of the content hash (UIDs).
    canonical: str = ""


def _segment(entry: dict, n: int) -> Segment:
    where = f"segment {n}"
    try:
        label_id = int(entry["labelID"])
    except (KeyError, TypeError, ValueError):
        raise SegError(f"{where}: labelID is missing or not an integer") from None
    if label_id < 1:
        raise SegError(f"{where}: labelID must be at least 1")
    label = entry.get("SegmentLabel") or entry.get("SegmentDescription")
    if not label:
        raise SegError(f"{where}: SegmentLabel or SegmentDescription is needed")
    algorithm_type = str(entry.get("SegmentAlgorithmType", "")).upper()
    if algorithm_type not in _ALGORITHM_TYPES:
        raise SegError(
            f"{where}: SegmentAlgorithmType '{algorithm_type}' is not one of "
            f"{', '.join(sorted(_ALGORITHM_TYPES))}"
        )
    algorithm_name = entry.get("SegmentAlgorithmName")
    if algorithm_type != "MANUAL" and not algorithm_name:
        raise SegError(f"{where}: SegmentAlgorithmName is needed for {algorithm_type}")
    rgb = entry.get("recommendedDisplayRGBValue")
    if rgb is not None and (len(rgb) != 3 or not all(0 <= int(v) <= 255 for v in rgb)):
        raise SegError(f"{where}: recommendedDisplayRGBValue must be three values 0-255")
    family = template_code(entry, "SegmentAlgorithmFamilyCodeSequence", where, required=False)
    return Segment(
        label_id=label_id,
        label=str(label),
        category=template_code(entry, "SegmentedPropertyCategoryCodeSequence", where),
        type=template_code(entry, "SegmentedPropertyTypeCodeSequence", where),
        algorithm_type=algorithm_type,
        algorithm_name=str(algorithm_name) if algorithm_name else None,
        algorithm_version=str(entry.get("SegmentAlgorithmVersion") or "unknown"),
        algorithm_family=family or codes.cid7162.ArtificialIntelligence,
        type_modifier=template_code(
            entry, "SegmentedPropertyTypeModifierCodeSequence", where, required=False
        ),
        anatomic_region=template_code(entry, "AnatomicRegionSequence", where, required=False),
        rgb=tuple(int(v) for v in rgb) if rgb is not None else None,
    )


def parse_metadata(data: dict) -> SegMetadata:
    """Check a dcmqi segmentation template (one label map) and turn it into SegMetadata."""
    if not isinstance(data, dict):
        raise SegError("metadata must be a JSON object (dcmqi template)")
    try:
        series_number = int(data["SeriesNumber"])
    except (KeyError, TypeError, ValueError):
        raise SegError("metadata: SeriesNumber is missing or not an integer") from None
    attributes = data.get("segmentAttributes")
    if not isinstance(attributes, list) or len(attributes) != 1:
        raise SegError("metadata: segmentAttributes must describe exactly one label map")
    entries = attributes[0]
    if not isinstance(entries, list) or not entries:
        raise SegError("metadata: segmentAttributes[0] lists no segments")
    segments = [_segment(entry, n) for n, entry in enumerate(entries, start=1)]
    ids = [s.label_id for s in segments]
    if len(set(ids)) != len(ids):
        raise SegError("metadata: duplicate labelID")
    content_label = str(data.get("ContentLabel") or "SEGMENTATION").upper()
    return SegMetadata(
        series_number=series_number,
        instance_number=int(data.get("InstanceNumber") or 1),
        series_description=str(data.get("SeriesDescription") or "Segmentation"),
        content_creator_name=data.get("ContentCreatorName") or None,
        content_label=content_label[:16],
        segments=sorted(segments, key=lambda s: s.label_id),
        canonical=json.dumps(data, sort_keys=True, separators=(",", ":")),
    )


def load_metadata(path: Path) -> SegMetadata:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SegError(f"metadata is not readable JSON ({type(exc).__name__})") from None
    return parse_metadata(data)


def _description(segment: Segment, number: int) -> hd.seg.SegmentDescription:
    algorithm = None
    if segment.algorithm_type != "MANUAL":
        algorithm = hd.AlgorithmIdentificationSequence(
            name=segment.algorithm_name,
            family=segment.algorithm_family,
            version=segment.algorithm_version,
        )
    description = hd.seg.SegmentDescription(
        segment_number=number,
        segment_label=segment.label,
        segmented_property_category=segment.category,
        segmented_property_type=segment.type,
        algorithm_type=segment.algorithm_type,
        algorithm_identification=algorithm,
        anatomic_regions=[segment.anatomic_region] if segment.anatomic_region else None,
        display_color=hd.color.CIELabColor.from_rgb(*segment.rgb) if segment.rgb else None,
    )
    if segment.type_modifier is not None:
        description.SegmentedPropertyTypeCodeSequence[
            0
        ].SegmentedPropertyTypeModifierCodeSequence = [
            hd.sr.CodedConcept.from_code(segment.type_modifier)
        ]
    return description


# --- conversion --------------------------------------------------------------------------------


@dataclass(frozen=True)
class SegResult:
    path: Path
    #: labelID in the label map -> segment number in the SEG (1..n).
    segment_numbers: dict[int, int]
    n_sources: int


def _label_values(labels_path: Path, source: Source, metadata: SegMetadata) -> np.ndarray:
    """Label map on the source grid as int64 (checked: grid, integers, declared labels)."""
    img = load_on_grid(labels_path, source, "label map")
    values = np.asanyarray(img.dataobj)
    if not np.all(np.isfinite(values)) or not np.array_equal(values, np.round(values)):
        raise SegError("label map has non-integer values")
    if (values < 0).any():
        raise SegError("label map has negative values")
    labels = values.astype(np.int64)
    present = set(np.unique(labels).tolist()) - {0}
    if not present:
        raise SegError("label map is empty (no non-zero voxels)")
    undeclared = present - {s.label_id for s in metadata.segments}
    if undeclared:
        listed = ", ".join(str(v) for v in sorted(undeclared)[:10])
        raise SegError(f"label values not described in the metadata: {listed}")
    return labels


def to_seg(
    source_nifti: Path, labels_nifti: Path, metadata: SegMetadata, output: Path
) -> SegResult:
    """Write a DICOM SEG for the label map ``labels_nifti`` on the grid of ``source_nifti``."""
    check_output(output)
    source = load_source(source_nifti)
    labels = _label_values(labels_nifti, source, metadata)

    numbering = {s.label_id: n for n, s in enumerate(metadata.segments, start=1)}
    lut = np.zeros(max(numbering) + 1, dtype=np.uint16)
    for label_id, number in numbering.items():
        lut[label_id] = number
    dtype = np.uint8 if len(numbering) < 256 else np.uint16
    pixel_array = source.frames(lut[labels]).astype(dtype)
    series_uid, sop_uid = uids_for(source, "seg", [metadata.canonical], pixel_array)

    with quiet_pydicom():
        segmentation = hd.seg.Segmentation(
            source_images=source.images,
            pixel_array=pixel_array,
            segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=[
                _description(s, numbering[s.label_id]) for s in metadata.segments
            ],
            series_instance_uid=series_uid,
            series_number=metadata.series_number,
            sop_instance_uid=sop_uid,
            instance_number=metadata.instance_number,
            software_versions=__version__,
            content_label=metadata.content_label,
            content_description=CONTENT_DESCRIPTION,
            content_creator_name=metadata.content_creator_name,
            omit_empty_frames=True,
            **EQUIPMENT,
        )
        segmentation.SeriesDescription = series_description(metadata.series_description)
    write(segmentation, output)
    return SegResult(output, numbering, len(source.images))
