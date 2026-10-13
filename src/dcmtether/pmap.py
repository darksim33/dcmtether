"""Map NIfTI (ADC, QSM, ...) + source NIfTI/tether pair -> DICOM Parametric Map.

The map must lie on the source NIfTI grid; every frame references its source image. Values are
stored losslessly: float32/float64 maps as Float/DoubleFloatPixelData, integer maps as unsigned
16-bit (or 8-bit) stored values whose real-world value mapping (slope/intercept) reproduces the
NIfTI values exactly. Metadata uses the dcmqi parametric map template
(https://qiicr.org/dcmqi/#/pm). Error messages never contain element values or file names.
"""

import json
from dataclasses import dataclass
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
    check_output,
    load_on_grid,
    load_source,
    series_description,
    template_code,
    template_instance_number,
    uids_for,
    write,
)

#: Failures of Parametric Map creation (all derived-object failures share one type).
PmapError = DerivedError

_UINT16_MAX = 65535


@dataclass(frozen=True)
class PmapMetadata:
    quantity: Code
    units: Code
    method: Code | None
    series_number: int
    instance_number: int
    series_description: str
    content_creator_name: str | None
    content_label: str
    #: Canonical JSON of the template, part of the content hash (UIDs).
    canonical: str


def parse_metadata(data: dict) -> PmapMetadata:
    """Check a dcmqi parametric map template and turn it into PmapMetadata."""
    if not isinstance(data, dict):
        raise PmapError("metadata must be a JSON object (dcmqi template)")
    try:
        series_number = int(data["SeriesNumber"])
    except (KeyError, TypeError, ValueError):
        raise PmapError("metadata: SeriesNumber is missing or not an integer") from None
    return PmapMetadata(
        quantity=template_code(data, "QuantityValueCode", "metadata"),
        units=template_code(data, "MeasurementUnitsCode", "metadata"),
        method=template_code(data, "MeasurementMethodCode", "metadata", required=False),
        series_number=series_number,
        instance_number=template_instance_number(data),
        series_description=str(data.get("SeriesDescription") or "Parametric map"),
        content_creator_name=data.get("ContentCreatorName") or None,
        content_label=str(data.get("ContentLabel") or "PARAMETRICMAP").upper()[:16],
        canonical=json.dumps(data, sort_keys=True, separators=(",", ":")),
    )


def load_metadata(path: Path) -> PmapMetadata:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PmapError(f"metadata is not readable JSON ({type(exc).__name__})") from None
    return parse_metadata(data)


@dataclass(frozen=True)
class PmapResult:
    path: Path
    #: Pixel data type written: float32, float64, uint16 or uint8.
    dtype: str
    n_sources: int


def _stored_values(img) -> tuple[np.ndarray, float, float]:
    """(stored array, slope, intercept) with real value = slope * stored + intercept, exact.

    Floats stay as they are (slope 1). Integers keep their NIfTI scaling; signed or wider
    integers are shifted into uint16 (intercept adjusted), never rounded.
    """
    raw = np.asarray(img.dataobj.get_unscaled())
    slope, intercept = (float(v) for v in (img.dataobj.slope, img.dataobj.inter))
    if raw.dtype.kind == "f":
        if not np.all(np.isfinite(raw)):
            raise PmapError("map has NaN or infinite values (replace them, e.g. with 0)")
        if (slope, intercept) != (1.0, 0.0):
            raw = raw * raw.dtype.type(slope) + raw.dtype.type(intercept)
        return raw.astype(raw.dtype.newbyteorder("=")), 1.0, 0.0
    if raw.dtype.kind not in "iu":
        raise PmapError("map must have integer or floating-point voxels")
    low, high = int(raw.min()), int(raw.max())
    if raw.dtype in (np.uint8, np.uint16):
        return raw.astype(raw.dtype.newbyteorder("=")), slope, intercept
    if high - low > _UINT16_MAX:
        raise PmapError("integer map range does not fit in 16-bit unsigned stored values")
    shift = min(low, 0) if high <= _UINT16_MAX else low
    stored = (raw.astype(np.int64) - shift).astype(np.uint16)
    return stored, slope, intercept + slope * shift


def to_pmap(
    source_nifti: Path, map_nifti: Path, metadata: PmapMetadata, output: Path
) -> PmapResult:
    """Write a DICOM Parametric Map for ``map_nifti`` on the grid of ``source_nifti``."""
    check_output(output)
    source = load_source(source_nifti)
    stored, slope, intercept = _stored_values(load_on_grid(map_nifti, source, "map"))
    pixel_array = source.frames(stored)
    series_uid, sop_uid = uids_for(
        source, "pmap", [metadata.canonical, repr((slope, intercept))], pixel_array
    )

    low, high = pixel_array.min(), pixel_array.max()
    floats = pixel_array.dtype.kind == "f"
    value_range = (float(low), float(high)) if floats else (int(low), int(high))
    mapping = hd.pm.RealWorldValueMapping(
        lut_label=metadata.quantity.meaning[:16],
        lut_explanation=f"{metadata.quantity.meaning} ({metadata.units.meaning})"[:64],
        unit=metadata.units,
        value_range=value_range,
        slope=slope,
        intercept=intercept,
        quantity_definition=metadata.quantity,
    )
    if metadata.method is not None:
        mapping.QuantityDefinitionSequence.append(
            hd.sr.CodeContentItem(
                name=codes.SCT.MeasurementMethod,
                value=metadata.method,
                relationship_type=hd.sr.RelationshipTypeValues.HAS_CONCEPT_MOD,
            )
        )
    with quiet_pydicom():
        parametric_map = hd.pm.ParametricMap(
            source_images=source.images,
            pixel_array=pixel_array,
            series_instance_uid=series_uid,
            series_number=metadata.series_number,
            sop_instance_uid=sop_uid,
            instance_number=metadata.instance_number,
            software_versions=__version__,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[mapping],
            # Window in stored values, covering the whole map.
            window_center=(float(low) + float(high)) / 2,
            window_width=max(float(high) - float(low), 1e-12 if floats else 1.0),
            content_label=metadata.content_label,
            content_description=CONTENT_DESCRIPTION,
            content_creator_name=metadata.content_creator_name,
            content_qualification=hd.ContentQualificationValues.RESEARCH,
            **EQUIPMENT,
        )
        parametric_map.SeriesDescription = series_description(metadata.series_description)
    write(parametric_map, output)
    return PmapResult(output, pixel_array.dtype.name, len(source.images))
