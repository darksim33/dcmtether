"""Phase 5c: derived MR images (DERIVED\\SECONDARY) as a fallback for PACS without SEG/PM support.

One classic MR image per source slice, built from that source image's header as a template:
new deterministic UIDs, ``ImageType DERIVED\\SECONDARY``, a SourceImageSequence reference, tool
equipment and research labeling; private tags, overlays, icons and values that described the
source pixels are removed. Pixel values:

- label map (dcmqi SEG template): label values as pixels, exact;
- map (dcmqi parametric map template): integer maps exact up to the rounding of
  RescaleSlope/Intercept to DS precision (reported if not zero); float maps
  quantized to 16 bits (MR images hold integers) with the maximum error computed and reported.
  The Parametric Map (``to-pmap``) remains the lossless representation.

Error messages never contain element values or file names.
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
from pydicom.valuerep import DS

from dcmtether import __version__, pmap, seg, uids
from dcmtether._quiet import quiet_pydicom
from dcmtether.derived import (
    CONTENT_DESCRIPTION,
    DerivedError,
    check_output,
    load_on_grid,
    load_source,
    series_description,
    uids_for,
)

#: Failures of derived MR creation (all derived-object failures share one type).
DerivedMRError = DerivedError

_UINT16_MAX = 65535
#: Removed from the template: they describe the source pixels or the acquiring device.
_REMOVED_KEYWORDS = (
    "IconImageSequence",
    "SmallestImagePixelValue",
    "LargestImagePixelValue",
    "PixelPaddingValue",
    "PixelPaddingRangeLimit",
    "WindowCenter",
    "WindowWidth",
    "WindowCenterWidthExplanation",
    "VOILUTFunction",
    "VOILUTSequence",
    "ModalityLUTSequence",
    "RescaleSlope",
    "RescaleIntercept",
    "RescaleType",
    "LossyImageCompression",
    "LossyImageCompressionRatio",
    "LossyImageCompressionMethod",
    "DeviceSerialNumber",
    "StationName",
    "ReferencedImageSequence",
    "DerivationCodeSequence",
    "SourceImageSequence",
    "InstanceCreationDate",
    "InstanceCreationTime",
)
_OUTPUT_PATTERN = "MR_{:04d}.dcm"


def load_metadata(path: Path) -> seg.SegMetadata | pmap.PmapMetadata:
    """A dcmqi SEG template (-> label map) or parametric map template (-> map)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DerivedMRError(f"metadata is not readable JSON ({type(exc).__name__})") from None
    is_seg = isinstance(data, dict) and "segmentAttributes" in data
    is_pmap = isinstance(data, dict) and "QuantityValueCode" in data
    if is_seg == is_pmap:
        raise DerivedMRError(
            "metadata must be a dcmqi SEG template (segmentAttributes) or a dcmqi parametric "
            "map template (QuantityValueCode), not both"
        )
    return seg.parse_metadata(data) if is_seg else pmap.parse_metadata(data)


@dataclass(frozen=True)
class DerivedMRResult:
    paths: list[Path]
    kind: str  # "map" or "labels"
    #: Largest |written real value - NIfTI value| (0 if exact).
    max_error: float
    #: The error comes from writing the rescale values as DS only (integer map: the stored
    #: values are exact), not from quantizing a float map to 16 bits.
    rounded: bool = False


def _label_pixels(path: Path, source, metadata: seg.SegMetadata):
    labels = seg._label_values(path, source, metadata)
    if labels.max() > _UINT16_MAX:
        raise DerivedMRError("label values above 65535 do not fit in MR pixel data")
    return labels.astype(np.uint16), 1.0, 0.0, None


def _ds(value: float) -> DS:
    return DS(value, auto_format=True)


def _as_written(value: float) -> float:
    """The number a reader gets: DS holds 16 characters (float(DS) still gives the original)."""
    return float(str(_ds(value)))


def _map_pixels(path: Path, source):
    """(uint16 stored values, slope, intercept, real values) in NIfTI space."""
    img = load_on_grid(path, source, "map")
    raw = np.asarray(img.dataobj.get_unscaled())
    if raw.dtype.kind in "iu":
        stored, slope, intercept = pmap._stored_values(img)
        real = stored.astype(np.float64) * slope + intercept
        return stored.astype(np.uint16), slope, intercept, real
    real = np.asanyarray(img.dataobj).astype(np.float64)
    if not np.all(np.isfinite(real)):
        raise DerivedMRError("map has NaN or infinite values (replace them, e.g. with 0)")
    low, high = float(real.min()), float(real.max())
    slope = (high - low) / _UINT16_MAX if high > low else 1.0
    slope, intercept = _as_written(slope), _as_written(low)
    stored = np.clip(np.rint((real - intercept) / slope), 0, _UINT16_MAX).astype(np.uint16)
    return stored, slope, intercept, real


def _image(
    template: Dataset, pixels: np.ndarray, number: int, uid: str, series_uid: str, info: dict
) -> Dataset:
    ds = copy.deepcopy(template)
    ds.remove_private_tags()
    for elem in list(ds):
        if 0x6000 <= elem.tag.group <= 0x601E and elem.tag.group % 2 == 0:  # overlays
            del ds[elem.tag]
    for keyword in _REMOVED_KEYWORDS:
        if keyword in ds:
            delattr(ds, keyword)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = uid
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = info["series_number"]
    ds.SeriesDescription = info["series_description"]
    ds.InstanceNumber = number
    ds.ImageType = ["DERIVED", "SECONDARY", "OTHER"]
    ds.DerivationDescription = info["derivation"]
    ds.ImageComments = info["comments"]
    ds.BurnedInAnnotation = "NO"
    source_ref = Dataset()
    source_ref.ReferencedSOPClassUID = template.SOPClassUID
    source_ref.ReferencedSOPInstanceUID = template.SOPInstanceUID
    ds.SourceImageSequence = [source_ref]
    ds.Manufacturer = "dcmtether"
    ds.ManufacturerModelName = "dcmtether"
    ds.SoftwareVersions = __version__
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    if info["rescale"] is not None:
        slope, intercept, units = info["rescale"]
        ds.RescaleSlope, ds.RescaleIntercept = _ds(slope), _ds(intercept)
        ds.RescaleType = units[:64]
    center, width = info["window"]
    ds.WindowCenter, ds.WindowWidth = _ds(center), _ds(width)
    ds.PixelData = np.ascontiguousarray(pixels, dtype="<u2").tobytes()
    ds["PixelData"].VR = "OW"
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta
    return ds


def to_derived_mr(
    source_nifti: Path,
    derivative_nifti: Path,
    metadata: seg.SegMetadata | pmap.PmapMetadata,
    output_dir: Path,
) -> DerivedMRResult:
    """Write one derived MR image per source slice into ``output_dir`` (new files only)."""
    source = load_source(source_nifti)
    if any("frames" in i for i in source.document["instances"]):
        raise DerivedMRError(
            "derived MR images from enhanced (multi-frame) sources are not supported"
        )
    targets = [output_dir / _OUTPUT_PATTERN.format(n) for n in range(1, len(source.slices) + 1)]
    for target in targets:
        check_output(target)

    is_labels = isinstance(metadata, seg.SegMetadata)
    rounded = False
    if is_labels:
        stored, slope, intercept, real = _label_pixels(derivative_nifti, source, metadata)
        derivation = "dcmtether: label map (pixel value = label value)"
        rescale = None
        max_error = 0.0
    else:
        stored, slope, intercept, real = _map_pixels(derivative_nifti, source)
        quantity, units = metadata.quantity.meaning, metadata.units.meaning
        derivation = f"dcmtether: {quantity} map ({units})"
        rescale = (slope, intercept, metadata.units.value)
        written = stored.astype(np.float64) * _as_written(slope) + _as_written(intercept)
        max_error = float(np.max(np.abs(written - real)))
        rounded = nib.load(derivative_nifti).get_data_dtype().kind in "iu"
    frames = source.frames(stored)
    real_low = float(frames.min()) * slope + intercept
    real_high = float(frames.max()) * slope + intercept
    comments = f"{CONTENT_DESCRIPTION}. {derivation}"
    if max_error > 0:
        cause = "rescale rounded to DS precision" if rounded else "quantized to 16 bits"
        comments += f"; {cause}, max error {max_error:.3g}"
    info = {
        "series_number": metadata.series_number,
        "series_description": series_description(metadata.series_description),
        "derivation": derivation[:1024],
        "comments": comments[:10240],
        "rescale": rescale,
        "window": ((real_low + real_high) / 2, max(real_high - real_low, 1e-12)),
    }
    series_uid, _ = uids_for(source, "derived-mr", [metadata.canonical], frames)
    content = series_uid.removeprefix("2.25.")

    written_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with quiet_pydicom():
            for n, (template, plane, target) in enumerate(
                zip(source.originals, frames, targets, strict=True), start=1
            ):
                uid = uids.derived_uid(template.SOPInstanceUID, "derived-mr", content, "instance")
                image = _image(template, plane, n, uid, series_uid, info)
                partial = target.with_name(target.name + ".partial")
                written_paths.append(partial)  # before writing: a failed write leaves a file
                image.save_as(partial, enforce_file_format=True)
            for partial, target in zip(written_paths, targets, strict=True):
                partial.rename(target)
    except BaseException:
        for path in (*written_paths, *targets):
            path.unlink(missing_ok=True)
        raise
    return DerivedMRResult(targets, "labels" if is_labels else "map", max_error, rounded)
