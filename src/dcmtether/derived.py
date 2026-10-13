"""Shared machinery for derived DICOM objects (SEG, Parametric Map, derived MR).

A derivative is a NIfTI on the grid of a converted source NIfTI. The source images are rebuilt
in memory from its tether.json (no original files needed), every derived frame references its
source image, UIDs are deterministic and content-hashed, and every object is labelled for
research use. Error messages never contain element values or file names.
"""

import copy
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.sr.coding import Code

from dcmtether import orientation, reverse, tether, uids
from dcmtether._quiet import quiet_pydicom
from dcmtether.errors import DcmtetherError
from dcmtether.scanning import ORIENTATION_TOL, POSITION_TOL_MM

#: Marks every derived object (appended to SeriesDescription, in the content description).
RESEARCH_LABEL = "RESEARCH USE ONLY"
CONTENT_DESCRIPTION = f"{RESEARCH_LABEL} - not for clinical use"
EQUIPMENT = {
    "manufacturer": "dcmtether",
    "manufacturer_model_name": "dcmtether",
    "device_serial_number": "none",
}
_LO_MAX = 64


class DerivedError(DcmtetherError, ValueError):
    """Expected failure with a safe (value-free) message."""


def template_code(entry: dict, key: str, where: str, required: bool = True) -> Code | None:
    """A dcmqi-template code ``{CodeValue, CodingSchemeDesignator, CodeMeaning}``."""
    value = entry.get(key)
    if value is None:
        if required:
            raise DerivedError(f"{where}: {key} is missing")
        return None
    try:
        return Code(
            str(value["CodeValue"]),
            str(value["CodingSchemeDesignator"]),
            str(value["CodeMeaning"]),
        )
    except (KeyError, TypeError):
        raise DerivedError(
            f"{where}: {key} needs CodeValue, CodingSchemeDesignator and CodeMeaning"
        ) from None


def template_instance_number(data: dict) -> int:
    """``InstanceNumber`` of a dcmqi template (default 1)."""
    value = data.get("InstanceNumber") or 1
    try:
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise ValueError
        return int(value)
    except ValueError:
        raise DerivedError("metadata: InstanceNumber is not an integer") from None


def series_description(description: str) -> str:
    """``<description> RESEARCH USE ONLY``, within the LO length limit."""
    return f"{description[: _LO_MAX - len(RESEARCH_LABEL) - 1]} {RESEARCH_LABEL}"


@dataclass
class Source:
    document: dict
    #: Source images for highdicom, one shared geometry (see :func:`_shared_geometry`).
    images: list[pydicom.Dataset]
    #: Per derived frame (in ``images`` / frame order): the slice index it shows.
    slices: list[int]
    nifti: nib.Nifti1Image
    #: The rebuilt source images as they are (same order as ``images``, own geometry).
    originals: list[pydicom.Dataset]

    @property
    def series_instance_uid(self) -> str:
        return self.document["source"]["series_instance_uid"]

    def frames(self, volume_ras: np.ndarray) -> np.ndarray:
        """A NIfTI-space volume on the source grid as derived frames (frames, rows, columns)."""
        ornt = np.asarray(self.document["pixel"]["orientation"]["ornt_transform"])
        dicom_order = orientation.from_closest_ras(volume_ras, ornt)  # (rows, columns, slices)
        return np.stack([dicom_order[:, :, k] for k in self.slices])


def _ordered(document: dict, datasets: list) -> tuple[list, list[int]]:
    if len(document["pixel"]["shape_dicom"]) != 3:
        raise DerivedError("derivatives of 4D sources are not supported yet")
    instances = document["instances"]
    if any("frames" in i for i in instances):
        if len(instances) > 1:
            raise DerivedError("sources split over several multi-frame instances are not supported")
        return datasets, [index[0] for index in instances[0]["frames"]]
    order = sorted(range(len(instances)), key=lambda n: instances[n]["index"][0])
    return [datasets[n] for n in order], [instances[n]["index"][0] for n in order]


def _shared_geometry(source_images: list) -> list:
    """Source copies with one shared orientation and pixel spacing, as highdicom requires.

    highdicom compares orientations exactly and wants multi-frame geometry in the shared
    functional groups. Scanners write rounding noise (GE) or per-frame groups (Philips); the
    scan accepted differences up to ORIENTATION_TOL, so the first image's values are used.
    Only the in-memory copies passed to highdicom change, never files or tether.json.
    """
    images = [copy.deepcopy(ds) for ds in source_images]
    if len(images) == 1 and "PerFrameFunctionalGroupsSequence" in images[0]:
        ds = images[0]
        shared = ds.SharedFunctionalGroupsSequence[0]
        per_frame = ds.PerFrameFunctionalGroupsSequence
        for keyword, attribute in (
            ("PlaneOrientationSequence", "ImageOrientationPatient"),
            ("PixelMeasuresSequence", "PixelSpacing"),
        ):
            if keyword in shared or keyword not in per_frame[0]:
                continue
            values = np.asarray(
                [[float(v) for v in item[keyword][0][attribute]] for item in per_frame]
            )
            if not np.allclose(values, values[0], rtol=0, atol=ORIENTATION_TOL):
                raise DerivedError(f"source frames differ in {attribute}")
            setattr(shared, keyword, per_frame[0][keyword])
            for item in per_frame:
                del item[keyword]
        return images
    orientations = np.asarray([[float(v) for v in ds.ImageOrientationPatient] for ds in images])
    if not np.allclose(orientations, orientations[0], rtol=0, atol=ORIENTATION_TOL):
        raise DerivedError("source images differ in ImageOrientationPatient")
    for ds in images[1:]:
        ds.ImageOrientationPatient = images[0].ImageOrientationPatient
    return images


def load_source(source_nifti: Path) -> Source:
    """Rebuild the source images of a converted NIfTI (checked against its tether.json)."""
    try:
        document, datasets = reverse.rebuild(source_nifti)
    except (reverse.ReverseError, tether.TetherError) as exc:
        raise DerivedError(f"source: {exc}") from None
    originals, slices = _ordered(document, datasets)
    with quiet_pydicom():
        images = _shared_geometry(originals)
    return Source(document, images, slices, nib.load(source_nifti), originals)


def load_on_grid(path: Path, source: Source, what: str) -> nib.Nifti1Image:
    """A NIfTI that must lie on the source grid (same 3D shape and affine)."""
    img = nib.load(path)
    if img.shape != source.nifti.shape[:3]:
        raise DerivedError(f"{what} shape does not match the source NIfTI (3D, same grid)")
    if not np.allclose(img.affine, source.nifti.affine, rtol=0, atol=POSITION_TOL_MM):
        raise DerivedError(f"{what} affine does not match the source NIfTI (same grid needed)")
    return img


def uids_for(source: Source, derivation: str, parts: Iterable[str], array: np.ndarray):
    """(series UID, SOP instance UID): the same inputs give the same UIDs, other content new."""
    content = hashlib.sha256()
    for part in (source.series_instance_uid, derivation, str(array.shape), str(array.dtype)):
        content.update(part.encode("utf-8") + b"\0")
    for part in parts:
        content.update(part.encode("utf-8") + b"\0")
    content.update(np.ascontiguousarray(array).tobytes())
    digest = content.hexdigest()
    return (
        uids.derived_uid(source.series_instance_uid, derivation, digest, "series"),
        uids.derived_uid(source.series_instance_uid, derivation, digest, "instance"),
    )


def check_output(output: Path) -> None:
    if output.exists():
        raise DerivedError("output file already exists (existing files are never overwritten)")


def write(dataset: pydicom.Dataset, output: Path) -> None:
    """Write via a ``.partial`` file; nothing is left behind on failure."""
    check_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    try:
        with quiet_pydicom():
            dataset.save_as(partial)
        partial.rename(output)
    finally:
        partial.unlink(missing_ok=True)
