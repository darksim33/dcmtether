"""Element-by-element comparison of an original and a rebuilt DICOM series (PLAN.md §6.2).

Identity means: same instances (paired by SOPInstanceUID), same elements with the same VR and
the same encoded value bytes (so DS/IS strings and character set encodings count), sequences
compared item by item, and equal decoded pixel arrays. Defined vs undefined lengths, element
order, the preamble and the file meta group length are not compared.

Reports name elements by keyword/tag and instances by InstanceNumber; they never contain
element values, UIDs or file names (PHI rules).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataelem import RawDataElement
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.uid import UID, ExplicitVRLittleEndian

from dcmtether._quiet import quiet_pydicom
from dcmtether.scanning import LOSSLESS_COMPRESSED_TRANSFER_SYNTAXES
from dcmtether.tether import element_name

PIXEL_DATA = 0x7FE00010
TRANSFER_SYNTAX = 0x00020010
#: Derived from the other file meta elements; a difference there is reported on its own.
FILE_META_GROUP_LENGTH = 0x00020000

Finding = tuple[str, str]  # (element path, kind)


@dataclass(frozen=True)
class Difference:
    instance: str
    where: str
    kind: str

    def line(self) -> str:
        return (
            f"{self.instance}: {self.where}: {self.kind}"
            if self.where
            else (f"{self.instance}: {self.kind}")
        )


@dataclass
class Report:
    n_compared: int = 0
    differences: list[Difference] = field(default_factory=list)
    #: Not differences under the identity definition, but worth knowing.
    notes: list[Difference] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not self.differences

    def lines(self) -> list[str]:
        return [d.line() for d in self.differences]


def dicom_files(directory: Path) -> list[Path]:
    """All regular files below ``directory``, sorted."""
    return sorted(p for p in directory.rglob("*") if p.is_file())


def _vr(raw, elem) -> str:
    if isinstance(raw, RawDataElement) and raw.VR:
        return raw.VR
    return elem.VR


def _compare_datasets(a: Dataset, b: Dataset, prefix: str, out: list[Finding]) -> None:
    for tag in sorted(set(a.keys()) | set(b.keys())):
        if not prefix and tag in (PIXEL_DATA, FILE_META_GROUP_LENGTH):
            continue
        where = prefix + element_name(tag)
        if tag not in b:
            out.append((where, "missing in rebuilt"))
            continue
        if tag not in a:
            out.append((where, "extra in rebuilt"))
            continue
        raw_a, raw_b = a.get_item(tag), b.get_item(tag)
        elem_a, elem_b = a[tag], b[tag]  # converted; the raw elements keep the bytes
        if _vr(raw_a, elem_a) != _vr(raw_b, elem_b):
            out.append((where, "VR differs"))
        elif elem_a.VR == "SQ":
            if len(elem_a.value) != len(elem_b.value):
                out.append((where, "number of items differs"))
                continue
            for n, (item_a, item_b) in enumerate(zip(elem_a.value, elem_b.value, strict=True)):
                _compare_datasets(item_a, item_b, f"{where}[{n}].", out)
        elif isinstance(raw_a, RawDataElement) and isinstance(raw_b, RawDataElement):
            if (raw_a.value or b"") != (raw_b.value or b""):
                out.append((where, "value differs"))
        elif elem_a.value != elem_b.value:
            out.append((where, "value differs"))


def _decompressed(a: Dataset, b: Dataset) -> bool:
    """``b`` is ``a`` written uncompressed: a lossless source as Explicit VR Little Endian."""
    return (
        a.file_meta.get("TransferSyntaxUID") in LOSSLESS_COMPRESSED_TRANSFER_SYNTAXES
        and b.file_meta.get("TransferSyntaxUID") == ExplicitVRLittleEndian
    )


def _compare_pixels(a: Dataset, b: Dataset, diffs: list[Finding], notes: list[Finding]) -> None:
    if PIXEL_DATA not in a or PIXEL_DATA not in b:
        if PIXEL_DATA in a or PIXEL_DATA in b:
            diffs.append(
                ("PixelData", "missing in rebuilt" if PIXEL_DATA in a else "extra in rebuilt")
            )
        return
    raw_a, raw_b = a.get_item(PIXEL_DATA), b.get_item(PIXEL_DATA)
    try:
        pixels_a, pixels_b = a.pixel_array, b.pixel_array
    except Exception as exc:
        diffs.append(("PixelData", f"cannot decode ({type(exc).__name__})"))
        return
    if pixels_a.shape != pixels_b.shape or not np.array_equal(pixels_a, pixels_b):
        diffs.append(("PixelData", "decoded pixels differ"))
    elif raw_a.value != raw_b.value and not _decompressed(a, b):
        notes.append(("PixelData", "raw bytes differ (decoded pixels equal)"))


def _compare(a: Dataset, b: Dataset) -> tuple[list[Finding], list[Finding]]:
    diffs: list[Finding] = []
    notes: list[Finding] = []
    with quiet_pydicom():
        meta_diffs: list[Finding] = []
        _compare_datasets(a.file_meta, b.file_meta, "", meta_diffs)
        if _decompressed(a, b):
            # Expected for a lossless source; elements and decoded pixels are still compared.
            meta_diffs.remove(("TransferSyntaxUID", "value differs"))
            source = UID(a.file_meta.TransferSyntaxUID).name
            change = f"changed from {source} to {ExplicitVRLittleEndian.name} (lossless)"
            notes.append(("TransferSyntaxUID", change))
        diffs.extend(meta_diffs)
        _compare_datasets(a, b, "", diffs)
        _compare_pixels(a, b, diffs, notes)
    return diffs, notes


def _read(path: Path) -> Dataset | None:
    try:
        with quiet_pydicom():
            ds = pydicom.dcmread(path)
    except (InvalidDicomError, OSError, ValueError, EOFError):
        return None
    return ds if "SOPInstanceUID" in ds else None


def compare_files(original: Path, rebuilt: Path) -> tuple[list[Finding], list[Finding]]:
    """(differences, notes) between two files, as (element path, kind) pairs."""
    a, b = _read(original), _read(rebuilt)
    if a is None or b is None:
        raise ValueError("not a readable DICOM file")
    return _compare(a, b)


def _label(ds: Dataset, fallback: str) -> str:
    number = ds.get("InstanceNumber")
    return f"instance {number}" if number not in (None, "") else fallback


def _index(paths: Sequence[Path], side: str, report: Report) -> dict[str, tuple[str, Dataset]]:
    by_uid: dict[str, tuple[str, Dataset]] = {}
    for n, path in enumerate(paths, start=1):
        ds = _read(path)
        label = f"{side} file #{n}"
        if ds is None:
            report.differences.append(Difference(label, "", "not a readable DICOM file"))
            continue
        uid = str(ds.SOPInstanceUID)
        if uid in by_uid:
            report.differences.append(Difference(label, "", f"duplicate SOPInstanceUID in {side}"))
            continue
        by_uid[uid] = (_label(ds, label), ds)
    return by_uid


def compare_series(original: Sequence[Path], rebuilt: Sequence[Path]) -> Report:
    report = Report()
    by_uid_a = _index(original, "original", report)
    by_uid_b = _index(rebuilt, "rebuilt", report)
    for uid, (label, a) in by_uid_a.items():
        if uid not in by_uid_b:
            report.differences.append(Difference(label, "", "missing in rebuilt"))
            continue
        report.n_compared += 1
        diffs, notes = _compare(a, by_uid_b[uid][1])
        report.differences.extend(Difference(label, w, k) for w, k in diffs)
        report.notes.extend(Difference(label, w, k) for w, k in notes)
    for uid, (label, _) in by_uid_b.items():
        if uid not in by_uid_a:
            report.differences.append(
                Difference(label, "", "extra instance (no matching SOPInstanceUID in original)")
            )
    return report
