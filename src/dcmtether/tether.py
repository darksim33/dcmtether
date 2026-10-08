"""``*.tether.json``: everything but pixel data, as DICOM JSON (PLAN.md §2 and §4)."""

import base64
import copy
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pydicom
from pydicom.charset import decode_bytes
from pydicom.datadict import keyword_for_tag
from pydicom.dataelem import DataElement, RawDataElement
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filebase import DicomBytesIO
from pydicom.filewriter import write_data_element
from pydicom.valuerep import DS, IS, PN_DELIMS, TEXT_VR_DELIMS

from dcmtether import __version__
from dcmtether._quiet import quiet_pydicom

FORMAT_VERSION = 2
TETHER_SUFFIX = ".tether.json"

#: Never stored in tether.json: the voxels live in the NIfTI.
PIXEL_TAGS = frozenset({"7FE00010", "7FE00008", "7FE00009"})  # Pixel / Float / Double Float

DICOM_ORDER = "dicom_rows_columns_slices"

#: Original strings of text elements, keyed by element path: ``TAG`` or ``TAG/item/TAG/...``
#: inside sequences; one string per value (VM), exactly as encoded (incl. padding).
Strings = dict[str, list[str]]

TEXT_VRS = frozenset(
    {"AE", "AS", "CS", "DA", "DS", "DT", "IS", "LO", "LT", "PN", "SH", "ST", "TM", "UC", "UI"}
    | {"UR", "UT"}
)
#: Text VRs that are encoded with SpecificCharacterSet (others are ASCII).
CHARSET_VRS = frozenset({"LO", "LT", "PN", "SH", "ST", "UC", "UT"})
#: Text VRs where a backslash is part of the value, not a value delimiter.
SINGLE_VALUE_VRS = frozenset({"LT", "ST", "UR", "UT"})
#: Converted by pydicom while reading (it needs it to decode text), so its raw bytes are gone.
_CONVERTED_ON_READ = frozenset({0x00080005})  # SpecificCharacterSet


class TetherError(ValueError):
    """The header cannot be stored losslessly. Messages never contain element values."""


def element_name(tag: int) -> str:
    return keyword_for_tag(tag) or f"({tag >> 16:04X},{tag & 0xFFFF:04X})"


def _to_json(ds: pydicom.Dataset) -> dict:
    with quiet_pydicom():
        try:
            data = ds.to_json_dict()
        except Exception as exc:
            raise TetherError(
                f"header cannot be represented as DICOM JSON ({type(exc).__name__})"
            ) from None
    return {tag: value for tag, value in data.items() if tag not in PIXEL_TAGS}


# --- original strings -------------------------------------------------------------------------


def _encoded(elem: DataElement, encodings) -> bytes:
    """The value bytes exactly as pydicom writes them (including the even-length pad)."""
    fp = DicomBytesIO()
    fp.is_little_endian, fp.is_implicit_VR = True, True
    write_data_element(fp, elem, encodings)
    return fp.getvalue()[8:]  # implicit VR header: tag + 4-byte length


def _decode(data: bytes, vr: str, encodings) -> list[str]:
    chunks = [data] if vr in SINGLE_VALUE_VRS else data.split(b"\\")
    if vr == "PN":  # like pydicom: each component group (``=``) is decoded on its own
        return [
            "=".join(decode_bytes(group, encodings, PN_DELIMS) for group in chunk.split(b"="))
            for chunk in chunks
        ]
    if vr in CHARSET_VRS:
        return [decode_bytes(chunk, encodings, TEXT_VR_DELIMS) for chunk in chunks]
    return [chunk.decode("latin-1") for chunk in chunks]


def _restore(elem: DataElement, parts: list[str]) -> None:
    """Give ``elem`` values that pydicom writes as exactly ``parts``.

    DS() and IS() strip spaces, so their ``original_string`` (which the writer uses) is forced.
    """
    values: list = list(parts)
    if elem.VR in ("DS", "IS"):
        cls = DS if elem.VR == "DS" else IS
        for n, part in enumerate(parts):
            if part.strip():
                values[n] = cls(part.strip())
                values[n].original_string = part
    elem.value = values if len(values) > 1 else values[0]


def capture_strings(ds: Dataset, prefix: str = "") -> Strings:
    """Original strings of text elements that a DICOM JSON round trip would not reproduce.

    Must run on a freshly read dataset, before its elements are accessed: pydicom strips spaces
    when it converts values, so only the raw bytes hold the original string.
    """
    strings: Strings = {}
    with quiet_pydicom():
        encodings = ds._character_set
        for tag in ds.keys():
            raw = ds.get_item(tag)
            key = f"{prefix}{tag:08X}"
            elem = ds[tag]  # converted; ``raw`` keeps the bytes
            if elem.VR == "SQ":
                for n, item in enumerate(elem.value):
                    strings.update(capture_strings(item, f"{key}/{n}/"))
                continue
            if elem.VR not in TEXT_VRS:
                continue
            if isinstance(raw, RawDataElement):
                data = raw.value or b""
            elif raw.value in (None, ""):  # pydicom converts zero-length elements while reading
                data = b""
            elif tag in _CONVERTED_ON_READ:
                continue
            else:
                raise TetherError(
                    f"{element_name(tag)}: element converted before its string was captured"
                )
            value = json.loads(json.dumps(elem.to_json_dict(None, 0))).get("Value")
            if value is None:  # empty element
                canonical = b""
            else:
                rebuilt = DataElement.from_json(Dataset, key[-8:], elem.VR, value, "Value")
                canonical = _encoded(rebuilt, encodings)
            if canonical == data:
                continue
            parts = _decode(data, elem.VR, encodings)
            trial = copy.deepcopy(elem)  # fail now, not when rebuilding
            _restore(trial, parts)
            if _encoded(trial, encodings) != data:
                raise TetherError(f"{element_name(tag)}: original string cannot be reproduced")
            strings[key] = parts
    return strings


def _apply_strings(ds: Dataset, strings: Strings) -> None:
    for path, parts in strings.items():
        steps = path.split("/")
        target = ds
        for tag, item in zip(steps[0::2], steps[1::2], strict=False):
            target = target[int(tag, 16)].value[int(item)]
        _restore(target[int(steps[-1], 16)], parts)


# --- document ----------------------------------------------------------------------------------


def split_shared(items: Sequence[dict]) -> tuple[dict, list[dict]]:
    """Elements present in every item with an equal value → shared; the rest per item."""
    if not items:
        return {}, []
    shared = {
        tag: value
        for tag, value in items[0].items()
        if all(tag in other and other[tag] == value for other in items[1:])
    }
    rest = [{tag: v for tag, v in item.items() if tag not in shared} for item in items]
    return shared, rest


def _split_with_strings(
    items: Sequence[dict], strings: Sequence[Strings]
) -> tuple[dict, Strings, list[dict], list[Strings]]:
    """:func:`split_shared`, where an element's DS/IS strings count as part of its value."""
    keyed = [
        {tag: (value, {p: s for p, s in st.items() if p[:8] == tag}) for tag, value in item.items()}
        for item, st in zip(items, strings, strict=True)
    ]
    shared, rest = split_shared(keyed)
    return (
        {tag: value for tag, (value, _) in shared.items()},
        {p: s for _, st in shared.values() for p, s in st.items()},
        [{tag: value for tag, (value, _) in r.items()} for r in rest],
        [{p: s for _, st in r.values() for p, s in st.items()} for r in rest],
    )


def _pixel_data_trailing(ds: pydicom.Dataset) -> str | None:
    """Bytes stored after the last pixel (beyond the even-length pad), base64; else ``None``."""
    n_bits = ds.Rows * ds.Columns * ds.SamplesPerPixel * ds.BitsAllocated
    expected = (n_bits * int(ds.get("NumberOfFrames") or 1) + 7) // 8
    extra = ds.PixelData[expected:]
    if not extra or (expected % 2 and extra == b"\0"):
        return None
    return base64.b64encode(extra).decode("ascii")


def _instance(index, name, file_meta, elements, strings, ds) -> dict:
    entry = {
        "index": list(index),
        "file_name": name,
        "file_meta": file_meta,
        "elements": elements,
        "original_strings": strings,
    }
    trailing = _pixel_data_trailing(ds)
    if trailing is not None:
        entry["pixel_data_trailing"] = trailing
    return entry


def build(
    datasets: Sequence[pydicom.Dataset],
    file_names: Sequence[str],
    indices: Sequence[Sequence[int]],
    dtype: np.dtype,
    shape_dicom: Sequence[int],
    ornt: np.ndarray,
    strings: Sequence[Strings],
) -> dict:
    """Build the tether document; ``datasets`` and ``strings`` in the order of ``indices``."""
    first = datasets[0]
    meta_shared, meta_rest = split_shared([_to_json(ds.file_meta) for ds in datasets])
    shared, shared_strings, rest, rest_strings = _split_with_strings(
        [_to_json(ds) for ds in datasets], strings
    )
    return {
        "format_version": FORMAT_VERSION,
        "tool": {"name": "dcmtether", "version": __version__},
        "source": {
            "series_instance_uid": str(first.SeriesInstanceUID),
            "sop_class_uid": str(first.SOPClassUID),
            "transfer_syntax_uid": str(first.file_meta.TransferSyntaxUID),
            "n_instances": len(datasets),
        },
        "file_meta": meta_shared,
        "shared": shared,
        "original_strings": shared_strings,
        "instances": [
            _instance(*entry)
            for entry in zip(
                indices, file_names, meta_rest, rest, rest_strings, datasets, strict=True
            )
        ],
        "pixel": {
            "dtype": np.dtype(dtype).name,
            "shape_dicom": [int(n) for n in shape_dicom],
            "pixel_data_vr": first["PixelData"].VR,
            "orientation": {
                "from": DICOM_ORDER,
                "to": "closest_RAS",
                "ornt_transform": np.asarray(ornt).astype(int).tolist(),
            },
        },
    }


def write(document: dict, path: Path) -> None:
    try:
        text = json.dumps(document, indent=1, allow_nan=False)
    except ValueError:
        raise TetherError("header contains NaN or infinite values (not storable as JSON)") from None
    path.write_text(text + "\n", encoding="utf-8")


def load(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "format_version" not in document:
        raise TetherError("not a tether document")
    version = document["format_version"]
    if version != FORMAT_VERSION:
        raise TetherError(
            f"tether format version {version} is not supported (expected {FORMAT_VERSION}); "
            "re-run to-nii"
        )
    return document


def rebuild_datasets(document: dict) -> list[Dataset]:
    """Per-instance datasets (with file meta, without pixel data), in ``instances`` order."""
    datasets = []
    with quiet_pydicom():
        for instance in document["instances"]:
            ds = Dataset.from_json({**document["shared"], **instance["elements"]})
            ds.file_meta = FileMetaDataset(
                Dataset.from_json({**document["file_meta"], **instance["file_meta"]})
            )
            _apply_strings(ds, {**document["original_strings"], **instance["original_strings"]})
            datasets.append(ds)
    return datasets
