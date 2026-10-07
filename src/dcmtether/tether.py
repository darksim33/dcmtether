"""``*.tether.json``: everything but pixel data, as DICOM JSON (PLAN.md §2 and §4)."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pydicom

from dcmtether import __version__
from dcmtether._quiet import quiet_pydicom

FORMAT_VERSION = 1
TETHER_SUFFIX = ".tether.json"

#: Never stored in tether.json: the voxels live in the NIfTI.
PIXEL_TAGS = frozenset({"7FE00010", "7FE00008", "7FE00009"})  # Pixel / Float / Double Float

DICOM_ORDER = "dicom_rows_columns_slices"


class TetherError(ValueError):
    """The header cannot be stored losslessly. Messages never contain element values."""


def _to_json(ds: pydicom.Dataset) -> dict:
    with quiet_pydicom():
        try:
            data = ds.to_json_dict()
        except Exception as exc:
            raise TetherError(
                f"header cannot be represented as DICOM JSON ({type(exc).__name__})"
            ) from None
    return {tag: value for tag, value in data.items() if tag not in PIXEL_TAGS}


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


def build(
    datasets: Sequence[pydicom.Dataset],
    file_names: Sequence[str],
    indices: Sequence[Sequence[int]],
    dtype: np.dtype,
    shape_dicom: Sequence[int],
    ornt: np.ndarray,
) -> dict:
    """Build the tether document; ``datasets`` in the order of ``indices``."""
    first = datasets[0]
    meta_shared, meta_rest = split_shared([_to_json(ds.file_meta) for ds in datasets])
    shared, rest = split_shared([_to_json(ds) for ds in datasets])
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
        "instances": [
            {"index": list(index), "file_name": name, "file_meta": m, "elements": e}
            for index, name, m, e in zip(indices, file_names, meta_rest, rest, strict=True)
        ],
        "pixel": {
            "dtype": np.dtype(dtype).name,
            "shape_dicom": [int(n) for n in shape_dicom],
            "orientation": {
                "from": DICOM_ORDER,
                "to": "closest_RAS",
                "ornt_transform": np.asarray(ornt).astype(int).tolist(),
            },
        },
    }


def write(document: dict, path: Path) -> None:
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
