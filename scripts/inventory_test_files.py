"""Inventory the MR content of the pinned dicom-test-files set.

Fetches candidate files (headers are read with ``stop_before_pixels``) and prints a
Markdown table grouped by SeriesInstanceUID. Public test data only.

    uv run python scripts/inventory_test_files.py > docs/dev/test-data-inventory.md
"""

import sys
from collections import defaultdict
from pathlib import Path

import pydicom
from pydicom.uid import UID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import testdata


def _is_candidate(name: str) -> bool:
    # WG04 holds one file per modality and codec; only the MR ones matter here.
    if name.startswith("WG04/"):
        return Path(name).name.startswith("MR")
    return name.endswith((".dcm", ".seg")) or name.startswith("pydicom/")


def _classify(ds: pydicom.Dataset) -> str:
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if "PerFrameFunctionalGroupsSequence" in ds:
        return f"enhanced multi-frame ({frames} frames)"
    if frames > 1:
        return f"multi-frame ({frames} frames)"
    return "single-frame"


def main() -> None:
    series: dict[str, list[tuple[str, pydicom.Dataset]]] = defaultdict(list)
    skipped: list[str] = []
    for name in testdata.available():
        if not _is_candidate(name):
            continue
        path = testdata.fetch(name)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception as exc:
            skipped.append(f"{name} ({type(exc).__name__})")
            continue
        if getattr(ds, "Modality", None) != "MR":
            continue
        series[str(getattr(ds, "SeriesInstanceUID", f"<none:{name}>"))].append((name, ds))

    sys.stdout.reconfigure(encoding="utf-8")
    columns = [
        "Files",
        "SOP class",
        "Distinct SOPInstanceUIDs",
        "Layout",
        "Rows x Cols",
        "Manufacturer",
        "Transfer syntaxes",
    ]
    print("| " + " | ".join(columns) + " |")
    print("|---|---|---|---|---|---|---|")
    for _uid, members in sorted(series.items(), key=lambda kv: kv[1][0][0]):
        names = "<br>".join(f"`{n}`" for n, _ in members)
        ds = members[0][1]
        sop = UID(getattr(ds, "SOPClassUID", "")).name or "?"
        # Codec variants may or may not keep the SOPInstanceUID, so this is not a slice count.
        n_images = len({str(getattr(m, "SOPInstanceUID", n)) for n, m in members})
        syntaxes = sorted(
            {
                UID(m.file_meta.TransferSyntaxUID).name
                if "TransferSyntaxUID" in getattr(m, "file_meta", {})
                else "(no file meta)"
                for _, m in members
            }
        )
        size = f"{getattr(ds, 'Rows', '?')} x {getattr(ds, 'Columns', '?')}"
        maker = getattr(ds, "Manufacturer", "") or "—"
        print(
            f"| {names} | {sop} | {n_images} | {_classify(ds)} | {size} | {maker} "
            f"| {'<br>'.join(syntaxes)} |"
        )
    if skipped:
        print("\nUnreadable candidates:", ", ".join(skipped))


if __name__ == "__main__":
    main()
