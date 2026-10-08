"""Per-frame views of DICOM instances (Enhanced MR functional groups, PLAN.md Phase 4c).

A *view* is a flat dataset holding what one frame "is": the instance's top-level elements plus
the contents of every functional group that applies to the frame (shared first, then
per-frame, which wins). Code that reads geometry, rescale or echo/diffusion values per image
(scan, volume assembly, sidecar, to-dcm checks) works on views and does not care whether an
image is a classic single-frame instance or one frame of an enhanced instance.

Views share element objects with the source dataset; treat them as read-only.
"""

from pydicom.dataset import Dataset

SHARED_GROUPS = 0x52009229  # SharedFunctionalGroupsSequence
PER_FRAME_GROUPS = 0x52009230  # PerFrameFunctionalGroupsSequence
PIXEL_DATA = 0x7FE00010
#: Sequences inside a functional group item whose contents are flattened too:
#: DiffusionGradientDirectionSequence (holds DiffusionGradientOrientation).
_NESTED = frozenset({0x00189076})


def has_functional_groups(ds: Dataset) -> bool:
    return PER_FRAME_GROUPS in ds or SHARED_GROUPS in ds


def n_frames(ds: Dataset) -> int:
    try:
        return max(1, int(ds.get("NumberOfFrames") or 1))
    except (TypeError, ValueError):
        return 1


def _add_group(view: Dataset, group: Dataset) -> None:
    for sequence in group:
        if sequence.VR != "SQ" or not sequence.value:
            continue
        for elem in sequence.value[0]:  # a functional group macro has exactly one item
            if elem.tag in _NESTED and elem.VR == "SQ" and elem.value:
                for nested in elem.value[0]:
                    view.add(nested)
            else:
                view.add(elem)


def frame_views(ds: Dataset) -> list[Dataset]:
    """One view per frame. Classic instances: ``[ds]``; multi-frame without functional groups:
    the instance itself once per frame (geometry then stays missing)."""
    if not has_functional_groups(ds):
        return [ds] * n_frames(ds)
    base = Dataset()
    for elem in ds:
        if elem.tag not in (SHARED_GROUPS, PER_FRAME_GROUPS, PIXEL_DATA):
            base.add(elem)
    shared = ds.get("SharedFunctionalGroupsSequence")
    per_frame = ds.get("PerFrameFunctionalGroupsSequence") or []
    views = []
    for n in range(n_frames(ds)):
        view = Dataset()
        view.update(base)
        if shared:
            _add_group(view, shared[0])
        if n < len(per_frame):
            _add_group(view, per_frame[n])
        views.append(view)
    return views
