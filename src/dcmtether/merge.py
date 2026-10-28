"""Merge series of one measurement into one 4D NIfTI (experimental).

Some scanners write one series per b-value (e.g. IVIM). Each is converted on its own, with its
own tether.json. Merging adds one 4D NIfTI, sidecar and tether.json per group under
``DATASET/derivatives/merged/`` (same subject/session/datatype folders as the dataset), or
``in_place`` beside the series. ``replace`` (in place) then removes the single series' files;
the manifest records the merged file as the first series' output (the others point to it in
``merged_into``), so nothing is converted again. Only with a tether.json: without one the single
series stay. The tether.json combines the series' own: ``to-dcm`` on the merged NIfTI writes the
source series back, identical to the originals. Only for unchanged voxels: the files keep their
original UIDs. Series merged as float (different stored scaling) get no tether.json.

Groups are found automatically, or named by manifest id. Every group must share subject,
session, study, frame of reference and voxel grid. Automatic groups, one per voxel grid, come
from two rules, in this order:

- Siemens XA split (``siemens-xa-split``): series that Siemens XA split from one enhanced
  acquisition (``MFSPLIT`` in ImageType, XA software) with the same SeriesTime, as dcm2niix
  groups XA classic data; any series type (diffusion, echoes, dynamics, ...), ORIGINAL/DERIVED
  and the pixel component (magnitude, phase, ...) kept apart. SeriesNumbers may differ freely.
- Diffusion protocol (``diffusion-protocol``): single-volume diffusion series (suffix ``dwi``)
  with the same protocol (ProtocolName, else SeriesDescription, without b-value tokens such as
  ``b50``) and echo time, whose b-value and gradient differ from each other (a repeated
  measurement is not merged).

Volumes are ordered by b-value if every series has one, then by gradient direction (in the
order the directions were acquired), else by SeriesNumber. Messages name series by manifest id only.
"""

import json
import posixpath
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from dcmtether import manifest, tether
from dcmtether.errors import DcmtetherError
from dcmtether.forward import NIFTI_SUFFIX, SIDECAR_SUFFIX, write_nifti
from dcmtether.progress import Progress, no_progress
from dcmtether.scanning import DERIVATIVES_DIR, MERGED_DIR

#: Sidecar keys that are always listed per volume (like in a converted diffusion series).
_ALWAYS_PER_VOLUME = (
    "DiffusionBValue",
    "DiffusionGradientOrientation",
    "DiffusionGradientVendor",
    "DiffusionBMatrix",
)
#: b-value tokens in a protocol name or description: "b50", "b_800", "B-1000", "b=0".
_B_TOKEN = re.compile(r"(?<![a-z0-9])b[\s_=-]?\d+(?![0-9])", re.I)
_AFFINE_ATOL = 1e-3


class MergeError(DcmtetherError, ValueError):
    pass


#: Grouping rules, recorded in the merged sidecar (``DcmTether.MergeRule``).
XA_SPLIT = "siemens-xa-split"
DIFFUSION_PROTOCOL = "diffusion-protocol"
IDS = "ids"


@dataclass(frozen=True)
class MergedGroup:
    #: Manifest ids, in volume order (b-value, gradient direction, SeriesNumber).
    series_ids: tuple[int, ...]
    n_volumes: int
    #: The merged NIfTI, or ``None`` if the group was not written.
    path: Path | None = None
    #: Why it was not written.
    skipped: str | None = None
    #: The merged tether.json (``to-dcm`` writes the source series back), or ``None``.
    tether_path: Path | None = None
    #: Why no tether.json was written.
    no_tether: str | None = None
    #: How the group was found: :data:`XA_SPLIT`, :data:`DIFFUSION_PROTOCOL` or :data:`IDS`.
    rule: str = ""
    #: ``replace=True``: the single series' files were removed in favor of the merged ones.
    replaced: bool = False
    #: ``replace=True``: why the single series were kept.
    not_replaced: str | None = None


@dataclass(frozen=True)
class MergeResult:
    groups: tuple[MergedGroup, ...]

    @property
    def merged(self) -> list[MergedGroup]:
        return [g for g in self.groups if g.path is not None]

    @property
    def skipped(self) -> list[MergedGroup]:
        return [g for g in self.groups if g.path is None]


@dataclass(frozen=True)
class _Member:
    series_id: int
    series_number: int | None
    subject: str
    session: str
    study: str | None
    nifti: Path
    sidecar: dict
    frame_of_reference: str | None
    shape: tuple[int, ...]
    affine: np.ndarray
    tether: Path
    #: Source folder, POSIX, relative to the source root.
    source_dir: str
    #: SeriesTime: a grouping key only, never printed, logged or written (PHI rules).
    series_time: str | None = None
    sidecar_file: Path | None = None

    @property
    def n_volumes(self) -> int:
        return self.shape[3] if len(self.shape) > 3 else 1

    @property
    def is_dwi(self) -> bool:
        return self.nifti.name.endswith(f"_dwi{NIFTI_SUFFIX}")

    def per_volume(self) -> dict[str, list]:
        """Every sidecar key as one value per volume."""
        info = self.sidecar.get("DcmTether", {})
        values = {k: [v] * self.n_volumes for k, v in self.sidecar.items() if k != "DcmTether"}
        values.update(info.get("PerVolume", {}))
        return values


def _first_value(document: dict, tag: str) -> str | None:
    """First value of an element: shared, else of the first instance."""
    element = document["shared"].get(tag) or document["instances"][0]["elements"].get(tag)
    return (element.get("Value") or [None])[0] if element else None


def _members(dataset: Path, progress: Progress = no_progress) -> dict[int, _Member | str]:
    """Converted series with one output, by id; a reason instead if one cannot be merged."""
    db = manifest.open_manifest(dataset)
    try:
        rows = db.execute(
            "SELECT s.id, s.series_number, s.study_instance_uid, s.subject, s.session, "
            "s.source_dir, o.nifti_path, o.sidecar_path, o.tether_path, "
            "(SELECT COUNT(*) FROM outputs o2 WHERE o2.series_id = s.id) AS n_outputs "
            "FROM series s JOIN outputs o ON o.series_id = s.id "
            "WHERE s.conversion_status = 'converted' AND o.chunk = 1 ORDER BY s.id"
        ).fetchall()
    finally:
        db.close()
    members: dict[int, _Member | str] = {}
    for row in progress(rows, total=len(rows), desc="Reading series", unit="series"):
        if row["n_outputs"] > 1:
            members[row["id"]] = "split into several NIfTI files"
            continue
        try:
            nifti = dataset / row["nifti_path"]
            header = nib.load(nifti).header
            sidecar = json.loads((dataset / row["sidecar_path"]).read_text("utf-8"))
            document = tether.load(dataset / row["tether_path"])
        except (OSError, ValueError, tether.TetherError):
            members[row["id"]] = "outputs missing or unreadable"
            continue
        members[row["id"]] = _Member(
            row["id"],
            row["series_number"],
            row["subject"],
            row["session"],
            row["study_instance_uid"],
            nifti,
            sidecar,
            _first_value(document, "00200052"),  # FrameOfReferenceUID
            tuple(int(n) for n in header.get_data_shape()),
            header.get_best_affine(),
            dataset / row["tether_path"],
            row["source_dir"],
            _first_value(document, "00080031"),  # SeriesTime
            dataset / row["sidecar_path"],
        )
    return members


def _b_value(member: _Member) -> float | None:
    """The smallest b-value of a series, or ``None`` if it has none."""
    values = [b for b in member.per_volume().get("DiffusionBValue", []) if b is not None]
    try:
        return min(float(b) for b in values) if values else None
    except (TypeError, ValueError):
        return None


def _direction(member: _Member) -> tuple | None:
    """Gradient direction of a series' first volume (rounded), or ``None`` (b0, trace)."""
    gradient = _diffusion_key(member)[1]
    try:
        return None if gradient is None else tuple(round(float(g), 4) for g in gradient)
    except (TypeError, ValueError):
        return None


def _ordered(members: list[_Member]) -> list[_Member]:
    """By b-value if every series has one, then gradient direction, then SeriesNumber, then id.

    Directions rank in the order they first appear by SeriesNumber (acquisition order), so
    every b-value lists them alike; no direction comes first.
    """

    def by_number(m: _Member) -> tuple:
        return (m.series_number is None, m.series_number or 0, m.series_id)

    b_values = {m.series_id: _b_value(m) for m in members}
    if not all(b is not None for b in b_values.values()):
        return sorted(members, key=by_number)
    directions = {m.series_id: _direction(m) for m in members}
    rank: dict[tuple, int] = {}
    for m in sorted(members, key=by_number):
        direction = directions[m.series_id]
        if direction is not None:
            rank.setdefault(direction, len(rank))
    return sorted(
        members,
        key=lambda m: (
            b_values[m.series_id],
            -1 if directions[m.series_id] is None else rank[directions[m.series_id]],
            *by_number(m),
        ),
    )


def _mismatch(members: list[_Member]) -> str | None:
    """Why ``members`` cannot be one 4D image, or ``None``."""
    first = members[0]
    checks = [
        ("subject or session", lambda m: (m.subject, m.session)),
        ("study", lambda m: m.study),
        ("frame of reference", lambda m: m.frame_of_reference),
        ("image size", lambda m: m.shape[:3]),
    ]
    for what, key in checks:
        if any(key(m) != key(first) for m in members):
            return f"different {what}"
    if any(not np.allclose(m.affine, first.affine, atol=_AFFINE_ATOL) for m in members):
        return "different slice positions or orientation"
    return None


def _protocol(member: _Member) -> str | None:
    text = member.sidecar.get("ProtocolName") or member.sidecar.get("SeriesDescription")
    if not text:
        return None
    return re.sub(r"[^a-z0-9]+", "", _B_TOKEN.sub("", str(text)).lower()) or None


def _diffusion_key(member: _Member) -> tuple:
    per_volume = member.per_volume()
    b_value = per_volume.get("DiffusionBValue", [None])[0]
    gradient = (
        per_volume.get("DiffusionGradientOrientation")
        or per_volume.get("DiffusionGradientVendor")
        or [None]
    )[0]
    return (b_value, None if gradient is None else tuple(gradient))


def _grid(m: _Member) -> tuple:
    """Separates acquisitions of one protocol at different positions."""
    return (m.shape[:3], tuple(np.round(m.affine, 2).ravel().tolist()))


def _image_type(m: _Member) -> list[str]:
    value = m.sidecar.get("ImageType") or []
    return [str(v).upper() for v in ([value] if isinstance(value, str) else value)]


def _xa_split(m: _Member) -> bool:
    """A classic series that Siemens XA split from one enhanced acquisition."""
    return (
        "MFSPLIT" in _image_type(m)
        and "SIEMENS" in str(m.sidecar.get("Manufacturer", "")).upper()
        and "XA" in str(m.sidecar.get("SoftwareVersions", "")).upper()
        and m.series_time is not None
    )


def _xa_split_groups(members: list[_Member]) -> list[list[_Member]]:
    """Series of one XA acquisition: same SeriesTime (as dcm2niix groups XA classic data)."""
    by_key: dict[tuple, list[_Member]] = defaultdict(list)
    for m in members:
        if not _xa_split(m):
            continue
        image_type = _image_type(m)
        # ORIGINAL/DERIVED and the pixel component (M, P, R, I) keep e.g. phase apart
        kind = (tuple(image_type[:1]), tuple(image_type[2:3]))
        key = (m.subject, m.session, m.study, m.frame_of_reference, m.series_time, kind, _grid(m))
        by_key[key].append(m)
    return [found for found in by_key.values() if len(found) > 1]


def find_groups(members: list[_Member]) -> list[tuple[list[_Member], str | None, str]]:
    """Automatic groups (see the module docstring): members, the reason a group is not merged
    (or ``None``), and the rule that found it."""
    groups = []
    for found in _xa_split_groups(members):
        found = _ordered(found)
        groups.append((found, _mismatch(found), XA_SPLIT))
    taken = {m.series_id for found, _, _ in groups for m in found}
    by_key: dict[tuple, list[_Member]] = defaultdict(list)
    for m in members:
        if m.series_id in taken:
            continue
        protocol = _protocol(m)
        if not m.is_dwi or m.n_volumes != 1 or protocol is None:
            continue
        echo_time = m.sidecar.get("EchoTime")
        key = (m.subject, m.session, m.study, m.frame_of_reference, protocol, echo_time, _grid(m))
        by_key[key].append(m)
    for found in by_key.values():
        if len(found) < 2:
            continue
        found = _ordered(found)
        reason = _mismatch(found)
        diffusion = [_diffusion_key(m) for m in found]
        if reason is None and any(b is None for b, _ in diffusion):
            reason = "b-value missing"
        if reason is None and len(set(diffusion)) < len(diffusion):
            reason = "b-values repeat (a repeated measurement?)"
        groups.append((found, reason, DIFFUSION_PROTOCOL))
    return sorted(groups, key=lambda g: g[0][0].series_id)


def _merged_sidecar(members: list[_Member]) -> dict:
    columns = [m.per_volume() for m in members]
    keys = list(dict.fromkeys(k for c in columns for k in c))
    out: dict = {}
    per_volume: dict = {}
    for key in keys:
        values = [
            v
            for m, c in zip(members, columns, strict=True)
            for v in c.get(key, [None] * m.n_volumes)
        ]
        if key not in _ALWAYS_PER_VOLUME and all(v == values[0] for v in values):
            if values[0] is not None:
                out[key] = values[0]
        else:
            per_volume[key] = values
    infos = [m.sidecar.get("DcmTether", {}) for m in members]
    info: dict = {"Version": infos[0].get("Version"), "MergedFrom": [m.series_id for m in members]}
    sources = list(dict.fromkeys(i["DiffusionSource"] for i in infos if "DiffusionSource" in i))
    if sources:
        info["DiffusionSource"] = "; ".join(sources)
    info["PerVolume"] = per_volume
    out["DcmTether"] = info
    return out


def _merged_data(members: list[_Member]) -> tuple[np.ndarray, tuple[float, float] | None, bool]:
    """4D data, NIfTI scaling, and whether the values had to be rescaled to float."""
    images = [nib.load(m.nifti) for m in members]
    # nibabel moves scl_slope/scl_inter from the header to the array proxy (1, 0 if unset)
    scalings = {(float(img.dataobj.slope), float(img.dataobj.inter)) for img in images}
    dtypes = {img.get_data_dtype() for img in images}
    if len(scalings) == 1 and len(dtypes) == 1:
        arrays = [np.asanyarray(img.dataobj.get_unscaled()) for img in images]
        scaling = scalings.pop()
        scaling = None if scaling == (1.0, 0.0) else scaling
        rescaled = False
    else:  # different stored values: the real values, as float
        arrays = [np.asanyarray(img.dataobj, dtype=np.float32) for img in images]
        scaling, rescaled = None, True
    arrays = [a if a.ndim == 4 else a[..., np.newaxis] for a in arrays]
    return np.concatenate(arrays, axis=3), scaling, rescaled


def _output_path(dataset: Path, first: _Member, in_place: bool) -> Path:
    relative = first.nifti.relative_to(dataset)
    stem, suffix = relative.name.removesuffix(NIFTI_SUFFIX).rsplit("_", 1)
    name = f"{stem}_desc-merged_{suffix}{NIFTI_SUFFIX}"
    folder = dataset if in_place else dataset / DERIVATIVES_DIR / MERGED_DIR
    return folder / relative.parent / name


def _merged_tether(members: list[_Member]) -> dict:
    """The members' tether documents as one (:func:`dcmtether.tether.combine`)."""
    folders = [m.source_dir or "." for m in members]
    common = (posixpath.commonpath(folders) or ".") if len(set(folders)) > 1 else folders[0]
    prefixes = [posixpath.relpath(f, common) for f in folders]
    prefixes = ["" if p == "." else p for p in prefixes]
    return tether.combine([tether.load(m.tether) for m in members], prefixes)


def _replace(dataset: Path, members: list[_Member], targets: tuple[Path, ...], n: int) -> None:
    """The merged files become the first series' output; then the single files go."""
    nifti, sidecar, tether_file = (t.relative_to(dataset).as_posix() for t in targets)
    output = manifest.Output(1, nifti, tether_file, sidecar, n)
    db = manifest.open_manifest(dataset)
    try:
        manifest.record_merged(db, members[0].series_id, [m.series_id for m in members[1:]], output)
    finally:
        db.close()
    for m in members:
        for path in (m.nifti, m.sidecar_file, m.tether):
            if path is not None:
                path.unlink(missing_ok=True)


def _merged_from(targets: tuple[Path, ...]) -> set[int] | None:
    """Series ids of complete merged files (NIfTI, sidecar and tether.json), else ``None``."""
    if not all(t.exists() for t in targets):
        return None
    try:
        info = json.loads(targets[1].read_text("utf-8"))["DcmTether"]
        return {int(i) for i in info["MergedFrom"]}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_group(
    dataset: Path,
    members: list[_Member],
    rule: str,
    *,
    in_place: bool = False,
    replace: bool = False,
) -> MergedGroup:
    """Write one group to ``derivatives/merged`` or, ``in_place``, beside its series;
    ``replace`` (implies ``in_place``) then removes the single series if a tether.json was
    written."""
    ids = tuple(m.series_id for m in members)
    n_volumes = sum(m.n_volumes for m in members)
    nifti_path = _output_path(dataset, members[0], in_place or replace)
    stem = nifti_path.name.removesuffix(NIFTI_SUFFIX)
    sidecar_path = nifti_path.with_name(stem + SIDECAR_SUFFIX)
    tether_path = nifti_path.with_name(stem + tether.TETHER_SUFFIX)
    targets = (nifti_path, sidecar_path, tether_path)
    if any(t.exists() for t in targets):
        if replace and _merged_from(targets) == set(ids):
            # merged in place before (--in-place, or interrupted): replace with those files
            _replace(dataset, members, targets, n_volumes)
            return MergedGroup(
                ids, n_volumes, nifti_path, tether_path=tether_path, rule=rule, replaced=True
            )
        return MergedGroup(ids, n_volumes, skipped="already merged", rule=rule)
    data, scaling, rescaled = _merged_data(members)
    info = _merged_sidecar(members)
    info["DcmTether"]["MergeRule"] = rule
    document, no_tether = None, None
    if rescaled:
        info["DcmTether"]["Rescaled"] = True  # stored values differed: real values as float
        no_tether = "stored values rescaled to float"
    else:
        try:
            document = _merged_tether(members)
        except tether.TetherError as exc:
            no_tether = str(exc)
    nifti_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_nifti(nifti_path, data, members[0].affine, scaling)
        sidecar_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        if document is not None:
            tether.write(document, tether_path)
    except BaseException:
        for target in targets:
            target.unlink(missing_ok=True)
        raise
    written = tether_path if document is not None else None
    replaced, not_replaced = False, None
    if replace and written is None:
        not_replaced = "no tether.json"
    elif replace:
        _replace(dataset, members, targets, n_volumes)
        replaced = True
    return MergedGroup(
        ids,
        n_volumes,
        nifti_path,
        tether_path=written,
        no_tether=no_tether,
        rule=rule,
        replaced=replaced,
        not_replaced=not_replaced,
    )


def merge(
    dataset: Path,
    ids: list[int] | None = None,
    progress: Progress = no_progress,
    *,
    in_place: bool = False,
    replace: bool = False,
) -> MergeResult:
    """Merge the automatic groups of ``dataset``, or the series ``ids`` as one group
    (placement: :func:`write_group`)."""
    place = {"in_place": in_place, "replace": replace}
    members = _members(dataset, progress)
    if ids is None:
        candidates = [m for m in members.values() if isinstance(m, _Member)]
        found_groups = find_groups(candidates)
        results = []
        for group, reason, rule in progress(
            found_groups, total=len(found_groups), desc="Merging", unit="group"
        ):
            if reason is None:
                results.append(write_group(dataset, group, rule, **place))
            else:
                n = sum(m.n_volumes for m in group)
                found = tuple(m.series_id for m in group)
                results.append(MergedGroup(found, n, skipped=reason, rule=rule))
        return MergeResult(tuple(results))

    if len(set(ids)) < 2:
        raise MergeError("name at least two different series")
    chosen = []
    for series_id in sorted(set(ids)):
        member = members.get(series_id)
        if member is None:
            raise MergeError(f"series #{series_id} is not a converted series")
        if isinstance(member, str):
            raise MergeError(f"series #{series_id}: {member}")
        chosen.append(member)
    chosen = _ordered(chosen)
    reason = _mismatch(chosen)
    if reason is not None:
        raise MergeError(f"series cannot be merged: {reason}")
    [written] = progress([chosen], total=1, desc="Merging", unit="group")
    return MergeResult((write_group(dataset, written, IDS, **place),))
