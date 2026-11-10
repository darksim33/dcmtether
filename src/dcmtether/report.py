"""What the user is told about a result, as lines of text (shared by the CLI and the GUI).

Lines never name patients, UIDs or paths: series are referred to by manifest id.
"""

from collections import Counter

from dcmtether.api import ConversionResult, DcmResult, ScanSummary
from dcmtether.comparison import Report
from dcmtether.derived_mr import DerivedMRResult
from dcmtether.manifest import ManifestUpdate
from dcmtether.merge import MergeResult
from dcmtether.pmap import PmapResult
from dcmtether.scanning import Status, plural
from dcmtether.seg import SegResult


def scan_lines(
    summary: ScanSummary, only: set[int] | None = None, body_part: bool = False
) -> list[str]:
    """Counts, then the non-supported series (all, or those in ``only``) and skipped files.

    With ``body_part``, every series is listed with its BodyPartExamined."""
    lines = [
        f"Scanned {summary.n_files} files: {summary.n_instances} DICOM instances in "
        f"{len(summary.series)} series, {plural(summary.n_skipped, 'file')} skipped."
    ]
    lines += [f"  {status.value + ':':<13}{summary.count(status)}" for status in Status]
    for series in summary.series:
        if only is not None and series.id not in only:
            continue
        if series.status is Status.SUPPORTED and not body_part:
            continue
        details = plural(series.n_instances, "instance")
        if body_part:
            details += f", body part {series.body_part or 'not set'}"
        line = f"  series #{series.id} ({details}) {series.status.value}"
        if series.status is not Status.SUPPORTED:
            line += f": {'; '.join(series.reasons)}"
        lines.append(line)
    lines += [f"  skipped {plural(n, 'file')}: {reason}" for reason, n in summary.skipped.items()]
    return lines


def update_lines(
    summary: ScanSummary, update: ManifestUpdate, body_part: bool = False
) -> list[str]:
    """A rescan of a dataset: the scan (new and rescanned series only), then the manifest."""
    lines = scan_lines(summary, only=set(update.new) | set(update.rescanned), body_part=body_part)
    lines.append(
        f"Manifest: {len(update.new)} new series, {len(update.rescanned)} rescanned, "
        f"{len(update.unchanged)} unchanged."
    )
    lines += [
        f"  series #{series_id}: source files changed after conversion (not reconverted; "
        "remove its outputs and manifest entry to redo it)"
        for series_id in update.changed_after_conversion
    ]
    lines += [
        f"  series #{series_id}: no longer found in SOURCE (kept in the manifest)"
        for series_id in update.missing
    ]
    return lines


def conversion_lines(result: ConversionResult, verbose: bool = False) -> list[str]:
    lines = [f"Converted {len(result.converted)} series, {len(result.failed)} failed."]
    for outcome in result.outcomes:
        if outcome.ok:
            if verbose:
                lines.append(
                    f"  series #{outcome.series_id} converted ({plural(outcome.outputs, 'output')})"
                )
            if outcome.warning:
                lines.append(f"  series #{outcome.series_id} warning: {outcome.warning}")
        else:
            lines.append(f"  series #{outcome.series_id} failed: {outcome.message}")
    return lines


def dcm_lines(result: DcmResult) -> list[str]:
    lines = [f"Wrote {plural(len(result.paths), 'DICOM file')}."]
    if result.transfer_syntax_change:
        source, written = result.transfer_syntax_change
        lines.append(f"Transfer syntax changed from {source} to {written} (lossless).")
    return lines


def compare_lines(report: Report) -> list[str]:
    compared = plural(report.n_compared, "instance")
    if report.identical:
        lines = [f"Identical: {compared} compared."]
    else:
        lines = [f"{plural(len(report.differences), 'difference')} in {compared} compared:"]
        lines += [f"  {line}" for line in report.lines()]
    # Notes repeat per instance (e.g. a transfer syntax change): each kind once.
    counts = Counter((note.where, note.kind) for note in report.notes)
    lines += [
        f"  note: {where}: {kind} ({plural(n, 'instance')})" for (where, kind), n in counts.items()
    ]
    return lines


UID_FALLBACK = "note: source UID stem not usable, synthetic 2.25 UIDs written"


def _uid_note(uid_fallback: bool) -> list[str]:
    return [UID_FALLBACK] if uid_fallback else []


def seg_lines(result: SegResult) -> list[str]:
    lines = [
        f"Wrote DICOM SEG with {plural(len(result.segment_numbers), 'segment')} on "
        f"{plural(result.n_sources, 'source image')}."
    ]
    renumbered = {k: v for k, v in result.segment_numbers.items() if k != v}
    if renumbered:
        pairs = ", ".join(f"{k}->{v}" for k, v in sorted(renumbered.items()))
        lines.append(f"Label values renumbered to segment numbers: {pairs}.")
    return lines + _uid_note(result.uid_fallback)


def pmap_lines(result: PmapResult) -> list[str]:
    return [
        f"Wrote DICOM Parametric Map ({result.dtype}) on "
        f"{plural(result.n_sources, 'source image')}.",
        *_uid_note(result.uid_fallback),
    ]


def mr_lines(result: DerivedMRResult) -> list[str]:
    lines = [f"Wrote {plural(len(result.paths), 'derived MR image')} ({result.kind})."]
    if result.max_error > 0:
        cause = "Rescale rounded to DS precision" if result.rounded else "Quantized to 16 bits"
        lines.append(f"{cause}: max error {result.max_error:.3g} (to-pmap is exact).")
    return lines + _uid_note(result.uid_fallback)


def merge_lines(result: MergeResult) -> list[str]:
    if not result.groups:
        return ["No series to merge."]
    lines = []
    for group in result.groups:
        listed = ", ".join(f"#{i}" for i in group.series_ids)
        if group.path is not None:
            replaced = "; replaced the single series" if group.replaced else ""
            lines.append(f"Merged series {listed} ({plural(group.n_volumes, 'volume')}){replaced}.")
            if group.no_tether:
                lines.append(
                    f"  no tether.json ({group.no_tether}): to-dcm needs the single series"
                )
            if group.not_replaced:
                lines.append(f"  not replaced ({group.not_replaced}): the single series stay")
        else:
            lines.append(f"Not merged: series {listed}: {group.skipped}.")
    return lines


def merge_ok(result: MergeResult, ids_given: bool) -> bool:
    """False if a named group was not merged, or a group was not replaced."""
    return not ((ids_given and result.skipped) or any(g.not_replaced for g in result.groups))
