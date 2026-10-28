"""What the user is told about a result, as lines of text (shared by the CLI and the GUI).

Lines never name patients, UIDs or paths: series are referred to by manifest id.
"""

from collections import Counter

from dcmtether.api import ConversionResult, DcmResult, ScanSummary
from dcmtether.comparison import Report
from dcmtether.derived_mr import DerivedMRResult
from dcmtether.manifest import ManifestUpdate
from dcmtether.pmap import PmapResult
from dcmtether.scanning import Status, plural
from dcmtether.seg import SegResult


def scan_lines(summary: ScanSummary, only: set[int] | None = None) -> list[str]:
    """Counts, then the non-supported series (all, or those in ``only``) and skipped files."""
    lines = [
        f"Scanned {summary.n_files} files: {summary.n_instances} DICOM instances in "
        f"{len(summary.series)} series, {plural(summary.n_skipped, 'file')} skipped."
    ]
    lines += [f"  {status.value + ':':<13}{summary.count(status)}" for status in Status]
    for series in summary.series:
        if only is not None and series.id not in only:
            continue
        if series.status is not Status.SUPPORTED:
            lines.append(
                f"  series #{series.id} ({plural(series.n_instances, 'instance')}) "
                f"{series.status.value}: {'; '.join(series.reasons)}"
            )
    lines += [f"  skipped {plural(n, 'file')}: {reason}" for reason, n in summary.skipped.items()]
    return lines


def update_lines(summary: ScanSummary, update: ManifestUpdate) -> list[str]:
    """A rescan of a dataset: the scan (new and rescanned series only), then the manifest."""
    lines = scan_lines(summary, only=set(update.new) | set(update.rescanned))
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


def seg_lines(result: SegResult) -> list[str]:
    lines = [
        f"Wrote DICOM SEG with {plural(len(result.segment_numbers), 'segment')} on "
        f"{plural(result.n_sources, 'source image')}."
    ]
    renumbered = {k: v for k, v in result.segment_numbers.items() if k != v}
    if renumbered:
        pairs = ", ".join(f"{k}->{v}" for k, v in sorted(renumbered.items()))
        lines.append(f"Label values renumbered to segment numbers: {pairs}.")
    return lines


def pmap_lines(result: PmapResult) -> list[str]:
    return [
        f"Wrote DICOM Parametric Map ({result.dtype}) on "
        f"{plural(result.n_sources, 'source image')}."
    ]


def mr_lines(result: DerivedMRResult) -> list[str]:
    lines = [f"Wrote {plural(len(result.paths), 'derived MR image')} ({result.kind})."]
    if result.max_error > 0:
        cause = "Rescale rounded to DS precision" if result.rounded else "Quantized to 16 bits"
        lines.append(f"{cause}: max error {result.max_error:.3g} (to-pmap is exact).")
    return lines
