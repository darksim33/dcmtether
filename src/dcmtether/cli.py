"""Command-line interface (see docs/usage.md): the commands of :mod:`dcmtether.api`, printed.

Output and logs never name patients, UIDs or paths: series are referred to by manifest id.
Commands that work on a dataset append structured records to ``DATASET/logs/dcmtether_<time>.jsonl``
(:mod:`dcmtether.log`).
"""

import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import click
from tqdm import tqdm

from dcmtether import __version__, api
from dcmtether.api import ConversionResult, ScanSummary
from dcmtether.errors import DcmtetherError
from dcmtether.manifest import ManifestUpdate
from dcmtether.scanning import Status, plural

_existing_dir = click.Path(exists=True, file_okay=False, path_type=Path)
_existing_file = click.Path(exists=True, dir_okay=False, path_type=Path)
_output_dir = click.Path(file_okay=False, path_type=Path)
_verbose = click.option("--verbose", "-v", is_flag=True, help="Also list every series.")
_full_name = click.option(
    "--full-name",
    "-f",
    is_flag=True,
    help="Use the full patient name in subject labels instead of 5+5 letters; fixed for a "
    "dataset on its first conversion.",
)
_body_part = click.option(
    "--body-part",
    "-b",
    is_flag=True,
    help="List every scanned series with its BodyPartExamined.",
)


def _progress(iterable: Iterable, **kwargs) -> Iterable:
    """Progress bar on stderr, only on a terminal (``TQDM_DISABLE=1`` turns it off); it is
    removed when done, so the summary on stdout reads as without it."""
    return tqdm(iterable, file=sys.stderr, leave=False, disable=None, dynamic_ncols=True, **kwargs)


@contextmanager
def _errors() -> Iterator[None]:
    """Expected failures become one-line errors (exit code 1); their messages are safe."""
    try:
        yield
    except DcmtetherError as exc:
        raise click.ClickException(str(exc)) from None


@click.group(help="Lossless DICOM <-> NIfTI round trips for MRI series. Research use only.")
@click.version_option(__version__, prog_name="dcmtether")
def main() -> None:
    pass


# --- scanning and conversion -------------------------------------------------------------------


def _report_scan(
    summary: ScanSummary, only: set[int] | None = None, body_part: bool = False
) -> None:
    """Counts, then the non-supported series (by manifest id; every series with
    ``body_part``) and the skipped files."""
    click.echo(
        f"Scanned {summary.n_files} files: {summary.n_instances} DICOM instances in "
        f"{len(summary.series)} series, {plural(summary.n_skipped, 'file')} skipped."
    )
    for status in Status:
        click.echo(f"  {status.value + ':':<13}{summary.count(status)}")
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
        click.echo(line)
    for reason, n in summary.skipped.items():
        click.echo(f"  skipped {plural(n, 'file')}: {reason}")


def _report_update(summary: ScanSummary, update: ManifestUpdate, body_part: bool) -> None:
    _report_scan(summary, only=set(update.new) | set(update.rescanned), body_part=body_part)
    click.echo(
        f"Manifest: {len(update.new)} new series, {len(update.rescanned)} rescanned, "
        f"{len(update.unchanged)} unchanged."
    )
    for series_id in update.changed_after_conversion:
        click.echo(
            f"  series #{series_id}: source files changed after conversion (not "
            "reconverted; remove its outputs and manifest entry to redo it)"
        )
    for series_id in update.missing:
        click.echo(f"  series #{series_id}: no longer found in SOURCE (kept in the manifest)")


def _report_conversion(result: ConversionResult, verbose: bool) -> None:
    """Console lines for a conversion; exit code 1 if a series failed."""
    click.echo(f"Converted {len(result.converted)} series, {len(result.failed)} failed.")
    for outcome in result.outcomes:
        if outcome.ok:
            if verbose:
                click.echo(
                    f"  series #{outcome.series_id} converted ({plural(outcome.outputs, 'output')})"
                )
            if outcome.warning:
                click.echo(f"  series #{outcome.series_id} warning: {outcome.warning}")
        else:
            click.echo(f"  series #{outcome.series_id} failed: {outcome.message}")
    if not result.ok:
        raise click.exceptions.Exit(1)


@main.command()
@click.argument("source", type=_existing_dir)
@click.argument("dataset", type=_output_dir)
@_body_part
def scan(source: Path, dataset: Path, body_part: bool) -> None:
    """Index DICOM series under SOURCE and classify them (headers only).

    Writes the manifest to the DATASET directory.
    """
    with _errors():
        summary = api.scan(source, dataset, progress=_progress)
    _report_scan(summary, body_part=body_part)
    click.echo("Manifest written.")


@main.command("to-nii")
@click.argument("dataset", type=_existing_dir)
@_full_name
@_verbose
def to_nii(dataset: Path, full_name: bool, verbose: bool) -> None:
    """Convert scanned series in DATASET to NIfTI + tether.json.

    Converts every 'supported' or 'split' series not converted yet; failed ones are retried.
    File names contain patient names and study dates, so the dataset must not be shared as is.
    """
    with _errors():
        result = api.to_nii(dataset, full_name=full_name, progress=_progress)
    _report_conversion(result, verbose)


@main.command()
@click.argument("source", type=_existing_dir)
@click.option(
    "--output",
    "-o",
    "dataset",
    type=_output_dir,
    help="Dataset directory (default: SOURCE itself).",
)
@_full_name
@_body_part
@_verbose
def run(
    source: Path, dataset: Path | None, full_name: bool, body_part: bool, verbose: bool
) -> None:
    """Scan SOURCE and convert it into a dataset; safe to run again.

    The dataset is written into SOURCE unless --output names another directory. The first run
    creates it. Later runs (same SOURCE and output) add new series, rescan series that are not
    converted yet and retry failed ones; converted series are never touched (if their source
    files changed, that is reported).
    """

    def scanned(summary: ScanSummary, update: ManifestUpdate) -> None:
        _report_update(summary, update, body_part)

    with _errors():
        result = api.run(source, dataset, full_name=full_name, progress=_progress, _scanned=scanned)
    _report_conversion(result.conversion, verbose)


def _series_ids(ctx, param, value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        return [int(v) for v in value.split(",") if v.strip()]
    except ValueError:
        raise click.BadParameter("expected manifest ids like 4,5,6") from None


@main.command()
@click.argument("dataset", type=_existing_dir)
@click.option(
    "--ids",
    callback=_series_ids,
    help="Merge these series (manifest ids, e.g. 4,5,6) instead of finding groups.",
)
def merge(dataset: Path, ids: list[int] | None) -> None:
    """Merge series of one measurement into one 4D NIfTI (experimental).

    For scanners that write one series per b-value (e.g. IVIM). Without --ids, groups are found
    automatically. Writes DATASET/derivatives/merged/; the converted series stay as they are.
    Exit code 1 if a named group was not merged.
    """
    with _errors():
        result = api.merge_series(dataset, ids)
    if not result.groups:
        click.echo("No series to merge.")
    for group in result.groups:
        listed = ", ".join(f"#{i}" for i in group.series_ids)
        if group.path is not None:
            click.echo(f"Merged series {listed} ({plural(group.n_volumes, 'volume')}).")
        else:
            click.echo(f"Not merged: series {listed}: {group.skipped}.")
    if ids is not None and result.skipped:
        raise click.exceptions.Exit(1)


# --- reverse and comparison --------------------------------------------------------------------


@main.command("to-dcm")
@click.argument("nifti", type=_existing_file)
@click.argument("output", type=_output_dir)
def to_dcm(nifti: Path, output: Path) -> None:
    """Rebuild DICOM from a NIfTI file and its *.tether.json.

    Reads NIFTI (plus the matching *.tether.json next to it) and writes DICOM files to OUTPUT,
    under their original names relative to the series' source directory. Existing files are
    never overwritten.
    """
    with _errors():
        result = api.to_dcm(nifti, output)
    click.echo(f"Wrote {plural(len(result.paths), 'DICOM file')}.")
    if result.transfer_syntax_change:
        source, written = result.transfer_syntax_change
        click.echo(f"Transfer syntax changed from {source} to {written} (lossless).")


@main.command()
@click.argument("original", type=_existing_dir)
@click.argument("rebuilt", type=_existing_dir)
def compare(original: Path, rebuilt: Path) -> None:
    """Compare ORIGINAL and REBUILT DICOM series element by element.

    Instances are paired by SOPInstanceUID. Exit code 0 if identical, 1 if not.
    """
    report = api.compare(original, rebuilt)
    compared = plural(report.n_compared, "instance")
    if report.identical:
        click.echo(f"Identical: {compared} compared.")
    else:
        click.echo(f"{plural(len(report.differences), 'difference')} in {compared} compared:")
        for line in report.lines():
            click.echo(f"  {line}")
    # Notes repeat per instance (e.g. a transfer syntax change): print each kind once.
    for (where, kind), n in Counter((note.where, note.kind) for note in report.notes).items():
        click.echo(f"  note: {where}: {kind} ({plural(n, 'instance')})")
    if not report.identical:
        raise click.exceptions.Exit(1)


# --- derivatives -------------------------------------------------------------------------------


@main.command("to-seg")
@click.argument("nifti", type=_existing_file)
@click.argument("labels", type=_existing_file)
@click.argument("metadata", type=_existing_file)
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def to_seg_command(nifti: Path, labels: Path, metadata: Path, output: Path) -> None:
    """Write a DICOM SEG for the label map LABELS on the grid of NIFTI.

    NIFTI is a converted source (with its *.tether.json); LABELS must have the same shape and
    affine. METADATA is a dcmqi segmentation template (JSON). Writes OUTPUT (a new file).
    Research use only.
    """
    with _errors():
        result = api.to_seg(nifti, labels, metadata, output)
    click.echo(
        f"Wrote DICOM SEG with {plural(len(result.segment_numbers), 'segment')} on "
        f"{plural(result.n_sources, 'source image')}."
    )
    renumbered = {k: v for k, v in result.segment_numbers.items() if k != v}
    if renumbered:
        pairs = ", ".join(f"{k}->{v}" for k, v in sorted(renumbered.items()))
        click.echo(f"Label values renumbered to segment numbers: {pairs}.")


@main.command("to-pmap")
@click.argument("nifti", type=_existing_file)
@click.argument("map_nifti", metavar="MAP", type=_existing_file)
@click.argument("metadata", type=_existing_file)
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def to_pmap_command(nifti: Path, map_nifti: Path, metadata: Path, output: Path) -> None:
    """Write a DICOM Parametric Map for MAP (e.g. ADC, QSM) on the grid of NIFTI.

    NIFTI is a converted source (with its *.tether.json); MAP must have the same shape and
    affine. METADATA is a dcmqi parametric map template (JSON: quantity, units, method).
    Writes OUTPUT (a new file). Research use only.
    """
    with _errors():
        result = api.to_pmap(nifti, map_nifti, metadata, output)
    click.echo(
        f"Wrote DICOM Parametric Map ({result.dtype}) on "
        f"{plural(result.n_sources, 'source image')}."
    )


@main.command("to-mr")
@click.argument("nifti", type=_existing_file)
@click.argument("derivative", type=_existing_file)
@click.argument("metadata", type=_existing_file)
@click.argument("output", type=_output_dir)
def to_mr_command(nifti: Path, derivative: Path, metadata: Path, output: Path) -> None:
    """Write DERIVATIVE as derived MR images (one per source slice) into OUTPUT.

    Fallback for PACS without SEG/Parametric Map support. DERIVATIVE lies on the grid of
    NIFTI; METADATA is a dcmqi SEG template (label map: label values as pixels) or parametric
    map template (map: float maps are quantized to 16 bits). Research use only.
    """
    with _errors():
        result = api.to_mr(nifti, derivative, metadata, output)
    click.echo(f"Wrote {plural(len(result.paths), 'derived MR image')} ({result.kind}).")
    if result.max_error > 0:
        cause = "Rescale rounded to DS precision" if result.rounded else "Quantized to 16 bits"
        click.echo(f"{cause}: max error {result.max_error:.3g} (to-pmap is exact).")
