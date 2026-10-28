"""Command-line interface (see docs/usage.md): the commands of :mod:`dcmtether.api`, printed.

Output and logs never name patients, UIDs or paths: series are referred to by manifest id.
Commands that work on a dataset append structured records to ``DATASET/logs/dcmtether_<time>.jsonl``
(:mod:`dcmtether.log`). The wording of the output lives in :mod:`dcmtether.report`.
"""

import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import click
from tqdm import tqdm

from dcmtether import __version__, api, report
from dcmtether.api import ScanSummary
from dcmtether.errors import DcmtetherError
from dcmtether.manifest import ManifestUpdate

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


#: GPL notice for --version (dcmtether is not interactive).
_NOTICE = (
    "%(prog)s %(version)s  Copyright (C) 2026 Thomas Thiel\n"
    "This program comes with ABSOLUTELY NO WARRANTY. It is free software under\n"
    "GPL-3.0-or-later, and you are welcome to redistribute it under its conditions;\n"
    "see the LICENSE file or <https://www.gnu.org/licenses/>. Research use only."
)


@contextmanager
def _errors() -> Iterator[None]:
    """Expected failures become one-line errors (exit code 1); their messages are safe."""
    try:
        yield
    except DcmtetherError as exc:
        raise click.ClickException(str(exc)) from None


def _echo(lines: list[str]) -> None:
    for line in lines:
        click.echo(line)


@click.group(help="Lossless DICOM <-> NIfTI round trips for MRI series. Research use only.")
@click.version_option(__version__, prog_name="dcmtether", message=_NOTICE)
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
    _echo(report.conversion_lines(result, verbose))
    if not result.ok:
        raise click.exceptions.Exit(1)


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
@click.option(
    "--in-place",
    is_flag=True,
    help="Write the merged files beside the converted series instead of derivatives/merged.",
)
@click.option(
    "--replace",
    is_flag=True,
    help="As --in-place, then remove the single series' files (only with a merged tether.json).",
)
def merge(dataset: Path, ids: list[int] | None, in_place: bool, replace: bool) -> None:
    """Merge series of one measurement into one 4D NIfTI (experimental).

    For scanners that write one series per b-value (e.g. IVIM). Without --ids, groups are found
    automatically. Writes DATASET/derivatives/merged/, or beside the converted series with
    --in-place; --replace also removes the single series. Exit code 1 if a named group was not
    merged, or a group was not replaced.
    """
    with _errors():
        result = api.merge_series(
            dataset, ids, in_place=in_place, replace=replace, progress=_progress
        )
    if not result.groups:
        click.echo("No series to merge.")
    for group in result.groups:
        listed = ", ".join(f"#{i}" for i in group.series_ids)
        if group.path is not None:
            replaced = "; replaced the single series" if group.replaced else ""
            click.echo(f"Merged series {listed} ({plural(group.n_volumes, 'volume')}){replaced}.")
            if group.no_tether:
                click.echo(f"  no tether.json ({group.no_tether}): to-dcm needs the single series")
            if group.not_replaced:
                click.echo(f"  not replaced ({group.not_replaced}): the single series stay")
        else:
            click.echo(f"Not merged: series {listed}: {group.skipped}.")
    not_replaced = any(g.not_replaced for g in result.groups)
    if (ids is not None and result.skipped) or not_replaced:
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
    _echo(report.dcm_lines(result))


@main.command()
@click.argument("original", type=_existing_dir)
@click.argument("rebuilt", type=_existing_dir)
def compare(original: Path, rebuilt: Path) -> None:
    """Compare ORIGINAL and REBUILT DICOM series element by element.

    Instances are paired by SOPInstanceUID. Exit code 0 if identical, 1 if not.
    """
    result = api.compare(original, rebuilt)
    _echo(report.compare_lines(result))
    if not result.identical:
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
    _echo(report.seg_lines(result))


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
    _echo(report.pmap_lines(result))


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
    _echo(report.mr_lines(result))
