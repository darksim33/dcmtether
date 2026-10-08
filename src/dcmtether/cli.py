"""Command-line interface (see docs/usage.md).

Output and logs never name patients, UIDs or paths: series are referred to by manifest id.
Commands that work on a dataset append structured records to ``DATASET/logs/dcmtether.jsonl``
(:mod:`dcmtether.log`).
"""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click

from dcmtether import __version__
from dcmtether.compare import compare_series, dicom_files
from dcmtether.derived_mr import DerivedMRError, to_derived_mr
from dcmtether.derived_mr import load_metadata as load_derived_metadata
from dcmtether.forward import Outcome, convert_dataset
from dcmtether.log import DatasetLog, find_dataset
from dcmtether.manifest import (
    ManifestError,
    manifest_path,
    series_for_output,
    series_ids,
    update_manifest,
    write_manifest,
)
from dcmtether.pmap import PmapError, to_pmap
from dcmtether.pmap import load_metadata as load_pmap_metadata
from dcmtether.reverse import ReverseError, tether_path_for, to_dicom, transfer_syntax_change
from dcmtether.scan import ScanResult, Status, plural
from dcmtether.scan import scan as scan_directory
from dcmtether.seg import SegError, load_metadata, to_seg
from dcmtether.tether import TetherError
from dcmtether.tether import load as tether_load

#: Exit code for commands that are not implemented yet.
EXIT_NOT_IMPLEMENTED = 3

_existing_dir = click.Path(exists=True, file_okay=False, path_type=Path)
_existing_file = click.Path(exists=True, dir_okay=False, path_type=Path)
_output_dir = click.Path(file_okay=False, path_type=Path)
_verbose = click.option("--verbose", "-v", is_flag=True, help="Also list every series.")


@contextmanager
def _logged(dataset: Path | None, command: str) -> Iterator[DatasetLog]:
    """start/end records around a command; ClickExceptions are logged with their message."""
    logger = DatasetLog(dataset, command)
    logger.end_fields = {}
    logger.event("start")
    try:
        yield logger
    except click.ClickException as exc:
        logger.event("error", reason=exc.message)
        raise
    finally:
        logger.event("end", **logger.end_fields)


@click.group(help="Lossless DICOM <-> NIfTI round trips for MRI series. Research use only.")
@click.version_option(__version__, prog_name="dcmtether")
def main() -> None:
    pass


# --- scanning and conversion -------------------------------------------------------------------


def _scan_summary(result: ScanResult) -> None:
    n_instances = sum(len(s.files) for s in result.series)  # files, not frames
    click.echo(
        f"Scanned {result.n_files} files: {n_instances} DICOM instances in "
        f"{len(result.series)} series, {plural(len(result.skipped), 'file')} skipped."
    )
    counts = Counter(s.status for s in result.series)
    for status in Status:
        click.echo(f"  {status.value + ':':<13}{counts[status]}")


def _report_series(result: ScanResult, ids: dict[str, int], logger, only=None) -> None:
    """Non-supported series (by manifest id) on the console; every series in the log."""
    for series in sorted(result.series, key=lambda s: ids[s.series_instance_uid]):
        series_id = ids[series.series_instance_uid]
        if only is not None and series_id not in only:
            continue
        logger.event(
            "series_classified",
            series_id=series_id,
            status=series.status.value,
            reason="; ".join(series.reasons),
        )
        if series.status is not Status.SUPPORTED:
            click.echo(
                f"  series #{series_id} ({plural(len(series.files), 'instance')}) "
                f"{series.status.value}: {'; '.join(series.reasons)}"
            )
    for reason, n in sorted(Counter(s.reason for s in result.skipped).items()):
        click.echo(f"  skipped {plural(n, 'file')}: {reason}")


def _log_scanned(logger, result: ScanResult) -> None:
    counts = Counter(s.status for s in result.series)
    logger.event(
        "scanned",
        n_files=result.n_files,
        n_series=len(result.series),
        n_skipped=len(result.skipped),
        n_supported=counts[Status.SUPPORTED],
        n_split=counts[Status.SPLIT],
        n_unsupported=counts[Status.UNSUPPORTED],
    )


def _report_conversion(outcomes: list[Outcome], logger, verbose: bool) -> int:
    """Console + log lines for a conversion; returns the number of failures."""
    converted = [o for o in outcomes if o.ok]
    failed = [o for o in outcomes if not o.ok]
    click.echo(f"Converted {len(converted)} series, {len(failed)} failed.")
    for outcome in outcomes:
        if outcome.ok:
            logger.event("series_converted", series_id=outcome.series_id, outputs=outcome.outputs)
            if verbose:
                click.echo(
                    f"  series #{outcome.series_id} converted ({plural(outcome.outputs, 'output')})"
                )
        else:
            logger.event("series_failed", series_id=outcome.series_id, reason=outcome.message)
            click.echo(f"  series #{outcome.series_id} failed: {outcome.message}")
    logger.end_fields = {"n_converted": len(converted), "n_failed": len(failed)}
    return len(failed)


@main.command()
@click.argument("source", type=_existing_dir)
@click.argument("dataset", type=_output_dir)
def scan(source: Path, dataset: Path) -> None:
    """Index DICOM series under SOURCE and classify them (headers only).

    Writes the manifest to the DATASET directory.
    """
    target = manifest_path(dataset)
    if target.exists():
        raise click.ClickException(f"manifest already exists: {target}")
    result = scan_directory(source, exclude=dataset)
    write_manifest(dataset, result)
    with _logged(dataset, "scan") as logger:
        _log_scanned(logger, result)
        _scan_summary(result)
        _report_series(result, series_ids(dataset), logger)
        click.echo(f"Manifest: {target}")


@main.command("to-nii")
@click.argument("dataset", type=_existing_dir)
@_verbose
def to_nii(dataset: Path, verbose: bool) -> None:
    """Convert scanned series in DATASET to NIfTI + tether.json.

    Converts every 'supported' or 'split' series not converted yet; failed ones are retried.
    File names contain patient names and study dates, so the dataset must not be shared as is.
    """
    has_manifest = manifest_path(dataset).exists()
    with _logged(dataset if has_manifest else None, "to-nii") as logger:
        try:
            outcomes = convert_dataset(dataset)
        except ManifestError as exc:
            raise click.ClickException(str(exc)) from None
        failed = _report_conversion(outcomes, logger, verbose)
    if failed:
        raise click.exceptions.Exit(1)


@main.command()
@click.argument("source", type=_existing_dir)
@click.argument("dataset", type=_output_dir)
@_verbose
def run(source: Path, dataset: Path, verbose: bool) -> None:
    """Scan SOURCE and convert it into DATASET; safe to run again.

    The first run creates the dataset. Later runs (same SOURCE) add new series, rescan series
    that are not converted yet and retry failed ones; converted series are never touched (if
    their source files changed, that is reported).
    """
    first_run = not manifest_path(dataset).exists()
    result = scan_directory(source, exclude=dataset)
    if first_run:
        write_manifest(dataset, result)
    with _logged(dataset, "run") as logger:
        if first_run:
            new, rescanned, unchanged = sorted(series_ids(dataset).values()), [], []
            changed: list[int] = []
            missing: list[int] = []
        else:
            try:
                update = update_manifest(dataset, result)
            except ManifestError as exc:
                raise click.ClickException(str(exc)) from None
            new, rescanned, unchanged = update.new, update.rescanned, update.unchanged
            changed, missing = update.changed_after_conversion, update.missing
        _log_scanned(logger, result)
        logger.event(
            "manifest_updated",
            new=new,
            rescanned=rescanned,
            unchanged=unchanged,
            changed_after_conversion=changed,
            missing=missing,
        )
        _scan_summary(result)
        _report_series(result, series_ids(dataset), logger, only=set(new) | set(rescanned))
        click.echo(
            f"Manifest: {len(new)} new series, {len(rescanned)} rescanned, "
            f"{len(unchanged)} unchanged."
        )
        for series_id in changed:
            click.echo(
                f"  series #{series_id}: source files changed after conversion (not "
                "reconverted; remove its outputs and manifest entry to redo it)"
            )
        for series_id in missing:
            click.echo(f"  series #{series_id}: no longer found in SOURCE (kept in the manifest)")
        failed = _report_conversion(convert_dataset(dataset), logger, verbose)
    if failed:
        raise click.exceptions.Exit(1)


# --- reverse and comparison --------------------------------------------------------------------


def _source_log(nifti: Path, command: str):
    """Log into the dataset ``nifti`` belongs to (if any), with its series id."""
    dataset = find_dataset(nifti)
    series_id = series_for_output(dataset, nifti) if dataset is not None else None
    return _logged(dataset, command), series_id


@main.command("to-dcm")
@click.argument("nifti", type=_existing_file)
@click.argument("output", type=_output_dir)
def to_dcm(nifti: Path, output: Path) -> None:
    """Rebuild DICOM from a NIfTI file and its *.tether.json.

    Reads NIFTI (plus the matching *.tether.json next to it) and writes DICOM files to OUTPUT,
    under their original names relative to the series' source directory. Existing files are
    never overwritten.
    """
    context, series_id = _source_log(nifti, "to-dcm")
    with context as logger:
        try:
            written = to_dicom(nifti, output)
            change = transfer_syntax_change(tether_load(tether_path_for(nifti)))
        except (ReverseError, TetherError) as exc:
            raise click.ClickException(str(exc)) from None
        logger.event("written", series_id=series_id, count=len(written))
        click.echo(f"Wrote {plural(len(written), 'DICOM file')}.")
        if change:
            click.echo(f"Transfer syntax changed from {change[0]} to {change[1]} (lossless).")


@main.command()
@click.argument("original", type=_existing_dir)
@click.argument("rebuilt", type=_existing_dir)
def compare(original: Path, rebuilt: Path) -> None:
    """Compare ORIGINAL and REBUILT DICOM series element by element.

    Instances are paired by SOPInstanceUID. Exit code 0 if identical, 1 if not.
    """
    report = compare_series(dicom_files(original), dicom_files(rebuilt))
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
    context, series_id = _source_log(nifti, "to-seg")
    with context as logger:
        try:
            result = to_seg(nifti, labels, load_metadata(metadata), output)
        except SegError as exc:
            raise click.ClickException(str(exc)) from None
        logger.event(
            "written",
            series_id=series_id,
            count=result.n_sources,
            outputs=len(result.segment_numbers),
        )
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
    context, series_id = _source_log(nifti, "to-pmap")
    with context as logger:
        try:
            result = to_pmap(nifti, map_nifti, load_pmap_metadata(metadata), output)
        except PmapError as exc:
            raise click.ClickException(str(exc)) from None
        logger.event("written", series_id=series_id, count=result.n_sources, kind=result.dtype)
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
    map template (map: float maps are quantised to 16 bits). Research use only.
    """
    context, series_id = _source_log(nifti, "to-mr")
    with context as logger:
        try:
            result = to_derived_mr(nifti, derivative, load_derived_metadata(metadata), output)
        except DerivedMRError as exc:
            raise click.ClickException(str(exc)) from None
        logger.event("written", series_id=series_id, count=len(result.paths), kind=result.kind)
        click.echo(f"Wrote {plural(len(result.paths), 'derived MR image')} ({result.kind}).")
        if result.max_error > 0:
            click.echo(
                f"Quantized to 16 bits: max error {result.max_error:.3g} (to-pmap is exact)."
            )
