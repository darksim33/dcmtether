"""Command-line interface (see docs/dev/PLAN.md). Output never names patients, UIDs or paths."""

from collections import Counter
from pathlib import Path

import click

from dcmtether import __version__
from dcmtether.compare import compare_series, dicom_files
from dcmtether.forward import convert_dataset
from dcmtether.manifest import ManifestError, manifest_path, write_manifest
from dcmtether.reverse import ReverseError, tether_path_for, to_dicom, transfer_syntax_change
from dcmtether.scan import Status, plural
from dcmtether.scan import scan as scan_directory
from dcmtether.tether import TetherError
from dcmtether.tether import load as tether_load

#: Exit code for commands that are not implemented yet.
EXIT_NOT_IMPLEMENTED = 3

_existing_dir = click.Path(exists=True, file_okay=False, path_type=Path)
_existing_file = click.Path(exists=True, dir_okay=False, path_type=Path)
_output_dir = click.Path(file_okay=False, path_type=Path)


@click.group(help="Lossless DICOM <-> NIfTI round trips for MRI series. Research use only.")
@click.version_option(__version__, prog_name="dcmtether")
def main() -> None:
    pass


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

    # Output names series by manifest id only: no UIDs, descriptions or paths (PHI rules).
    n_instances = sum(len(s.files) for s in result.series)  # files, not frames
    click.echo(
        f"Scanned {result.n_files} files: {n_instances} DICOM instances in "
        f"{len(result.series)} series, {plural(len(result.skipped), 'file')} skipped."
    )
    counts = Counter(s.status for s in result.series)
    for status in Status:
        click.echo(f"  {status.value + ':':<13}{counts[status]}")
    for series_id, series in enumerate(result.series, start=1):
        if series.status is not Status.SUPPORTED:
            click.echo(
                f"  series #{series_id} ({plural(len(series.files), 'instance')}) "
                f"{series.status.value}: {'; '.join(series.reasons)}"
            )
    for reason, n in sorted(Counter(s.reason for s in result.skipped).items()):
        click.echo(f"  skipped {plural(n, 'file')}: {reason}")
    click.echo(f"Manifest: {target}")


@main.command("to-nii")
@click.argument("dataset", type=_existing_dir)
def to_nii(dataset: Path) -> None:
    """Convert scanned series in DATASET to NIfTI + tether.json.

    Converts every 'supported' series not converted yet; failed ones are retried. File names
    contain patient names and study dates, so the dataset must not be shared as is.
    """
    try:
        outcomes = convert_dataset(dataset)
    except ManifestError as exc:
        raise click.ClickException(str(exc)) from None

    # Output names series by manifest id only (PHI rules); file names are not printed.
    converted = [o for o in outcomes if o.ok]
    failed = [o for o in outcomes if not o.ok]
    click.echo(f"Converted {len(converted)} series, {len(failed)} failed.")
    for outcome in failed:
        click.echo(f"  series #{outcome.series_id} failed: {outcome.message}")
    if failed:
        raise click.exceptions.Exit(1)


@main.command("to-dcm")
@click.argument("nifti", type=_existing_file)
@click.argument("output", type=_output_dir)
def to_dcm(nifti: Path, output: Path) -> None:
    """Rebuild DICOM from a NIfTI file and its *.tether.json.

    Reads NIFTI (plus the matching *.tether.json next to it) and writes DICOM files to OUTPUT,
    under their original names relative to the series' source directory. Existing files are
    never overwritten.
    """
    try:
        written = to_dicom(nifti, output)
        change = transfer_syntax_change(tether_load(tether_path_for(nifti)))
    except (ReverseError, TetherError) as exc:
        raise click.ClickException(str(exc)) from None
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
