"""Command-line interface. Subcommands are stubs until their phase lands (see docs/dev/PLAN.md)."""

from collections import Counter
from pathlib import Path

import click

from dcmtether import __version__
from dcmtether.forward import convert_dataset
from dcmtether.manifest import ManifestError, manifest_path, write_manifest
from dcmtether.scan import Status, plural
from dcmtether.scan import scan as scan_directory

EXIT_NOT_IMPLEMENTED = 3

_existing_dir = click.Path(exists=True, file_okay=False, path_type=Path)
_existing_file = click.Path(exists=True, dir_okay=False, path_type=Path)
_output_dir = click.Path(file_okay=False, path_type=Path)


def _not_implemented(command: str) -> None:
    click.echo(f"dcmtether {command}: not implemented yet", err=True)
    raise click.exceptions.Exit(EXIT_NOT_IMPLEMENTED)


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
    n_instances = sum(len(s.instances) for s in result.series)
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
                f"  series #{series_id} ({plural(len(series.instances), 'instance')}) "
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

    Reads NIFTI (plus the matching *.tether.json next to it) and writes DICOM files to OUTPUT.
    """
    _not_implemented("to-dcm")


@main.command()
@click.argument("original", type=_existing_dir)
@click.argument("rebuilt", type=_existing_dir)
def compare(original: Path, rebuilt: Path) -> None:
    """Compare ORIGINAL and REBUILT DICOM series element by element."""
    _not_implemented("compare")
