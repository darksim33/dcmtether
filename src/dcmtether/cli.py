"""Command-line interface. Subcommands are stubs until their phase lands (see docs/dev/PLAN.md)."""

from pathlib import Path

import click

from dcmtether import __version__

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
    _not_implemented("scan")


@main.command("to-nii")
@click.argument("dataset", type=_existing_dir)
def to_nii(dataset: Path) -> None:
    """Convert scanned series in DATASET to NIfTI + tether.json."""
    _not_implemented("to-nii")


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
