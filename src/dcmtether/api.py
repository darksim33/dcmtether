"""The Python API: one function per command of the CLI (see docs/api.md).

The functions do what the commands do, on a dataset folder, and return what the CLI prints as
result objects. A series that cannot be converted does not raise: it is listed in the result.
Errors that stop a command raise a :class:`dcmtether.errors.DcmtetherError`; their messages
never contain element values, names, UIDs or paths. Unless ``log=False``, commands that work
on a dataset write to its log (:mod:`dcmtether.log`), with the same records as the CLI.
"""

import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dcmtether import derived_mr, pmap, reverse, scanning, seg, tether
from dcmtether.comparison import Report, compare_series, dicom_files
from dcmtether.derived_mr import DerivedMRResult
from dcmtether.forward import Outcome, convert_dataset
from dcmtether.log import DatasetLog, find_dataset, logged
from dcmtether.manifest import (
    ManifestError,
    ManifestUpdate,
    manifest_path,
    series_for_output,
    series_ids,
    update_manifest,
    write_manifest,
)
from dcmtether.pmap import PmapResult
from dcmtether.progress import Progress, no_progress
from dcmtether.scanning import ScanResult, Status
from dcmtether.seg import SegResult

type StrPath = str | os.PathLike[str]


@dataclass(frozen=True)
class SeriesInfo:
    """One scanned series, by manifest id (no UIDs, no paths)."""

    id: int
    status: Status
    #: Why it is ``split`` or ``unsupported`` (empty for ``supported``).
    reasons: tuple[str, ...]
    #: Files, not frames.
    n_instances: int


@dataclass(frozen=True)
class ScanSummary:
    n_files: int
    #: DICOM files (the rest was skipped).
    n_instances: int
    #: Every series the scan found, ordered by id.
    series: tuple[SeriesInfo, ...]
    #: Reason -> number of skipped files.
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def n_skipped(self) -> int:
        return sum(self.skipped.values())

    def count(self, status: Status) -> int:
        return sum(1 for s in self.series if s.status is status)


@dataclass(frozen=True)
class ConversionResult:
    #: One per series that was attempted, ordered by id.
    outcomes: tuple[Outcome, ...]

    @property
    def converted(self) -> list[int]:
        return [o.series_id for o in self.outcomes if o.ok]

    @property
    def failed(self) -> dict[int, str]:
        """Series id -> reason."""
        return {o.series_id: o.message for o in self.outcomes if not o.ok}

    @property
    def warnings(self) -> dict[int, str]:
        """Series id -> non-fatal problem (e.g. a numbered duplicate name)."""
        return {o.series_id: o.warning for o in self.outcomes if o.ok and o.warning}

    @property
    def ok(self) -> bool:
        return not self.failed


@dataclass(frozen=True)
class RunResult:
    scan: ScanSummary
    #: Series ids by what the scan meant for the manifest.
    new: tuple[int, ...]
    rescanned: tuple[int, ...]
    unchanged: tuple[int, ...]
    #: Converted, but the source files differ now (reported, never reconverted).
    changed_after_conversion: tuple[int, ...]
    #: In the manifest, but no longer found in SOURCE (kept).
    missing: tuple[int, ...]
    conversion: ConversionResult

    @property
    def converted(self) -> list[int]:
        return self.conversion.converted

    @property
    def failed(self) -> dict[int, str]:
        return self.conversion.failed

    @property
    def ok(self) -> bool:
        return self.conversion.ok


@dataclass(frozen=True)
class DcmResult:
    #: The DICOM files written.
    paths: tuple[Path, ...]
    #: (source, written) transfer syntax names if they differ (lossless compressed sources).
    transfer_syntax_change: tuple[str, str] | None = None


# --- helpers -----------------------------------------------------------------------------------


def _existing_dir(path: StrPath, what: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{what} does not exist")
    if not path.is_dir():
        raise NotADirectoryError(f"{what} is not a directory")
    return path


def _existing_file(path: StrPath, what: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} is not an existing file")
    return path


def _check_first_in_place(source: Path, dataset: Path) -> None:
    """Refuse to start a dataset inside a SOURCE that has ``sub-*`` folders of its own.

    Scans of an in-place dataset skip its subject folders; folders that were there before
    would be skipped with them (and mixed with the dataset's). Nothing is written.
    """
    if source.resolve() != dataset.resolve():
        return
    n = scanning.subject_folders(source)
    if n:
        raise ManifestError(f"SOURCE already has sub-* folders ({n}); use --output")


def _summary(result: ScanResult, ids: dict[str, int]) -> ScanSummary:
    series = sorted(
        (
            SeriesInfo(ids[s.series_instance_uid], s.status, tuple(s.reasons), len(s.files))
            for s in result.series
        ),
        key=lambda info: info.id,
    )
    skipped = dict(sorted(Counter(s.reason for s in result.skipped).items()))
    return ScanSummary(result.n_files, sum(s.n_instances for s in series), tuple(series), skipped)


def _log_scanned(logger: DatasetLog, summary: ScanSummary) -> None:
    logger.event(
        "scanned",
        n_files=summary.n_files,
        n_series=len(summary.series),
        n_skipped=summary.n_skipped,
        n_supported=summary.count(Status.SUPPORTED),
        n_split=summary.count(Status.SPLIT),
        n_unsupported=summary.count(Status.UNSUPPORTED),
    )


def _log_classified(logger: DatasetLog, summary: ScanSummary, only: set[int] | None = None) -> None:
    for series in summary.series:
        if only is None or series.id in only:
            logger.event(
                "series_classified",
                series_id=series.id,
                status=series.status.value,
                reason="; ".join(series.reasons),
            )


def _convert(
    dataset: Path, logger: DatasetLog, full_name: bool, progress: Progress
) -> ConversionResult:
    result = ConversionResult(tuple(convert_dataset(dataset, progress, full_name or None)))
    for outcome in result.outcomes:
        if outcome.ok:
            logger.event("series_converted", series_id=outcome.series_id, outputs=outcome.outputs)
            if outcome.warning:
                logger.event("series_warning", series_id=outcome.series_id, reason=outcome.warning)
        else:
            logger.event("series_failed", series_id=outcome.series_id, reason=outcome.message)
    logger.end_fields = {"n_converted": len(result.converted), "n_failed": len(result.failed)}
    return result


def _source_log(nifti: Path, command: str, log: bool):
    """The log of the dataset ``nifti`` belongs to (if any), and its series id there."""
    dataset = find_dataset(nifti)
    series_id = series_for_output(dataset, nifti) if dataset is not None else None
    return logged(dataset if log else None, command), series_id


def _metadata(metadata: StrPath | dict, parse: Callable, load: Callable):
    return parse(metadata) if isinstance(metadata, dict) else load(Path(metadata))


# --- scanning and conversion -------------------------------------------------------------------


def scan(
    source: StrPath, dataset: StrPath, *, log: bool = True, progress: Progress | None = None
) -> ScanSummary:
    """Index the DICOM series under ``source`` and classify them (headers only).

    Writes the manifest to ``dataset`` (created if needed); refuses an existing one.
    """
    source, dataset = _existing_dir(source, "SOURCE"), Path(dataset)
    if manifest_path(dataset).exists():
        raise ManifestError("manifest already exists in DATASET")
    _check_first_in_place(source, dataset)
    result = scanning.scan(source, exclude=dataset, progress=progress or no_progress)
    write_manifest(dataset, result)
    with logged(dataset if log else None, "scan", new_file=True) as logger:
        summary = _summary(result, series_ids(dataset))
        _log_scanned(logger, summary)
        _log_classified(logger, summary)
    return summary


def to_nii(
    dataset: StrPath,
    *,
    full_name: bool = False,
    log: bool = True,
    progress: Progress | None = None,
) -> ConversionResult:
    """Convert the scanned series of ``dataset`` to NIfTI + tether.json.

    Converts every ``supported`` or ``split`` series not converted yet; failed ones are
    retried. ``full_name`` uses the full patient name in subject labels; it is fixed for a
    dataset on its first conversion. File names contain patient names and study dates.
    """
    dataset = _existing_dir(dataset, "DATASET")
    has_manifest = manifest_path(dataset).exists()
    with logged(dataset if log and has_manifest else None, "to-nii", new_file=True) as logger:
        return _convert(dataset, logger, full_name, progress or no_progress)


def run(
    source: StrPath,
    output: StrPath | None = None,
    *,
    full_name: bool = False,
    log: bool = True,
    progress: Progress | None = None,
    _scanned: Callable[[ScanSummary, ManifestUpdate], None] | None = None,
) -> RunResult:
    """Scan ``source`` and convert it into a dataset; safe to call again.

    The dataset is written into ``source`` unless ``output`` names another directory. The
    first call creates it. Later calls (same source and output) add new series, rescan series
    that are not converted yet and retry failed ones; converted series are never touched.
    """
    source = _existing_dir(source, "SOURCE")
    dataset = source if output is None else Path(output)
    progress = progress or no_progress
    first_run = not manifest_path(dataset).exists()
    if first_run:
        _check_first_in_place(source, dataset)
    result = scanning.scan(source, exclude=dataset, progress=progress)
    if first_run:
        write_manifest(dataset, result)
    with logged(dataset if log else None, "run", new_file=True) as logger:
        if first_run:
            update = ManifestUpdate(new=sorted(series_ids(dataset).values()))
        else:
            update = update_manifest(dataset, result)
        summary = _summary(result, series_ids(dataset))
        _log_scanned(logger, summary)
        logger.event(
            "manifest_updated",
            new=update.new,
            rescanned=update.rescanned,
            unchanged=update.unchanged,
            changed_after_conversion=update.changed_after_conversion,
            missing=update.missing,
        )
        _log_classified(logger, summary, only=set(update.new) | set(update.rescanned))
        if _scanned is not None:  # the CLI reports the scan before the conversion starts
            _scanned(summary, update)
        conversion = _convert(dataset, logger, full_name, progress)
    return RunResult(
        summary,
        tuple(update.new),
        tuple(update.rescanned),
        tuple(update.unchanged),
        tuple(update.changed_after_conversion),
        tuple(update.missing),
        conversion,
    )


# --- reverse and comparison --------------------------------------------------------------------


def to_dcm(nifti: StrPath, output: StrPath, *, log: bool = True) -> DcmResult:
    """Rebuild the DICOM files of ``nifti`` from it and the ``*.tether.json`` next to it.

    Writes into ``output`` under the original names, relative to the series' source folder.
    Existing files are never overwritten.
    """
    nifti = _existing_file(nifti, "NIFTI")
    context, series_id = _source_log(nifti, "to-dcm", log)
    with context as logger:
        written = reverse.to_dicom(nifti, Path(output))
        document = tether.load(reverse.tether_path_for(nifti))
        logger.event("written", series_id=series_id, count=len(written))
        return DcmResult(tuple(written), reverse.transfer_syntax_change(document))


def compare(original: StrPath, rebuilt: StrPath) -> Report:
    """Compare two DICOM series (folders) element by element, paired by SOPInstanceUID."""
    original, rebuilt = _existing_dir(original, "ORIGINAL"), _existing_dir(rebuilt, "REBUILT")
    return compare_series(dicom_files(original), dicom_files(rebuilt))


# --- derivatives -------------------------------------------------------------------------------


def to_seg(
    nifti: StrPath, labels: StrPath, metadata: StrPath | dict, output: StrPath, *, log: bool = True
) -> SegResult:
    """Write a DICOM SEG for the label map ``labels`` on the grid of ``nifti``.

    ``nifti`` is a converted source (with its ``*.tether.json``); ``metadata`` is a dcmqi
    segmentation template (path or loaded dict). Writes ``output`` (a new file).
    """
    nifti, labels = _existing_file(nifti, "NIFTI"), _existing_file(labels, "LABELS")
    context, series_id = _source_log(nifti, "to-seg", log)
    with context as logger:
        template = _metadata(metadata, seg.parse_metadata, seg.load_metadata)
        result = seg.to_seg(nifti, labels, template, Path(output))
        logger.event(
            "written",
            series_id=series_id,
            count=result.n_sources,
            outputs=len(result.segment_numbers),
        )
        return result


def to_pmap(
    nifti: StrPath, map: StrPath, metadata: StrPath | dict, output: StrPath, *, log: bool = True
) -> PmapResult:
    """Write a DICOM Parametric Map for ``map`` (e.g. ADC, QSM) on the grid of ``nifti``.

    ``metadata`` is a dcmqi parametric map template (path or loaded dict). Writes ``output``
    (a new file).
    """
    nifti, map = _existing_file(nifti, "NIFTI"), _existing_file(map, "MAP")
    context, series_id = _source_log(nifti, "to-pmap", log)
    with context as logger:
        template = _metadata(metadata, pmap.parse_metadata, pmap.load_metadata)
        result = pmap.to_pmap(nifti, map, template, Path(output))
        logger.event("written", series_id=series_id, count=result.n_sources, kind=result.dtype)
        return result


def to_mr(
    nifti: StrPath,
    derivative: StrPath,
    metadata: StrPath | dict,
    output: StrPath,
    *,
    log: bool = True,
) -> DerivedMRResult:
    """Write ``derivative`` as derived MR images (one per source slice) into ``output``.

    Fallback for PACS without SEG / Parametric Map support. ``metadata`` is a dcmqi SEG
    template (label map: label values as pixels) or parametric map template (map).
    """
    nifti = _existing_file(nifti, "NIFTI")
    derivative = _existing_file(derivative, "DERIVATIVE")
    context, series_id = _source_log(nifti, "to-mr", log)
    with context as logger:
        template = _metadata(metadata, derived_mr.parse_metadata, derived_mr.load_metadata)
        result = derived_mr.to_derived_mr(nifti, derivative, template, Path(output))
        logger.event("written", series_id=series_id, count=len(result.paths), kind=result.kind)
        return result
