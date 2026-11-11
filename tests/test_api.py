"""The Python API (docs/api.md): the CLI's commands as functions with result objects."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from click.testing import CliRunner
from roundtrip import output_rows
from synthetic import SeriesSpec, make_localizer, make_series, write_series
from test_pmap import META as PMAP_META
from test_seg import META as SEG_META

import dcmtether as dt
from dcmtether import log
from dcmtether.cli import main

API = {
    "ConversionError",
    "ConversionResult",
    "DcmResult",
    "DcmtetherError",
    "DerivedError",
    "DerivedMRResult",
    "Difference",
    "ManifestError",
    "MergeError",
    "MergeResult",
    "MergedGroup",
    "Outcome",
    "PmapResult",
    "Report",
    "ReverseError",
    "RunResult",
    "ScanSummary",
    "SegResult",
    "SeriesInfo",
    "Status",
    "TetherError",
    "UidError",
    "__version__",
    "compare",
    "merge_series",
    "run",
    "scan",
    "to_dcm",
    "to_mr",
    "to_nii",
    "to_pmap",
    "to_seg",
}


def _series(number: int, **kw):
    return make_series(SeriesSpec(series_number=number, **kw))


def _source(tmp_path: Path) -> Path:
    """Three series: a supported one, a localizer (split) and one without a PatientName."""
    src = tmp_path / "src"
    write_series(_series(1), src / "a")
    nameless = _series(2)
    for ds in nameless.datasets:
        ds.PatientName = ""
    write_series(nameless, src / "b")
    write_series(make_localizer(), src / "loc")
    (src / "notes.txt").write_text("x", "utf-8")
    return src


def _events(dataset: Path) -> list[tuple]:
    return [
        tuple((k, v) for k, v in json.loads(line).items() if k != "time")
        for path in log.log_files(dataset)
        for line in path.read_text("utf-8").splitlines()
    ]


@pytest.fixture
def converted(tmp_path) -> tuple[Path, Path]:
    """(source NIfTI in a dataset, its DICOM folder)."""
    src = tmp_path / "src"
    write_series(_series(1), src / "a")
    dt.run(src, tmp_path / "dataset")
    return tmp_path / "dataset" / output_rows(tmp_path / "dataset")[0]["nifti_path"], src / "a"


def _derivatives(tmp_path: Path, nifti: Path) -> tuple[Path, Path]:
    img = nib.load(nifti)
    labels = np.zeros(img.shape, np.uint8)
    labels[1, 1, 1], labels[2, 2, 2] = 1, 2
    nib.save(nib.Nifti1Image(labels, img.affine), tmp_path / "labels.nii.gz")
    values = np.linspace(0, 1, labels.size, dtype=np.float32).reshape(img.shape)
    nib.save(nib.Nifti1Image(values, img.affine), tmp_path / "map.nii.gz")
    return tmp_path / "labels.nii.gz", tmp_path / "map.nii.gz"


# --- surface -----------------------------------------------------------------------------------


def test_public_names():
    assert set(dt.__all__) == API
    assert all(hasattr(dt, name) for name in API)


def test_functions_do_not_shadow_modules():
    import dcmtether.comparison as comparison
    import dcmtether.scanning as scanning

    assert scanning.scan is not dt.scan and callable(dt.scan)
    assert comparison.compare_series is not dt.compare and callable(dt.compare)


def test_every_error_is_a_dcmtether_error():
    for error in (
        dt.ManifestError,
        dt.ConversionError,
        dt.ReverseError,
        dt.TetherError,
        dt.DerivedError,
    ):
        assert issubclass(error, dt.DcmtetherError)
    # what callers could catch before still works
    assert issubclass(dt.ManifestError, RuntimeError) and issubclass(dt.TetherError, ValueError)


# --- run, scan, to_nii -------------------------------------------------------------------------


def test_run_returns_what_the_cli_prints(tmp_path):
    src = _source(tmp_path)
    result = dt.run(str(src), str(tmp_path / "dataset"))  # plain strings are fine

    assert isinstance(result, dt.RunResult)
    assert (result.scan.n_files, result.scan.n_instances) == (14, 13)
    assert result.scan.skipped == {"not a DICOM file": 1}
    assert [(s.id, s.status, s.n_instances) for s in result.scan.series] == [
        (1, dt.Status.SUPPORTED, 5),
        (2, dt.Status.SUPPORTED, 5),
        (3, dt.Status.SPLIT, 3),
    ]
    assert result.scan.series[2].reasons == (
        "3 distinct image planes (orientation, matrix or pixel spacing)",
    )
    assert result.new == (1, 2, 3) and result.unchanged == ()
    assert result.converted == [1, 3]
    assert result.failed == {2: "PatientName is missing"}  # listed, not raised
    assert not result.ok
    assert [o.outputs for o in result.conversion.outcomes if o.ok] == [1, 3]
    assert len(output_rows(tmp_path / "dataset")) == 4


def test_rerun_reports_unchanged_and_retries_failures(tmp_path):
    src = _source(tmp_path)
    dt.run(src, tmp_path / "dataset")
    write_series(_series(4, seed=4), src / "d")
    result = dt.run(src, tmp_path / "dataset")
    assert (result.new, result.rescanned, result.unchanged) == ((4,), (2,), (1, 3))
    assert result.converted == [4] and list(result.failed) == [2]


def test_run_without_output_writes_into_source(tmp_path):
    src = tmp_path / "src"
    write_series(_series(1), src / "a")
    assert dt.run(src).ok
    assert (src / "manifest.sqlite").is_file()
    assert dt.run(src).unchanged == (1,)


def test_scan_then_to_nii_equals_run(tmp_path):
    src = _source(tmp_path)
    summary = dt.scan(src, tmp_path / "stepwise")
    conversion = dt.to_nii(tmp_path / "stepwise")
    result = dt.run(src, tmp_path / "at-once")
    assert summary == result.scan
    assert conversion == result.conversion
    names = [Path(r["nifti_path"]).name for r in output_rows(tmp_path / "stepwise")]
    assert names == [Path(r["nifti_path"]).name for r in output_rows(tmp_path / "at-once")]


def test_full_name(tmp_path):
    src = tmp_path / "src"
    write_series(_series(1), src / "a")
    assert dt.run(src, tmp_path / "dataset", full_name=True).ok
    [row] = output_rows(tmp_path / "dataset")
    assert row["nifti_path"].startswith("sub-SyntheticPhantom/")


def test_warnings_are_listed(tmp_path):
    src = tmp_path / "src"
    write_series(_series(1, seed=1), src / "a")
    twin = _series(1, seed=2)
    for ds, other in zip(twin.datasets, _series(1, seed=1).datasets, strict=True):
        ds.StudyInstanceUID, ds.PatientID = other.StudyInstanceUID, other.PatientID
    write_series(twin, src / "b")
    result = dt.run(src, tmp_path / "dataset")
    assert result.ok and result.converted == [1, 2]
    assert set(result.conversion.warnings) == {1, 2}
    assert all(w.startswith("duplicate name") for w in result.conversion.warnings.values())


def test_errors_that_stop_a_command_raise(tmp_path):
    src = tmp_path / "src"
    write_series(_series(1), src / "a")
    dt.scan(src, tmp_path / "dataset")
    with pytest.raises(dt.ManifestError, match=r"^manifest already exists in DATASET$"):
        dt.scan(src, tmp_path / "dataset")
    with pytest.raises(dt.ManifestError, match=r"^no manifest in DATASET"):
        dt.to_nii(tmp_path / "src")
    write_series(_series(2), tmp_path / "other" / "a")
    with pytest.raises(dt.DcmtetherError, match="SOURCE is not the folder"):
        dt.run(tmp_path / "other", tmp_path / "dataset")

    write_series(_series(3), tmp_path / "labelled" / "sub-01")
    with pytest.raises(dt.ManifestError, match=r"^SOURCE already has sub-\* folders \(1\)"):
        dt.run(tmp_path / "labelled")
    with pytest.raises(FileNotFoundError, match=r"^SOURCE does not exist$"):
        dt.run(tmp_path / "Mueller_Thomas")


def test_progress_callable_is_used(tmp_path):
    src = tmp_path / "src"
    write_series(_series(1), src / "a")
    seen = []

    def progress(iterable, **kwargs):
        seen.append(kwargs["desc"])
        return iterable

    dt.run(src, tmp_path / "dataset", progress=progress)
    assert seen == ["Reading headers", "Classifying", "Converting"]


# --- to_dcm, compare ---------------------------------------------------------------------------


def test_to_dcm_and_compare(tmp_path, converted):
    nifti, dicom = converted
    result = dt.to_dcm(nifti, tmp_path / "rebuilt")
    assert isinstance(result, dt.DcmResult)
    assert len(result.paths) == 5 and all(p.is_file() for p in result.paths)
    assert result.transfer_syntax_change is None

    report = dt.compare(dicom, tmp_path / "rebuilt")
    assert isinstance(report, dt.Report)
    assert report.identical and report.n_compared == 5

    with pytest.raises(dt.ReverseError, match="already exists"):
        dt.to_dcm(nifti, tmp_path / "rebuilt")


def test_compare_reports_differences(tmp_path, converted):
    _, dicom = converted
    write_series(_series(1, seed=9), tmp_path / "other")
    report = dt.compare(dicom, tmp_path / "other")
    assert not report.identical and report.differences


# --- derivatives -------------------------------------------------------------------------------


@pytest.mark.parametrize("as_dict", [False, True], ids=["path", "dict"])
def test_derivatives(tmp_path, converted, as_dict):
    nifti, _ = converted
    labels, values = _derivatives(tmp_path, nifti)

    def template(meta: dict, name: str):
        if as_dict:
            return meta
        (tmp_path / name).write_text(json.dumps(meta), "utf-8")
        return tmp_path / name

    seg = dt.to_seg(nifti, labels, template(SEG_META, "seg.json"), tmp_path / "seg.dcm")
    assert isinstance(seg, dt.SegResult)
    assert seg.path.is_file() and seg.segment_numbers == {1: 1, 2: 2} and seg.n_sources == 5

    pm = dt.to_pmap(nifti, values, template(PMAP_META, "pmap.json"), tmp_path / "pm.dcm")
    assert isinstance(pm, dt.PmapResult)
    assert pm.path.is_file() and pm.dtype == "float32"

    mr = dt.to_mr(nifti, values, template(PMAP_META, "pmap.json"), tmp_path / "mr")
    assert isinstance(mr, dt.DerivedMRResult)
    assert (len(mr.paths), mr.kind) == (5, "map") and mr.max_error > 0 and not mr.rounded
    labels_mr = dt.to_mr(nifti, labels, template(SEG_META, "seg.json"), tmp_path / "mr-labels")
    assert labels_mr.kind == "labels"


def test_bad_template_raises_a_derived_error(tmp_path, converted):
    nifti, _ = converted
    labels, _ = _derivatives(tmp_path, nifti)
    with pytest.raises(dt.DerivedError, match="SeriesNumber"):
        dt.to_seg(nifti, labels, {"segmentAttributes": []}, tmp_path / "seg.dcm")
    with pytest.raises(dt.DerivedError, match="dcmqi"):
        dt.to_mr(nifti, labels, {}, tmp_path / "mr")


# --- derived UIDs ------------------------------------------------------------------------------

#: Synthetic source series UIDs: a short one (stem "1.2.3.4") and one whose stem is too long.
SHORT_SERIES_UID = "1.2.3.4.5"
LONG_SERIES_UID = "1." + "9" * 38 + ".5"


def _converted_with_series_uid(tmp_path: Path, series_uid: str | None) -> tuple[Path, list]:
    """(source NIfTI, source datasets) for one synthetic series, optionally with another UID."""
    series = _series(1)
    if series_uid is not None:
        for ds in series.datasets:
            ds.SeriesInstanceUID = series_uid
    write_series(series, tmp_path / "src" / "a")
    dt.run(tmp_path / "src", tmp_path / "dataset")
    nifti = tmp_path / "dataset" / output_rows(tmp_path / "dataset")[0]["nifti_path"]
    return nifti, series.datasets


def _write_all(tmp_path: Path, nifti: Path, out: str, **uid_options) -> dict:
    """Write SEG, Parametric Map and derived MR; their results and written datasets."""
    labels, values = _derivatives(tmp_path, nifti)
    base = tmp_path / out
    results = {
        "seg": dt.to_seg(nifti, labels, SEG_META, base / "seg.dcm", **uid_options),
        "pmap": dt.to_pmap(nifti, values, PMAP_META, base / "pm.dcm", **uid_options),
        "mr": dt.to_mr(nifti, values, PMAP_META, base / "mr", **uid_options),
    }
    written = {
        "seg": [pydicom.dcmread(results["seg"].path)],
        "pmap": [pydicom.dcmread(results["pmap"].path)],
        "mr": [pydicom.dcmread(p) for p in results["mr"].paths],
    }
    return {"results": results, "written": written}


class TestDerivedUids:
    """New Series/SOP UIDs of SEG, Parametric Map and derived MR in all three modes."""

    @pytest.mark.parametrize(
        ("series_uid", "options", "prefix"),
        [
            (None, {}, "2.25."),
            (None, {"uid_root": "1.5"}, "1.5."),
            (SHORT_SERIES_UID, {"uid_source": True}, "1.2.3.4."),
        ],
        ids=["synthetic", "org-root", "source-stem"],
    )
    def test_prefix_and_identity(self, tmp_path, series_uid, options, prefix):
        """Every new UID has the mode's prefix; Study and Frame of Reference are kept."""
        nifti, sources = _converted_with_series_uid(tmp_path, series_uid)
        out = _write_all(tmp_path, nifti, "out", **options)
        source_sops = {ds.SOPInstanceUID for ds in sources}
        for kind, datasets in out["written"].items():
            assert not out["results"][kind].uid_fallback
            assert len({ds.SeriesInstanceUID for ds in datasets}) == 1, f"{kind}: one series"
            for ds in datasets:
                for uid in (ds.SeriesInstanceUID, ds.SOPInstanceUID):
                    assert uid.startswith(prefix) and len(uid) <= 64, kind
                assert ds.SeriesInstanceUID != sources[0].SeriesInstanceUID, kind
                assert ds.SOPInstanceUID not in source_sops, kind
                assert ds.StudyInstanceUID == sources[0].StudyInstanceUID, kind
                assert ds.FrameOfReferenceUID == sources[0].FrameOfReferenceUID, kind
            assert len({ds.SOPInstanceUID for ds in datasets}) == len(datasets), kind

    @pytest.mark.parametrize(
        "options",
        [{}, {"uid_root": "1.5"}, {"uid_source": True}],
        ids=["synthetic", "root", "stem"],
    )
    def test_rerun_gives_same_uids(self, tmp_path, options):
        """Writing the same derivatives again gives the same UIDs."""
        nifti, _ = _converted_with_series_uid(tmp_path, SHORT_SERIES_UID)
        first = _write_all(tmp_path, nifti, "first", **options)["written"]
        second = _write_all(tmp_path, nifti, "second", **options)["written"]
        for kind in first:
            assert [(d.SeriesInstanceUID, d.SOPInstanceUID) for d in first[kind]] == [
                (d.SeriesInstanceUID, d.SOPInstanceUID) for d in second[kind]
            ], kind

    def test_org_root_keeps_hash_digits(self, tmp_path):
        """An org root changes only the prefix; the digits match the synthetic UIDs."""
        nifti, _ = _converted_with_series_uid(tmp_path, None)
        synthetic = _write_all(tmp_path, nifti, "synthetic")["written"]
        rooted = _write_all(tmp_path, nifti, "rooted", uid_root="1.5")["written"]
        for kind in synthetic:
            for plain, with_root in zip(synthetic[kind], rooted[kind], strict=True):
                assert with_root.SOPInstanceUID == "1.5." + plain.SOPInstanceUID.removeprefix(
                    "2.25."
                ), kind

    def test_unusable_stem_falls_back(self, tmp_path):
        """A source stem that is too long gives synthetic UIDs and says so."""
        nifti, _ = _converted_with_series_uid(tmp_path, LONG_SERIES_UID)
        out = _write_all(tmp_path, nifti, "out", uid_source=True)
        for kind, datasets in out["written"].items():
            assert out["results"][kind].uid_fallback, kind
            assert all(ds.SOPInstanceUID.startswith("2.25.") for ds in datasets), kind

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"uid_root": "1.02"}, "digits separated by dots"),
            ({"uid_root": "1." + "9" * 38}, "longer than 39"),
            ({"uid_root": "1.5", "uid_source": True}, "cannot be combined"),
        ],
        ids=["malformed", "too-long", "combined"],
    )
    def test_bad_options_fail_before_any_file(self, tmp_path, options, message):
        """Bad UID options raise ``UidError`` before inputs are read or outputs written."""
        missing = tmp_path / "missing.nii.gz"
        calls = [
            lambda: dt.to_seg(missing, missing, SEG_META, tmp_path / "seg.dcm", **options),
            lambda: dt.to_pmap(missing, missing, PMAP_META, tmp_path / "pm.dcm", **options),
            lambda: dt.to_mr(missing, missing, PMAP_META, tmp_path / "mr", **options),
        ]
        for call in calls:
            with pytest.raises(dt.UidError, match=message):
                call()
        assert list(tmp_path.iterdir()) == []


def _cli_uid_commands(tmp_path: Path, nifti: Path) -> dict[str, list[str]]:
    """CLI arguments (without UID options) of the three derivative commands."""
    labels, values = _derivatives(tmp_path, nifti)
    seg_meta, pmap_meta = tmp_path / "seg.json", tmp_path / "pmap.json"
    seg_meta.write_text(json.dumps(SEG_META))
    pmap_meta.write_text(json.dumps(PMAP_META))
    out = tmp_path / "cli"
    return {
        "to-seg": ["to-seg", str(nifti), str(labels), str(seg_meta), str(out / "seg.dcm")],
        "to-pmap": ["to-pmap", str(nifti), str(values), str(pmap_meta), str(out / "pm.dcm")],
        "to-mr": ["to-mr", str(nifti), str(values), str(pmap_meta), str(out / "mr")],
    }


def _cli_written(command: str, tmp_path: Path) -> list:
    out = tmp_path / "cli"
    paths = sorted((out / "mr").iterdir()) if command == "to-mr" else list(out.iterdir())
    return [pydicom.dcmread(p) for p in paths]


class TestDerivedUidsCli:
    """``--uid-root`` / ``--uid-source`` on to-seg, to-pmap and to-mr."""

    @pytest.mark.parametrize("command", ["to-seg", "to-pmap", "to-mr"])
    @pytest.mark.parametrize(
        ("options", "prefix"),
        [([], "2.25."), (["--uid-root", "1.5"], "1.5."), (["--uid-source"], "1.2.3.4.")],
        ids=["synthetic", "org-root", "source-stem"],
    )
    def test_options_set_the_prefix(self, tmp_path, command, options, prefix):
        nifti, _ = _converted_with_series_uid(tmp_path, SHORT_SERIES_UID)
        args = _cli_uid_commands(tmp_path, nifti)[command]
        result = CliRunner().invoke(main, [*args, *options])
        assert result.exit_code == 0, result.output
        assert "note:" not in result.output
        for ds in _cli_written(command, tmp_path):
            assert ds.SeriesInstanceUID.startswith(prefix)
            assert ds.SOPInstanceUID.startswith(prefix)

    @pytest.mark.parametrize("command", ["to-seg", "to-pmap", "to-mr"])
    @pytest.mark.parametrize(
        "options",
        [["--uid-root", "1..2"], ["--uid-root", "1.5", "--uid-source"]],
        ids=["invalid-root", "both"],
    )
    def test_bad_options_are_one_line_errors(self, tmp_path, command, options):
        nifti, _ = _converted_with_series_uid(tmp_path, SHORT_SERIES_UID)
        args = _cli_uid_commands(tmp_path, nifti)[command]
        result = CliRunner().invoke(main, [*args, *options])
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert result.output.startswith("Error: ")
        assert "Traceback" not in result.output and "1..2" not in result.output
        assert not (tmp_path / "cli").exists()

    @pytest.mark.parametrize("command", ["to-seg", "to-pmap", "to-mr"])
    def test_unusable_stem_prints_a_note(self, tmp_path, command):
        nifti, _ = _converted_with_series_uid(tmp_path, LONG_SERIES_UID)
        args = _cli_uid_commands(tmp_path, nifti)[command]
        result = CliRunner().invoke(main, [*args, "--uid-source"])
        assert result.exit_code == 0, result.output
        assert result.output.splitlines()[-1] == (
            "note: source UID stem not usable, synthetic 2.25 UIDs written"
        )
        assert LONG_SERIES_UID not in result.output
        assert all(ds.SOPInstanceUID.startswith("2.25.") for ds in _cli_written(command, tmp_path))


def test_damaged_tether_raises_from_every_single_series_function(tmp_path, converted):
    nifti, _ = converted
    labels, values = _derivatives(tmp_path, nifti)
    tether_path = nifti.with_name(nifti.name.removesuffix(".nii.gz") + ".tether.json")
    tether_path.write_text(tether_path.read_text("utf-8")[:200], "utf-8")
    calls = [
        lambda: dt.to_dcm(nifti, tmp_path / "out"),
        lambda: dt.to_seg(nifti, labels, SEG_META, tmp_path / "seg.dcm"),
        lambda: dt.to_pmap(nifti, values, PMAP_META, tmp_path / "pm.dcm"),
        lambda: dt.to_mr(nifti, values, PMAP_META, tmp_path / "mr"),
    ]
    for call in calls:
        with pytest.raises(dt.DcmtetherError, match=r"tether.json is not readable") as caught:
            call()
        message = str(caught.value)
        assert "sub-" not in message and "Synth" not in message and tmp_path.name not in message


# --- logging -----------------------------------------------------------------------------------


def test_log_false_writes_no_log(tmp_path, converted):
    src = tmp_path / "quiet-src"
    write_series(_series(1), src / "a")
    dataset = tmp_path / "quiet"
    dt.scan(src, dataset, log=False)
    dt.to_nii(dataset, log=False)
    dt.run(src, dataset, log=False)
    nifti = dataset / output_rows(dataset)[0]["nifti_path"]
    labels, values = _derivatives(tmp_path, nifti)
    dt.to_dcm(nifti, tmp_path / "out", log=False)
    dt.to_seg(nifti, labels, SEG_META, tmp_path / "seg.dcm", log=False)
    dt.to_pmap(nifti, values, PMAP_META, tmp_path / "pm.dcm", log=False)
    dt.to_mr(nifti, values, PMAP_META, tmp_path / "mr", log=False)
    assert not (dataset / "logs").exists()


def test_api_and_cli_write_the_same_log_records(tmp_path):
    src = _source(tmp_path)
    dt.run(src, tmp_path / "api")
    dt.run(src, tmp_path / "api")
    nifti = tmp_path / "api" / output_rows(tmp_path / "api")[0]["nifti_path"]
    dt.to_dcm(nifti, tmp_path / "api-out")
    with pytest.raises(dt.ReverseError):
        dt.to_dcm(nifti, tmp_path / "api-out")  # recorded as an error

    run = ["run", str(src), "-o", str(tmp_path / "cli")]
    CliRunner().invoke(main, run)
    CliRunner().invoke(main, run)
    nifti = tmp_path / "cli" / output_rows(tmp_path / "cli")[0]["nifti_path"]
    for _ in range(2):
        CliRunner().invoke(main, ["to-dcm", str(nifti), str(tmp_path / "cli-out")])

    events = _events(tmp_path / "api")
    assert events == _events(tmp_path / "cli")
    assert ("event", "error") in events[-2]
    assert len(log.log_files(tmp_path / "api")) == 2  # one per run; to_dcm appended
