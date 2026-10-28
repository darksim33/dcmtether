"""Merging series of one measurement (one series per b-value) into one 4D NIfTI."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from click.testing import CliRunner
from roundtrip import output_rows
from synthetic import SeriesSpec, make_series, write_series

import dcmtether as dt
from dcmtether.cli import main

DIFFUSION = ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE"]


def _b_series(b_value: float, number: int, protocol="ep2d_ivim", gradient=None, **spec):
    """One single-volume diffusion series; pixel values offset by the series number."""
    extra = {"ImageType": DIFFUSION, "ProtocolName": protocol}
    series = make_series(
        SeriesSpec(series_number=number, series_description=f"IVIM_b{int(b_value)}", **spec)
    )
    for ds in series.datasets:
        for keyword, value in extra.items():
            setattr(ds, keyword, value)
        ds.add_new(0x00189087, "FD", float(b_value))  # DiffusionBValue
        if gradient is not None:
            ds.add_new(0x00189089, "FD", list(gradient))  # DiffusionGradientOrientation
        pixels = np.frombuffer(ds.PixelData, dtype=series.volume.dtype) + number * 100
        ds.PixelData = pixels.astype(series.volume.dtype).tobytes()
    return series


def _dataset(tmp_path: Path, *series) -> Path:
    for s in series:
        write_series(s, tmp_path / "src" / f"s{s.spec.series_number}")
    dataset = tmp_path / "dataset"
    result = dt.run(tmp_path / "src", dataset)
    assert result.ok, result.failed
    return dataset


def _ids_by_number(dataset: Path) -> dict[int, int]:
    import sqlite3

    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        return dict(db.execute("SELECT series_number, id FROM series").fetchall())


def test_one_series_per_b_value_is_merged(tmp_path):
    # written out of order: volumes follow SeriesNumber
    dataset = _dataset(tmp_path, _b_series(800, 7), _b_series(0, 5), _b_series(50, 6))
    ids = _ids_by_number(dataset)
    result = dt.merge_series(dataset)
    [group] = result.groups
    assert group.series_ids == (ids[5], ids[6], ids[7])
    assert group.n_volumes == 3 and group.skipped is None

    path = group.path
    assert path.relative_to(dataset).parts[:5] == (
        "derivatives",
        "merged",
        "sub-SynthPhant",
        "ses-20260101",
        "dwi",
    )
    assert path.name.endswith("_desc-merged_dwi.nii.gz")
    merged = nib.load(path)
    assert merged.shape[3] == 3
    by_number = {
        number: dataset / row["nifti_path"]
        for row in output_rows(dataset)
        for number, sid in ids.items()
        if sid == row["series_id"]
    }
    for volume, number in enumerate((5, 6, 7)):
        source = nib.load(by_number[number])
        assert np.array_equal(merged.dataobj[..., volume], source.dataobj)
        assert np.allclose(merged.affine, source.affine)

    sidecar = json.loads(path.with_name(path.name.replace(".nii.gz", ".json")).read_text())
    info = sidecar["DcmTether"]
    assert info["MergedFrom"] == [ids[5], ids[6], ids[7]]
    assert info["PerVolume"]["DiffusionBValue"] == [0.0, 50.0, 800.0]
    assert info["PerVolume"]["SeriesNumber"] == [5, 6, 7]
    assert sidecar["ProtocolName"] == "ep2d_ivim"  # the same in every series
    assert "SeriesDescription" in info["PerVolume"]


def test_merging_again_does_not_overwrite(tmp_path):
    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6))
    first = dt.merge_series(dataset).groups[0].path
    before = first.read_bytes()
    [group] = dt.merge_series(dataset).groups
    assert group.skipped == "already merged" and group.path is None
    assert first.read_bytes() == before


def test_repeated_b_values_are_not_merged_automatically(tmp_path):
    dataset = _dataset(tmp_path, _b_series(800, 5), _b_series(800, 6))
    [group] = dt.merge_series(dataset).groups
    assert group.path is None
    assert "repeat" in group.skipped
    assert not (dataset / "derivatives").exists()


@pytest.mark.parametrize(
    "second",
    [
        {"protocol": "t2_tse"},  # another protocol
        {"origin": (-10.0, -20.0, 45.0)},  # another position
        {"rows": 7},  # another grid
    ],
    ids=["protocol", "position", "grid"],
)
def test_series_that_do_not_belong_together_are_not_grouped(tmp_path, second):
    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6, **second))
    result = dt.merge_series(dataset)
    assert [g for g in result.groups if g.path is not None] == []


def test_b_value_tokens_in_the_protocol_name_are_ignored(tmp_path):
    dataset = _dataset(
        tmp_path, _b_series(0, 5, protocol="IVIM_b0"), _b_series(50, 6, protocol="IVIM_b50")
    )
    [group] = dt.merge_series(dataset).groups
    assert group.path is not None


def test_explicit_ids_skip_the_protocol_checks(tmp_path):
    dataset = _dataset(
        tmp_path, _b_series(0, 5, protocol="ivim_a"), _b_series(800, 6, protocol="other")
    )
    ids = _ids_by_number(dataset)
    assert dt.merge_series(dataset).groups == ()
    [group] = dt.merge_series(dataset, [ids[6], ids[5]]).groups
    assert group.series_ids == (ids[5], ids[6])
    assert group.path is not None


def test_explicit_ids_still_need_the_same_grid(tmp_path):
    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6, rows=7))
    ids = list(_ids_by_number(dataset).values())
    with pytest.raises(dt.MergeError, match="different image size"):
        dt.merge_series(dataset, ids)
    with pytest.raises(dt.MergeError, match="at least two"):
        dt.merge_series(dataset, ids[:1])
    with pytest.raises(dt.MergeError, match="not a converted series"):
        dt.merge_series(dataset, [*ids, 99])


def test_different_scaling_is_merged_as_float(tmp_path):
    dataset = _dataset(
        tmp_path, _b_series(0, 5, rescale=(1.0, 0.0)), _b_series(50, 6, rescale=(2.0, -5.0))
    )
    [group] = dt.merge_series(dataset).groups
    merged = nib.load(group.path)
    assert merged.get_data_dtype() == np.float32
    sidecar = json.loads(
        group.path.with_name(group.path.name.replace(".nii.gz", ".json")).read_text()
    )
    assert sidecar["DcmTether"]["Rescaled"] is True
    sources = sorted(output_rows(dataset), key=lambda r: r["series_id"])
    for volume, row in enumerate(sources):
        expected = nib.load(dataset / row["nifti_path"]).get_fdata()
        assert np.allclose(merged.dataobj[..., volume], expected)


def test_in_place_rescans_skip_the_merged_files(tmp_path):
    for s in (_b_series(0, 5), _b_series(50, 6)):
        write_series(s, tmp_path / "src" / f"s{s.spec.series_number}")
    assert dt.run(tmp_path / "src").ok
    dt.merge_series(tmp_path / "src")
    again = dt.run(tmp_path / "src")
    assert again.scan.n_skipped == 0


def test_cli_merge_names_series_by_id_only(tmp_path):
    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6))
    ids = sorted(_ids_by_number(dataset).values())
    result = CliRunner().invoke(main, ["merge", str(dataset)])
    assert result.exit_code == 0, result.output
    assert result.output == f"Merged series #{ids[0]}, #{ids[1]} (2 volumes).\n"
    again = CliRunner().invoke(main, ["merge", str(dataset), "--ids", f"{ids[0]},{ids[1]}"])
    assert again.exit_code == 1
    assert again.output == f"Not merged: series #{ids[0]}, #{ids[1]}: already merged.\n"
    bad = CliRunner().invoke(main, ["merge", str(dataset), "--ids", "a,b"])
    assert bad.exit_code == 2
    nothing = CliRunner().invoke(main, ["merge", str(tmp_path / "dataset"), "--ids", "1"])
    assert nothing.exit_code == 1 and "at least two" in nothing.output


def test_merge_is_logged(tmp_path):
    from dcmtether import log

    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6))
    dt.merge_series(dataset)
    records = [json.loads(line) for line in log.log_files(dataset)[-1].read_text().splitlines()]
    merged = [r for r in records if r["event"] == "merged"]
    assert merged and merged[0]["command"] == "merge" and merged[0]["count"] == 2


def test_shared_scaling_is_kept(tmp_path):
    rescale = (2.0, -5.0)
    dataset = _dataset(
        tmp_path, _b_series(0, 5, rescale=rescale), _b_series(50, 6, rescale=rescale)
    )
    [group] = dt.merge_series(dataset).groups
    merged = nib.load(group.path)
    assert merged.get_data_dtype() == np.uint16  # stored values unchanged
    assert (float(merged.dataobj.slope), float(merged.dataobj.inter)) == rescale
    sources = sorted(output_rows(dataset), key=lambda r: r["series_id"])
    for volume, row in enumerate(sources):
        expected = nib.load(dataset / row["nifti_path"]).get_fdata()
        assert np.array_equal(merged.get_fdata()[..., volume], expected)


def test_two_acquisitions_of_one_protocol_are_two_groups(tmp_path):
    """The same IVIM protocol measured twice at different positions: one group each."""
    other = {"origin": (-10.0, -20.0, 60.0)}
    dataset = _dataset(
        tmp_path,
        _b_series(0, 5),
        _b_series(50, 6),
        _b_series(0, 15, **other),
        _b_series(50, 16, **other),
    )
    ids = _ids_by_number(dataset)
    groups = dt.merge_series(dataset).groups
    assert sorted(g.series_ids for g in groups) == sorted([(ids[5], ids[6]), (ids[15], ids[16])])
    assert all(g.path is not None for g in groups)
    assert len({g.path for g in groups}) == 2


def test_merged_tether_writes_the_source_series_back(tmp_path):
    series = [_b_series(0, 5), _b_series(50, 6, gradient=(1.0, 0.0, 0.0)), _b_series(800, 7)]
    dataset = _dataset(tmp_path, *series)
    [group] = dt.merge_series(dataset).groups
    assert group.tether_path is not None and group.no_tether is None
    assert group.tether_path.name.endswith("_desc-merged_dwi.tether.json")

    rebuilt = tmp_path / "rebuilt"
    written = dt.to_dcm(group.path, rebuilt).paths
    assert len(written) == sum(len(s.datasets) for s in series)
    for s in series:  # one folder per source series, as in SOURCE
        folder = f"s{s.spec.series_number}"
        report = dt.compare(tmp_path / "src" / folder, rebuilt / folder)
        assert report.identical, report.lines()
        assert report.n_compared == len(s.datasets)


def test_merged_tether_with_every_series_in_one_folder(tmp_path):
    folder = tmp_path / "src" / "all"
    folder.mkdir(parents=True)
    for s in (_b_series(0, 5), _b_series(50, 6)):
        for i, ds in enumerate(s.datasets, start=1):
            ds.save_as(folder / f"S{s.spec.series_number}_IM_{i:04d}", enforce_file_format=True)
    dataset = tmp_path / "dataset"
    assert dt.run(tmp_path / "src", dataset).ok
    [group] = dt.merge_series(dataset).groups
    rebuilt = tmp_path / "rebuilt"
    dt.to_dcm(group.path, rebuilt)
    assert dt.compare(tmp_path / "src" / "all", rebuilt).identical


def test_no_merged_tether_when_rescaled_to_float(tmp_path):
    dataset = _dataset(
        tmp_path, _b_series(0, 5, rescale=(1.0, 0.0)), _b_series(50, 6, rescale=(2.0, -5.0))
    )
    [group] = dt.merge_series(dataset).groups
    assert group.tether_path is None
    assert group.no_tether == "stored values rescaled to float"
    assert not list(group.path.parent.glob("*.tether.json"))


def test_merged_tether_of_4d_series(tmp_path):
    from synthetic import make_4d

    series = [make_4d(SeriesSpec(series_number=n), n_volumes=2, kind="echo") for n in (5, 9)]
    dataset = _dataset(tmp_path, *series)
    [group] = dt.merge_series(dataset, list(_ids_by_number(dataset).values())).groups
    assert group.n_volumes == 4 and group.tether_path is not None
    merged = nib.load(group.path)
    sources = sorted(output_rows(dataset), key=lambda r: r["series_id"])
    for n, row in enumerate(sources):
        source = np.asanyarray(nib.load(dataset / row["nifti_path"]).dataobj)
        assert np.array_equal(merged.dataobj[..., 2 * n : 2 * n + 2], source)
    rebuilt = tmp_path / "rebuilt"
    dt.to_dcm(group.path, rebuilt)
    assert dt.compare(tmp_path / "src", rebuilt).identical


# --- Siemens XA split series (MFSPLIT) ---------------------------------------------------------

XA_MAGNITUDE = ["ORIGINAL", "PRIMARY", "M", "MFSPLIT", "NORM"]


def _xa_series(number: int, series_time="101500.123", echo_time=10.0, image_type=None, **attrs):
    """One classic series that Siemens XA split from an enhanced acquisition."""
    extra = {
        "Manufacturer": "SIEMENS",
        "SoftwareVersions": "syngo MR XA30",
        "ImageType": image_type or XA_MAGNITUDE,
        "SeriesTime": series_time,
        "ProtocolName": "gre_me",
        "EchoTime": echo_time,
        **attrs,
    }
    series = make_series(SeriesSpec(series_number=number, series_description="gre_me", extra=extra))
    for ds in series.datasets:
        pixels = np.frombuffer(ds.PixelData, dtype=series.volume.dtype) + number * 100
        ds.PixelData = pixels.astype(series.volume.dtype).tobytes()
    return series


def test_xa_split_echoes_are_merged_and_written_back(tmp_path):
    # SeriesNumbers of one acquisition need not follow any pattern
    series = [_xa_series(41, echo_time=30.0), _xa_series(7), _xa_series(23, echo_time=20.0)]
    dataset = _dataset(tmp_path, *series)
    ids = _ids_by_number(dataset)
    [group] = dt.merge_series(dataset).groups
    assert group.rule == "siemens-xa-split"
    assert group.series_ids == (ids[7], ids[23], ids[41])
    sidecar_path = group.path.with_name(group.path.name.replace(".nii.gz", ".json"))
    text = sidecar_path.read_text("utf-8")
    sidecar = json.loads(text)
    assert sidecar["DcmTether"]["MergeRule"] == "siemens-xa-split"
    assert sidecar["DcmTether"]["PerVolume"]["EchoTime"] == [0.01, 0.02, 0.03]
    assert "101500" not in text  # SeriesTime is a grouping key only

    rebuilt = tmp_path / "rebuilt"
    dt.to_dcm(group.path, rebuilt)
    assert dt.compare(tmp_path / "src", rebuilt).identical


def test_xa_split_needs_the_same_series_time(tmp_path):
    dataset = _dataset(tmp_path, _xa_series(7), _xa_series(8, series_time="103000.456"))
    assert dt.merge_series(dataset).groups == ()


def test_xa_split_keeps_magnitude_and_phase_apart(tmp_path):
    phase = ["ORIGINAL", "PRIMARY", "P", "MFSPLIT", "NORM"]
    dataset = _dataset(
        tmp_path,
        _xa_series(7),
        _xa_series(8, echo_time=20.0),
        _xa_series(9, image_type=phase),
        _xa_series(10, echo_time=20.0, image_type=phase),
    )
    ids = _ids_by_number(dataset)
    groups = dt.merge_series(dataset).groups
    assert sorted(g.series_ids for g in groups) == sorted([(ids[7], ids[8]), (ids[9], ids[10])])
    assert all(g.rule == "siemens-xa-split" and g.path is not None for g in groups)


@pytest.mark.parametrize(
    "change",
    [
        {"image_type": ["ORIGINAL", "PRIMARY", "M", "NORM"]},  # no MFSPLIT
        {"SoftwareVersions": "syngo MR E11"},  # not XA
        {"Manufacturer": "GE MEDICAL SYSTEMS"},
    ],
    ids=["no-mfsplit", "not-xa", "not-siemens"],
)
def test_xa_split_rule_needs_all_signals(tmp_path, change):
    dataset = _dataset(tmp_path, _xa_series(7, **change), _xa_series(8, echo_time=20.0, **change))
    assert dt.merge_series(dataset).groups == ()


def test_xa_split_diffusion_is_grouped_once(tmp_path):
    """dwi series split by XA match both rules: the XA rule takes them."""
    diffusion = ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE", "MFSPLIT"]
    xa = {"Manufacturer": "SIEMENS", "SoftwareVersions": "syngo MR XA30", "SeriesTime": "101500"}
    series = []
    for number, b_value in ((5, 0), (6, 50)):
        s = _b_series(b_value, number)
        for ds in s.datasets:
            ds.ImageType = diffusion
            for keyword, value in xa.items():
                setattr(ds, keyword, value)
        series.append(s)
    dataset = _dataset(tmp_path, *series)
    [group] = dt.merge_series(dataset).groups
    assert group.rule == "siemens-xa-split" and group.path is not None


@pytest.mark.parametrize("named", [False, True], ids=["automatic", "ids"])
def test_merge_rule_is_recorded(tmp_path, named):
    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6))
    ids = list(_ids_by_number(dataset).values()) if named else None
    [group] = dt.merge_series(dataset, ids).groups
    assert group.rule == ("ids" if named else "diffusion-protocol")


@pytest.mark.parametrize("named", [False, True], ids=["automatic", "ids"])
def test_merge_reports_progress(tmp_path, named):
    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6), _b_series(0, 7, "other"))
    ids = list(_ids_by_number(dataset).values())[:2] if named else None
    calls = []

    def progress(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    [group] = dt.merge_series(dataset, ids, progress=progress).groups
    assert group.path is not None
    assert [(c["desc"], c["unit"], c["total"]) for c in calls] == [
        ("Reading series", "series", 3),
        ("Merging", "group", 1),
    ]


# --- placement: --in-place and --replace ------------------------------------------------------


def _series_rows(dataset: Path) -> dict[int, dict]:
    import sqlite3

    with sqlite3.connect(dataset / "manifest.sqlite") as db:
        db.row_factory = sqlite3.Row
        return {
            r["series_number"]: dict(r)
            for r in db.execute(
                "SELECT s.series_number, s.merged_into, o.nifti_path, o.n_volumes "
                "FROM series s LEFT JOIN outputs o ON o.series_id = s.id"
            )
        }


def test_in_place_writes_beside_the_series(tmp_path):
    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6))
    singles = sorted(dataset.glob("sub-*/**/*.nii.gz"))
    [group] = dt.merge_series(dataset, in_place=True).groups
    assert group.path.parent == singles[0].parent
    assert not (dataset / "derivatives").exists()
    assert not group.replaced and all(p.exists() for p in singles)
    rebuilt = tmp_path / "rebuilt"
    dt.to_dcm(group.path, rebuilt)
    assert dt.compare(tmp_path / "src", rebuilt).identical
    [again] = dt.merge_series(dataset, in_place=True).groups
    assert again.skipped == "already merged"


def test_replace_removes_the_single_series(tmp_path):
    series = [_b_series(0, 5), _b_series(50, 6), _b_series(800, 7)]
    dataset = _dataset(tmp_path, *series)
    ids = _ids_by_number(dataset)
    singles = sorted(p for p in dataset.glob("sub-*/**/*") if p.is_file())
    [group] = dt.merge_series(dataset, replace=True).groups
    assert group.replaced and group.not_replaced is None
    left = sorted(p for p in dataset.glob("sub-*/**/*") if p.is_file())
    assert not set(singles) & set(left)
    assert {p.name.removesuffix("".join(p.suffixes)) for p in left} == {
        group.path.name.removesuffix(".nii.gz")
    }

    rows = _series_rows(dataset)
    merged = group.path.relative_to(dataset).as_posix()
    assert (rows[5]["nifti_path"], rows[5]["n_volumes"], rows[5]["merged_into"]) == (
        merged,
        3,
        None,
    )
    assert rows[6] == {
        "series_number": 6,
        "merged_into": ids[5],
        "nifti_path": None,
        "n_volumes": None,
    }
    assert rows[7]["merged_into"] == ids[5] and rows[7]["nifti_path"] is None

    assert dt.merge_series(dataset).groups == ()
    rerun = dt.run(tmp_path / "src", dataset)
    assert rerun.ok and not rerun.converted
    rebuilt = tmp_path / "rebuilt"
    dt.to_dcm(group.path, rebuilt)
    for s in series:
        folder = f"s{s.spec.series_number}"
        assert dt.compare(tmp_path / "src" / folder, rebuilt / folder).identical


def test_replace_keeps_the_series_without_a_merged_tether(tmp_path):
    dataset = _dataset(
        tmp_path, _b_series(0, 5, rescale=(1.0, 0.0)), _b_series(50, 6, rescale=(2.0, -5.0))
    )
    singles = sorted(dataset.glob("sub-*/**/*.nii.gz"))
    [group] = dt.merge_series(dataset, replace=True).groups
    assert group.path is not None and group.path.parent == singles[0].parent
    assert not group.replaced and group.not_replaced == "no tether.json"
    assert all(p.exists() for p in singles)
    assert all(r["merged_into"] is None for r in _series_rows(dataset).values())


def test_cli_replace(tmp_path):
    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6))
    ids = sorted(_ids_by_number(dataset).values())
    result = CliRunner().invoke(main, ["merge", str(dataset), "--replace"])
    assert result.exit_code == 0, result.output
    assert result.output == (
        f"Merged series #{ids[0]}, #{ids[1]} (2 volumes); replaced the single series.\n"
    )


def test_cli_replace_without_tether_fails(tmp_path):
    dataset = _dataset(
        tmp_path, _b_series(0, 5, rescale=(1.0, 0.0)), _b_series(50, 6, rescale=(2.0, -5.0))
    )
    result = CliRunner().invoke(main, ["merge", str(dataset), "--replace"])
    assert result.exit_code == 1
    assert "not replaced (no tether.json): the single series stay" in result.output


def test_replace_is_logged(tmp_path):
    from dcmtether import log

    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6))
    dt.merge_series(dataset, replace=True)
    records = [json.loads(line) for line in log.log_files(dataset)[-1].read_text().splitlines()]
    assert [r["event"] for r in records if r["event"] in ("merged", "replaced")] == [
        "merged",
        "replaced",
    ]


def test_volumes_are_sorted_by_b_value(tmp_path):
    # SeriesNumbers in another order than the b-values
    series = [_b_series(800, 5), _b_series(0, 6), _b_series(50, 7)]
    dataset = _dataset(tmp_path, *series)
    ids = _ids_by_number(dataset)
    [group] = dt.merge_series(dataset).groups
    assert group.series_ids == (ids[6], ids[7], ids[5])
    sidecar = json.loads(
        group.path.with_name(group.path.name.replace(".nii.gz", ".json")).read_text()
    )
    assert sidecar["DcmTether"]["PerVolume"]["DiffusionBValue"] == [0, 50, 800]
    merged = nib.load(group.path)
    for volume, number in enumerate((6, 7, 5)):
        single = next(r for r in output_rows(dataset) if r["series_id"] == ids[number])
        expected = nib.load(dataset / single["nifti_path"]).get_fdata()
        assert np.array_equal(merged.get_fdata()[..., volume], expected)
    rebuilt = tmp_path / "rebuilt"
    dt.to_dcm(group.path, rebuilt)
    assert dt.compare(tmp_path / "src", rebuilt).identical


def test_volumes_are_sorted_by_b_value_then_direction(tmp_path):
    x, y, z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    acquired = {5: (50, z), 6: (0, None), 7: (50, x), 8: (100, x), 9: (100, z), 10: (50, y)}
    acquired[11] = (100, y)
    series = [_b_series(b, n, gradient=g) for n, (b, g) in acquired.items()]
    dataset = _dataset(tmp_path, *series)
    ids = _ids_by_number(dataset)
    [group] = dt.merge_series(dataset).groups
    # directions first appear as z (#5), x (#7), y (#10): that order within every b-value
    assert group.series_ids == tuple(ids[n] for n in (6, 5, 7, 10, 9, 8, 11))
    sidecar = json.loads(
        group.path.with_name(group.path.name.replace(".nii.gz", ".json")).read_text()
    )
    assert sidecar["DcmTether"]["PerVolume"]["DiffusionBValue"] == [0, 50, 50, 50, 100, 100, 100]
    rebuilt = tmp_path / "rebuilt"
    dt.to_dcm(group.path, rebuilt)
    assert dt.compare(tmp_path / "src", rebuilt).identical


def test_replace_after_in_place_uses_the_merged_files(tmp_path):
    dataset = _dataset(tmp_path, _b_series(0, 5), _b_series(50, 6))
    ids = _ids_by_number(dataset)
    [first] = dt.merge_series(dataset, in_place=True).groups
    singles = [p for p in dataset.glob("sub-*/**/*.nii.gz") if p != first.path]
    [group] = dt.merge_series(dataset, replace=True).groups
    assert group.replaced and group.skipped is None and group.path == first.path
    assert not any(p.exists() for p in singles)
    assert _series_rows(dataset)[6]["merged_into"] == ids[5]
    rebuilt = tmp_path / "rebuilt"
    dt.to_dcm(group.path, rebuilt)
    assert dt.compare(tmp_path / "src", rebuilt).identical
