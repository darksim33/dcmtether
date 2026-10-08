"""dcm_qa validation data: fetching, and round trips of every series we can convert."""

import gzip
import io
import json
import re

import dcmqa
import pytest
import testdata
from roundtrip import manifest_rows, output_rows

from dcmtether.compare import compare_series, dicom_files
from dcmtether.forward import convert_dataset
from dcmtether.manifest import write_manifest
from dcmtether.reverse import to_dicom
from dcmtether.scan import Status, scan


def test_manifest_is_pinned():
    manifest = json.loads(dcmqa._MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["owner"] == "neurolabusc"
    assert manifest["repositories"]
    for repo in manifest["repositories"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", repo["commit"])
        assert all(path.startswith("In/") for path in repo["files"])
        for entry in repo["files"].values():
            assert re.fullmatch(r"[0-9a-f]{40}", entry["git_sha1"])


def test_git_blob_sha1():
    # `git hash-object` of a file containing "hello\n"
    assert dcmqa.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_unknown_repository_is_rejected():
    with pytest.raises(testdata.FetchError, match="unknown"):
        dcmqa.fetch("dcm_qa_does_not_exist")


def _fake_repo(monkeypatch, tmp_path, files: dict[str, bytes], served: dict[str, bytes]):
    manifest = {
        "owner": "x",
        "repositories": {
            "fake": {
                "commit": "0" * 40,
                "files": {
                    p: {"git_sha1": dcmqa.git_blob_sha1(d), "size": len(d)}
                    for p, d in files.items()
                },
            }
        },
    }
    path = tmp_path / "dcm_qa.json"
    path.write_text(json.dumps(manifest), "utf-8")
    monkeypatch.setattr(dcmqa, "_MANIFEST_PATH", path)
    monkeypatch.setenv("DCMTETHER_DCM_QA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(
        dcmqa.urllib.request,
        "urlopen",
        lambda url, **k: io.BytesIO(served[url.rsplit("/", 1)[-1]]),
    )


def test_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    _fake_repo(monkeypatch, tmp_path, {"In/a/x.dcm": b"good"}, {"x.dcm": b"bad"})
    with pytest.raises(testdata.FetchError, match="checksum"):
        dcmqa.fetch("fake")
    assert not (dcmqa.cache_dir("fake") / "In" / "a" / "x.dcm").exists()


def test_gz_files_are_verified_then_decompressed(tmp_path, monkeypatch):
    packed = gzip.compress(b"dicom bytes")
    files = {"In/a/x.dcm.gz": packed, "In/a/notes.pdf": b"pdf"}
    _fake_repo(monkeypatch, tmp_path, files, {"x.dcm.gz": packed})
    folder = dcmqa.fetch("fake", "a")
    assert [p.name for p in folder.iterdir()] == ["x.dcm"]  # .pdf is not fetched
    assert (folder / "x.dcm").read_bytes() == b"dicom bytes"


# --- real data ---------------------------------------------------------------------------------

#: Series converted (and round-tripped) per repository. A change here is deliberate: either a new
#: input type became supported, or something regressed. See docs/dev/test-data.md.
EXPECTED_CONVERTED = {
    "dcm_qa_12bit": 0,  # Siemens mosaic
    "dcm_qa_enh": 0,  # enhanced multi-frame (Phase 4c)
    "dcm_qa_me": 2,  # Siemens multi-echo GRE (4D, EchoNumbers) + a 3D series
    "dcm_qa_nih": 0,  # GE fMRI (volumes told apart only by InstanceNumber) + Siemens mosaic
    "dcm_qa_pdt2": 1,  # Siemens dual-echo PD/T2 (4D, EchoNumbers)
    "dcm_qa_philips_dwi": 0,  # classic + enhanced exports share one series UID
    "dcm_qa_ts": 0,  # one series in 7 transfer syntaxes (Phase 4b)
}


def test_expected_list_covers_every_pinned_repository():
    assert sorted(EXPECTED_CONVERTED) == dcmqa.repositories()


@pytest.mark.network
@pytest.mark.parametrize("repository", sorted(EXPECTED_CONVERTED))
def test_round_trip_real_data(tmp_path, dcm_qa, repository):
    root = dcm_qa(repository)
    dataset = tmp_path / "dataset"
    write_manifest(dataset, scan(root))
    outcomes = convert_dataset(dataset)
    assert all(o.ok for o in outcomes), [o.message for o in outcomes]
    assert len(outcomes) == EXPECTED_CONVERTED[repository]

    series = {r["id"]: r for r in manifest_rows(dataset)}
    for row in output_rows(dataset):
        rebuilt = tmp_path / f"rebuilt-{row['series_id']}"
        to_dicom(dataset / row["nifti_path"], rebuilt)
        source = root / series[row["series_id"]]["source_dir"]
        report = compare_series(dicom_files(source), dicom_files(rebuilt))
        assert report.identical, report.lines()
        assert report.notes == []


@pytest.mark.network
def test_unsupported_real_data_is_classified_not_guessed(dcm_qa):
    reasons = [r for s in scan(dcm_qa("dcm_qa_nih")).series for r in s.reasons]
    assert reasons.count("Siemens mosaic images") == 4
    assert (
        reasons.count(
            "10 volumes at the same 15 slice positions, but no attribute tells them apart"
        )
        == 4
    )
    statuses = {s.status for s in scan(dcm_qa("dcm_qa_me")).series}
    assert statuses == {Status.SUPPORTED, Status.UNSUPPORTED}
