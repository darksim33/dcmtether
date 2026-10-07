import io
import json
import re

import pydicom
import pytest
import testdata


def test_manifest_is_pinned_and_complete():
    manifest = json.loads(testdata._MANIFEST_PATH.read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["commit"])
    assert len(manifest["files"]) > 100
    for entry in manifest["files"].values():
        assert entry["compression"] in {"none", "zstd"}
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])


def test_unknown_file_is_rejected():
    with pytest.raises(testdata.FetchError, match="unknown"):
        testdata.fetch("pydicom/does-not-exist.dcm")


def test_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DCMTETHER_TEST_DATA_CACHE", str(tmp_path))
    monkeypatch.setattr(testdata.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    with pytest.raises(testdata.FetchError, match="checksum"):
        testdata.fetch("pydicom/MR_small.dcm")
    assert not (testdata.cache_dir() / "pydicom" / "MR_small.dcm").exists()


@pytest.mark.network
@pytest.mark.parametrize(
    "name", ["pydicom/MR_small.dcm", "dcmqi/paramaps/701-ADCb500/000000.dcm"]
)  # plain + zstd-compressed
def test_fetch_returns_readable_dicom(name, test_file):
    ds = pydicom.dcmread(test_file(name), stop_before_pixels=True)
    assert ds.Modality == "MR"
