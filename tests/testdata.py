"""Fetch public test files from ``robyoung/dicom-test-files`` at a pinned commit.

Only files listed in ``tests/data/dicom_test_files.json`` can be fetched, and every
download is checked against its SHA-256 before it is used. Files are cached under
``.cache/dicom-test-files/<commit>/`` (override with ``DCMTETHER_TEST_DATA_CACHE``).
"""

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import zstandard

_MANIFEST_PATH = Path(__file__).parent / "data" / "dicom_test_files.json"
_REPO_ROOT = Path(__file__).resolve().parent.parent


class FetchError(RuntimeError):
    """A test file is unknown or failed verification."""


class DownloadError(FetchError):
    """A test file could not be downloaded (e.g. offline)."""


def _manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def available() -> list[str]:
    """Names of all fetchable test files (e.g. ``pydicom/MR_small.dcm``)."""
    return sorted(_manifest()["files"])


def cache_dir() -> Path:
    base = os.environ.get("DCMTETHER_TEST_DATA_CACHE")
    root = Path(base) if base else _REPO_ROOT / ".cache" / "dicom-test-files"
    return root / _manifest()["commit"]


def fetch(name: str) -> Path:
    """Return the local path of test file ``name``, downloading it if needed."""
    manifest = _manifest()
    entry = manifest["files"].get(name)
    if entry is None:
        raise FetchError(f"unknown test file: {name}")

    target = cache_dir() / name
    if target.exists():
        return target

    remote_name = name + (".zst" if entry["compression"] == "zstd" else "")
    url = (
        f"https://raw.githubusercontent.com/{manifest['repository']}/"
        f"{manifest['commit']}/data/{remote_name}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        download = Path(tmp) / "download"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, download.open("wb") as f:
                shutil.copyfileobj(resp, f)
        except OSError as exc:
            raise DownloadError(f"failed to download {url}: {exc}") from exc

        digest = hashlib.sha256(download.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise FetchError(f"checksum mismatch for {name}: {digest}")

        staged = Path(tmp) / "staged"
        if entry["compression"] == "zstd":
            with download.open("rb") as src, staged.open("wb") as dst:
                zstandard.ZstdDecompressor().copy_stream(src, dst)
        else:
            download.replace(staged)
        staged.replace(target)
    return target
