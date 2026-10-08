"""Fetch dcm_qa validation data (Chris Rorden's dcm2niix test repositories) at pinned commits.

Only the ``In/`` folders listed in ``tests/data/dcm_qa.json`` (``scripts/pin_dcm_qa.py``) can be
fetched. Every file is checked against its git blob SHA-1 before it is used; ``.gz`` files are
decompressed after the check. Files are cached under ``.cache/dcm_qa/<repo>/<commit>/``
(override the root with ``DCMTETHER_DCM_QA_CACHE``).
"""

import gzip
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from testdata import DownloadError, FetchError

_MANIFEST_PATH = Path(__file__).parent / "data" / "dcm_qa.json"
_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Not DICOM and not needed (documentation, macOS metadata).
_SKIPPED_SUFFIXES = (".pdf", ".DS_Store")


def _manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def repositories() -> list[str]:
    return sorted(_manifest()["repositories"])


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def cache_dir(repository: str) -> Path:
    base = os.environ.get("DCMTETHER_DCM_QA_CACHE")
    root = Path(base) if base else _REPO_ROOT / ".cache" / "dcm_qa"
    commit = _manifest()["repositories"][repository]["commit"]
    return root / repository / commit


def _local_name(path: str) -> str:
    return path.removesuffix(".gz")


def _fetch_file(owner: str, repository: str, commit: str, path: str, entry: dict) -> None:
    target = cache_dir(repository) / _local_name(path)
    if target.exists():
        return
    url = (
        f"https://raw.githubusercontent.com/{owner}/{repository}/{commit}/"
        f"{urllib.parse.quote(path)}"
    )
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except OSError as exc:
        raise DownloadError(f"failed to download {repository}/{path}: {exc}") from exc
    if git_blob_sha1(data) != entry["git_sha1"]:
        raise FetchError(f"checksum mismatch for {repository}/{path}")
    if path.endswith(".gz"):
        data = gzip.decompress(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp.write(data)
    Path(tmp.name).replace(target)


def fetch(repository: str, subdir: str = "") -> Path:
    """Local copy of ``In/<subdir>`` of ``repository`` (all files below), downloading as needed."""
    manifest = _manifest()
    repo = manifest["repositories"].get(repository)
    if repo is None:
        raise FetchError(f"unknown dcm_qa repository: {repository}")
    prefix = str(PurePosixPath("In", subdir)) + "/"
    paths = [p for p in repo["files"] if p.startswith(prefix) and not p.endswith(_SKIPPED_SUFFIXES)]
    if not paths:
        raise FetchError(f"nothing under {prefix} in {repository}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                _fetch_file, manifest["owner"], repository, repo["commit"], p, repo["files"][p]
            )
            for p in paths
        ]
        for future in futures:
            future.result()
    return cache_dir(repository) / prefix.rstrip("/")
