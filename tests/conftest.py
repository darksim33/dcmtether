import os
from collections.abc import Callable
from pathlib import Path

import pytest
import testdata
from dicom_validator.spec_reader.edition_reader import EditionReader
from dicom_validator.validator.dicom_file_validator import DicomFileValidator
from dicom_validator.validator.error_handler import ValidationResultHandlerBase
from dicom_validator.validator.validation_result import ErrorCode, ValidationResult
from pydicom.datadict import keyword_for_tag

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Pinned so validation results don't change when NEMA publishes a new edition.
DICOM_EDITION = "2026d"

IODError = tuple[str, str, ErrorCode]
"""(module name, tag keyword or hex tag, error code)"""


class _CollectingHandler(ValidationResultHandlerBase):
    """Keeps results for assertions instead of logging them."""


@pytest.fixture(scope="session")
def iod_errors() -> Callable[[Path], list[IODError]]:
    """Validate one DICOM file against its IOD; return all errors found."""
    path = Path(
        os.environ.get("DCMTETHER_DICOM_STANDARD_CACHE", REPO_ROOT / ".cache" / "dicom-validator")
    )
    path.mkdir(parents=True, exist_ok=True)  # EditionReader assumes it exists
    reader = EditionReader(path)
    if reader.get_edition_path(DICOM_EDITION) is None:
        pytest.skip(f"DICOM standard {DICOM_EDITION} unavailable (needs network on first run)")
    validator = DicomFileValidator(
        reader.load_dicom_info(DICOM_EDITION), error_handler=_CollectingHandler()
    )

    def run(file_path: Path) -> list[IODError]:
        results: dict[str, ValidationResult] = validator.validate(file_path)
        errors = []
        for result in results.values():
            for module, tag_errors in result.module_errors.items():
                for dicom_tag, error in tag_errors.items():
                    name = keyword_for_tag(dicom_tag.tag) or str(dicom_tag.tag)
                    errors.append((module, name, error.code))
            if result.errors and not result.module_errors:
                errors.append(("<file>", str(result.status), ErrorCode.NoError))
        return errors

    return run


@pytest.fixture(scope="session")
def test_file() -> Callable[[str], Path]:
    """Fetch a pinned public test file by name; skip the test if offline."""

    def get(name: str) -> Path:
        try:
            return testdata.fetch(name)
        except testdata.DownloadError as exc:
            pytest.skip(str(exc))

    return get
