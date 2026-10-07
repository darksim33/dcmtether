"""Keep pydicom from quoting element values in warnings and log records (PHI rules)."""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

from pydicom import config


@contextmanager
def quiet_pydicom() -> Iterator[None]:
    """Suppress value-validation warnings while reading/serialising; values are kept as read."""
    previous = config.settings.reading_validation_mode
    config.settings.reading_validation_mode = config.IGNORE
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        config.settings.reading_validation_mode = previous
