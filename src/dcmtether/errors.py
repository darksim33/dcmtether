"""The base class of every error dcmtether raises on purpose."""


class DcmtetherError(Exception):
    """An expected failure. Messages never contain element values, names, UIDs or paths."""
