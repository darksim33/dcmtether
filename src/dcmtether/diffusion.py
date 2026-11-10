"""Diffusion b-value and gradient of one image (classic instance or frame view).

Standard ``DiffusionBValue (0018,9087)`` / ``DiffusionGradientOrientation (0018,9089)`` first;
older classic exports store them only in vendor private tags, which are read as a fallback
(Siemens, GE, Philips). The b-matrix comes from the standard ``DiffusionBValueXX…ZZ`` elements
(enhanced MR) or Siemens' ``(0019,xx27)``; GE and Philips store none. Private blocks are found
by their private creator, never by a fixed element number. Vendor gradients are returned as
stored: only Siemens' are in patient LPS like the standard element; GE (scanner frame) and
Philips (RL/AP/FH) are not converted.
"""

import math
import struct
from dataclasses import dataclass

import pydicom

STANDARD = "standard"
SIEMENS = "siemens"
GE = "ge"
PHILIPS = "philips"

#: Where each source's values live, for the sidecar (``DcmTether.DiffusionSource``).
SOURCE_TAGS = {
    STANDARD: "(0018,9087)/(0018,9089)/(0018,9602-9607)",
    SIEMENS: "SIEMENS MR HEADER (0019,xx0C)/(0019,xx0E)/(0019,xx27)",
    GE: "GEMS_PARM_01 (0043,xx39) / GEMS_ACQU_01 (0019,xxBB-xxBD)",
    PHILIPS: "Philips Imaging DD 001 (2001,xx03) / Philips MR Imaging DD 001 (2005,xxB0-xxB2)",
}

#: Suffixes of the standard ``DiffusionBValueXX`` … ``DiffusionBValueZZ`` b-matrix elements.
_B_MATRIX_AXES = ("XX", "XY", "XZ", "YY", "YZ", "ZZ")

#: GE adds this to the b-value in slop_int_6 on some software versions.
_GE_B_OFFSET = 1_000_000_000


@dataclass(frozen=True)
class Diffusion:
    b_value: float | None = None
    #: As stored (b0 images often have none, or ``(0, 0, 0)``).
    gradient: tuple[float, float, float] | None = None
    #: :data:`STANDARD`, :data:`SIEMENS`, :data:`GE`, :data:`PHILIPS`; ``None`` if no b-value.
    source: str | None = None
    #: Upper triangle (xx, xy, xz, yy, yz, zz), if the source stores one.
    b_matrix: tuple[float, float, float, float, float, float] | None = None

    @property
    def gradient_lps(self) -> bool:
        """The gradient is in patient LPS, like DiffusionGradientOrientation."""
        return self.source in (STANDARD, SIEMENS)


def _numbers(value, fmt: str | None = None) -> list[float] | None:
    """Numeric values of an element value; ``fmt`` decodes raw bytes (VR UN) as binary."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, bytes):
            if fmt is None:
                value = value.decode("ascii").strip("\0 ").split("\\")
            else:
                size = struct.calcsize(fmt)
                value = [v for (v,) in struct.iter_unpack(fmt, value[: len(value) // size * size])]
        elif isinstance(value, str | int | float) or not hasattr(value, "__iter__"):
            value = [value]
        numbers = [float(v) for v in value]
    except (TypeError, ValueError, UnicodeDecodeError, struct.error):
        return None
    return numbers if numbers and all(math.isfinite(n) for n in numbers) else None


def _private(ds: pydicom.Dataset, group: int, creator: str, offset: int, fmt=None):
    try:
        return _numbers(ds.private_block(group, creator)[offset].value, fmt)
    except KeyError:
        return None


def _direction(values: list[float] | None) -> tuple[float, float, float] | None:
    if values is None or len(values) != 3:
        return None
    return (values[0], values[1], values[2])


def _first(values: list[float] | None) -> float | None:
    return None if not values else values[0]


def _b_matrix(values: list[float] | None) -> tuple[float, float, float, float, float, float] | None:
    if values is None or len(values) != 6:
        return None
    return (values[0], values[1], values[2], values[3], values[4], values[5])


def _standard(ds: pydicom.Dataset) -> Diffusion | None:
    b_value = _first(_numbers(ds.get("DiffusionBValue")))
    if b_value is None:
        return None
    gradient = ds.get("DiffusionGradientOrientation")
    parts = [_first(_numbers(ds.get(f"DiffusionBValue{axes}"))) for axes in _B_MATRIX_AXES]
    b_matrix = None if any(p is None for p in parts) else _b_matrix(parts)
    return Diffusion(b_value, _direction(_numbers(gradient)), STANDARD, b_matrix)


def _siemens(ds: pydicom.Dataset) -> Diffusion | None:
    b_value = _first(_private(ds, 0x0019, "SIEMENS MR HEADER", 0x0C))
    if b_value is None:
        return None
    gradient = _private(ds, 0x0019, "SIEMENS MR HEADER", 0x0E, "<d")
    b_matrix = _b_matrix(_private(ds, 0x0019, "SIEMENS MR HEADER", 0x27, "<d"))
    return Diffusion(b_value, _direction(gradient), SIEMENS, b_matrix)


def _ge(ds: pydicom.Dataset) -> Diffusion | None:
    # slop_int_6 holds other data on non-diffusion sequences: require the gradient tags too.
    parts = [_first(_private(ds, 0x0019, "GEMS_ACQU_01", off)) for off in (0xBB, 0xBC, 0xBD)]
    b_value = _first(_private(ds, 0x0043, "GEMS_PARM_01", 0x39))
    if b_value is None or any(p is None for p in parts):
        return None
    return Diffusion(b_value % _GE_B_OFFSET, _direction(parts), GE)


def _philips(ds: pydicom.Dataset) -> Diffusion | None:
    b_value = _first(_private(ds, 0x2001, "Philips Imaging DD 001", 0x03, "<f"))
    if b_value is None:
        return None
    parts = [
        _first(_private(ds, 0x2005, "Philips MR Imaging DD 001", off, "<f"))
        for off in (0xB0, 0xB1, 0xB2)
    ]
    gradient = None if any(p is None for p in parts) else _direction(parts)
    return Diffusion(b_value, gradient, PHILIPS)


_VENDORS = (("SIEMENS", _siemens), ("GE", _ge), ("PHILIPS", _philips))


def diffusion_values(ds: pydicom.Dataset) -> Diffusion:
    """Standard tags, else the private tags of the instance's manufacturer (MR only)."""
    standard = _standard(ds)
    if standard is not None:
        return standard
    if str(ds.get("Modality", "")).upper() != "MR":
        return Diffusion()
    words = str(ds.get("Manufacturer", "")).upper().split()
    for name, read in _VENDORS:
        if name in words:
            return read(ds) or Diffusion()
    return Diffusion()
