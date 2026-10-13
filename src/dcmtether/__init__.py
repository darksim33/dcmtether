"""DCMtether: lossless DICOM ⇄ NIfTI round trips for MRI series (research use only).

The names in ``__all__`` are the Python API (docs/api.md); everything else is internal.
"""

from importlib.metadata import version

__version__ = version("dcmtether")

# After __version__: the modules below import it from here.
from dcmtether.api import (
    ConversionResult,
    DcmResult,
    RunResult,
    ScanSummary,
    SeriesInfo,
    compare,
    run,
    scan,
    to_dcm,
    to_mr,
    to_nii,
    to_pmap,
    to_seg,
)
from dcmtether.comparison import Difference, Report
from dcmtether.derived import DerivedError
from dcmtether.derived_mr import DerivedMRResult
from dcmtether.errors import DcmtetherError
from dcmtether.forward import ConversionError, Outcome
from dcmtether.manifest import ManifestError
from dcmtether.pmap import PmapResult
from dcmtether.reverse import ReverseError
from dcmtether.scanning import Status
from dcmtether.seg import SegResult
from dcmtether.tether import TetherError

__all__ = [
    "ConversionError",
    "ConversionResult",
    "DcmResult",
    "DcmtetherError",
    "DerivedError",
    "DerivedMRResult",
    "Difference",
    "ManifestError",
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
    "__version__",
    "compare",
    "run",
    "scan",
    "to_dcm",
    "to_mr",
    "to_nii",
    "to_pmap",
    "to_seg",
]
