# PyInstaller spec for the portable Windows app (a folder with dcmtether-gui.exe).
# Build with scripts/build_windows.ps1 from the repository root.
from pathlib import Path

from PIL import Image
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

ROOT = Path(SPECPATH).parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

icon = BUILD / "dcmtether.ico"
Image.open(ROOT / "docs" / "assets" / "logo.png").save(
    icon, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
)

# The window icons (dcmtether/gui/icon-*.png) are package data, collected below.
# Found through package metadata at run time, so invisible to PyInstaller's import analysis:
# dcmtether.__version__ reads its own metadata, and pydicom/pylibjpeg find their pixel data
# decoders through entry points.
metadata = []
for package in (
    "dcmtether",
    "pydicom",
    "pylibjpeg",
    "pylibjpeg-libjpeg",
    "pylibjpeg-openjpeg",
    "pylibjpeg-rle",
    "pyjpegls",
    "highdicom",
    "nibabel",
):
    metadata += copy_metadata(package)

analysis = Analysis(
    [str(ROOT / "packaging" / "launch_gui.py")],
    pathex=[str(ROOT / "src")],
    datas=metadata + collect_data_files("highdicom") + collect_data_files("dcmtether"),
    hiddenimports=["libjpeg", "openjpeg", "rle", "jpeg_ls"],
    excludes=[
        "pytest",
        "_pytest",
        "dicom_validator",
        "dcm2niix",
        "ruff",
        "IPython",
        "matplotlib",
        "scipy",
    ],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    exclude_binaries=True,
    name="dcmtether-gui",
    icon=str(icon),
    console=False,
    upx=False,
)
COLLECT(exe, analysis.binaries, analysis.datas, name="dcmtether-gui", upx=False)
