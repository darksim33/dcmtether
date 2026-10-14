# Build the portable Windows app: dist/dcmtether-gui-<version>-win64.zip
# Run from the repository root: powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
$ErrorActionPreference = "Stop"

$version = (uv run python -c "import dcmtether; print(dcmtether.__version__)").Trim()
$app = "dist/dcmtether-gui"
$zip = "dist/dcmtether-gui-$version-win64.zip"

uv sync --no-dev
uv run --no-dev --with pyinstaller pyinstaller --noconfirm --clean packaging/dcmtether-gui.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# The app has no console: the self-test writes its result next to the exe.
Remove-Item "$app/self-test.txt" -ErrorAction SilentlyContinue
$test = Start-Process "$app/dcmtether-gui.exe" -ArgumentList "--self-test" -Wait -PassThru
$result = Get-Content "$app/self-test.txt" -Raw
Remove-Item "$app/self-test.txt"
if ($test.ExitCode -ne 0) { throw "self-test failed:`n$result" }

Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path $app -DestinationPath $zip
uv sync  # restore the development environment

$unpacked = (Get-ChildItem $app -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
$packed = (Get-Item $zip).Length / 1MB
"Built {0} ({1:N0} MB, {2:N0} MB unpacked)" -f $zip, $packed, $unpacked
