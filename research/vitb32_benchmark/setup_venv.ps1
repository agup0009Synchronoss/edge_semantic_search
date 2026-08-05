# setup_venv.ps1
# Creates venv_clip and installs all dependencies needed for the VG embeddings
# verification scripts. Uses --trusted-host flags to bypass corporate SSL inspection.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvPath  = Join-Path $ScriptDir "venv_clip"

Write-Host "=== Creating virtual environment: $VenvPath ===" -ForegroundColor Cyan
python -m venv $VenvPath

$PipExe = Join-Path $VenvPath "Scripts\pip.exe"

# Common trusted-host flags for corporate SSL bypass
$TrustedHosts = @(
    "--trusted-host", "pypi.org",
    "--trusted-host", "files.pythonhosted.org",
    "--trusted-host", "download.pytorch.org"
)

Write-Host "`n=== Upgrading pip ===" -ForegroundColor Cyan
& $PipExe install --upgrade pip @TrustedHosts

Write-Host "`n=== Installing core dependencies ===" -ForegroundColor Cyan
& $PipExe install @TrustedHosts `
    "numpy" `
    "Pillow" `
    "scikit-learn" `
    "tqdm" `
    "requests"

Write-Host "`n=== Installing PyTorch (CPU) ===" -ForegroundColor Cyan
& $PipExe install @TrustedHosts `
    torch torchvision `
    --index-url https://download.pytorch.org/whl/cpu

Write-Host "`n=== Installing sentence-transformers ===" -ForegroundColor Cyan
& $PipExe install @TrustedHosts `
    "sentence-transformers"

Write-Host "`n=== All dependencies installed successfully ===" -ForegroundColor Green
Write-Host "Activate with:  .\venv_clip\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Run scripts:"
Write-Host "  .\venv_clip\Scripts\python.exe 01_verify_pkl_coverage.py"
Write-Host "  .\venv_clip\Scripts\python.exe 02_cosim_verification.py"
