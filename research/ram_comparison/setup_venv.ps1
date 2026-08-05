# setup_venv.ps1 — create venv_ramclip for the RAM++ vs TinyCLIP app.
#
# Separate from research/vitb32_benchmark/venv_clip on purpose: RAM's vendored
# BLIP-era BERT needs transformers 4.x, while venv_clip runs 5.12.1 for
# sentence-transformers. See requirements.txt for the details.
#
# Usage:  ./setup_venv.ps1

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $here "venv_ramclip"

# Corporate TLS interception — same trusted-host set the sibling script uses.
$trusted = @(
    "--trusted-host", "pypi.org",
    "--trusted-host", "files.pythonhosted.org",
    "--trusted-host", "download.pytorch.org"
)

if (-not (Test-Path $venv)) {
    Write-Host "Creating venv at $venv"
    python -m venv $venv
}

$py = Join-Path $venv "Scripts\python.exe"

& $py -m pip install --upgrade pip @trusted

# torch CPU wheels come from the pytorch index, everything else from PyPI.
Write-Host "`nInstalling torch (CPU)..."
& $py -m pip install @trusted --index-url https://download.pytorch.org/whl/cpu `
    torch==2.5.1 torchvision==0.20.1

Write-Host "`nInstalling the rest..."
& $py -m pip install @trusted `
    timm==0.9.16 transformers==4.44.2 "pandas>=2.0" `
    fairscale==0.4.13 scipy==1.13.1 `
    onnxruntime==1.27.0 onnxruntime_extensions==0.15.0 `
    "gradio>=5,<6" "numpy>=1.26,<3" "pillow>=11" "tqdm>=4.66"

Write-Host "`nDone. Verify with:"
Write-Host "  .\venv_ramclip\Scripts\python.exe verify_env.py"
