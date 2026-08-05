#!/usr/bin/env bash
# setup_venv.sh — create venv_ramclip for the RAM++ vs TinyCLIP app (Linux/macOS).
#
# Separate from ../vitb32_benchmark/venv_clip on purpose: RAM's vendored
# BLIP-era BERT needs transformers 4.x. See requirements.txt.
#
# Torch defaults to CUDA when nvidia-smi is present, CPU wheels otherwise.
# RAM++ is the one component that genuinely uses the GPU (ram_tagger.py picks
# cuda automatically): ~1.5 s/image on CPU vs well under a second on GPU.
#
# Usage:
#   ./setup_venv.sh                 # auto-detect GPU
#   ./setup_venv.sh --cpu           # force CPU torch wheels (smaller)
#   ./setup_venv.sh --gpu           # force CUDA torch
#   ./setup_venv.sh --ort-gpu       # also swap onnxruntime -> onnxruntime-gpu,
#                                   # for the TinyCLIP side. OPT-IN, and you must
#                                   # then run:
#                                   #   parity_assets.py --check-providers
#   ./setup_venv.sh --system-site   # let the venv see the image's preinstalled
#                                   # packages (useful on prebuilt DS images)
#
# Env:
#   PYTHON=python3.11   pick a specific interpreter
#   PIP_TRUSTED=1       add --trusted-host flags for TLS-intercepting networks
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/venv_ramclip"
PYTHON="${PYTHON:-python3}"

if command -v nvidia-smi >/dev/null 2>&1; then GPU=1; else GPU=0; fi
ORT_GPU=0
VENV_FLAGS=()
for arg in "$@"; do
  case "$arg" in
    --cpu)         GPU=0 ;;
    --gpu)         GPU=1 ;;
    --ort-gpu)     ORT_GPU=1 ;;
    --system-site) VENV_FLAGS+=(--system-site-packages) ;;
    -h|--help)     sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$GPU" = 1 ]; then
  TORCH_INDEX=()
  echo "==> GPU detected (or forced): installing CUDA torch"
else
  TORCH_INDEX=(--index-url https://download.pytorch.org/whl/cpu)
  echo "==> no GPU: installing CPU torch wheels"
fi

TRUSTED=()
if [ "${PIP_TRUSTED:-0}" = "1" ]; then
  TRUSTED=(--trusted-host pypi.org --trusted-host files.pythonhosted.org
           --trusted-host download.pytorch.org)
fi

if [ ! -d "$VENV" ]; then
  echo "==> creating venv at $VENV"
  "$PYTHON" -m venv "${VENV_FLAGS[@]}" "$VENV"
fi
PY="$VENV/bin/python"

echo "==> $("$PY" --version)"
"$PY" -m pip install --upgrade pip "${TRUSTED[@]}"

# torch first, from its own index, so the CPU wheels win before the shared
# resolve pulls a CUDA build as a transitive dependency.
echo "==> installing torch"
"$PY" -m pip install "${TRUSTED[@]}" "${TORCH_INDEX[@]}" torch torchvision

echo "==> installing the rest"
"$PY" -m pip install "${TRUSTED[@]}" -r "$HERE/requirements.txt"

if [ "$ORT_GPU" = 1 ]; then
  echo "==> swapping onnxruntime -> onnxruntime-gpu"
  "$PY" -m pip uninstall -y onnxruntime || true
  "$PY" -m pip install "${TRUSTED[@]}" onnxruntime-gpu
fi

echo
"$PY" - <<'PYEOF'
import torch, onnxruntime as ort
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}"
      + (f"  device={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
print(f"  onnxruntime {ort.__version__}  providers={ort.get_available_providers()}")
PYEOF

cat <<EOF

Done. Next:

  $VENV/bin/python verify_env.py          # preflight, no checkpoint load
  $VENV/bin/python verify_env.py --full   # also loads the 2.8 GB checkpoint
  $VENV/bin/python app.py                 # http://127.0.0.1:7863

RAM++ uses the GPU automatically. TinyCLIP (onnxruntime) stays on CPU unless
you opt in -- see docs/jovyan_deployment.md, section "Using the GPU".

If artifacts are missing, run the bootstrap from the repo root:
  ./scripts/bootstrap_jovyan.sh --ram
EOF
