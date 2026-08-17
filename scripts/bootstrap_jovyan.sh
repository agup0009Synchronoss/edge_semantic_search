#!/usr/bin/env bash
# bootstrap_jovyan.sh — get the two Gradio apps running on a fresh Linux box.
#
# Written for a jovyan-style image (JupyterHub/datascience-notebook), but there
# is nothing jovyan-specific in it beyond the default paths.
#
# Usage, from the repo root:
#   ./scripts/bootstrap_jovyan.sh --ram              RAM++ vs TinyCLIP app
#   ./scripts/bootstrap_jovyan.sh --clip             TinyCLIP vs CLIP ViT-B/32 app
#   ./scripts/bootstrap_jovyan.sh --all              both
#
# Useful flags:
#   --skip-venv            reuse existing venvs, only fix up assets
#   --regenerate           build every derivable asset here instead of copying
#                          it over. On a GPU box this is usually the right call
#                          -- it removes the 18 MB RAM transfer entirely.
#   --cpu / --gpu          force torch flavour (default: auto-detect nvidia-smi)
#   --ort-gpu              also put onnxruntime on CUDA. OPT-IN: verify with
#                          parity_assets.py --check-providers before trusting
#                          anything rebuilt this way.
#   --system-site          venvs inherit the image's preinstalled packages
#
# Env:
#   PYTHON=python3.11               interpreter for the venvs
#   PIP_TRUSTED=1                   --trusted-host flags for TLS interception
#   VG_DATA_ROOT=/path/to/vg        Visual Genome corpus (the CLIP app only)
#
# This script is idempotent: re-running it skips whatever is already in place.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAM="$REPO/research/ram_comparison"
CLIP="$REPO/research/vitb32_benchmark"

DO_RAM=0; DO_CLIP=0; SKIP_VENV=0; BUILD_CLS=0
VENV_ARGS=()

[ $# -eq 0 ] && { sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0; }
for arg in "$@"; do
  case "$arg" in
    --ram)  DO_RAM=1 ;;
    --clip) DO_CLIP=1 ;;
    --all)  DO_RAM=1; DO_CLIP=1 ;;
    --skip-venv)  SKIP_VENV=1 ;;
    --regenerate) BUILD_CLS=1 ;;
    --build-classifiers) BUILD_CLS=1 ;;   # kept as an alias
    --cpu|--gpu|--ort-gpu|--system-site) VENV_ARGS+=("$arg") ;;
    -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*"; }
ok()   { printf '  ok %s\n' "$*"; }

MISSING=0
need_file() {  # need_file <path> <human description>
  if [ -s "$1" ]; then ok "$2"
  else warn "MISSING: $2"; warn "         expected at $1"; MISSING=1; fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared: the ONNX encoders are committed, so this should always pass. If it
# does not, the clone is damaged — stop before wasting a 2.8 GB download.
# ─────────────────────────────────────────────────────────────────────────────
step "checking committed ONNX encoders"
"${PYTHON:-python3}" "$REPO/research/common/verify_assets.py"

# ─────────────────────────────────────────────────────────────────────────────
# RAM++ vs TinyCLIP
# ─────────────────────────────────────────────────────────────────────────────
if [ "$DO_RAM" = 1 ]; then
  step "RAM++ app: virtualenv"
  if [ "$SKIP_VENV" = 1 ] && [ -x "$RAM/venv_ramclip/bin/python" ]; then
    ok "reusing $RAM/venv_ramclip"
  else
    ( cd "$RAM" && ./setup_venv.sh "${VENV_ARGS[@]+"${VENV_ARGS[@]}"}" )
  fi
  PY_RAM="$RAM/venv_ramclip/bin/python"

  step "RAM++ app: seeding committed artifacts into data/text/"
  # thresholds_4585.npy, tag_order_4585.json and the mapping CSV are committed
  # under results/. The app reads them from data/text/, which is gitignored, so
  # copy rather than re-deriving them with 02_build_tag_mapping.py.
  mkdir -p "$RAM/data/text" "$RAM/data/ram" "$RAM/data/cache" "$RAM/data/llm_desc"
  for f in thresholds_4585.npy tag_order_4585.json lvis_to_ram_mapping.csv; do
    if [ ! -s "$RAM/data/text/$f" ]; then
      cp "$RAM/results/$f" "$RAM/data/text/$f"
      ok "seeded $f from results/"
    else
      ok "$f already present"
    fi
  done

  step "RAM++ app: vendored recognize-anything source"
  if [ -d "$RAM/vendor/recognize-anything" ]; then
    ok "already vendored"
  else
    mkdir -p "$RAM/vendor"
    GIT_SSL_NO_VERIFY="${GIT_SSL_NO_VERIFY:-0}" git clone --depth 1 \
      https://github.com/xinyu1205/recognize-anything.git \
      "$RAM/vendor/recognize-anything"
  fi

  step "RAM++ app: checkpoint + tag lists (2.8 GB, resumable)"
  if [ -s "$RAM/data/ram/ram_plus_swin_large_14m.pth" ]; then
    ok "checkpoint already present"
  else
    ( cd "$RAM" && "$PY_RAM" 00_download_ram.py )
  fi

  step "RAM++ app: TinyCLIP classifier matrices"
  # ~9 MB each. Neither committed nor downloadable, but fully DERIVABLE from
  # things that are: the committed descriptions asset and the committed
  # templates module. So on any box with cores to spare, --regenerate beats
  # copying. Note these encode through onnxruntime, which is CPU unless you
  # opted into --ort-gpu; the work parallelises across cores, not the GPU.
  if [ "$BUILD_CLS" = 1 ]; then
    if [ ! -s "$RAM/data/text/classifiers_llm.npy" ]; then
      echo "  building llm classifiers (~92k strings)"
      ( cd "$RAM" && "$PY_RAM" 01_build_text_classifiers.py --source llm )
    fi
    if [ ! -s "$RAM/data/text/classifiers_templates.npy" ]; then
      echo "  building template classifiers (~403k strings, the slow one)"
      ( cd "$RAM" && "$PY_RAM" 01_build_text_classifiers.py --source templates )
    fi
  fi
  need_file "$RAM/data/text/classifiers_llm.npy"       "classifiers_llm.npy (9 MB)"
  need_file "$RAM/data/text/classifiers_templates.npy" "classifiers_templates.npy (9 MB)"
  if [ "$MISSING" = 1 ]; then
    warn "re-run with --regenerate to build them here, or copy them across"
    warn "see docs/jovyan_deployment.md"
  fi

  step "RAM++ app: preflight"
  ( cd "$RAM" && "$PY_RAM" verify_env.py ) || warn "verify_env reported problems (see above)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# TinyCLIP vs CLIP ViT-B/32
# ─────────────────────────────────────────────────────────────────────────────
if [ "$DO_CLIP" = 1 ]; then
  step "CLIP app: virtualenv"
  if [ "$SKIP_VENV" = 1 ] && [ -x "$CLIP/venv_clip/bin/python" ]; then
    ok "reusing $CLIP/venv_clip"
  else
    ( cd "$CLIP" && ./setup_venv.sh "${VENV_ARGS[@]+"${VENV_ARGS[@]}"}" )
  fi

  step "CLIP app: Visual Genome corpus"
  PY_CLIP="$CLIP/venv_clip/bin/python"
  VG="${VG_DATA_ROOT:-$HOME/data/visual_genome}"
  echo "  looking in $VG   (override with VG_DATA_ROOT)"

  # The images are the ONE thing that is neither committed, nor downloadable
  # from here, nor derivable. Both .pkl files are computed FROM them, so if the
  # images are absent there is nothing to regenerate and the app cannot run.
  HAVE_IMAGES=0
  if [ -d "$VG/resized_224_x_224" ]; then
    ok "resized_224_x_224/ present"
    HAVE_IMAGES=1
  else
    warn "MISSING: $VG/resized_224_x_224/"
    warn "         untar 224.tar into $VG/ — the embeddings are derived from these,"
    warn "         so without them neither .pkl can be built and the app has"
    warn "         nothing to display."
    MISSING=1
  fi

  if [ "$BUILD_CLS" = 1 ] && [ "$HAVE_IMAGES" = 1 ]; then
    if [ ! -s "$VG/vg_clip_embeddings.pkl" ]; then
      echo "  building vg_clip_embeddings.pkl (CLIP ViT-B/32, GPU-accelerated)"
      ( cd "$CLIP" && VG_DATA_ROOT="$VG" "$PY_CLIP" build_vit_embeddings.py )
    fi
    if [ ! -s "$CLIP/vg_tinyclip_embeddings.pkl" ]; then
      echo "  building vg_tinyclip_embeddings.pkl (TinyCLIP ONNX, CPU pool)"
      ( cd "$CLIP" && VG_DATA_ROOT="$VG" "$PY_CLIP" precompute_tinyclip.py )
    fi
  fi

  need_file "$VG/vg_clip_embeddings.pkl"       "vg_clip_embeddings.pkl (215 MB)"
  need_file "$CLIP/vg_tinyclip_embeddings.pkl" "vg_tinyclip_embeddings.pkl (216 MB)"
  if [ "$MISSING" = 1 ] && [ "$HAVE_IMAGES" = 1 ] && [ "$BUILD_CLS" = 0 ]; then
    warn "you have the images — re-run with --regenerate to build both .pkl here"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
step "summary"
if [ "$MISSING" = 1 ]; then
  warn "some artifacts are missing — see docs/jovyan_deployment.md for what to copy"
else
  ok "everything present"
fi

cat <<EOF

Launch:
EOF
[ "$DO_RAM" = 1 ]  && echo "  cd research/ram_comparison    && ./venv_ramclip/bin/python app.py   # :7863"
[ "$DO_CLIP" = 1 ] && echo "  cd research/vitb32_benchmark  && ./venv_clip/bin/python app.py      # :7862"
cat <<EOF

Both bind 127.0.0.1 by default. On a remote box you want:
  GRADIO_HOST=0.0.0.0 GRADIO_PORT=7863 ./venv_ramclip/bin/python app.py
On JupyterHub, serve under the proxy instead:
  GRADIO_ROOT_PATH=/user/jovyan/proxy/7863 ./venv_ramclip/bin/python app.py
EOF

[ "$MISSING" = 1 ] && exit 1 || exit 0
