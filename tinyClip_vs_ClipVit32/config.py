"""
config.py

Single source of truth for the TinyCLIP vs CLIP ViT-B/32 benchmark: paths,
constants, and the sys.path wiring that makes the shared encoder importable.

Mirrors the same pattern as the sibling workstreams' config.py, so all three
resolve shared code the same way.

External data: this benchmark runs over a local Visual Genome corpus that is
not part of this repo (~15 GB). Six scripts here used to hardcode one machine's
absolute path to it; they now go through `research/common/paths.py`, which
reads $VG_DATA_ROOT and raises an actionable error when it is missing. The
three VG_* variables below stay supported for per-file overrides, because
app.py already used them.
"""

from __future__ import annotations

import os
import pathlib
import sys

# ── Repo layout ───────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent           # tinyClip_vs_ClipVit32/
REPO_ROOT = HERE.parent                                   # edge_semantic_search/
COMMON_DIR = REPO_ROOT / "research" / "common"            # shared encoder + ssl_bypass

# Make the shared modules importable:
#   from tinyclip_encoder import TinyClipEncoder
#   import ssl_bypass          (must precede transformers / sentence-transformers)
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import paths  # noqa: E402  (needs COMMON_DIR on sys.path first)

ASSETS_DIR = paths.ASSETS_DIR                             # the canonical *.onnx encoders

# ── Local artifacts ───────────────────────────────────────────────────────────
# 226 MB, gitignored; produced by precompute_tinyclip.py.
TINY_PKL = pathlib.Path(os.environ.get("VG_TINY_PKL", HERE / "vg_tinyclip_embeddings.pkl"))

# Expected key count, asserted by verify_tinyclip_pkl.py.
EXPECTED_KEYS = 108_079


# ── External Visual Genome corpus ─────────────────────────────────────────────
def resized_dir(required: bool = True) -> pathlib.Path:
    """108,079 pre-resized 224x224 .jpg images.

    $VG_RESIZED_DIR overrides; otherwise derived from $VG_DATA_ROOT.
    """
    env = os.environ.get("VG_RESIZED_DIR")
    if env:
        return pathlib.Path(env).expanduser()
    return paths.vg_images_dir(required)


def vit_pkl(required: bool = True) -> pathlib.Path:
    """CLIP ViT-B/32 embeddings, keyed "<id>.jpg".

    $VG_VIT_PKL overrides; otherwise derived from $VG_DATA_ROOT.
    """
    env = os.environ.get("VG_VIT_PKL")
    if env:
        return pathlib.Path(env).expanduser()
    return paths.vg_clip_pkl(required)
