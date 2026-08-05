"""
paths.py

Repo-root and external-corpus path resolution, shared by every research
workstream.

Two separate jobs:

1. **Locating this repo's own files.** Everything is derived from `REPO_ROOT`,
   which is computed from this file's location, so scripts work regardless of
   the caller's working directory.

2. **Locating corpora that live outside this repo.** The Visual Genome images
   and the CLIP ViT-B/32 embedding pickle used by the vitb32 benchmark are not
   distributable here (~15 GB, and VG carries its own terms). Six scripts used
   to hardcode `C:\\Users\\agup0009\\code\\edge_object_detection\\...`, which
   meant nobody but one machine could run them. They now call `vg_data_root()`,
   which reads the `VG_DATA_ROOT` environment variable and fails with an
   actionable message rather than a confusing FileNotFoundError.
"""

from __future__ import annotations

import os
import pathlib
import sys

# ── This repo ─────────────────────────────────────────────────────────────────
COMMON_DIR = pathlib.Path(__file__).resolve().parent
RESEARCH_DIR = COMMON_DIR.parent
REPO_ROOT = RESEARCH_DIR.parent

ASSETS_DIR = COMMON_DIR / "assets"          # the canonical ONNX encoders
ANDROID_ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets"


def add_common_to_sys_path() -> None:
    """Make `tinyclip_encoder`, `templates`, `ssl_bypass` importable.

    Every workstream's config.py calls this. Idempotent.
    """
    p = str(COMMON_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


# ── External corpora (not in this repo) ───────────────────────────────────────
VG_ENV_VAR = "VG_DATA_ROOT"

# Where it happened to live on the machine this work was done on. Kept only as
# a fallback so the original setup keeps working without extra configuration.
_VG_DEFAULT = pathlib.Path.home() / "code" / "edge_object_detection" / "data" / "visual_genome"

_VG_HELP = f"""
Visual Genome data not found.

The vitb32 benchmark needs a local Visual Genome corpus that is not part of
this repo (~15 GB). Point {VG_ENV_VAR} at a directory containing:

    resized_224_x_224/          108,079 pre-resized .jpg images
    vg_clip_embeddings.pkl      CLIP ViT-B/32 embeddings, keyed "<id>.jpg"

    PowerShell:  $env:{VG_ENV_VAR} = "D:\\data\\visual_genome"
    bash:        export {VG_ENV_VAR}=/data/visual_genome

See docs/reproducibility.md for how to build both from the Visual Genome
release. Without them, the calibration and RAM-comparison workstreams still
run — only the vitb32 benchmark depends on this corpus.
""".strip()


def vg_data_root(required: bool = True) -> pathlib.Path:
    """Resolve the Visual Genome corpus root.

    Order: $VG_DATA_ROOT, then the original machine's default location.
    Raises FileNotFoundError with setup instructions when `required` and absent.
    """
    env = os.environ.get(VG_ENV_VAR)
    candidate = pathlib.Path(env).expanduser() if env else _VG_DEFAULT
    if candidate.is_dir():
        return candidate
    if required:
        raise FileNotFoundError(
            f"{_VG_HELP}\n\nTried: {candidate}"
            + (f"  (from ${VG_ENV_VAR})" if env else "  (default; $%s is unset)" % VG_ENV_VAR)
        )
    return candidate


def vg_images_dir(required: bool = True) -> pathlib.Path:
    return vg_data_root(required) / "resized_224_x_224"


def vg_clip_pkl(required: bool = True) -> pathlib.Path:
    return vg_data_root(required) / "vg_clip_embeddings.pkl"
