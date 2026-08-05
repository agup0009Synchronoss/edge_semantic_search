"""
config.py

Single source of truth for the RAM++ vs TinyCLIP comparison app: paths,
constants, and the sys.path wiring that lets us reuse the already-built
TinyCLIP encoder and LVIS calibration artifacts.

Reuse map (nothing here is re-implemented):
  - tinyClip_vs_ClipVit32/tinyclip_encoder.py  -> TinyClipEncoder (Android-exact)
  - tinyClipCaliberation/templates.py          -> the 88 template strings
  - tinyClipCaliberation/data/calibration/     -> balanced per-tag thresholds
"""

from __future__ import annotations

import pathlib
import sys

# ── Repo layout ───────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent          # gradio_tinyClip_vs_RAMProd/
CALIB_ROOT = HERE.parent                                 # tinyClipCaliberation/
REPO_ROOT = CALIB_ROOT.parent                            # edge_semantic_search/
ENCODER_DIR = REPO_ROOT / "tinyClip_vs_ClipVit32"        # reused ONNX encoder + assets
ASSETS_DIR = ENCODER_DIR / "assets"

# Make the reused modules importable:
#   from tinyclip_encoder import TinyClipEncoder   (encoder + ONNX assets)
#   import templates                               (the 88 template strings)
for _p in (str(ENCODER_DIR), str(CALIB_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Data tree (gitignored — multi-GB) ─────────────────────────────────────────
DATA = HERE / "data"
RAM_DIR = DATA / "ram"                  # tag list, thresholds, checkpoint
TEXT_DIR = DATA / "text"                # TinyCLIP classifiers for the 4585
LLM_DESC_DIR = DATA / "llm_desc"        # ChatGPT description JSON drops
CACHE_DIR = DATA / "cache"
VENDOR_DIR = HERE / "vendor"            # vendored recognize-anything source
TAG_CHUNK_DIR = HERE / "ram_tag_chunks"  # tag files to upload to ChatGPT

# ── RAM++ artifacts ───────────────────────────────────────────────────────────
RAM_TAG_LIST = RAM_DIR / "ram_tag_list.txt"                # 4585 tags
RAM_TAG_THRESHOLDS = RAM_DIR / "ram_tag_list_threshold.txt"  # 4585 per-tag cutoffs
RAM_CHECKPOINT = RAM_DIR / "ram_plus_swin_large_14m.pth"   # ~3 GB

RAM_REPO_URL = "https://github.com/xinyu1205/recognize-anything.git"
RAM_RAW_BASE = "https://raw.githubusercontent.com/xinyu1205/recognize-anything/main/"
RAM_CKPT_URL = (
    "https://huggingface.co/xinyu1205/recognize-anything-plus-model/"
    "resolve/main/ram_plus_swin_large_14m.pth"
)

# RAM++ input size. The OSS checkpoint was TRAINED at 384 and
# ram_tag_list_threshold.txt was tuned at 384; running at 224 is ~3x faster but
# shifts the sigmoid scores, so those per-tag cutoffs become approximate.
# Default is 224 per project decision; the app exposes 384 as a toggle.
RAM_IMAGE_SIZE_DEFAULT = 224
RAM_IMAGE_SIZES = (224, 384)

# ── TinyCLIP artifacts ────────────────────────────────────────────────────────
EMBED_DIM = 512
N_TAGS = 4585

# Two independent classifier sets, built from disjoint text sources so they can
# be A/B'd in the UI (project decision):
#   templates -> 88 strings/tag  (80 OpenAI + 5 LVIS + 3 question templates)
#   llm       -> 10 strings/tag  (ChatGPT visual descriptions)
CLASSIFIERS_TEMPLATES = TEXT_DIR / "classifiers_templates.npy"   # (4585, 512) f32
CLASSIFIERS_LLM = TEXT_DIR / "classifiers_llm.npy"               # (4585, 512) f32
TAG_ORDER_JSON = TEXT_DIR / "tag_order_4585.json"                # row -> tag metadata
STRINGS_TEMPLATES_JSONL = TEXT_DIR / "strings_templates.jsonl"
STRINGS_LLM_JSONL = TEXT_DIR / "strings_llm.jsonl"

CLASSIFIER_SETS = {
    "templates": CLASSIFIERS_TEMPLATES,
    "llm": CLASSIFIERS_LLM,
}

# ── LVIS calibration reuse ────────────────────────────────────────────────────
# Project decision: balanced thresholds are authoritative. They were calibrated
# on balanced pos/neg subsets, so they are not crushed by the 100k-image
# negative base rate the way the naive full-split thresholds are.
LVIS_BALANCED_THRESHOLDS = CALIB_ROOT / "data" / "calibration" / "balanced_thresholds.npy"
LVIS_NAIVE_THRESHOLDS = CALIB_ROOT / "data" / "calibration" / "thresholds.npy"
LVIS_TAG_ORDER = CALIB_ROOT / "data" / "text" / "tag_order.json"

# Mapping artifacts (LVIS 1203 -> RAM 4585)
TAG_MAPPING_CSV = TEXT_DIR / "lvis_to_ram_mapping.csv"
THRESHOLDS_4585 = TEXT_DIR / "thresholds_4585.npy"   # NaN where uncalibrated

# Sentinel for "no calibrated threshold — use the UI knob"
UNCALIBRATED = float("nan")

# ── UI defaults ───────────────────────────────────────────────────────────────
# Global cosine cutoff applied to TinyCLIP tags that have no LVIS calibration.
# Calibrated tags observed range 0.20-0.40 (balanced), so 0.30 is a sane middle.
DEFAULT_UNCALIBRATED_THRESHOLD = 0.30
UNCALIBRATED_THRESHOLD_RANGE = (0.10, 0.60)
DEFAULT_TOP_K = 25          # matches prod RAM's max_tags
GRADIO_PORT_DEFAULT = 7863  # 7862 is the existing tinyClip_vs_ClipVit32 app


def ensure_dirs() -> None:
    """Create all output directories (safe to call repeatedly)."""
    for d in (RAM_DIR, TEXT_DIR, LLM_DESC_DIR, CACHE_DIR, VENDOR_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_ram_tags() -> list[str]:
    """The 4585 RAM++ tags, in checkpoint row order."""
    return [l.strip() for l in RAM_TAG_LIST.read_text(encoding="utf8").splitlines() if l.strip()]


def load_ram_thresholds() -> list[float]:
    """The 4585 RAM++ per-tag sigmoid cutoffs, aligned with load_ram_tags()."""
    return [
        float(l.strip())
        for l in RAM_TAG_THRESHOLDS.read_text(encoding="utf8").splitlines()
        if l.strip()
    ]
