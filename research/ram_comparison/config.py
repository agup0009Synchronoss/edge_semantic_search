"""
config.py

Single source of truth for the RAM++ vs TinyCLIP comparison app: paths,
constants, and the sys.path wiring that lets us reuse the already-built
TinyCLIP encoder and LVIS calibration artifacts.

Reuse map (nothing here is re-implemented):
  - research/common/tinyclip_encoder.py    -> TinyClipEncoder (Android-exact)
  - research/common/templates.py           -> the 88 template strings
  - research/common/ssl_bypass.py          -> corporate-TLS workaround
  - research/lvis_calibration/results/          -> balanced per-tag thresholds
"""

from __future__ import annotations

import pathlib
import sys

# ── Repo layout ───────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent          # research/ram_comparison/
RESEARCH_DIR = HERE.parent                               # research/
REPO_ROOT = RESEARCH_DIR.parent                          # edge_semantic_search/
COMMON_DIR = RESEARCH_DIR / "common"                     # shared encoder, templates, ssl_bypass
CALIB_ROOT = RESEARCH_DIR / "lvis_calibration"           # sibling project; source of the thresholds

# Make the shared modules importable:
#   from tinyclip_encoder import TinyClipEncoder   (encoder + ONNX assets)
#   import templates                               (the 88 template strings)
#   import ssl_bypass                              (must precede transformers)
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import paths  # noqa: E402  (needs COMMON_DIR on sys.path first)

ASSETS_DIR = paths.ASSETS_DIR

# ── Data tree (gitignored — multi-GB) ─────────────────────────────────────────
DATA = HERE / "data"
RAM_DIR = DATA / "ram"                  # tag list, thresholds, checkpoint
TEXT_DIR = DATA / "text"                # TinyCLIP classifiers for the 4585
LLM_DESC_DIR = DATA / "llm_desc"        # drop zone for NEW ChatGPT batches
CACHE_DIR = DATA / "cache"
VENDOR_DIR = HERE / "vendor"            # vendored recognize-anything source

# ── Committed text assets ─────────────────────────────────────────────────────
# The generated descriptions are committed, not regenerable: re-prompting yields
# different text, which would silently invalidate classifiers_llm.npy and every
# number in the benchmark. This file is the canonical copy — it previously
# existed in three places on disk (here, data/llm_desc/, and the sibling
# calibration project's data/), with nothing keeping them in sync.
ASSETS_LOCAL = HERE / "assets"
CLIP_DESCRIPTIONS = ASSETS_LOCAL / "clip_descriptions_4585.json"   # 10 descriptions x 4585 tags
TAG_CHUNK_DIR = ASSETS_LOCAL / "tag_chunks"   # the batches uploaded to produce it


def description_sources() -> list[pathlib.Path]:
    """Every JSON to read descriptions from: the committed asset first, then any
    new drops in data/llm_desc/. Lets you extend the set without editing the
    committed file, while a fresh clone still works with no drops at all."""
    out = [CLIP_DESCRIPTIONS] if CLIP_DESCRIPTIONS.is_file() else []
    if LLM_DESC_DIR.is_dir():
        out += [p for p in sorted(LLM_DESC_DIR.glob("*.json")) if p.name != CLIP_DESCRIPTIONS.name]
    return out

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
#
# These read from the sibling project's committed results/ rather than its
# gitignored data/, so a fresh clone can run this app without first reproducing
# a multi-hour calibration.
LVIS_RESULTS = CALIB_ROOT / "results"
LVIS_BALANCED_THRESHOLDS = LVIS_RESULTS / "balanced_thresholds.npy"
LVIS_NAIVE_THRESHOLDS = LVIS_RESULTS / "thresholds.npy"
LVIS_TAG_ORDER = LVIS_RESULTS / "tag_order.json"

# Mapping artifacts (LVIS 1203 -> RAM 4585)
TAG_MAPPING_CSV = TEXT_DIR / "lvis_to_ram_mapping.csv"
THRESHOLDS_4585 = TEXT_DIR / "thresholds_4585.npy"   # NaN where uncalibrated

# Sentinel for "no calibrated threshold — use the UI knob"
UNCALIBRATED = float("nan")

# ── Precision-target calibration (04_precision_calibration.py) ────────────────
# Fbeta calibration OPTIMIZES for precision; these sets GUARANTEE it. A tag is
# calibrated to >= its target on a balanced LVIS subset, or it is NaN (NA).
#
# These are small and committed, so they are written straight to results/ rather
# than into the gitignored data/ tree.
#
# IMPORTANT: the arrays are (1203,) indexed by LVIS row, but they are derived
# from the RAM-tag TEMPLATE embeddings (classifiers_templates.npy), not from the
# calibration project's own classifiers_combined.npy. Same shape as
# lvis_calibration/results/balanced_thresholds.npy, different meaning — they are
# NOT interchangeable with it.
#
# And note what "p90" means: >= 90% precision ON A BALANCED 50/50 SUBSET. At a
# realistic ~1% prevalence the same operating point yields far lower precision
# (see the README). These measure intrinsic separability, not deployment.
RESULTS = HERE / "results"
PRECISION_TARGETS = (0.80, 0.85, 0.90)


def precision_set_name(target: float) -> str:
    """0.85 -> 'p85'."""
    return f"p{int(round(target * 100))}"


PRECISION_THRESHOLDS = {
    precision_set_name(t): RESULTS / f"precision_thresholds_{precision_set_name(t)}.npy"
    for t in PRECISION_TARGETS
}
PRECISION_CALIB_CSV = RESULTS / "precision_calibration.csv"

# ── UI defaults ───────────────────────────────────────────────────────────────
# Global cosine cutoff applied to TinyCLIP tags that have no LVIS calibration
# (or, with the "apply to all tags" toggle, to every tag). Calibrated tags
# observed range 0.20-0.40 (balanced); 0.375 sits toward the top of that range
# — project decision, tightened from the original 0.30 midpoint to cut down
# knob-driven false positives on the ~3548 uncalibrated tags.
DEFAULT_UNCALIBRATED_THRESHOLD = 0.375
UNCALIBRATED_THRESHOLD_RANGE = (0.10, 0.60)
DEFAULT_TOP_K = 25          # matches prod RAM's max_tags
GRADIO_PORT_DEFAULT = 7863  # 7862 is the existing research/vitb32_benchmark app


def ensure_dirs() -> None:
    """Create all output directories (safe to call repeatedly)."""
    for d in (RAM_DIR, TEXT_DIR, LLM_DESC_DIR, CACHE_DIR, VENDOR_DIR, RESULTS):
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
