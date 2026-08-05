"""
config.py

Central paths, constants and small helpers for the TinyCLIP LVIS calibration
pipeline. Everything downstream (00_..04_) imports from here so the pipeline has
a single source of truth for paths, the Fbeta weighting, the cosine-threshold
grid and the confidence buckets.

Model reuse: we reuse the Android-exact encoders from research/common
(tinyclip_encoder.py + assets/*.onnx — the same bytes the APK ships, held
identical by ASSETS.sha256). This module puts that folder on sys.path so
`from tinyclip_encoder import TinyClipEncoder` and `import templates` work.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

# ── Repo layout ───────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent               # tinyClipCaliberation/
REPO_ROOT = HERE.parent                                       # edge_semantic_search/
COMMON_DIR = REPO_ROOT / "research" / "common"                # shared encoder + templates

# Make the shared modules importable before anything else needs them.
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import paths  # noqa: E402  (needs COMMON_DIR on sys.path first)

ASSETS_DIR = paths.ASSETS_DIR                                 # *.onnx encoders

# ── Data tree (all gitignored — multi-GB) ─────────────────────────────────────
DATA = HERE / "data"
ANN_DIR = DATA / "annotations"
ANN_JSON = ANN_DIR / "lvis_v1_train.json"
IMAGES_DIR = DATA / "images" / "train2017"
TEXT_DIR = DATA / "text"
IMG_EMB_DIR = DATA / "image_embeds"
CALIB_DIR = DATA / "calibration"

# Text-side artifacts
CLASSIFIERS_COMBINED = TEXT_DIR / "classifiers_combined.npy"       # (T,512) f32
CLASSIFIERS_BY_SOURCE = TEXT_DIR / "classifiers_by_source.npz"     # prompts/descriptions/questions
TAG_ORDER_JSON = TEXT_DIR / "tag_order.json"                       # row -> tag metadata
STRINGS_JSONL = TEXT_DIR / "strings.jsonl"                         # every raw text string
PER_STRING_EMBEDS = TEXT_DIR / "per_string_embeds.npz"            # optional (--save-per-string)

# Image-side artifacts
IMG_MATRIX = IMG_EMB_DIR / "img_matrix.npy"                        # (N,512) f32 L2-normed
IMG_IDS = IMG_EMB_DIR / "img_ids.npy"                             # (N,) int64 coco image_id

# Calibration artifacts
THRESHOLDS_NPY = CALIB_DIR / "thresholds.npy"                      # (T,) effective thresholds
CALIB_TABLE_JSON = CALIB_DIR / "calibration_table.json"
CALIB_TABLE_PARQUET = CALIB_DIR / "calibration_table.parquet"
BUCKET_THRESHOLDS_JSON = CALIB_DIR / "bucket_thresholds.json"

# ── Model constants ───────────────────────────────────────────────────────────
EMBED_DIM = 512

# ── Fbeta objective (precision-leaning) ───────────────────────────────────────
# Fbeta = (1+b^2) * P*R / (b^2*P + R).  b^2 = w_recall / w_precision.
# Target weighting ~80% precision / 20% recall -> b^2 = 0.20/0.80 = 0.25 -> b=0.5.
# (Was 0.816 / 60-40: too mild a lean for tags with compressed score
# distributions, e.g. "person" picked threshold 0.31 at precision=0.50 because
# raising the threshold bought too little precision for the recall it cost.)
FBETA = 0.5                       # set 1.0 to recover the standard F1 score
FBETA_SQ = FBETA * FBETA

# ── Cosine-threshold search grid ──────────────────────────────────────────────
THRESH_MIN = 0.20
THRESH_MAX = 0.65
THRESH_STEP = 0.01
# inclusive of THRESH_MAX (46 points at step 0.01)
T_GRID = np.round(np.arange(THRESH_MIN, THRESH_MAX + THRESH_STEP / 2, THRESH_STEP), 4).astype(np.float32)

# ── Confidence buckets (from LVIS per-tag positive image_count) ───────────────
# 1-10 -> weak (use bucket fallback), 11-100 -> medium, >100 -> high.
BUCKET_WEAK = "weak"
BUCKET_MEDIUM = "medium"
BUCKET_HIGH = "high"


def bucket_for_count(n: int) -> str:
    """Map a tag's positive-image count to a calibration-confidence bucket."""
    if n <= 10:
        return BUCKET_WEAK
    if n <= 100:
        return BUCKET_MEDIUM
    return BUCKET_HIGH


def fbeta_score(precision: np.ndarray, recall: np.ndarray, beta_sq: float = FBETA_SQ) -> np.ndarray:
    """Vectorized Fbeta. Safe against 0/0 (returns 0 where denom==0)."""
    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)
    denom = beta_sq * precision + recall
    num = (1.0 + beta_sq) * precision * recall
    out = np.zeros_like(denom)
    np.divide(num, denom, out=out, where=denom > 0)
    return out


def ensure_dirs() -> None:
    """Create all output directories (safe to call repeatedly)."""
    for d in (ANN_DIR, IMAGES_DIR, TEXT_DIR, IMG_EMB_DIR, CALIB_DIR):
        d.mkdir(parents=True, exist_ok=True)
