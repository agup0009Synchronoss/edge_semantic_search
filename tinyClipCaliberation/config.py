"""
config.py

Central paths, constants and small helpers for the TinyCLIP LVIS calibration
pipeline. Everything downstream (00_..04_) imports from here so the pipeline has
a single source of truth for paths, the Fbeta weighting, the cosine-threshold
grid and the confidence buckets.

Model reuse: we reuse the Android-exact encoders shipped with the Gradio app
(tinyClip_vs_ClipVit32/tinyclip_encoder.py + assets/*.onnx). This module adds
that folder to sys.path so `from tinyclip_encoder import TinyClipEncoder` works.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

# ── Repo layout ───────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent               # tinyClipCaliberation/
REPO_ROOT = HERE.parent                                       # edge_semantic_search/
ENCODER_DIR = REPO_ROOT / "tinyClip_vs_ClipVit32"            # reused encoder + assets
ASSETS_DIR = ENCODER_DIR / "assets"                          # *.onnx encoders

# Make the reused encoder importable: `from tinyclip_encoder import TinyClipEncoder`
if str(ENCODER_DIR) not in sys.path:
    sys.path.insert(0, str(ENCODER_DIR))

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
# Target weighting ~60% precision / 40% recall  ->  b^2 = 0.40/0.60 = 0.667.
FBETA = 0.816                     # set 1.0 to recover the standard F1 score
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
