"""
02_cosim_verification.py

Re-generates CLIP ViT-B/32 embeddings for 10 randomly sampled images from
resized_224_x_224 using sentence-transformers, then computes cosine similarity
against the stored vectors in vg_clip_embeddings.pkl.

Expected result: cosine similarity ≈ 1.00 ± 0.02 for every image, confirming
that the pkl was produced by the same CLIP ViT-B/32 model.

Requires venv_clip (run setup_venv.ps1 first).
"""

# ── SSL bypass — ORDER IS LOAD-BEARING ────────────────────────────────────────
# config puts research/common on sys.path and pins HF_HOME; ssl_bypass applies
# the corporate-TLS workaround. Both must land before sentence-transformers,
# which reaches the Hub at import time.
import config       # noqa: F401  (sys.path wiring + HF_HOME)
import ssl_bypass   # noqa: F401  (MUST precede sentence-transformers)

import pickle
import pathlib
import random
import sys
import numpy as np
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────

RESIZED_DIR = config.resized_dir()
PKL_PATH = config.vit_pkl()

N_SAMPLES       = 10
COSIM_THRESHOLD = 0.98   # ≥ this is a PASS (plan specifies 1.00 ± 0.02)
RANDOM_SEED     = 42

# ── Sanity checks ──────────────────────────────────────────────────────────────
for p, label in [(RESIZED_DIR, "RESIZED_DIR"), (PKL_PATH, "PKL_PATH")]:
    if not p.exists():
        print(f"[ERROR] {label} not found: {p}")
        sys.exit(1)

# ── Load embeddings pkl ────────────────────────────────────────────────────────
print(f"Loading pkl ({PKL_PATH.stat().st_size / 1_048_576:.1f} MB) …", end=" ", flush=True)
with open(PKL_PATH, "rb") as fh:
    embeddings = pickle.load(fh)
print("done.")

if not isinstance(embeddings, dict):
    print(f"[ERROR] pkl is not a dict (got {type(embeddings).__name__}). Aborting.")
    sys.exit(1)

# Normalise key lookup: support int keys, str keys ("1023"), and fname keys ("1023.jpg")
_first_key = next(iter(embeddings))
if isinstance(_first_key, int):
    def _lookup(stem: str):
        return embeddings.get(int(stem))
elif isinstance(_first_key, str) and _first_key.endswith(".jpg"):
    def _lookup(stem: str):
        return embeddings.get(stem + ".jpg")
else:
    def _lookup(stem: str):
        return embeddings.get(stem)

print(f"pkl key type : {type(_first_key).__name__}  |  total keys : {len(embeddings):,}")
sample_val = embeddings[_first_key]
print(f"embedding    : shape={sample_val.shape}  dtype={sample_val.dtype}\n")

# ── Sample images ──────────────────────────────────────────────────────────────
all_images = list(RESIZED_DIR.glob("*.jpg"))
random.seed(RANDOM_SEED)
sampled = random.sample(all_images, min(N_SAMPLES, len(all_images)))
print(f"Sampled {len(sampled)} images (seed={RANDOM_SEED}):")
for f in sampled:
    print(f"  {f.name}")

# ── Load sentence-transformers CLIP model ─────────────────────────────────────
print("\nLoading SentenceTransformer('clip-ViT-B-32') …")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR] sentence-transformers not installed. Run setup_venv.ps1 first.")
    sys.exit(1)

model = SentenceTransformer('clip-ViT-B-32')
print(f"Model loaded on device: {model.device}\n")

# ── Generate fresh embeddings & compare ───────────────────────────────────────
print(f"{'Image':<20} {'StoredNorm':>10} {'FreshNorm':>10} {'CosSim':>10}  Status")
print("-" * 60)

results = []
skipped = []

for img_path in sampled:
    stem = img_path.stem

    # Look up stored vector
    stored_vec = _lookup(stem)
    if stored_vec is None:
        print(f"{img_path.name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}  [SKIP – not in pkl]")
        skipped.append(img_path.name)
        continue

    stored_vec = np.array(stored_vec, dtype=np.float32).flatten()

    # Generate fresh embedding
    pil_img   = Image.open(img_path).convert("RGB")
    fresh_vec = model.encode(pil_img, convert_to_numpy=True).flatten().astype(np.float32)

    # Cosine similarity (both vectors may or may not be L2-normalised in pkl)
    def _cosim(a, b):
        a = a / (np.linalg.norm(a) + 1e-10)
        b = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a, b))

    cos_sim = _cosim(stored_vec, fresh_vec)
    status  = "PASS" if cos_sim >= COSIM_THRESHOLD else "FAIL"

    print(f"{img_path.name:<20} {np.linalg.norm(stored_vec):>10.4f} {np.linalg.norm(fresh_vec):>10.4f} {cos_sim:>10.6f}  [{status}]")
    results.append((img_path.name, cos_sim, status))

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n-- SUMMARY ---------------------------------------------------------------")
if not results:
    print("[ERROR] No images could be compared (all skipped or pkl lookup failed).")
    sys.exit(1)

cos_values  = [r[1] for r in results]
mean_cosim  = np.mean(cos_values)
min_cosim   = np.min(cos_values)
max_cosim   = np.max(cos_values)
n_pass      = sum(1 for r in results if r[2] == "PASS")
n_fail      = len(results) - n_pass

print(f"  Images compared   : {len(results)}")
print(f"  Skipped (no pkl)  : {len(skipped)}")
print(f"  Mean cos_sim      : {mean_cosim:.6f}")
print(f"  Min  cos_sim      : {min_cosim:.6f}")
print(f"  Max  cos_sim      : {max_cosim:.6f}")
print(f"  PASS (>={COSIM_THRESHOLD}) : {n_pass}  /  FAIL: {n_fail}")

print("\n-- VERDICT ---------------------------------------------------------------")
if n_fail == 0 and len(skipped) == 0:
    print(f"PASS  All {len(results)} embeddings match CLIP ViT-B/32 (mean cosine = {mean_cosim:.4f}).")
    print("      The pkl was produced by the same sentence-transformers clip-ViT-B-32 model.")
elif n_fail == 0:
    print(f"PARTIAL  {len(results)} matched; {len(skipped)} image(s) not found in pkl.")
else:
    print(f"FAIL  {n_fail}/{len(results)} image(s) have cosine similarity below {COSIM_THRESHOLD}.")
    print("      The pkl may have been generated by a different model or preprocessing pipeline.")
    if min_cosim > 0.90:
        print(f"      Note: min cosine = {min_cosim:.4f} — close but below threshold. Check normalisation.")
