"""
verify_tinyclip_pkl.py  (Stage 3 gate)

Confirms vg_tinyclip_embeddings.pkl is complete and aligned with the ViT-B/32
pkl:

  - exactly 108,079 keys
  - key set identical to vg_clip_embeddings.pkl
  - every embedding is shape (512,), float32, finite, and unit-norm (L2 ~ 1.0)
"""

import sys
import pickle
import pathlib
import numpy as np

import config

TINY_PKL = config.TINY_PKL
VIT_PKL  = config.vit_pkl()
EXPECTED = 108_079

for p in (TINY_PKL, VIT_PKL):
    if not p.exists():
        sys.exit(f"ERROR: not found: {p}")

print(f"Loading {TINY_PKL.name} ...", end=" ", flush=True)
with open(TINY_PKL, "rb") as fh:
    tiny = pickle.load(fh)
print(f"{len(tiny):,} keys")

print(f"Loading {VIT_PKL.name} ...", end=" ", flush=True)
with open(VIT_PKL, "rb") as fh:
    vit = pickle.load(fh)
print(f"{len(vit):,} keys")

tiny_keys = set(tiny.keys())
vit_keys  = set(vit.keys())

count_ok = len(tiny) == EXPECTED
keys_ok  = tiny_keys == vit_keys
missing  = vit_keys - tiny_keys   # images without a TinyCLIP embedding
extra    = tiny_keys - vit_keys   # TinyCLIP keys not in ViT set

print("\n-- Key alignment --------------------------------------------------------")
print(f"  TinyCLIP keys           : {len(tiny_keys):,}")
print(f"  ViT-B/32 keys           : {len(vit_keys):,}")
print(f"  Expected                : {EXPECTED:,}")
print(f"  Missing from TinyCLIP   : {len(missing):,}")
print(f"  Extra in TinyCLIP       : {len(extra):,}")
if missing:
    print(f"    e.g. {sorted(missing)[:10]}")
if extra:
    print(f"    e.g. {sorted(extra)[:10]}")

# ── Embedding quality: shape, dtype, finiteness, unit-norm ────────────────────
print("\n-- Embedding quality ----------------------------------------------------")
bad_shape = bad_dtype = bad_finite = bad_norm = 0
norms = []
for k, v in tiny.items():
    arr = np.asarray(v)
    if arr.shape != (512,):
        bad_shape += 1
        continue
    if arr.dtype != np.float32:
        bad_dtype += 1
    if not np.isfinite(arr).all():
        bad_finite += 1
        continue
    n = float(np.linalg.norm(arr))
    norms.append(n)
    if abs(n - 1.0) > 1e-2:
        bad_norm += 1

norms = np.array(norms) if norms else np.array([0.0])
print(f"  bad shape (!= (512,))   : {bad_shape:,}")
print(f"  non-float32             : {bad_dtype:,}")
print(f"  non-finite              : {bad_finite:,}")
print(f"  norm not ~1.0 (>1e-2)   : {bad_norm:,}")
print(f"  norm min/mean/max       : {norms.min():.4f} / {norms.mean():.4f} / {norms.max():.4f}")

quality_ok = bad_shape == 0 and bad_finite == 0 and bad_norm == 0

print("\n-- STAGE 3 VERDICT ------------------------------------------------------")
if count_ok and keys_ok and quality_ok:
    print(f"PASS  {len(tiny):,} TinyCLIP embeddings, keys identical to ViT-B/32, all unit-norm.")
else:
    reasons = []
    if not count_ok:  reasons.append(f"count {len(tiny):,} != {EXPECTED:,}")
    if not keys_ok:   reasons.append("key set mismatch")
    if not quality_ok: reasons.append("embedding quality issues")
    sys.exit("FAIL  " + "; ".join(reasons))
