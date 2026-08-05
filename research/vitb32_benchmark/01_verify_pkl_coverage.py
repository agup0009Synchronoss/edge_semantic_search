"""
01_verify_pkl_coverage.py

Verifies that every image in resized_224_x_224 has a matching embedding entry
in vg_clip_embeddings.pkl, and reports any gaps in either direction.

No GPU / model required — only stdlib + pickle + pathlib.
"""

import pickle
import pathlib
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
import config

RESIZED_DIR = config.resized_dir()
PKL_PATH = config.vit_pkl()

# ── Sanity checks ──────────────────────────────────────────────────────────────
for p, label in [(RESIZED_DIR, "RESIZED_DIR"), (PKL_PATH, "PKL_PATH")]:
    if not p.exists():
        print(f"[ERROR] {label} not found: {p}")
        sys.exit(1)

# ── Step 1: collect image IDs from filesystem ─────────────────────────────────
print("Scanning image directory …", end=" ", flush=True)
image_files = list(RESIZED_DIR.glob("*.jpg"))
print(f"{len(image_files):,} .jpg files found.")

# Build a set of integer IDs (filenames are pure numeric, e.g. "1023.jpg")
image_ids_int: set[int]  = set()
image_ids_str: set[str]  = set()   # "1023"
image_ids_fname: set[str] = set()  # "1023.jpg"

for f in image_files:
    stem = f.stem                   # "1023"
    image_ids_str.add(stem)
    image_ids_fname.add(f.name)     # "1023.jpg"
    try:
        image_ids_int.add(int(stem))
    except ValueError:
        pass  # non-numeric filenames are noted but not fatal

print(f"  Non-numeric filenames : {len(image_files) - len(image_ids_int):,}")

# ── Step 2: load the pkl ───────────────────────────────────────────────────────
print(f"\nLoading pkl ({PKL_PATH.stat().st_size / 1_048_576:.1f} MB) …", end=" ", flush=True)
with open(PKL_PATH, "rb") as fh:
    embeddings = pickle.load(fh)
print("done.")

# ── Step 3: inspect pkl structure ─────────────────────────────────────────────
print(f"\npkl type : {type(embeddings).__name__}")

if isinstance(embeddings, dict):
    all_keys = list(embeddings.keys())
    n_keys   = len(all_keys)
    sample   = all_keys[:5]
    print(f"  keys    : {n_keys:,}")
    print(f"  sample  : {sample}")
    print(f"  key type: {type(sample[0]).__name__}")

    # Detect key format and build a normalised set of string IDs
    first_key = sample[0]
    if isinstance(first_key, int):
        pkl_ids_int  = set(embeddings.keys())
        pkl_ids_str  = {str(k) for k in pkl_ids_int}
    elif isinstance(first_key, str) and first_key.endswith(".jpg"):
        pkl_ids_fname = set(embeddings.keys())
        pkl_ids_str   = {k[:-4] for k in pkl_ids_fname}   # strip ".jpg"
        pkl_ids_int   = set()
        for s in pkl_ids_str:
            try:
                pkl_ids_int.add(int(s))
            except ValueError:
                pass
    elif isinstance(first_key, str):
        pkl_ids_str  = set(embeddings.keys())
        pkl_ids_int  = set()
        for s in pkl_ids_str:
            try:
                pkl_ids_int.add(int(s))
            except ValueError:
                pass
    else:
        print(f"[WARN] Unexpected key type: {type(first_key).__name__}. Manual inspection needed.")
        pkl_ids_str = set()
        pkl_ids_int = set()

    # Probe one embedding shape
    sample_val = embeddings[all_keys[0]]
    try:
        shape = sample_val.shape
        dtype = sample_val.dtype
    except AttributeError:
        shape = f"len={len(sample_val)}"
        dtype = type(sample_val).__name__
    print(f"  embedding shape : {shape}  dtype: {dtype}")

else:
    print("[ERROR] pkl is not a dict — manual inspection required.")
    print(f"  repr (first 200 chars): {repr(embeddings)[:200]}")
    sys.exit(1)

# ── Step 4: set-difference analysis ───────────────────────────────────────────
# Normalise both sides to integer IDs where possible; fall back to string
use_int = len(image_ids_int) > 0 and len(pkl_ids_int) > 0

if use_int:
    images_set = image_ids_int
    pkl_set    = pkl_ids_int
    label      = "integer ID"
else:
    images_set = image_ids_str
    pkl_set    = pkl_ids_str
    label      = "string ID"

print(f"\n-- Coverage analysis (matching by {label}) -------------------------")
print(f"  Images in folder      : {len(images_set):>10,}")
print(f"  Keys in pkl           : {len(pkl_set):>10,}")

missing_embed  = images_set - pkl_set   # images with NO embedding
extra_in_pkl   = pkl_set    - images_set  # pkl keys with NO matching image file

print(f"\n  Images missing from pkl : {len(missing_embed):,}")
print(f"  Pkl keys with no image  : {len(extra_in_pkl):,}")

# ── Step 5: detailed reports ───────────────────────────────────────────────────
MAX_PRINT = 30

if missing_embed:
    sorted_missing = sorted(missing_embed)
    print(f"\n[!] IMAGES MISSING EMBEDDINGS (showing first {min(MAX_PRINT, len(missing_embed))}):")
    for k in sorted_missing[:MAX_PRINT]:
        print(f"      {k}")
    if len(sorted_missing) > MAX_PRINT:
        print(f"      … and {len(sorted_missing) - MAX_PRINT} more")
else:
    print("\n[OK] Every image has a corresponding embedding.")

if extra_in_pkl:
    sorted_extra = sorted(extra_in_pkl)
    print(f"\n[!] PKL KEYS WITH NO IMAGE FILE (showing first {min(MAX_PRINT, len(extra_in_pkl))}):")
    for k in sorted_extra[:MAX_PRINT]:
        print(f"      {k}")
    if len(sorted_extra) > MAX_PRINT:
        print(f"      … and {len(sorted_extra) - MAX_PRINT} more")
else:
    print("[OK] No orphan keys in pkl.")

# ── Step 6: final verdict ──────────────────────────────────────────────────────
print("\n-- VERDICT --------------------------------------------------------------")
if not missing_embed and not extra_in_pkl:
    print("PASS  Perfect 1-to-1 coverage: all 108K images have embeddings.")
elif not missing_embed:
    print(f"PARTIAL  All images have embeddings, but {len(extra_in_pkl):,} orphan pkl keys exist.")
else:
    print(f"FAIL  {len(missing_embed):,} image(s) are missing embeddings.")
