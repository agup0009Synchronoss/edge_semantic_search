"""
precompute_tinyclip.py  (Stage 3)

Runs the Android-exact TinyCLIP image encoder over all 108K resized Visual
Genome images and writes vg_tinyclip_embeddings.pkl:

    { "<id>.jpg": np.ndarray(512,) float32 (L2-normalized), ... }

Same key scheme as vg_clip_embeddings.pkl so the two pkls align 1-to-1.

Features:
  - Multiprocessing across CPU cores (each worker = 1 ORT session, 1 intra-op
    thread to avoid oversubscription).
  - Resumable: existing keys in the pkl are skipped.
  - Periodic checkpoint saves so a crash never loses much work.

Usage:
    python precompute_tinyclip.py [--workers N] [--limit N]
"""

import os
import sys
import time
import pickle
import pathlib
import argparse
import numpy as np
from PIL import Image

import config

RESIZED_DIR = config.resized_dir(required=False)
OUT_PKL = config.TINY_PKL
CHECKPOINT_EVERY = 5000

# Per-process encoder (lazy global so each worker builds its own ORT sessions)
_ENC = None


def _init_worker():
    global _ENC
    from tinyclip_encoder import TinyClipEncoder
    _ENC = TinyClipEncoder(intra_op_threads=1)


def _encode_one(path_str: str):
    global _ENC
    try:
        img = Image.open(path_str)
        emb = _ENC.encode_image(img)
        return (pathlib.Path(path_str).name, emb.astype(np.float32))
    except Exception as e:  # noqa: BLE001
        return (pathlib.Path(path_str).name, None, str(e))


def _save(embeddings: dict):
    tmp = OUT_PKL.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(embeddings, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(OUT_PKL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="process only first N (debug)")
    args = ap.parse_args()

    if not RESIZED_DIR.exists():
        sys.exit(f"ERROR: image dir not found: {RESIZED_DIR}")

    all_files = sorted(RESIZED_DIR.glob("*.jpg"))
    if args.limit:
        all_files = all_files[: args.limit]
    print(f"Found {len(all_files):,} images.")

    # Resume: load existing pkl and skip done keys
    embeddings: dict = {}
    if OUT_PKL.exists():
        with open(OUT_PKL, "rb") as fh:
            embeddings = pickle.load(fh)
        print(f"Resuming: {len(embeddings):,} embeddings already present.")

    todo = [str(f) for f in all_files if f.name not in embeddings]
    print(f"To encode: {len(todo):,}  |  workers: {args.workers}")
    if not todo:
        print("Nothing to do.")
        return

    from multiprocessing import Pool
    from tqdm import tqdm

    errors = []
    t0 = time.time()
    done_since_save = 0

    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for result in tqdm(
            pool.imap_unordered(_encode_one, todo, chunksize=16),
            total=len(todo), desc="TinyCLIP encode", unit="img",
        ):
            if len(result) == 3:           # error tuple
                errors.append((result[0], result[2]))
                continue
            name, emb = result
            embeddings[name] = emb
            done_since_save += 1
            if done_since_save >= CHECKPOINT_EVERY:
                _save(embeddings)
                done_since_save = 0

    _save(embeddings)
    dt = time.time() - t0
    rate = len(todo) / dt if dt > 0 else 0
    print(f"\nDone. Encoded {len(todo):,} in {dt/60:.1f} min ({rate:.1f} img/s).")
    print(f"Total embeddings in pkl: {len(embeddings):,}")
    if errors:
        print(f"[!] {len(errors)} errors (first 10):")
        for name, err in errors[:10]:
            print(f"    {name}: {err}")


if __name__ == "__main__":
    main()
