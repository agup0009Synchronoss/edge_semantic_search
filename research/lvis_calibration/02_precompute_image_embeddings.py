"""
02_precompute_image_embeddings.py  (Step 2)

Run the Android-exact TinyCLIP *fp32 vision* encoder over every LVIS-train image
and store the embeddings as a matrix (not a dict) so calibration can do one big
matmul.

Mirrors research/vitb32_benchmark/precompute_tinyclip.py: multiprocessing pool with
one ORT session per worker (1 intra-op thread), resumable, periodic checkpoint.

Outputs (see config.py):
  data/image_embeds/img_matrix.npy  (N,512) f32, L2-normalized rows
  data/image_embeds/img_ids.npy     (N,)  int64 coco image_id  (row -> id)

Usage:
  python 02_precompute_image_embeddings.py [--workers N] [--limit N] [--all]
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import pathlib
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

import config

CHECKPOINT_EVERY = 5000

_ENC = None


def _init_worker():
    global _ENC
    from tinyclip_encoder import TinyClipEncoder
    _ENC = TinyClipEncoder(intra_op_threads=1)


def _encode_one(path_str: str):
    global _ENC
    name = pathlib.Path(path_str).name
    try:
        img = Image.open(path_str)
        emb = _ENC.encode_image(img).astype(np.float32)
        return name, emb
    except Exception as e:  # noqa: BLE001
        return name, None, str(e)


def _id_of(name: str) -> int:
    return int(pathlib.Path(name).stem)


def _save(embeds: dict) -> None:
    """Atomically write matrix + ids sorted by image_id."""
    if not embeds:
        return
    ids = np.array(sorted(embeds.keys()), dtype=np.int64)
    mat = np.stack([embeds[int(i)] for i in ids]).astype(np.float32)
    # NOTE: np.save appends ".npy" unless the path already ends in ".npy", so
    # the temp names must end in ".npy" or the replace() target won't exist.
    tmp_ids = config.IMG_IDS.with_name(config.IMG_IDS.stem + ".tmp.npy")
    tmp_mat = config.IMG_MATRIX.with_name(config.IMG_MATRIX.stem + ".tmp.npy")
    np.save(tmp_ids, ids)
    np.save(tmp_mat, mat)
    tmp_ids.replace(config.IMG_IDS)
    tmp_mat.replace(config.IMG_MATRIX)


def _load_existing() -> dict:
    if config.IMG_MATRIX.exists() and config.IMG_IDS.exists():
        ids = np.load(config.IMG_IDS)
        mat = np.load(config.IMG_MATRIX)
        return {int(i): mat[r] for r, i in enumerate(ids)}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="only first N images (dry run)")
    ap.add_argument("--all", action="store_true",
                    help="encode every jpg in the folder (skip LVIS-id filter)")
    args = ap.parse_args()

    config.ensure_dirs()
    if not config.IMAGES_DIR.exists():
        sys.exit(f"ERROR: image dir not found: {config.IMAGES_DIR}  (run 00 first)")

    all_files = sorted(config.IMAGES_DIR.glob("*.jpg"))
    if not args.all:
        import lvis_meta
        wanted = lvis_meta.load().image_ids
        all_files = [f for f in all_files if _id_of(f.name) in wanted]
    if args.limit:
        all_files = all_files[: args.limit]
    print(f"[img] {len(all_files):,} images to consider.")

    embeds = _load_existing()
    if embeds:
        print(f"[img] resuming: {len(embeds):,} embeddings already present.")

    todo = [str(f) for f in all_files if _id_of(f.name) not in embeds]
    print(f"[img] to encode: {len(todo):,} | workers {args.workers}")
    if not todo:
        _save(embeds)
        print("[img] nothing to do.")
        return

    errors = []
    t0 = time.time()
    since = 0
    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for result in tqdm(
            pool.imap_unordered(_encode_one, todo, chunksize=16),
            total=len(todo), desc="vision encode", unit="img",
        ):
            if len(result) == 3:
                errors.append((result[0], result[2]))
                continue
            name, emb = result
            embeds[_id_of(name)] = emb
            since += 1
            if since >= CHECKPOINT_EVERY:
                _save(embeds)
                since = 0

    _save(embeds)
    dt = time.time() - t0
    print(f"[img] done. {len(todo):,} encoded in {dt/60:.1f} min "
          f"({len(todo)/max(dt,1e-9):.1f} img/s). Total: {len(embeds):,}")
    if errors:
        print(f"[img][warn] {len(errors)} errors (first 5): {errors[:5]}")


if __name__ == "__main__":
    main()
