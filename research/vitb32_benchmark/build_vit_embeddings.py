"""
build_vit_embeddings.py

Builds `vg_clip_embeddings.pkl` — the CLIP ViT-B/32 lane of the side-by-side app.

Until now this file was an input the repo could not produce: it was created in a
sibling project and copied in, so `02_cosim_verification.py` could only *verify*
it, never rebuild it. That made the CLIP lane the one part of the benchmark a
new machine could not reconstruct. This closes that gap.

Output shape matches `precompute_tinyclip.py` exactly, because the two pkls must
align key-for-key:

    { "<id>.jpg": np.ndarray(512,) float32, L2-normalized }

The model must stay `clip-ViT-B-32`. It is the model that defines this lane —
swapping it puts the stored image vectors in a different space from the query
vectors the app computes at search time, which produces confident nonsense
rather than an obvious failure.

Unlike the TinyCLIP side (onnxruntime, CPU, multiprocessing), this runs through
torch and **uses the GPU automatically** when one is present.

Usage:
    python build_vit_embeddings.py                    # all images
    python build_vit_embeddings.py --limit 500        # quick smoke test
    python build_vit_embeddings.py --batch-size 512   # tune for your VRAM
"""

# ── SSL bypass — ORDER IS LOAD-BEARING ────────────────────────────────────────
import config       # noqa: F401  (sys.path wiring + HF_HOME)
import ssl_bypass   # noqa: F401  (MUST precede sentence-transformers)

import argparse
import pathlib
import pickle
import sys
import time

import numpy as np
from PIL import Image

MODEL_NAME = "clip-ViT-B-32"
CHECKPOINT_EVERY = 5000


def _save(path: pathlib.Path, store: dict) -> None:
    """Atomic write, so a crash mid-checkpoint cannot corrupt the pkl."""
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(store, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: $VG_DATA_ROOT/vg_clip_embeddings.pkl")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    images_dir = config.resized_dir()
    out_path = args.out or config.vit_pkl(required=False)

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model  : {MODEL_NAME}")
    print(f"device : {device}")
    print(f"images : {images_dir}")
    print(f"out    : {out_path}")

    files = sorted(images_dir.glob("*.jpg"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"ERROR: no .jpg found under {images_dir}", file=sys.stderr)
        return 1

    # Resume: keep whatever is already embedded.
    store: dict[str, np.ndarray] = {}
    if out_path.exists():
        with open(out_path, "rb") as fh:
            store = pickle.load(fh)
        print(f"resuming: {len(store):,} already embedded")

    todo = [p for p in files if p.name not in store]
    print(f"total {len(files):,} | to encode {len(todo):,}")
    if not todo:
        print("nothing to do")
        return 0

    model = SentenceTransformer(MODEL_NAME, device=device)

    t0 = time.time()
    done = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        imgs, names = [], []
        for p in batch:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                names.append(p.name)
            except Exception as e:  # a corrupt jpg should not kill a long run
                print(f"  ! skipping {p.name}: {e}")
        if not imgs:
            continue

        vecs = model.encode(imgs, batch_size=len(imgs), convert_to_numpy=True,
                            show_progress_bar=False, normalize_embeddings=True)
        for name, v in zip(names, vecs):
            store[name] = v.astype(np.float32)
        for im in imgs:
            im.close()

        done += len(names)
        if done % CHECKPOINT_EVERY < args.batch_size:
            _save(out_path, store)
            rate = done / max(time.time() - t0, 1e-6)
            eta = (len(todo) - done) / max(rate, 1e-6)
            print(f"  {done:,}/{len(todo):,}  {rate:.0f} img/s  eta {eta/60:.1f} min")

    _save(out_path, store)
    elapsed = time.time() - t0
    print(f"\nwrote {out_path}  ({len(store):,} keys, {elapsed/60:.1f} min)")

    dims = {v.shape for v in store.values()}
    norms = np.array([float(np.linalg.norm(v)) for v in list(store.values())[:1000]])
    print(f"  shapes: {dims}")
    print(f"  norms : min {norms.min():.4f} max {norms.max():.4f} (expect ~1.0)")
    print("\nNext: python 02_cosim_verification.py   # confirms provenance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
