"""
01_build_text_classifiers.py  (Step 1)

Tier-1 text-side classifier construction (prompt ensembling).

For every LVIS tag we generate many text strings grouped by source
(prompts / descriptions / questions), embed each with the Android-exact TinyCLIP
*int8* text encoder, then average the L2-normalized per-string vectors and
re-normalize to get:

  - one combined classifier vector per tag  (mean over ALL strings)  <- primary
  - one per-source classifier vector per tag (for later ablation)

Outputs (see config.py):
  data/text/classifiers_combined.npy   (T,512) f32, L2-normalized rows
  data/text/classifiers_by_source.npz  prompts/descriptions/questions (T,512)
  data/text/tag_order.json             row -> tag metadata (canonical row order)
  data/text/strings.jsonl              every raw string (audit / LLM-later)
  data/text/per_string_embeds.npz      optional (--save-per-string)

Usage:
  python 01_build_text_classifiers.py [--workers N] [--limit N] [--save-per-string]
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import config
import templates
import lvis_meta

SOURCES = ("prompts", "descriptions", "questions")

# Per-worker encoder (lazy global; each worker builds its own ORT sessions)
_ENC = None


def _init_worker():
    global _ENC
    from tinyclip_encoder import TinyClipEncoder
    _ENC = TinyClipEncoder(intra_op_threads=1)


def _encode(arg):
    idx, text = arg
    global _ENC
    return idx, _ENC.encode_text(text).astype(np.float32)


def _l2norm_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    out = np.zeros_like(mat)
    np.divide(mat, norms, out=out, where=norms > 1e-8)
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="only first N tags (dry run)")
    ap.add_argument("--save-per-string", action="store_true", help="also dump per-string embeds")
    args = ap.parse_args()

    config.ensure_dirs()
    meta = lvis_meta.load()
    tag_ids = meta.tag_order[: args.limit] if args.limit else meta.tag_order
    T = len(tag_ids)
    print(f"[text] building strings for {T} tags ...")

    # 1) Build the flat list of strings + per-string (tag_row, source).
    records_tag_row: list[int] = []
    records_src_id: list[int] = []
    texts: list[str] = []
    tag_meta_rows: list[dict] = []

    src_index = {s: i for i, s in enumerate(SOURCES)}
    with open(config.STRINGS_JSONL, "w", encoding="utf-8") as sfh:
        for row, cid in enumerate(tag_ids):
            cat = meta.categories[cid]
            grouped = templates.build_tag_strings(cat)
            n_per_src = {}
            for src in SOURCES:
                strings = grouped[src]
                n_per_src[src] = len(strings)
                for txt in strings:
                    records_tag_row.append(row)
                    records_src_id.append(src_index[src])
                    texts.append(txt)
                    sfh.write(json.dumps(
                        {"tag_row": row, "cat_id": cid, "source": src, "text": txt}
                    ) + "\n")
            pos = meta.positives_count(cid)
            tag_meta_rows.append({
                "row": row,
                "cat_id": cid,
                "name": cat["name"],
                "synonyms": cat["synonyms"],
                "def": cat["def"],
                "frequency": cat["frequency"],
                "pos_count": pos,
                "bucket": config.bucket_for_count(pos),
                "n_prompts": n_per_src["prompts"],
                "n_descriptions": n_per_src["descriptions"],
                "n_questions": n_per_src["questions"],
            })

    M = len(texts)
    print(f"[text] {M:,} strings total ({M / max(T,1):.1f} per tag). Encoding with {args.workers} workers ...")

    # 2) Encode every string (int8 text encoder) in a process pool.
    embeds = np.zeros((M, config.EMBED_DIM), dtype=np.float32)
    t0 = time.time()
    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for idx, vec in tqdm(
            pool.imap_unordered(_encode, enumerate(texts), chunksize=64),
            total=M, desc="encode text", unit="str",
        ):
            embeds[idx] = vec
    dt = time.time() - t0
    print(f"[text] encoded {M:,} strings in {dt/60:.1f} min ({M/max(dt,1e-9):.0f} str/s).")

    # 3) Aggregate: sum per (tag, source); normalize(mean) == normalize(sum).
    tag_row = np.asarray(records_tag_row, dtype=np.int64)
    src_id = np.asarray(records_src_id, dtype=np.int64)

    by_source = {}
    combined_sum = np.zeros((T, config.EMBED_DIM), dtype=np.float64)
    for si, src in enumerate(SOURCES):
        acc = np.zeros((T, config.EMBED_DIM), dtype=np.float64)
        m = src_id == si
        # scatter-add rows for this source into per-tag accumulator
        np.add.at(acc, tag_row[m], embeds[m].astype(np.float64))
        combined_sum += acc
        by_source[src] = _l2norm_rows(acc.astype(np.float32))

    classifiers_combined = _l2norm_rows(combined_sum.astype(np.float32))

    # 4) Save artifacts.
    np.save(config.CLASSIFIERS_COMBINED, classifiers_combined)
    np.savez(config.CLASSIFIERS_BY_SOURCE, **{s: by_source[s] for s in SOURCES})
    with open(config.TAG_ORDER_JSON, "w", encoding="utf-8") as fh:
        json.dump({
            "fbeta": config.FBETA,
            "embed_dim": config.EMBED_DIM,
            "sources": list(SOURCES),
            "tags": tag_meta_rows,
        }, fh, indent=2)
    if args.save_per_string:
        np.savez(
            config.PER_STRING_EMBEDS,
            embeds=embeds, tag_row=tag_row, src_id=src_id,
            sources=np.array(SOURCES),
        )

    # 5) Quick self-check.
    norms = np.linalg.norm(classifiers_combined, axis=1)
    nonzero = norms > 1e-6
    print(f"[text] saved classifiers_combined {classifiers_combined.shape} "
          f"(||row||~1 for {nonzero.sum()}/{T} tags).")
    print(f"[text] wrote {config.TAG_ORDER_JSON.name}, {config.STRINGS_JSONL.name}"
          + (", per_string_embeds.npz" if args.save_per_string else ""))


if __name__ == "__main__":
    main()
