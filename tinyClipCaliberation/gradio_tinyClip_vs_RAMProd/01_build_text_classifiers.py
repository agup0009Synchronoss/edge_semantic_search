"""
01_build_text_classifiers.py

Build TinyCLIP "super embeddings" for the 4585 RAM++ tags.

Two independent classifier sets are produced from disjoint text sources so they
can be A/B'd in the UI:

    --source templates   88 strings/tag  (80 OpenAI + 5 LVIS + 3 question
                                          templates, from tinyClipCaliberation/
                                          templates.py) -> classifiers_templates.npy
    --source llm         10 strings/tag  (ChatGPT visual descriptions dropped
                                          into data/llm_desc/) -> classifiers_llm.npy

Aggregation is the canonical OpenAI zero-shot recipe, identical to
tinyClipCaliberation/01_build_text_classifiers.py: every string is encoded and
L2-normalized by encode_text(), summed per tag in float64, then re-normalized.
normalize(mean) == normalize(sum), so the mean is never formed explicitly.

Usage:
    python 01_build_text_classifiers.py --source templates
    python 01_build_text_classifiers.py --source llm
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import config
import templates as tpl  # from tinyClipCaliberation/, via config's sys.path wiring


# ── Per-worker encoder (each worker builds its own ORT sessions) ──────────────
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


# ── Text-string sources ───────────────────────────────────────────────────────
def build_template_strings(tags: list[str]) -> tuple[list[str], list[int]]:
    """88 strings per tag via the shared templates.build_tag_strings().

    RAM tags are bare strings with no synonyms and no WordNet gloss, so each
    yields exactly 85 prompts + 0 descriptions + 3 questions = 88.
    """
    texts: list[str] = []
    tag_row: list[int] = []
    for row, tag in enumerate(tags):
        cat = {"name": tag, "synonyms": [], "def": ""}
        groups = tpl.build_tag_strings(cat)
        strings = groups["prompts"] + groups["descriptions"] + groups["questions"]
        texts.extend(strings)
        tag_row.extend([row] * len(strings))
    return texts, tag_row


def build_llm_strings(tags: list[str], allow_partial: bool = False) -> tuple[list[str], list[int]]:
    """Descriptions from data/llm_desc/*.json, rendered through the same
    DESCRIPTION_TEMPLATES used for LVIS glosses so the phrasing matches the
    "a photo of a {tag}, which is ..." framing the ChatGPT prompt asked for.

    Expected JSON shape (per file, as returned by ChatGPT):
        [{"tag": "<exact tag>", "descriptions": ["...", ... 10 total]}, ...]
    """
    by_tag = load_descriptions()
    missing = [t for t in tags if not by_tag.get(t)]
    if missing and not allow_partial:
        raise SystemExit(
            f"[llm] {len(missing)} of {len(tags)} tags have no descriptions "
            f"(e.g. {missing[:5]}).\n"
            f"Run `python ingest_descriptions.py --report` to see which chunks are "
            f"incomplete, or pass --allow-partial to build anyway."
        )

    texts: list[str] = []
    tag_row: list[int] = []
    for row, tag in enumerate(tags):
        for desc in by_tag.get(tag, []):
            d = desc.strip().rstrip(".")
            if not d:
                continue
            for t in tpl.DESCRIPTION_TEMPLATES:
                texts.append(t.format(label=tag, definition=d))
            tag_row.extend([row] * len(tpl.DESCRIPTION_TEMPLATES))
    return texts, tag_row


def load_descriptions() -> dict[str, list[str]]:
    """Merge the committed description asset plus any new drops in
    data/llm_desc/ into {tag: [descriptions]}."""
    by_tag: dict[str, list[str]] = {}
    for path in config.description_sources():
        try:
            payload = json.loads(path.read_text(encoding="utf8"))
        except json.JSONDecodeError as e:
            print(f"  ! {path.name}: invalid JSON ({e}) — skipped")
            continue
        records = payload if isinstance(payload, list) else payload.get("results", [])
        for rec in records:
            tag = rec.get("tag")
            descs = [d for d in rec.get("descriptions", []) if isinstance(d, str) and d.strip()]
            if tag and descs:
                by_tag.setdefault(tag, []).extend(descs)
    return by_tag


SOURCE_BUILDERS = {
    "templates": build_template_strings,
    "llm": build_llm_strings,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(SOURCE_BUILDERS), required=True)
    ap.add_argument("--workers", type=int, default=None, help="default: cpu_count()-1")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tags (smoke test)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="[llm] build even if some tags lack descriptions (zero rows)")
    args = ap.parse_args()

    config.ensure_dirs()
    import os
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    tags = config.load_ram_tags()
    if args.limit:
        tags = tags[: args.limit]
    T = len(tags)
    print(f"[{args.source}] {T} tags, {workers} workers")

    if args.source == "llm":
        by_tag = load_descriptions()
        covered = sum(1 for t in tags if by_tag.get(t))
        print(f"  descriptions available for {covered}/{T} tags")
        texts, tag_row_list = build_llm_strings(tags, allow_partial=args.allow_partial)
    else:
        texts, tag_row_list = SOURCE_BUILDERS[args.source](tags)
    M = len(texts)
    if M == 0:
        raise SystemExit(f"[{args.source}] produced 0 strings — nothing to build.")
    print(f"  {M} strings ({M/max(T,1):.1f} per tag)")

    # Persist the raw strings for auditability (mirrors the LVIS pipeline).
    strings_path = (config.STRINGS_TEMPLATES_JSONL if args.source == "templates"
                    else config.STRINGS_LLM_JSONL)
    with strings_path.open("w", encoding="utf8") as f:
        for row, text in zip(tag_row_list, texts):
            f.write(json.dumps({"tag_row": row, "tag": tags[row], "source": args.source,
                                "text": text}, ensure_ascii=False) + "\n")
    print(f"  wrote {strings_path.name}")

    # ── Encode ────────────────────────────────────────────────────────────────
    embeds = np.zeros((M, config.EMBED_DIM), dtype=np.float32)
    t0 = time.time()
    with Pool(processes=workers, initializer=_init_worker) as pool:
        for idx, vec in tqdm(
            pool.imap_unordered(_encode, enumerate(texts), chunksize=64),
            total=M, desc=f"encode {args.source}", unit="str",
        ):
            embeds[idx] = vec
    dt = time.time() - t0
    print(f"  encoded {M} strings in {dt/60:.1f} min ({M/max(dt,1e-9):.0f} str/s)")

    # ── Aggregate: normalize -> sum -> normalize ──────────────────────────────
    tag_row = np.asarray(tag_row_list, dtype=np.int64)
    acc = np.zeros((T, config.EMBED_DIM), dtype=np.float64)
    np.add.at(acc, tag_row, embeds.astype(np.float64))
    classifiers = _l2norm_rows(acc.astype(np.float32))

    n_empty = int((np.linalg.norm(classifiers, axis=1) < 1e-6).sum())
    out_path = config.CLASSIFIER_SETS[args.source]
    np.save(out_path, classifiers)
    norms = np.linalg.norm(classifiers, axis=1)
    nz = norms[norms > 1e-6]
    print(f"  saved {out_path.name}: {classifiers.shape} "
          f"row-norm min/max {nz.min():.6f}/{nz.max():.6f}"
          + (f"  [{n_empty} all-zero rows]" if n_empty else ""))

    # Tag order metadata (written once, by whichever source runs first)
    if not config.TAG_ORDER_JSON.exists():
        meta = {"n_tags": T, "embed_dim": config.EMBED_DIM,
                "tags": [{"row": i, "name": t} for i, t in enumerate(tags)]}
        config.TAG_ORDER_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf8")
        print(f"  wrote {config.TAG_ORDER_JSON.name}")


if __name__ == "__main__":
    main()
