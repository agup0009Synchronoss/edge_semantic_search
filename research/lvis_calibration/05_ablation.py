"""
05_ablation.py  (optional, post-calibration analysis)

Quantify what each text source contributes to the classifier. Runs the same
Fbeta threshold sweep as 03_ for every classifier variant:

  - prompts       (OpenAI-80 + LVIS templates only)
  - descriptions  (def-gloss only)
  - questions      (VQA-style only)
  - combined       (mean over all strings — the primary classifier)

and reports, per variant: macro Fbeta (mean of per-tag best Fbeta over calibrated
tags), micro Fbeta (single best global threshold), mean precision/recall, and
macro Fbeta per confidence bucket. This shows whether adding descriptions /
questions on top of prompts actually helps.

Positives (and the implicit-negative pool) are identical across variants, so
differences are purely due to the text side.

Outputs:
  data/calibration/ablation.json

Usage:
  python 05_ablation.py
"""

from __future__ import annotations

import json
import time

import numpy as np

import config
import lvis_meta
import calib_core


def _positive_rows(meta, tags, id_to_row, N):
    """Ragged per-tag positive row indices (built once, reused per variant)."""
    rows = []
    for tag in tags:
        idx = [id_to_row[int(i)] for i in meta.positives.get(tag["cat_id"], ())
               if int(i) in id_to_row]
        rows.append(np.asarray(idx, dtype=np.int64))
    return rows


def _calibrate(classifiers, img_mat, pos_rows, buckets, grid, beta_sq):
    """Return aggregate metrics for one (T,512) classifier matrix."""
    S = img_mat @ classifiers.T                       # (N,T)
    T = classifiers.shape[0]
    N = img_mat.shape[0]
    K = len(grid)

    TP_all = np.zeros((T, K), dtype=np.int64)
    PP_all = np.zeros((T, K), dtype=np.int64)
    Ptot = np.zeros(T, dtype=np.int64)
    best_fb = np.full(T, np.nan)
    best_p = np.full(T, np.nan)
    best_r = np.full(T, np.nan)
    best_thr = np.full(T, np.nan)

    mask = np.zeros(N, dtype=bool)
    for r in range(T):
        idx = pos_rows[r]
        mask[:] = False
        mask[idx] = True
        col = S[:, r]
        tp, pp, ptot = calib_core.counts_over_grid(col, mask, grid)
        TP_all[r], PP_all[r], Ptot[r] = tp, pp, ptot
        if ptot > 0:
            p, rc, fb = calib_core.prf_from_counts(tp, pp, ptot, beta_sq)
            k = int(np.argmax(fb))
            best_fb[r], best_p[r], best_r[r], best_thr[r] = fb[k], p[k], rc[k], grid[k]

    cal = Ptot > 0
    # micro over all calibrated tags
    p_m, r_m, fb_m = calib_core.micro_prf(
        TP_all[cal].sum(0), PP_all[cal].sum(0), int(Ptot[cal].sum()), beta_sq)
    km = int(np.argmax(fb_m))

    def macro(sel):
        v = best_fb[sel]
        v = v[~np.isnan(v)]
        return round(float(v.mean()), 4) if len(v) else None

    by_bucket = {}
    for b in (config.BUCKET_HIGH, config.BUCKET_MEDIUM, config.BUCKET_WEAK):
        sel = np.array([t == b for t in buckets]) & cal
        by_bucket[b] = macro(sel)

    return {
        "macro_fbeta": macro(cal),
        "mean_precision": round(float(np.nanmean(best_p)), 4),
        "mean_recall": round(float(np.nanmean(best_r)), 4),
        "micro_fbeta": round(float(fb_m[km]), 4),
        "micro_threshold": round(float(grid[km]), 4),
        "micro_precision": round(float(p_m[km]), 4),
        "micro_recall": round(float(r_m[km]), 4),
        "by_bucket_macro_fbeta": by_bucket,
        "n_calibrated": int(cal.sum()),
    }


def main() -> None:
    with open(config.TAG_ORDER_JSON, "r", encoding="utf-8") as fh:
        tag_order = json.load(fh)
    tags = tag_order["tags"]
    buckets = [t["bucket"] for t in tags]

    img_mat = np.load(config.IMG_MATRIX).astype(np.float32)
    img_ids = np.load(config.IMG_IDS)
    N = img_mat.shape[0]
    id_to_row = {int(i): r for r, i in enumerate(img_ids)}

    meta = lvis_meta.load()
    grid = config.T_GRID
    beta_sq = config.FBETA_SQ
    pos_rows = _positive_rows(meta, tags, id_to_row, N)

    # Assemble variants: per-source + combined.
    variants = {}
    src = np.load(config.CLASSIFIERS_BY_SOURCE)
    for name in tag_order.get("sources", ["prompts", "descriptions", "questions"]):
        if name in src.files:
            variants[name] = src[name].astype(np.float32)
    variants["combined"] = np.load(config.CLASSIFIERS_COMBINED).astype(np.float32)

    print(f"[ablation] N={N:,} images x T={len(tags)} tags | beta={config.FBETA}")
    results = {}
    for name, C in variants.items():
        t0 = time.time()
        results[name] = _calibrate(C, img_mat, pos_rows, buckets, grid, beta_sq)
        print(f"[ablation] {name:<12} done in {time.time()-t0:.1f}s")

    # Report
    order = [v for v in ("prompts", "descriptions", "questions", "combined") if v in results]
    print("\n-- Ablation (per-tag best Fbeta) --")
    print(f"  {'variant':<12} {'macroF':>7} {'microF':>7} {'meanP':>7} {'meanR':>7} "
          f"{'high':>6} {'med':>6} {'weak':>6}")
    for name in order:
        r = results[name]
        bb = r["by_bucket_macro_fbeta"]
        print(f"  {name:<12} {str(r['macro_fbeta']):>7} {str(r['micro_fbeta']):>7} "
              f"{str(r['mean_precision']):>7} {str(r['mean_recall']):>7} "
              f"{str(bb[config.BUCKET_HIGH]):>6} {str(bb[config.BUCKET_MEDIUM]):>6} "
              f"{str(bb[config.BUCKET_WEAK]):>6}")

    if "prompts" in results and "combined" in results:
        d = (results["combined"]["macro_fbeta"] or 0) - (results["prompts"]["macro_fbeta"] or 0)
        print(f"\n  combined - prompts macro Fbeta delta: {d:+.4f} "
              f"({'ensembling helps' if d > 0 else 'no gain from desc/questions'})")

    config.ensure_dirs()
    with open(config.CALIB_DIR / "ablation.json", "w", encoding="utf-8") as fh:
        json.dump({"fbeta": config.FBETA, "n_images": N, "variants": results}, fh, indent=2)
    print(f"\n[ablation] saved {(config.CALIB_DIR / 'ablation.json')}")


if __name__ == "__main__":
    main()
