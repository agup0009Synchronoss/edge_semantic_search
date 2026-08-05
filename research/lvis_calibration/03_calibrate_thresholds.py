"""
03_calibrate_thresholds.py  (Step 3)

Per-tag cosine-threshold calibration maximizing a precision-leaning Fbeta
(beta from config, ~60% precision / 40% recall).

  1. Score matrix S = img_matrix @ classifiers.T  (N,T), both unit-norm -> cosine.
  2. Positive columns from LVIS annotations (positives only; every other image is
     an implicit negative).
  3. For each tag, sweep the cosine grid [0.20, 0.65] and pick the threshold with
     the best Fbeta. Also keep TP/PP count arrays for bucket + global micro-averages.
  4. Bucket fallback: micro-average Fbeta over each confidence bucket -> one
     threshold per bucket. Weak tags use the weak-bucket threshold as their
     "effective" threshold; medium/high use their own per-tag threshold.

Outputs (see config.py):
  data/calibration/thresholds.npy          (T,) effective thresholds (device-ready)
  data/calibration/calibration_table.json  full per-tag rows
  data/calibration/calibration_table.parquet (if pyarrow present)
  data/calibration/bucket_thresholds.json   bucket + global micro thresholds

Usage:
  python 03_calibrate_thresholds.py
"""

from __future__ import annotations

import json
import time
import argparse

import numpy as np

import config
import lvis_meta
import calib_core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifiers", default=str(config.CLASSIFIERS_COMBINED),
                    help="which (T,512) classifier matrix to calibrate")
    args = ap.parse_args()

    # ── Load artifacts ────────────────────────────────────────────────────────
    classifiers = np.load(args.classifiers).astype(np.float32)          # (T,512)
    with open(config.TAG_ORDER_JSON, "r", encoding="utf-8") as fh:
        tag_order = json.load(fh)
    tags = tag_order["tags"]
    T = classifiers.shape[0]
    assert len(tags) == T, f"tag_order ({len(tags)}) != classifiers ({T})"

    img_mat = np.load(config.IMG_MATRIX).astype(np.float32)             # (N,512)
    img_ids = np.load(config.IMG_IDS)                                    # (N,)
    N = img_mat.shape[0]
    id_to_row = {int(i): r for r, i in enumerate(img_ids)}

    meta = lvis_meta.load()
    grid = config.T_GRID
    K = len(grid)
    beta_sq = config.FBETA_SQ

    print(f"[calib] N={N:,} images x T={T} tags | grid {grid[0]}..{grid[-1]} "
          f"({K} pts) | beta={config.FBETA}")

    # ── Score matrix (one BLAS matmul) ────────────────────────────────────────
    t0 = time.time()
    S = img_mat @ classifiers.T                                         # (N,T) f32
    print(f"[calib] score matrix {S.shape} in {time.time()-t0:.1f}s")

    # ── Per-tag sweep ─────────────────────────────────────────────────────────
    TP_all = np.zeros((T, K), dtype=np.int64)
    PP_all = np.zeros((T, K), dtype=np.int64)
    Ptot = np.zeros(T, dtype=np.int64)

    rows = []
    t0 = time.time()
    for r, tag in enumerate(tags):
        cid = tag["cat_id"]
        pos_ids = meta.positives.get(cid, ())
        pos_mask = np.zeros(N, dtype=bool)
        for iid in pos_ids:
            row = id_to_row.get(int(iid))
            if row is not None:
                pos_mask[row] = True

        col = S[:, r]
        tp, pp, p_total = calib_core.counts_over_grid(col, pos_mask, grid)
        TP_all[r], PP_all[r], Ptot[r] = tp, pp, p_total
        precision, recall, fbeta = calib_core.prf_from_counts(tp, pp, p_total, beta_sq)

        if p_total > 0:
            k, thr, fb = calib_core.best_on_grid(fbeta, grid)
            rows.append({
                "row": r, "cat_id": cid, "name": tag["name"],
                "bucket": tag["bucket"], "frequency": tag["frequency"],
                "pos_support": p_total,
                "per_tag_threshold": round(thr, 4),
                "per_tag_fbeta": round(fb, 4),
                "precision": round(float(precision[k]), 4),
                "recall": round(float(recall[k]), 4),
            })
        else:
            rows.append({
                "row": r, "cat_id": cid, "name": tag["name"],
                "bucket": tag["bucket"], "frequency": tag["frequency"],
                "pos_support": 0,
                "per_tag_threshold": None, "per_tag_fbeta": None,
                "precision": None, "recall": None,
            })
    print(f"[calib] per-tag sweep in {time.time()-t0:.1f}s")

    # ── Bucket + global micro-averaged thresholds ─────────────────────────────
    def micro_threshold(idx: np.ndarray):
        if len(idx) == 0:
            return None
        tp_sum = TP_all[idx].sum(0)
        pp_sum = PP_all[idx].sum(0)
        p_sum = int(Ptot[idx].sum())
        p, rc, fb = calib_core.micro_prf(tp_sum, pp_sum, p_sum, beta_sq)
        k, thr, fbv = calib_core.best_on_grid(fb, grid)
        return {"threshold": round(thr, 4), "micro_fbeta": round(fbv, 4),
                "micro_precision": round(float(p[k]), 4),
                "micro_recall": round(float(rc[k]), 4), "n_tags": int(len(idx))}

    buckets = {}
    row_bucket = np.array([t["bucket"] for t in tags])
    for b in (config.BUCKET_WEAK, config.BUCKET_MEDIUM, config.BUCKET_HIGH):
        buckets[b] = micro_threshold(np.where(row_bucket == b)[0])
    global_thr = micro_threshold(np.arange(T))

    weak_thr = (buckets[config.BUCKET_WEAK] or global_thr or {}).get("threshold")

    # ── Effective threshold: weak -> weak-bucket fallback, else per-tag ────────
    thresholds = np.zeros(T, dtype=np.float32)
    for row in rows:
        r = row["row"]
        b = row["bucket"]
        per_tag = row["per_tag_threshold"]
        if b == config.BUCKET_WEAK or per_tag is None:
            eff = weak_thr if weak_thr is not None else config.THRESH_MIN
            row["confidence"] = "low" if row["pos_support"] > 0 else "none"
            row["effective_source"] = "weak_bucket_fallback"
        else:
            eff = per_tag
            row["confidence"] = "high" if b == config.BUCKET_HIGH else "medium"
            row["effective_source"] = "per_tag"
        row["effective_threshold"] = round(float(eff), 4)
        thresholds[r] = eff

    # ── Save ──────────────────────────────────────────────────────────────────
    config.ensure_dirs()
    np.save(config.THRESHOLDS_NPY, thresholds)
    table = {"fbeta": config.FBETA, "grid": [float(x) for x in grid],
             "n_images": int(N), "rows": rows}
    with open(config.CALIB_TABLE_JSON, "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2)
    with open(config.BUCKET_THRESHOLDS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"fbeta": config.FBETA, "buckets": buckets, "global": global_thr},
                  fh, indent=2)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), config.CALIB_TABLE_PARQUET)
    except Exception as e:  # noqa: BLE001
        print(f"[calib] (parquet skipped: {e})")

    calibrated = sum(1 for row in rows if row["pos_support"] > 0)
    print(f"[calib] saved thresholds.npy ({T} tags, {calibrated} with positives).")
    print(f"[calib] bucket thresholds: "
          + ", ".join(f"{b}={(buckets[b] or {}).get('threshold')}" for b in buckets)
          + f" | global={ (global_thr or {}).get('threshold') }")


if __name__ == "__main__":
    main()
