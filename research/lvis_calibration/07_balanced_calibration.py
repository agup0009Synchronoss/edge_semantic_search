"""
07_balanced_calibration.py  (balanced per-tag calibration)

Fixes the naive-negative base-rate floor from 03_ by calibrating each tag on a
BALANCED per-tag subset instead of the whole split:

    subset(tag) = all positives of the tag  +  an equal number of random negatives
                = P positives + P negatives  (50/50, so 2P images; varies per tag)

We sweep the cosine grid [0.20, 0.65] on that balanced subset and pick the
threshold that maximizes the precision-leaning weighted Fbeta (beta from config).
Because the subset is balanced, precision is no longer structurally floored, so
the thresholds and stats are meaningful even for rare tags.

In the same run we emit, per tag/subset, the full confusion matrix and metrics
AT the chosen threshold (evaluated on that balanced subset).

Negative sampling is seeded (seed = --seed + cat_id) so the subsets — and hence
the reported stats — are reproducible.

Outputs:
  data/calibration/balanced_thresholds.npy      (T,) per-tag thresholds
  data/calibration/balanced_per_tag_metrics.csv  per-subset confusion + metrics

Usage:
  python 07_balanced_calibration.py [--neg-per-pos 1.0] [--seed 1234]
"""

from __future__ import annotations

import csv
import json
import argparse

import numpy as np

import config
import lvis_meta
import calib_core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-per-pos", type=float, default=1.0,
                    help="negatives sampled per positive (1.0 = 50/50 balance)")
    ap.add_argument("--seed", type=int, default=1234, help="base RNG seed")
    ap.add_argument("--classifiers", default=str(config.CLASSIFIERS_COMBINED))
    args = ap.parse_args()

    classifiers = np.load(args.classifiers).astype(np.float32)                # (T,512)
    img_mat = np.load(config.IMG_MATRIX).astype(np.float32)                   # (N,512)
    img_ids = np.load(config.IMG_IDS)                                          # (N,)
    N = img_mat.shape[0]
    T = classifiers.shape[0]
    id_to_row = {int(i): r for r, i in enumerate(img_ids)}

    with open(config.TAG_ORDER_JSON, "r", encoding="utf-8") as fh:
        tags = json.load(fh)["tags"]

    meta = lvis_meta.load()
    grid = config.T_GRID
    beta_sq = config.FBETA_SQ
    all_rows = np.arange(N)

    print(f"[balanced] {T} tags | grid {grid[0]}..{grid[-1]} | beta={config.FBETA} "
          f"| neg/pos={args.neg_per_pos}")
    S = img_mat @ classifiers.T                                              # (N,T)

    thresholds = np.zeros(T, dtype=np.float32)
    out_csv = config.CALIB_DIR / "balanced_per_tag_metrics.csv"
    config.ensure_dirs()
    header = ["row", "cat_id", "name", "bucket", "frequency", "threshold",
              "subset_total", "num_positives", "num_negatives",
              "TP", "FP", "TN", "FN", "precision", "recall", "f1", "weighted_f1"]

    wf1_all, f1_all = [], []
    by_bucket: dict[str, list] = {}

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r, tag in enumerate(tags):
            cid = tag["cat_id"]
            pos_rows = np.array(
                [id_to_row[int(i)] for i in meta.positives.get(cid, ())
                 if int(i) in id_to_row], dtype=np.int64)
            P = len(pos_rows)
            if P == 0:
                thresholds[r] = config.THRESH_MIN
                continue

            # sample balanced negatives (seeded, reproducible)
            rng = np.random.default_rng(args.seed + cid)
            neg_mask = np.ones(N, dtype=bool)
            neg_mask[pos_rows] = False
            neg_pool = all_rows[neg_mask]
            n_neg = min(int(round(P * args.neg_per_pos)), len(neg_pool))
            neg_rows = rng.choice(neg_pool, size=n_neg, replace=False)

            subset = np.concatenate([pos_rows, neg_rows])
            scores = S[subset, r]
            pos_submask = np.zeros(subset.shape[0], dtype=bool)
            pos_submask[:P] = True

            # sweep Fbeta on the balanced subset
            tp_g, pp_g, ptot = calib_core.counts_over_grid(scores, pos_submask, grid)
            _, _, fb_g = calib_core.prf_from_counts(tp_g, pp_g, ptot, beta_sq)
            k = int(np.argmax(fb_g))
            thr = float(grid[k])
            thresholds[r] = thr

            # confusion + metrics at chosen threshold, ON the balanced subset
            pred = scores >= thr
            TP = int(np.count_nonzero(pred & pos_submask))
            FP = int(np.count_nonzero(pred) - TP)
            FN = P - TP
            TN = subset.shape[0] - TP - FP - FN
            precision = TP / (TP + FP) if (TP + FP) else 0.0
            recall = TP / (TP + FN) if (TP + FN) else 0.0
            f1 = (2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) else 0.0
            denom = beta_sq * precision + recall
            wf1 = ((1 + beta_sq) * precision * recall / denom) if denom > 0 else 0.0

            wf1_all.append(wf1)
            f1_all.append(f1)
            by_bucket.setdefault(tag["bucket"], []).append(wf1)

            w.writerow([
                r, cid, tag["name"], tag["bucket"], tag["frequency"],
                round(thr, 4), int(subset.shape[0]), P, n_neg,
                TP, FP, TN, FN,
                round(precision, 4), round(recall, 4),
                round(f1, 4), round(wf1, 4),
            ])

    np.save(config.CALIB_DIR / "balanced_thresholds.npy", thresholds)

    print(f"[balanced] wrote {out_csv}")
    print(f"[balanced] saved balanced_thresholds.npy ({T} tags)")
    print(f"[balanced] mean weighted_f1 = {np.mean(wf1_all):.4f} | mean f1 = {np.mean(f1_all):.4f}")
    for b in (config.BUCKET_HIGH, config.BUCKET_MEDIUM, config.BUCKET_WEAK):
        if b in by_bucket:
            v = by_bucket[b]
            print(f"[balanced]   {b:6}: n={len(v):4}  mean weighted_f1={np.mean(v):.4f}")


if __name__ == "__main__":
    main()
