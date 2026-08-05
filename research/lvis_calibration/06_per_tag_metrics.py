"""
06_per_tag_metrics.py  (post-calibration, per-tag scorecard)

For every tag, evaluate the confusion matrix and metrics at the FINAL chosen
threshold (the effective threshold from 03_ = per-tag for medium/high, weak-bucket
fallback for weak). Writes a flat CSV you can open in Excel.

Columns:
  row, cat_id, name, bucket, confidence, frequency, threshold,
  total_images, num_positives, num_negatives,
  TP, FP, TN, FN, precision, recall, f1, weighted_f1

  - total_images  = N (every image is a candidate for every tag)
  - num_positives = images annotated with the tag (TP+FN)
  - f1            = standard F1 (beta=1)
  - weighted_f1   = the precision-leaning Fbeta we optimized (beta from config)

Outputs:
  data/calibration/per_tag_metrics.csv

Usage:
  python 06_per_tag_metrics.py
"""

from __future__ import annotations

import csv
import json

import numpy as np

import config
import lvis_meta


def main() -> None:
    classifiers = np.load(config.CLASSIFIERS_COMBINED).astype(np.float32)     # (T,512)
    thresholds = np.load(config.THRESHOLDS_NPY).astype(np.float32)            # (T,) effective
    img_mat = np.load(config.IMG_MATRIX).astype(np.float32)                   # (N,512)
    img_ids = np.load(config.IMG_IDS)                                          # (N,)
    N = img_mat.shape[0]
    T = classifiers.shape[0]
    id_to_row = {int(i): r for r, i in enumerate(img_ids)}

    with open(config.TAG_ORDER_JSON, "r", encoding="utf-8") as fh:
        tags = json.load(fh)["tags"]
    # confidence + effective_source come from the calibration table (03_)
    with open(config.CALIB_TABLE_JSON, "r", encoding="utf-8") as fh:
        calib_rows = {r["row"]: r for r in json.load(fh)["rows"]}

    meta = lvis_meta.load()
    beta_sq = config.FBETA_SQ

    print(f"[metrics] scoring {T} tags x {N:,} images at their final thresholds ...")
    S = img_mat @ classifiers.T                                              # (N,T)

    out_path = config.CALIB_DIR / "per_tag_metrics.csv"
    config.ensure_dirs()
    header = ["row", "cat_id", "name", "bucket", "confidence", "frequency",
              "threshold", "total_images", "num_positives", "num_negatives",
              "TP", "FP", "TN", "FN", "precision", "recall", "f1", "weighted_f1"]

    pos_mask = np.zeros(N, dtype=bool)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r, tag in enumerate(tags):
            cid = tag["cat_id"]
            thr = float(thresholds[r])

            pos_mask[:] = False
            for iid in meta.positives.get(cid, ()):
                row = id_to_row.get(int(iid))
                if row is not None:
                    pos_mask[row] = True

            pred = S[:, r] >= thr
            P = int(pos_mask.sum())
            TP = int(np.count_nonzero(pred & pos_mask))
            FP = int(np.count_nonzero(pred) - TP)
            FN = P - TP
            TN = N - TP - FP - FN

            precision = TP / (TP + FP) if (TP + FP) else 0.0
            recall = TP / (TP + FN) if (TP + FN) else 0.0
            f1 = (2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) else 0.0
            denom = beta_sq * precision + recall
            wf1 = ((1 + beta_sq) * precision * recall / denom) if denom > 0 else 0.0

            crow = calib_rows.get(r, {})
            w.writerow([
                r, cid, tag["name"], tag["bucket"],
                crow.get("confidence", ""), tag["frequency"],
                round(thr, 4), N, P, N - P,
                TP, FP, TN, FN,
                round(precision, 4), round(recall, 4),
                round(f1, 4), round(wf1, 4),
            ])

    print(f"[metrics] wrote {out_path}")
    print(f"[metrics] columns: {', '.join(header)}")


if __name__ == "__main__":
    main()
