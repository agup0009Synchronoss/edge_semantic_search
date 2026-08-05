"""
04_report.py  (Step 4)

Human-readable summary of the calibration: Fbeta / precision / recall by bucket,
threshold histogram, best/worst tags, and the per-tag vs single-global-threshold
comparison (does prompt-ensembling + per-tag calibration beat one global cut?).

Usage:
  python 04_report.py
"""

from __future__ import annotations

import json
import statistics as stats
from collections import Counter

import numpy as np

import config


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(stats.fmean(xs), 4) if xs else None


def main() -> None:
    with open(config.CALIB_TABLE_JSON, "r", encoding="utf-8") as fh:
        table = json.load(fh)
    with open(config.BUCKET_THRESHOLDS_JSON, "r", encoding="utf-8") as fh:
        bt = json.load(fh)
    rows = table["rows"]
    calibrated = [r for r in rows if r["pos_support"] > 0]

    print("=" * 68)
    print(f"TinyCLIP LVIS calibration report   (Fbeta={table['fbeta']}, "
          f"N={table['n_images']:,} images, {len(rows)} tags)")
    print("=" * 68)

    # Overall
    print("\n-- Overall (per-tag, calibrated tags) --")
    print(f"  tags with positives : {len(calibrated)} / {len(rows)}")
    mp = _mean([r["precision"] for r in calibrated])
    mr = _mean([r["recall"] for r in calibrated])
    mf = _mean([r["per_tag_fbeta"] for r in calibrated])
    print(f"  mean Fbeta          : {mf}")
    print(f"  mean precision      : {mp}")
    print(f"  mean recall         : {mr}")
    lean = "precision > recall (beta lean OK)" if (mp or 0) > (mr or 0) else "recall >= precision (check beta!)"
    print(f"  -> {lean}")

    # By bucket
    print("\n-- By confidence bucket --")
    print(f"  {'bucket':8} {'n':>5} {'meanF':>7} {'meanP':>7} {'meanR':>7} {'thr(bkt)':>9}")
    for b in (config.BUCKET_HIGH, config.BUCKET_MEDIUM, config.BUCKET_WEAK):
        sub = [r for r in calibrated if r["bucket"] == b]
        bthr = (bt["buckets"].get(b) or {}).get("threshold")
        print(f"  {b:8} {len(sub):>5} {str(_mean([r['per_tag_fbeta'] for r in sub])):>7} "
              f"{str(_mean([r['precision'] for r in sub])):>7} "
              f"{str(_mean([r['recall'] for r in sub])):>7} {str(bthr):>9}")

    # Global baseline comparison
    g = bt.get("global") or {}
    print("\n-- Single global-threshold baseline (micro-averaged) --")
    print(f"  threshold={g.get('threshold')}  micro Fbeta={g.get('micro_fbeta')}  "
          f"P={g.get('micro_precision')}  R={g.get('micro_recall')}")
    if mf is not None and g.get("micro_fbeta") is not None:
        print(f"  mean per-tag Fbeta {mf} vs global micro Fbeta {g.get('micro_fbeta')} "
              f"-> per-tag {'higher' if mf > g['micro_fbeta'] else 'not higher'}")

    # Threshold histogram (per-tag thresholds, calibrated non-weak tags)
    print("\n-- Per-tag threshold histogram --")
    thr_counts = Counter(round(r["per_tag_threshold"], 2)
                         for r in calibrated if r["per_tag_threshold"] is not None)
    for thr in sorted(thr_counts):
        bar = "#" * min(50, thr_counts[thr])
        print(f"  {thr:0.2f}: {thr_counts[thr]:>4} {bar}")

    # Best / worst
    ranked = sorted(calibrated, key=lambda r: r["per_tag_fbeta"], reverse=True)
    print("\n-- Top 10 tags by Fbeta --")
    for r in ranked[:10]:
        print(f"  {r['per_tag_fbeta']:.3f}  thr={r['per_tag_threshold']:.2f}  "
              f"P={r['precision']:.2f} R={r['recall']:.2f}  n={r['pos_support']:<5} {r['name']}")
    print("\n-- Bottom 10 tags by Fbeta --")
    for r in ranked[-10:]:
        print(f"  {r['per_tag_fbeta']:.3f}  thr={r['per_tag_threshold']:.2f}  "
              f"P={r['precision']:.2f} R={r['recall']:.2f}  n={r['pos_support']:<5} {r['name']}")

    print("\n" + "=" * 68)


if __name__ == "__main__":
    main()
