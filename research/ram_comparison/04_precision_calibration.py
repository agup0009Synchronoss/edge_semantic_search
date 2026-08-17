"""
04_precision_calibration.py

Per-tag cosine thresholds that GUARANTEE a precision floor, instead of
maximizing Fbeta.

Every other threshold in this repo comes from `argmax(Fbeta)` — an objective,
not a promise. A tag whose Fbeta peaks at precision 0.52 still gets that
threshold (`person` is the standing example). Here we invert it: for each
target in {0.80, 0.85, 0.90}, pick the threshold that actually delivers at
least that precision on a balanced LVIS subset, and mark the tag NA when no
threshold can. A consumer can then reason about "calibrated to >=85%, or not
calibrated at all".

Ground truth is LVIS. The embeddings are the OpenAI-style TEMPLATE classifiers
built for the RAM comparison (`classifiers_templates.npy`), NOT the LLM
description classifiers — so each tag is scored with the same vector the Gradio
app uses in `templates` mode.

Method, per eligible tag:
  1. subset = all P positives + P seeded random negatives (50/50), using the
     exact RNG scheme of 07_balanced_calibration.py (default_rng(seed + cat_id),
     sampled without replacement, positives first) so the subsets are identical
     and the two calibrations are comparable.
  2. sweep the cosine grid, computing precision/recall at every point.
  3. take the LOWEST threshold meeting the target with >= --min-pp predicted
     positives. Lowest qualifying == highest recall, since recall is
     monotonically non-increasing in threshold.

Eligibility: the tag must map to a RAM tag (lvis_to_ram_mapping.csv) AND have
more than --min-gt ground-truth images. Rare tags are excluded outright because
a precision estimate off a handful of positives is noise, not calibration.

What this does NOT mean
-----------------------
Precision on a forced 50/50 subset is not deployment precision. With f/r pinned
by the balanced operating point, real precision at prevalence pi is
    pi / (pi + (1 - pi) * f/r)
so at a realistic ~1% prevalence, an 80% balanced operating point yields ~4%,
and 90% yields ~8%. These are intrinsic-separability operating points. Read
"p90" as "cleared 90% on a balanced subset", never as "90% precise in the wild".

Outputs (small, committed — written to results/, not the gitignored data/):
    results/precision_thresholds_p80.npy   (1203,) f32, NaN = NA
    results/precision_thresholds_p85.npy
    results/precision_thresholds_p90.npy
    results/precision_calibration.csv      one row per (tag, target), audit trail

Usage:
    ./venv_ramclip/Scripts/python.exe 04_precision_calibration.py
    ./venv_ramclip/Scripts/python.exe 04_precision_calibration.py --targets 0.80 0.95
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict

import numpy as np

import config

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass


def _load_from(path, name):
    """Load a module from an explicit path, bypassing sys.path resolution."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_calib_modules():
    """Import lvis_meta / calib_core bound to the CALIBRATION config, not ours.

    Both packages have a config.py. lvis_meta and calib_core do a bare
    `import config`, which would resolve to this app's config (no ANN_JSON, no
    T_GRID). So we bind `config` to the calibration one for the duration of
    those two imports, then put ours back.
    """
    ours = sys.modules["config"]
    lvis_cfg = _load_from(config.CALIB_ROOT / "config.py", "config")
    try:
        lvis_meta = _load_from(config.CALIB_ROOT / "lvis_meta.py", "lvis_meta")
        calib_core = _load_from(config.CALIB_ROOT / "calib_core.py", "calib_core")
    finally:
        sys.modules["config"] = ours
    return lvis_meta, calib_core, lvis_cfg


CSV_FIELDS = [
    "lvis_row", "cat_id", "name", "ram_row", "ram_tag", "bucket", "frequency",
    "target", "threshold", "subset_total", "num_positives", "num_negatives",
    "TP", "FP", "TN", "FN", "precision", "recall", "target_met",
    "best_precision", "reason",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=float, nargs="+", default=list(config.PRECISION_TARGETS),
                    help="precision floors to calibrate (default: 0.80 0.85 0.90)")
    ap.add_argument("--min-gt", type=int, default=20,
                    help="tag needs MORE than this many GT images to be eligible")
    ap.add_argument("--min-pp", type=int, default=5,
                    help="a grid point needs this many predicted positives to be trusted")
    ap.add_argument("--neg-per-pos", type=float, default=1.0,
                    help="negatives sampled per positive (1.0 = 50/50, matches 07_)")
    ap.add_argument("--seed", type=int, default=1234,
                    help="base RNG seed; per-tag seed is seed + cat_id, as in 07_")
    ap.add_argument("--grid-min", type=float, default=0.20)
    ap.add_argument("--grid-max", type=float, default=0.95)
    ap.add_argument("--grid-step", type=float, default=0.005)
    ap.add_argument("--classifiers", default=None,
                    help="path to a (4585,512) .npy (default: classifiers_templates.npy)")
    args = ap.parse_args()

    config.ensure_dirs()
    lvis_meta, calib_core, lvis_cfg = _load_calib_modules()

    # Deliberately NOT lvis_cfg.T_GRID: that is capped at 0.65 for the Fbeta
    # sweep. The cap is not what limits the high targets (qualifying thresholds
    # sit around 0.34-0.41), but there is no reason to carry an artificial
    # ceiling into a constraint-satisfaction search.
    grid = np.round(
        np.arange(args.grid_min, args.grid_max + args.grid_step / 2, args.grid_step), 4
    ).astype(np.float32)
    targets = sorted(set(round(float(t), 4) for t in args.targets))
    print(f"grid {grid[0]:.3f}..{grid[-1]:.3f} ({len(grid)} pts) | "
          f"targets {targets} | min_gt>{args.min_gt} | min_pp>={args.min_pp}")

    # ── inputs ────────────────────────────────────────────────────────────────
    clf_path = args.classifiers or config.CLASSIFIERS_TEMPLATES
    C = np.load(clf_path).astype(np.float32)
    if C.shape != (config.N_TAGS, config.EMBED_DIM):
        raise SystemExit(f"{clf_path}: expected {(config.N_TAGS, config.EMBED_DIM)}, got {C.shape}")
    print(f"classifiers: {getattr(clf_path, 'name', clf_path)} {C.shape}")

    img_mat = np.load(lvis_cfg.IMG_MATRIX).astype(np.float32)   # (N,512) L2-normed
    img_ids = np.load(lvis_cfg.IMG_IDS)
    N = img_mat.shape[0]
    id_to_row = {int(i): r for r, i in enumerate(img_ids)}
    print(f"images: {N}")

    lvis_tags = json.loads(config.LVIS_TAG_ORDER.read_text(encoding="utf8"))["tags"]
    T = len(lvis_tags)
    meta = lvis_meta.load()

    # LVIS row -> (ram_row, ram_tag) for applied mappings only
    mapped: dict[int, tuple[int, str]] = {}
    with config.TAG_MAPPING_CSV.open(encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["applied"] == "1":
                mapped[int(row["lvis_row"])] = (int(row["ram_row"]), row["ram_tag"])
    print(f"mapped LVIS->RAM: {len(mapped)} / {T}")

    # ── calibrate ─────────────────────────────────────────────────────────────
    thresholds = {t: np.full(T, np.nan, dtype=np.float32) for t in targets}
    rows_out: list[dict] = []
    all_rows = np.arange(N)
    stats = defaultdict(lambda: defaultdict(int))   # target -> reason -> count
    recalls = defaultdict(list)

    for r, tag in enumerate(lvis_tags):
        cid = tag["cat_id"]
        base = {"lvis_row": r, "cat_id": cid, "name": tag["name"],
                "bucket": tag["bucket"], "frequency": tag["frequency"]}

        if r not in mapped:
            for t in targets:
                stats[t]["unmapped"] += 1
                rows_out.append({**base, "ram_row": "", "ram_tag": "", "target": t,
                                 "threshold": "", "reason": "unmapped", "target_met": 0})
            continue
        ram_row, ram_tag = mapped[r]
        base |= {"ram_row": ram_row, "ram_tag": ram_tag}

        pos_rows = np.array(
            [id_to_row[int(i)] for i in meta.positives.get(cid, ()) if int(i) in id_to_row],
            dtype=np.int64)
        P = len(pos_rows)
        if P <= args.min_gt:
            for t in targets:
                stats[t]["low_gt"] += 1
                rows_out.append({**base, "target": t, "threshold": "",
                                 "num_positives": P, "reason": "low_gt", "target_met": 0})
            continue

        # Balanced subset — identical construction to 07_balanced_calibration.py
        rng = np.random.default_rng(args.seed + cid)
        neg_mask = np.ones(N, dtype=bool)
        neg_mask[pos_rows] = False
        neg_pool = all_rows[neg_mask]
        n_neg = min(int(round(P * args.neg_per_pos)), len(neg_pool))
        neg_rows = rng.choice(neg_pool, size=n_neg, replace=False)
        subset = np.concatenate([pos_rows, neg_rows])          # positives FIRST
        pos_submask = np.zeros(subset.shape[0], dtype=bool)
        pos_submask[:P] = True

        # Only the subset needs scoring — far cheaper than the full N x T matrix.
        scores = (img_mat[subset] @ C[ram_row]).astype(np.float32)

        tp_g, pp_g, ptot = calib_core.counts_over_grid(scores, pos_submask, grid)
        prec_g, rec_g, _ = calib_core.prf_from_counts(tp_g, pp_g, ptot, 1.0)

        supported = pp_g >= args.min_pp
        best_prec = float(prec_g[supported].max()) if supported.any() else 0.0

        for t in targets:
            hit = calib_core.best_at_precision(prec_g, rec_g, pp_g, grid, t, args.min_pp)
            if hit is None:
                reason = "low_support" if not supported.any() else "unreachable"
                stats[t][reason] += 1
                rows_out.append({**base, "target": t, "threshold": "",
                                 "subset_total": int(subset.shape[0]),
                                 "num_positives": P, "num_negatives": n_neg,
                                 "best_precision": round(best_prec, 4),
                                 "reason": reason, "target_met": 0})
                continue

            k, thr, prec, rec = hit
            thresholds[t][r] = np.float32(thr)

            # Confusion matrix at the chosen threshold, on the balanced subset.
            pred = scores >= thr
            TP = int(np.count_nonzero(pred & pos_submask))
            FP = int(np.count_nonzero(pred) - TP)
            FN = P - TP
            TN = int(subset.shape[0]) - TP - FP - FN

            stats[t]["ok"] += 1
            recalls[t].append(rec)
            rows_out.append({
                **base, "target": t, "threshold": round(thr, 4),
                "subset_total": int(subset.shape[0]), "num_positives": P,
                "num_negatives": n_neg, "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                "precision": round(prec, 4), "recall": round(rec, 4),
                "target_met": 1, "best_precision": round(best_prec, 4), "reason": "ok",
            })

    # ── write ─────────────────────────────────────────────────────────────────
    for t in targets:
        name = config.precision_set_name(t)
        out = config.RESULTS / f"precision_thresholds_{name}.npy"
        np.save(out, thresholds[t])

    with config.PRECISION_CALIB_CSV.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows_out, key=lambda x: (x["target"], x["lvis_row"])):
            w.writerow(row)

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\n{'target':>8s} {'calibrated':>11s} {'NA':>6s} "
          f"{'unreach':>8s} {'low_gt':>7s} {'unmapped':>9s} "
          f"{'thr med':>8s} {'thr max':>8s} {'recall med':>11s}")
    print("-" * 84)
    for t in targets:
        arr = thresholds[t]
        fin = arr[np.isfinite(arr)]
        rc = recalls[t]
        print(f"{t:>8.2f} {len(fin):>11d} {T - len(fin):>6d} "
              f"{stats[t]['unreachable']:>8d} {stats[t]['low_gt']:>7d} "
              f"{stats[t]['unmapped']:>9d} "
              + (f"{np.median(fin):>8.3f} {fin.max():>8.3f} " if len(fin) else f"{'-':>8s} {'-':>8s} ")
              + (f"{np.median(rc):>11.3f}" if rc else f"{'-':>11s}"))
    if any(stats[t]["low_support"] for t in targets):
        print(f"\n  low_support (no grid point with >={args.min_pp} predicted positives): "
              + ", ".join(f"{config.precision_set_name(t)}={stats[t]['low_support']}" for t in targets))
    print(f"\nwrote {len(targets)} threshold arrays + {config.PRECISION_CALIB_CSV.name} "
          f"to {config.RESULTS}")


if __name__ == "__main__":
    main()
