"""
03_benchmark_classifiers.py

Quantitative A/B of the TinyCLIP text sources: 88 template strings per tag vs
10 LLM visual descriptions per tag.

The comparison is only possible because 1037 of the 4585 RAM tags map onto an
LVIS category (see 02_build_tag_mapping.py), which gives us real ground truth:
the 100,169 precomputed LVIS-train image embeddings and their per-category
positives. We score BOTH classifier sets over the same images, the same tags,
and — critically — the same seeded negative samples, so the only variable is
the text that produced the embedding.

Two regimes, mirroring the existing calibration pipeline:

  balanced  each tag scored on (all its positives + an equal number of random
            negatives). This is the meaningful number: precision is not
            structurally floored by the 100k-image negative base rate, and it is
            the regime balanced_thresholds.npy was calibrated in.
            Reuses the sampling recipe from 07_balanced_calibration.py.

  naive     each tag scored against the full split. Reported for continuity with
            03_calibrate_thresholds.py; the absolute numbers are crushed by the
            base rate and should only be read set-vs-set, never in isolation.

For each tag we sweep the cosine grid and take the best achievable Fbeta, so
this measures the intrinsic separability of each text representation rather
than the quality of any particular threshold choice.

Outputs:
    data/text/benchmark_templates_vs_llm.csv   per-tag, both sets, both regimes
    stdout summary                             macro Fbeta / P / R + head-to-head

Usage:
    ./venv_ramclip/Scripts/python.exe 03_benchmark_classifiers.py
    ./venv_ramclip/Scripts/python.exe 03_benchmark_classifiers.py --regime balanced
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=("balanced", "naive", "both"), default="both")
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sets", nargs="+", default=["templates", "llm"])
    ap.add_argument("--combined", action="store_true",
                    help="also score normalize(templates + llm) — the union of both "
                         "text sources, free to evaluate since no re-encoding is needed")
    args = ap.parse_args()
    if args.combined and "combined" not in args.sets:
        args.sets = list(args.sets) + ["combined"]

    lvis_meta, calib_core, lvis_cfg = _load_calib_modules()
    grid = lvis_cfg.T_GRID
    beta_sq = lvis_cfg.FBETA_SQ
    print(f"grid {grid[0]:.2f}..{grid[-1]:.2f} ({len(grid)} pts) | beta={lvis_cfg.FBETA}")

    # ── ground truth ──────────────────────────────────────────────────────────
    img_mat = np.load(lvis_cfg.IMG_MATRIX)          # (N, 512) L2-normed
    img_ids = np.load(lvis_cfg.IMG_IDS)             # (N,) sorted coco image_id
    N = img_mat.shape[0]
    id_to_row = {int(i): r for r, i in enumerate(img_ids)}
    print(f"images: {N}")

    meta = lvis_meta.load()
    lvis_tags = json.loads(lvis_cfg.TAG_ORDER_JSON.read_text(encoding="utf8"))["tags"]

    # ── the mapped tag pairs ──────────────────────────────────────────────────
    pairs = []
    with config.TAG_MAPPING_CSV.open(encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["applied"] != "1":
                continue
            l_row = int(row["lvis_row"])
            lt = lvis_tags[l_row]
            pairs.append({
                "ram_row": int(row["ram_row"]), "ram_tag": row["ram_tag"],
                "lvis_row": l_row, "cat_id": lt["cat_id"], "lvis_name": lt["name"],
                "bucket": lt["bucket"], "frequency": lt["frequency"],
            })
    print(f"mapped tags with ground truth: {len(pairs)}")

    # Positive rows + the seeded balanced subset, computed ONCE and shared by
    # both classifier sets so the only difference is the embedding.
    all_rows = np.arange(N)
    for p in pairs:
        pos_rows = np.array(
            [id_to_row[int(i)] for i in meta.positives.get(p["cat_id"], ())
             if int(i) in id_to_row], dtype=np.int64)
        p["pos_rows"] = pos_rows
        P = len(pos_rows)
        p["P"] = P
        if P == 0:
            p["subset"] = None
            continue
        rng = np.random.default_rng(args.seed + p["cat_id"])
        neg_mask = np.ones(N, dtype=bool)
        neg_mask[pos_rows] = False
        neg_pool = all_rows[neg_mask]
        n_neg = min(int(round(P * args.neg_per_pos)), len(neg_pool))
        neg_rows = rng.choice(neg_pool, size=n_neg, replace=False)
        p["subset"] = np.concatenate([pos_rows, neg_rows])
        p["n_neg"] = n_neg

    usable = [p for p in pairs if p["subset"] is not None]
    print(f"tags with >=1 positive: {len(usable)}")

    regimes = (["balanced", "naive"] if args.regime == "both" else [args.regime])
    results: dict[tuple[str, str], dict] = {}

    def _classifiers(set_name: str) -> np.ndarray | None:
        """Load a named set, or synthesize 'combined' as the L2-normed sum of
        the two real sets — equal weight per SOURCE, not per string, so the 20
        LLM strings are not drowned by the 88 template strings."""
        if set_name != "combined":
            p = config.CLASSIFIER_SETS[set_name]
            return np.load(p) if p.exists() else None
        parts = []
        for s in ("templates", "llm"):
            p = config.CLASSIFIER_SETS[s]
            if not p.exists():
                return None
            parts.append(np.load(p))
        acc = np.sum(parts, axis=0, dtype=np.float64)
        norms = np.linalg.norm(acc, axis=1, keepdims=True)
        out = np.zeros_like(acc)
        np.divide(acc, norms, out=out, where=norms > 1e-8)
        return out.astype(np.float32)

    for set_name in args.sets:
        C = _classifiers(set_name)
        if C is None:
            print(f"  ! {set_name}: source classifiers not built — skipping")
            continue
        ram_rows = np.array([p["ram_row"] for p in usable], dtype=np.int64)
        print(f"\n[{set_name}] scoring {len(ram_rows)} tags over {N} images...")
        S = img_mat @ C[ram_rows].T            # (N, n_tags) float32
        print(f"  score matrix {S.shape} ({S.nbytes/1024**3:.2f} GB)")

        for regime in regimes:
            per_tag = []
            for j, p in enumerate(usable):
                if regime == "balanced":
                    scores = S[p["subset"], j]
                    pos_mask = np.zeros(p["subset"].shape[0], dtype=bool)
                    pos_mask[: p["P"]] = True
                else:
                    scores = S[:, j]
                    pos_mask = np.zeros(N, dtype=bool)
                    pos_mask[p["pos_rows"]] = True

                tp_g, pp_g, ptot = calib_core.counts_over_grid(scores, pos_mask, grid)
                prec_g, rec_g, fb_g = calib_core.prf_from_counts(tp_g, pp_g, ptot, beta_sq)
                k = int(np.argmax(fb_g))
                per_tag.append({
                    "ram_tag": p["ram_tag"], "lvis_name": p["lvis_name"],
                    "bucket": p["bucket"], "frequency": p["frequency"], "P": p["P"],
                    "thr": float(grid[k]), "fbeta": float(fb_g[k]),
                    "precision": float(prec_g[k]), "recall": float(rec_g[k]),
                })
            results[(set_name, regime)] = {"per_tag": per_tag}
            fb = np.array([t["fbeta"] for t in per_tag])
            pr = np.array([t["precision"] for t in per_tag])
            rc = np.array([t["recall"] for t in per_tag])
            print(f"  {regime:9s} macro Fbeta {fb.mean():.4f} | P {pr.mean():.4f} "
                  f"| R {rc.mean():.4f}")
        del S

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'regime':10s} {'set':10s} {'macroFb':>9s} {'P':>8s} {'R':>8s} "
          f"{'high':>8s} {'medium':>8s} {'weak':>8s}")
    print("-" * 72)
    for regime in regimes:
        for set_name in args.sets:
            key = (set_name, regime)
            if key not in results:
                continue
            per_tag = results[key]["per_tag"]
            fb = np.array([t["fbeta"] for t in per_tag])
            pr = np.array([t["precision"] for t in per_tag])
            rc = np.array([t["recall"] for t in per_tag])
            by_b = defaultdict(list)
            for t in per_tag:
                by_b[t["bucket"]].append(t["fbeta"])
            print(f"{regime:10s} {set_name:10s} {fb.mean():9.4f} {pr.mean():8.4f} "
                  f"{rc.mean():8.4f} "
                  + " ".join(f"{np.mean(by_b.get(b, [0])):8.4f}"
                             for b in ("high", "medium", "weak")))

    # Head-to-head, only meaningful when exactly two sets are present
    present = [s for s in args.sets if (s, regimes[0]) in results]
    if len(present) == 2:
        a, b = present
        for regime in regimes:
            fa = np.array([t["fbeta"] for t in results[(a, regime)]["per_tag"]])
            fb_ = np.array([t["fbeta"] for t in results[(b, regime)]["per_tag"]])
            wins_b = int((fb_ > fa).sum())
            wins_a = int((fa > fb_).sum())
            ties = len(fa) - wins_a - wins_b
            delta = fb_.mean() - fa.mean()
            print(f"\n[{regime}] {b} vs {a}: "
                  f"{b} wins {wins_b}, {a} wins {wins_a}, ties {ties} "
                  f"| mean delta {delta:+.4f} "
                  f"({100*delta/max(fa.mean(),1e-9):+.1f}%)")

    # ── per-tag CSV ───────────────────────────────────────────────────────────
    out = config.TEXT_DIR / "benchmark_templates_vs_llm.csv"
    fields = ["ram_tag", "lvis_name", "bucket", "frequency", "P", "regime"]
    for s in args.sets:
        fields += [f"{s}_thr", f"{s}_fbeta", f"{s}_precision", f"{s}_recall"]
    fields.append("winner")
    with out.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for regime in regimes:
            sets_here = [s for s in args.sets if (s, regime) in results]
            if not sets_here:
                continue
            n = len(results[(sets_here[0], regime)]["per_tag"])
            for i in range(n):
                base = results[(sets_here[0], regime)]["per_tag"][i]
                row = {"ram_tag": base["ram_tag"], "lvis_name": base["lvis_name"],
                       "bucket": base["bucket"], "frequency": base["frequency"],
                       "P": base["P"], "regime": regime}
                scores = {}
                for s in sets_here:
                    t = results[(s, regime)]["per_tag"][i]
                    row[f"{s}_thr"] = f"{t['thr']:.2f}"
                    row[f"{s}_fbeta"] = f"{t['fbeta']:.4f}"
                    row[f"{s}_precision"] = f"{t['precision']:.4f}"
                    row[f"{s}_recall"] = f"{t['recall']:.4f}"
                    scores[s] = t["fbeta"]
                row["winner"] = max(scores, key=scores.get) if scores else ""
                w.writerow(row)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
