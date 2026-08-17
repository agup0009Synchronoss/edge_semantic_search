"""
calib_core.py

Pure-numpy calibration math, separated from the 03_ script so it can be unit
tested without any downloaded data.

Given per-image cosine scores for one tag and a boolean positive mask, we compute
TP / predicted-positive counts at every threshold in the grid using searchsorted
(no N x K boolean blow-up), then Precision/Recall/Fbeta. "Predicted positive"
means score >= threshold. Negatives are implicit (every non-positive image).
"""

from __future__ import annotations

import numpy as np


def counts_over_grid(scores: np.ndarray, pos_mask: np.ndarray, grid: np.ndarray):
    """For one tag, return (TP, PP, P_total) over the threshold grid.

      TP[k] = #{ positives with score >= grid[k] }
      PP[k] = #{ all images with score >= grid[k] }   (predicted positives)
      P_total = total positives

    O((N + P) log N) via sorting + vectorized searchsorted over the grid.
    """
    scores = np.asarray(scores, dtype=np.float32)
    pos_mask = np.asarray(pos_mask, dtype=bool)
    p_total = int(pos_mask.sum())

    sorted_all = np.sort(scores)
    # count(score >= t) = N - searchsorted(sorted_all, t, 'left')
    pp = scores.shape[0] - np.searchsorted(sorted_all, grid, side="left")
    if p_total:
        sorted_pos = np.sort(scores[pos_mask])
        tp = p_total - np.searchsorted(sorted_pos, grid, side="left")
    else:
        tp = np.zeros_like(pp)
    return tp.astype(np.int64), pp.astype(np.int64), p_total


def prf_from_counts(tp: np.ndarray, pp: np.ndarray, p_total: int, beta_sq: float):
    """Precision, Recall, Fbeta arrays over the grid from count arrays."""
    tp = np.asarray(tp, dtype=np.float64)
    pp = np.asarray(pp, dtype=np.float64)
    precision = np.zeros_like(tp)
    np.divide(tp, pp, out=precision, where=pp > 0)
    recall = (tp / p_total) if p_total else np.zeros_like(tp)
    denom = beta_sq * precision + recall
    num = (1.0 + beta_sq) * precision * recall
    fbeta = np.zeros_like(denom)
    np.divide(num, denom, out=fbeta, where=denom > 0)
    return precision, recall, fbeta


def best_on_grid(fbeta: np.ndarray, grid: np.ndarray):
    """Return (best_index, best_threshold, best_fbeta). Ties -> lowest threshold
    index (argmax picks first max), which is the more precise/higher-recall end
    depending on grid direction; grid is ascending so first max = lowest thr."""
    k = int(np.argmax(fbeta))
    return k, float(grid[k]), float(fbeta[k])


def best_at_precision(precision: np.ndarray, recall: np.ndarray, pp: np.ndarray,
                      grid: np.ndarray, target: float, min_pp: int = 5):
    """Lowest threshold that DELIVERS `precision >= target`, or None.

    This is the constraint-satisfaction counterpart to `best_on_grid`, which
    maximizes an objective. Fbeta-argmax says "lean toward precision"; this says
    "clear this precision bar or report that you cannot".

    Lowest qualifying threshold == highest recall: recall is monotonically
    non-increasing in threshold, so among all grid points meeting the target the
    first one is the best trade. Scanning for the first qualifying point (rather
    than argmax of precision) is also robust to the non-monotonic wobble that
    real precision curves have near the top of the grid.

    `min_pp` guards the estimate itself. At high thresholds precision is computed
    from a handful of predictions, so 2/2 == 1.0 would otherwise beat a genuine
    40/45. Points with fewer than `min_pp` predicted positives are not eligible.

    Returns (index, threshold, precision, recall), or None when no grid point
    satisfies both the target and the support guard.
    """
    precision = np.asarray(precision, dtype=np.float64)
    pp = np.asarray(pp, dtype=np.int64)
    eligible = (pp >= int(min_pp)) & (precision >= float(target))
    hits = np.flatnonzero(eligible)
    if hits.size == 0:
        return None
    k = int(hits[0])
    return k, float(grid[k]), float(precision[k]), float(np.asarray(recall)[k])


def micro_prf(tp_sum: np.ndarray, pp_sum: np.ndarray, p_total_sum: int, beta_sq: float):
    """Micro-averaged Precision/Recall/Fbeta over the grid from summed counts."""
    return prf_from_counts(tp_sum, pp_sum, p_total_sum, beta_sq)
