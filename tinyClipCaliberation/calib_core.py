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


def micro_prf(tp_sum: np.ndarray, pp_sum: np.ndarray, p_total_sum: int, beta_sq: float):
    """Micro-averaged Precision/Recall/Fbeta over the grid from summed counts."""
    return prf_from_counts(tp_sum, pp_sum, p_total_sum, beta_sq)
