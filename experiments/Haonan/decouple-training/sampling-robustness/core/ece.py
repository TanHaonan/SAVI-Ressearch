"""Reliability curve + Expected Calibration Error (ECE) — the shared 'calibration' scalar.

Both functions are pure (numpy/python only), deterministic, model-free, and unit-tested.

A "confidence" is a number in [0,1] the model assigns to its top choice; "correct" is the
0/1 indicator that the top choice was right. We bin samples by confidence into n_bins equal-
width bins over [0,1], and in each bin compare the mean confidence to the empirical accuracy.

  reliability_curve(confidences, correct, n_bins=10)
      -> [(mean_conf_k, acc_k, count_k), ...] for each OCCUPIED bin (empty bins are dropped),
         ordered by bin index.
  ece(confidences, correct, n_bins=10)
      -> sum_k (count_k / N) * |acc_k - mean_conf_k|   (0 = perfectly calibrated).
         Empty input -> nan (no calibration is defined on zero samples).

Conventions:
- confidences are clipped into [0,1] before binning (a stray 1.0 lands in the top bin, not
  an overflow bin; a stray 0.0 lands in the bottom bin).
- ece() weights each bin's gap by that bin's share of the samples, so it equals exactly the
  count-weighted sum over reliability_curve(...).
"""
import numpy as np


def _bin_index(confidences, n_bins):
    """Equal-width bin index in [0, n_bins-1] for each confidence in [0,1].
    The closed right edge 1.0 is folded into the last bin so it never overflows."""
    c = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    idx = np.floor(c * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    return idx


def reliability_curve(confidences, correct, n_bins=10):
    """Per-occupied-bin (mean_conf, acc, count). Empty bins are omitted."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidences.size == 0:
        return []
    idx = _bin_index(confidences, n_bins)
    curve = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        curve.append((mean_conf, acc, count))
    return curve


def ece(confidences, correct, n_bins=10):
    """Expected Calibration Error = sum_k (count_k/N) * |acc_k - mean_conf_k|.
    Returns nan on empty input."""
    n = len(confidences)
    if n == 0:
        return float("nan")
    curve = reliability_curve(confidences, correct, n_bins=n_bins)
    return float(sum((count / n) * abs(acc - mean_conf) for (mean_conf, acc, count) in curve))
