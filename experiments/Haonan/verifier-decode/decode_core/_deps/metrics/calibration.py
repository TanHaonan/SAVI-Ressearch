"""Calibration metrics: gap between confidence and correctness.

Domain-agnostic. The caller supplies parallel lists of per-item confidence
(a probability in [0, 1]) and a correctness bool. The reliability curve and ECE
use equal-width bins over [0, 1].
"""
from __future__ import annotations


def _bin_index(c: float, n_bins: int) -> int:
    """Index of the equal-width bin in [0, 1] that confidence c falls in.

    Bins are [0, 1/n), [1/n, 2/n), ..., with the right edge 1.0 placed in the
    last bin. Values are clamped to [0, 1].
    """
    if c < 0.0:
        c = 0.0
    elif c > 1.0:
        c = 1.0
    idx = int(c * n_bins)
    if idx == n_bins:  # c == 1.0
        idx = n_bins - 1
    return idx


def reliability_curve(conf: list[float], correct: list[bool], n_bins: int = 10) -> list[dict]:
    """Per-bin reliability data.

    Returns a list of length n_bins; each entry is
    {"conf_mean": float, "acc": float, "count": int}. For an empty bin,
    conf_mean and acc are 0.0 and count is 0.
    """
    sums_conf = [0.0] * n_bins
    sums_acc = [0] * n_bins
    counts = [0] * n_bins
    for c, ok in zip(conf, correct):
        b = _bin_index(c, n_bins)
        sums_conf[b] += c
        sums_acc[b] += 1 if ok else 0
        counts[b] += 1

    curve = []
    for b in range(n_bins):
        if counts[b] == 0:
            curve.append({"conf_mean": 0.0, "acc": 0.0, "count": 0})
        else:
            curve.append(
                {
                    "conf_mean": sums_conf[b] / counts[b],
                    "acc": sums_acc[b] / counts[b],
                    "count": counts[b],
                }
            )
    return curve


def ece(conf: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Expected calibration error: count-weighted mean of |acc - conf_mean|
    across non-empty bins. Empty input -> 0.0.
    """
    n = len(conf)
    if n == 0:
        return 0.0
    curve = reliability_curve(conf, correct, n_bins)
    err = 0.0
    for bin_ in curve:
        if bin_["count"] == 0:
            continue
        err += (bin_["count"] / n) * abs(bin_["acc"] - bin_["conf_mean"])
    return err
