"""End-to-end accuracy and significance metrics.

Domain-agnostic. Correctness is supplied by the caller as a list of bools (or
0/1 ints). The bootstrap is deterministic given its seed.
"""
from __future__ import annotations

import numpy as np


def pass_at_1(correct_flags: list[bool]) -> float:
    """Mean correctness. Empty list -> 0.0."""
    n = len(correct_flags)
    if n == 0:
        return 0.0
    return sum(1 for f in correct_flags if f) / n


def paired_bootstrap(
    a_correct: list,
    b_correct: list,
    n: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a) - mean(b) with a 95% percentile CI.

    a_correct and b_correct are parallel per-instance correctness lists (bools
    or 0/1) over the SAME instances; resampling draws instance indices jointly
    so the pairing is preserved. Returns (delta, lo, hi) where delta is the
    observed mean(a) - mean(b) and (lo, hi) is the 2.5/97.5 percentile interval
    of the bootstrap deltas. Deterministic given `seed` (numpy default_rng).

    Empty inputs -> (0.0, 0.0, 0.0).
    """
    a = np.asarray(a_correct, dtype=float)
    b = np.asarray(b_correct, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a_correct and b_correct must have the same length")
    m = a.shape[0]
    delta = float(a.mean() - b.mean()) if m > 0 else 0.0
    if m == 0:
        return (0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    # n bootstrap resamples of m instance indices each, sampled jointly.
    idx = rng.integers(0, m, size=(n, m))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return (delta, lo, hi)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni corrected p-values, returned in the INPUT order.

    For m p-values sorted ascending p_(1) <= ... <= p_(m), the step-down
    adjusted value at rank i is max over j<=i of min(1, (m - j + 1) * p_(j)),
    i.e. the running maximum of (m - rank) * p enforces monotonicity, and each
    is clipped to 1.0. Empty list -> [].
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank  # m, m-1, ..., 1
        val = factor * pvals[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted
