"""Shared diagnostic metrics for "semantic-graph + global decode" vs an LM's
greedy/local output.

The central question these metrics answer: when a structured decoder changes the
LM's answer, did it make the answer FEASIBLE (satisfies the graph's hard
constraints) or did it make it CORRECT (matches gold)? Those are different. A
decoder can churn out feasible-but-wrong answers -- a tidy formatter, not a
reasoner. `feasibility_vs_accuracy_split` is the function that isolates that
warning bucket; the rest are supporting measurements.

Pure functions, numpy only. Inputs are lists/arrays of int assignments; `gold`
is the true assignment per instance. No file/IO, no imports from tests.
"""
from __future__ import annotations

import numpy as np


def accuracy(preds, golds) -> float:
    """Mean per-position accuracy over instances.

    preds/golds: lists of equal-length int sequences (one per instance). Each
    instance contributes its own per-position fraction-correct; the result is the
    unweighted mean of those per-instance fractions, so every instance counts the
    same regardless of length.
    """
    if len(preds) != len(golds):
        raise ValueError(f"len(preds)={len(preds)} != len(golds)={len(golds)}")
    if len(preds) == 0:
        return 0.0
    per_instance = []
    for p, g in zip(preds, golds):
        p = np.asarray(p)
        g = np.asarray(g)
        if p.shape != g.shape:
            raise ValueError(f"pred shape {p.shape} != gold shape {g.shape}")
        per_instance.append(float(np.mean(p == g)) if p.size else 0.0)
    return float(np.mean(per_instance))


def exact_match(preds, golds) -> float:
    """Fraction of instances whose whole sequence matches gold exactly."""
    if len(preds) != len(golds):
        raise ValueError(f"len(preds)={len(preds)} != len(golds)={len(golds)}")
    if len(preds) == 0:
        return 0.0
    hits = 0
    for p, g in zip(preds, golds):
        p = np.asarray(p)
        g = np.asarray(g)
        if p.shape == g.shape and bool(np.all(p == g)):
            hits += 1
    return hits / len(preds)


def feasible_rate(preds, graphs) -> float:
    """Fraction of instances whose pred satisfies all hard factors.

    feasible(i) := NOT graphs[i].violations(preds[i]).
    """
    if len(preds) != len(graphs):
        raise ValueError(f"len(preds)={len(preds)} != len(graphs)={len(graphs)}")
    if len(preds) == 0:
        return 0.0
    n_feasible = sum(0 if g.violations(p) else 1 for p, g in zip(preds, graphs))
    return n_feasible / len(preds)


def phi(err_a, err_b) -> float:
    """Phi (Matthews-style) correlation of two boolean error vectors.

    err_a, err_b: boolean vectors of equal length (per-position OR per-instance
    errors; the function is agnostic). Measures how the two error patterns line
    up: +1 = identical errors, -1 = perfectly opposite, 0 = unrelated. Used as a
    decorrelation / independence diagnostic between a decoder's errors and the
    LM's greedy errors.

    Degenerate case (either vector is all-True or all-False -> a variance of
    zero) returns 0.0 (no correlation defined), and never raises.
    """
    a = np.asarray(err_a, dtype=bool).astype(float)
    b = np.asarray(err_b, dtype=bool).astype(float)
    if a.shape != b.shape:
        raise ValueError(f"err_a shape {a.shape} != err_b shape {b.shape}")
    if a.size == 0:
        return 0.0
    # 2x2 contingency: n11=both error, n00=both ok, n10/n01=disagree.
    n11 = float(np.sum((a == 1) & (b == 1)))
    n00 = float(np.sum((a == 0) & (b == 0)))
    n10 = float(np.sum((a == 1) & (b == 0)))
    n01 = float(np.sum((a == 0) & (b == 1)))
    denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if denom <= 0.0:  # a margin is all-same -> phi undefined
        return 0.0
    return (n11 * n00 - n10 * n01) / np.sqrt(denom)


def coverage(score_pairs) -> float:
    """Fraction of candidate-score pairs with a non-zero margin (s1 != s2).

    score_pairs: list of [s1, s2]. A zero margin (s1 == s2) means the source
    produced no preference / abstained; coverage is the fraction where it did
    commit.
    """
    pairs = np.asarray(score_pairs, dtype=float)
    if pairs.size == 0:
        return 0.0
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"score_pairs must be (N,2); got shape {pairs.shape}")
    nonzero = np.sum(pairs[:, 0] != pairs[:, 1])
    return float(nonzero) / pairs.shape[0]


def feasibility_vs_accuracy_split(preds, graphs, golds) -> dict:
    """Split instances into three mutually-exclusive buckets (fractions sum to 1).

    For each instance:
      - infeasible:        pred violates a hard factor (graph.violations -> True).
      - feasible_correct:  feasible AND pred == gold exactly.
      - feasible_wrong:    feasible AND pred != gold.   <-- the warning bucket:
                           the decoder produced a valid-but-wrong answer.

    Infeasibility takes precedence: an infeasible pred is never counted as correct
    even if it happens to equal gold (a gold that itself violates the graph is an
    adversarial / mislabeled case, and we surface it as infeasible).

    Returns {"feasible_correct": f, "feasible_wrong": f, "infeasible": f}.
    """
    if not (len(preds) == len(graphs) == len(golds)):
        raise ValueError(
            f"length mismatch: preds={len(preds)} graphs={len(graphs)} golds={len(golds)}")
    n = len(preds)
    if n == 0:
        return {"feasible_correct": 0.0, "feasible_wrong": 0.0, "infeasible": 0.0}
    fc = fw = inf = 0
    for p, gr, g in zip(preds, graphs, golds):
        if gr.violations(p):
            inf += 1
            continue
        p_arr = np.asarray(p)
        g_arr = np.asarray(g)
        if p_arr.shape == g_arr.shape and bool(np.all(p_arr == g_arr)):
            fc += 1
        else:
            fw += 1
    return {
        "feasible_correct": fc / n,
        "feasible_wrong": fw / n,
        "infeasible": inf / n,
    }


def _bootstrap_ci(diffs, n_boot=1000, alpha=0.05, seed=0):
    """1000x bootstrap (2.5%, 97.5%) percentile CI for the mean of `diffs`."""
    diffs = np.asarray(diffs, dtype=float)
    m = diffs.size
    if m == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    boot_means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return [lo, hi]


def gain_vs_density(rows, key_a, key_b) -> list:
    """Per-density mean gain (key_a - key_b) with a bootstrap 95% CI.

    rows: list of {"density": int, key_a: per-instance-acc-array,
                   key_b: per-instance-acc-array}. The two arrays must be aligned
    per instance (same length) so element-wise differences are paired.

    For each row: gain = mean(key_a - key_b) over instances, and a 1000x
    bootstrap 95% CI over those per-instance differences.

    Returns [{"density": d, "gain": mean, "ci": [lo, hi]}], one per input row.
    """
    out = []
    for row in rows:
        a = np.asarray(row[key_a], dtype=float)
        b = np.asarray(row[key_b], dtype=float)
        if a.shape != b.shape:
            raise ValueError(
                f"density {row.get('density')}: {key_a} shape {a.shape} "
                f"!= {key_b} shape {b.shape}")
        diffs = a - b
        gain = float(diffs.mean()) if diffs.size else 0.0
        ci = _bootstrap_ci(diffs)
        out.append({"density": int(row["density"]), "gain": gain, "ci": ci})
    return out
