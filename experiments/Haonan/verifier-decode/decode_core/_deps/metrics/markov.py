"""Markov-divergence metrics over next-step KEY distributions.

For two histories that the domain map Phi merges into the same semantic state,
the divergence between their empirical next-step key distributions tells us how
lossy the merge is (near zero -> the induced process is close to Markov, a
precondition for the trellis). Distributions are dicts key -> probability; the
caller need not pre-normalize for the divergence functions (they normalize over
the union support), but `next_step_dist` returns a normalized dist.

Log base conventions:
  - js_divergence: base-2, so the value lies in [0, 1].
  - kl_divergence: natural log (nats).
"""
from __future__ import annotations

import math
from collections import Counter


def next_step_dist(samples: list[str]) -> dict[str, float]:
    """Normalized empirical distribution over next-step keys. Empty -> {}."""
    total = len(samples)
    if total == 0:
        return {}
    counts = Counter(samples)
    return {k: c / total for k, c in counts.items()}


def _normalized(d: dict) -> dict:
    s = sum(d.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in d.items()}


def js_divergence(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence, base-2, in [0, 1].

    Symmetric. 0.0 for identical distributions; 1.0 for disjoint support.
    Inputs are normalized over their own support before mixing, so unnormalized
    count dicts are accepted.
    """
    p = _normalized(p)
    q = _normalized(q)
    if not p and not q:
        return 0.0
    keys = set(p) | set(q)
    js = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, 0.0)
        mk = 0.5 * (pk + qk)
        if pk > 0:
            js += 0.5 * pk * math.log2(pk / mk)
        if qk > 0:
            js += 0.5 * qk * math.log2(qk / mk)
    # Numerical guard: keep within [0, 1].
    if js < 0.0:
        js = 0.0
    elif js > 1.0:
        js = 1.0
    return js


def kl_divergence(p: dict, q: dict) -> float:
    """KL(p || q) in nats (natural log).

    Inputs are normalized over their own support before comparison. Returns
    +inf if q assigns zero mass to a key where p > 0 (the divergence is
    genuinely unbounded there). Terms where p = 0 contribute 0.
    """
    p = _normalized(p)
    q = _normalized(q)
    kl = 0.0
    for k, pk in p.items():
        if pk <= 0:
            continue
        qk = q.get(k, 0.0)
        if qk <= 0:
            return float("inf")
        kl += pk * math.log(pk / qk)
    return kl
