"""Candidate-diversity metrics over behavioral KEYS.

A "key" is a string the caller has already produced by mapping a candidate
through the domain map Phi (for code: an output-vector hash; for Countdown: the
canonical state string). This module is domain-agnostic: it never imports the
domain and only sees the keys.
"""
from __future__ import annotations

import math
from collections import Counter


def bar(keys: list[str]) -> float:
    """Behavioral-duplicate rate: 1 - distinct/total.

    The fraction of slots that are behavioral duplicates. 0.0 when every key is
    distinct, approaches 1.0 as duplicates dominate. Empty list -> 0.0.
    """
    total = len(keys)
    if total == 0:
        return 0.0
    distinct = len(set(keys))
    return 1.0 - distinct / total


def cmr(keys: list[str]) -> float:
    """Candidate-multiplicity rate, "before pruning" reporting alias of `bar`.

    Same measure as `bar`; the separate name is kept so callers can report the
    duplicate rate over the *pre-pruning* candidate list under its own label.
    It equals `bar` over whatever list it is given. Empty list -> 0.0.
    """
    total = len(keys)
    if total == 0:
        return 0.0
    distinct = len(set(keys))
    return 1.0 - distinct / total


def k_eff(keys: list[str]) -> float:
    """Effective number of distinct keys = perplexity of the empirical dist.

    perplexity = exp(H) where H is the Shannon entropy (in nats) of the
    normalized key counts. For ["A","A","B","B"] -> 2.0; all-identical -> 1.0;
    empty -> 0.0. Matches the emission line's "K_eff ~= j" semantics.
    """
    total = len(keys)
    if total == 0:
        return 0.0
    counts = Counter(keys)
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p)
    return math.exp(h)
