"""Coverage metrics: how much of what a beam retains is usable / correct.

Domain-agnostic. Validity is supplied by the caller as a precomputed
key -> bool map (the domain executor was already run by the caller).
"""
from __future__ import annotations


def topk_valid_coverage(retained_keys: list[str], is_valid: dict[str, bool]) -> float:
    """Fraction of retained keys that are valid.

    `is_valid` maps a key to whether it is valid (parses / executes / satisfies
    constraints). A retained key absent from the map counts as not valid.
    Empty retained list -> 0.0.
    """
    total = len(retained_keys)
    if total == 0:
        return 0.0
    n_valid = sum(1 for k in retained_keys if is_valid.get(k, False))
    return n_valid / total


def beam_k_curve(per_instance_correct_in_topk: dict[int, list[bool]]) -> dict[int, float]:
    """Coverage curve: for each beam width K, the mean over instances of
    "a correct path is in the top-K".

    Input maps K -> list of per-instance booleans (one entry per instance).
    Output maps K -> mean of that list. An empty instance list -> 0.0.
    """
    out: dict[int, float] = {}
    for k, flags in per_instance_correct_in_topk.items():
        if len(flags) == 0:
            out[k] = 0.0
        else:
            out[k] = sum(1 for f in flags if f) / len(flags)
    return out
