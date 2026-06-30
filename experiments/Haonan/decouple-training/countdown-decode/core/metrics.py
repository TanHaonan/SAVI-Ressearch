"""Aggregation helpers for the countdown-decode arm matrix (PREREG sec 5).

Three pure, GPU-free pieces the harness leans on, in the terse
``controllable-posterior/core/metrics.py`` style (exact truth where available, no
proxy metrics):

  * ``boot_ci`` -- item-level bootstrap CI of a mean over per-instance 0/1 outcomes
    (resample instances with replacement; same percentile convention as the carrier).
  * ``modal_state`` -- the self-consistency picker: Phi-merge a bag of terminal states
    by ``domain.canon`` and return the modal canonical terminal state (its State object
    + count + share), deterministic tie-break by canon key.
  * ``leaf_check`` -- the per-arm correctness gate. In COUNTDOWN the "solution" is the
    TARGET and correctness is exactly ``is_goal``: the terminal is a SINGLE remaining
    value that equals the target. There is no separate substitution check (unlike the
    algebra version, where ``is_goal`` only confirmed the solved SHAPE ``x = c`` and the
    constant still had to be substitution-verified). For countdown ``is_goal`` already
    decides truth, so ``leaf_check`` is a thin, target-keyed wrapper over it.

These are the only numbers the run writes that are not raw ``decode_core`` fields.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap CI (item-level; mirrors controllable-posterior/core/metrics.boot_ci)
# ---------------------------------------------------------------------------

def boot_ci(vals, n_boot=2000, seed=0):
    """[2.5, 50, 97.5] percentiles of the bootstrap distribution of the mean.

    ``vals`` is the per-instance outcome list (0/1 pass@1 here, but any floats work).
    Resamples the items with replacement ``n_boot`` times. Empty input -> [nan]*3.
    """
    v = np.asarray(vals, float)
    if len(v) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [float(np.percentile(means, q)) for q in (2.5, 50, 97.5)]


# ---------------------------------------------------------------------------
# Self-consistency modal-state picker (Phi-merge a bag of terminal states)
# ---------------------------------------------------------------------------

def modal_state(domain, states):
    """Modal canonical terminal state of a bag of ``State`` objects, Phi-merged.

    Buckets ``states`` by ``domain.canon`` (the Phi key) and returns the bucket with
    the most members -- self-consistency over SEMANTIC terminal states, not raw text.
    Ties are broken deterministically by the (process-stable) canon key so the pick is
    reproducible. Returns ``(state, count, share)`` where ``state`` is a representative
    State of the modal bucket, ``count`` its size, ``share`` count / len(states); or
    ``(None, 0, 0.0)`` for an empty bag.

    The canon key is ``(tuple[Fraction, ...], Fraction)`` (countdown's Phi key), which
    is hashable and totally ordered, so the tie-break is well defined and deterministic.
    """
    if not states:
        return None, 0, 0.0
    buckets = {}  # canon key -> (representative state, count)
    for s in states:
        key = domain.canon(s)
        if key in buckets:
            rep, cnt = buckets[key]
            buckets[key] = (rep, cnt + 1)
        else:
            buckets[key] = (s, 1)
    # Most members; tie-break by the canon key for determinism (smallest key wins among
    # ties: the countdown canon key is a tuple of Fractions, totally ordered).
    best_key = min(buckets, key=lambda k: (-buckets[k][1], k))
    rep, cnt = buckets[best_key]
    return rep, cnt, cnt / len(states)


# ---------------------------------------------------------------------------
# Leaf correctness check (countdown: terminal value == target, i.e. is_goal)
# ---------------------------------------------------------------------------

def leaf_check(target, terminal_state, domain):
    """True iff ``terminal_state`` is a goal: one remaining value equal to ``target``.

    Countdown's correctness IS ``domain.is_goal``: the terminal is a single-value state
    whose value equals the target. We pass ``target`` through for signature parity with
    the algebra harness (where the solution had to be substitution-verified), but for
    countdown ``is_goal`` already encodes "the remaining value equals the target", so
    correctness is exactly ``terminal is not None and domain.is_goal(terminal)``.

    ``target`` is accepted (and not otherwise needed) so the harness can call
    ``leaf_check(target, terminal, domain)`` uniformly across arms; a ``None`` terminal
    (the arm did not reach a goal) is never correct.
    """
    return terminal_state is not None and domain.is_goal(terminal_state)
