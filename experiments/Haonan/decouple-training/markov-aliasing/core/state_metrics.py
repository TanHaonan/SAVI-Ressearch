"""Model-free state-level metrics for the state-emission experiment.

Every function operates purely on lists of state labels (the Phi-pushforward of
candidate texts) plus, where needed, the state alphabet, the survivor set, and a
known true posterior. Labels are either members of the alphabet (committed
states) or one of the abstention markers 'hedge' / 'none' / 'multi'. All metrics
are deterministic, pure numpy/python, and take no model or randomness.

This file holds the calibration + Markov JS/KL implementations. It is
generalized from sampling-diversity/divmetrics.py (which keyed on option
letters) to an arbitrary `alphabet` of states.

Conventions:
- "committed state" = a label that is a member of `alphabet`.
- abstentions ('hedge', 'none', 'multi') are never committed states; 'hedge' and
  'none' count toward abstain_rate, 'multi' (ambiguous multi-commit) does not.
- calibration_tv returns nan when nothing was committed (legitimately, not a
  crash): a fully-abstaining cell has no empirical committed distribution.
"""
import numpy as np

ABSTAIN = ("hedge", "none", "multi")


def _committed(states, alphabet):
    """Labels in `states` that are committed states (members of the alphabet)."""
    alpha = set(alphabet)
    return [s for s in states if s in alpha]


# --------------------------------------------------------------------- diversity
def k_eff_distinct(states, alphabet):
    """Number of DISTINCT committed states (abstentions excluded)."""
    return len(set(_committed(states, alphabet)))


def k_eff_entropy(states, alphabet):
    """Effective number of committed states = exp(Shannon entropy of the
    empirical committed-state distribution). 1.0 when collapsed to a single
    state; 0.0 when nothing was committed."""
    vs = _committed(states, alphabet)
    if not vs:
        return 0.0
    _, c = np.unique(vs, return_counts=True)
    p = c / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def coverage(states, survivors):
    """Fraction of the true survivors that appear at least once among the
    samples. survivors is the set of states the true posterior puts mass on
    (the carrier guarantees j >= 1). Returns nan if survivors is empty (no true
    posterior to cover), rather than dividing by zero."""
    survivors = list(survivors)
    if not survivors:
        return float("nan")
    survset = set(survivors)
    seen = set(s for s in states if s in survset)
    return len(seen) / len(survivors)


def eliminated_mass(states, survivors, alphabet):
    """Fraction of ALL samples that commit to a state that is in the alphabet but
    NOT a survivor (a genuine calibration error: the true posterior puts zero
    mass there). Empty input -> 0.0."""
    if not states:
        return 0.0
    survset = set(survivors)
    alpha = set(alphabet)
    bad = sum(1 for s in states if s in alpha and s not in survset)
    return bad / len(states)


def abstain_rate(states):
    """Fraction of samples that hedge or commit to nothing ('hedge'/'none').
    'multi' (ambiguous multi-commit) is excluded. Empty input -> 0.0."""
    if not states:
        return 0.0
    return sum(1 for s in states if s in ("hedge", "none")) / len(states)


# ------------------------------------------------------------------- calibration
def calibration_tv(states, target, alphabet):
    """Total-variation distance between the empirical committed-state
    distribution and the known true posterior `target`.

    Only committed states enter the empirical distribution; abstentions are
    excluded (and reported separately by abstain_rate). `target` is a dict over
    (a subset of) `alphabet`; missing entries are treated as 0 mass. This is the
    headline calibration scalar (same math as divmetrics.gen_state_tv,
    generalized to a state alphabet).

    Returns nan when nothing was committed (the empirical distribution is then
    undefined) -- this is a legitimate value for a fully-abstaining cell, not an
    error. Callers guard for it.
    """
    cnt = {a: 0 for a in alphabet}
    n = 0
    for s in states:
        if s in cnt:
            cnt[s] += 1
            n += 1
    if n == 0:
        return float("nan")
    p = np.array([cnt[a] / n for a in alphabet])
    t = np.array([float(target.get(a, 0.0)) for a in alphabet])
    return 0.5 * float(np.abs(p - t).sum())


# ----------------------------------------------------------------------- bar/cmr
def bar(states):
    """Beam-duplicate / collapse rate = 1 - distinct/total over ALL labels
    (committed states and abstention markers alike: this measures surface
    state-collapse, not validity). 0.0 = all distinct, ->1 = total collapse.
    Empty input -> 0.0."""
    n = len(states)
    if n == 0:
        return 0.0
    return 1.0 - len(set(states)) / n


def cmr(states):
    """Collapse / merge rate, measured pre-pruning. Here the sample set IS the
    pre-pruning set, so this equals bar(states); kept as a named alias so the
    pre/post distinction stays explicit at call sites."""
    return bar(states)


def bar_cmr(states_before, states_after=None):
    """Return both the beam-duplicate rate on the pre-pruning set (`bar`) and the
    collapse/merge rate (`cmr`). When `states_after` is omitted, cmr is measured
    on the same pre-pruning set; otherwise cmr is measured on the post-prune set.
    """
    after = states_before if states_after is None else states_after
    return {"bar": bar(states_before), "cmr": cmr(after)}


# ------------------------------------------------------- Markov next-step JS / KL
def _union_pmf(p, q):
    """Align two next-step state dicts to their shared (union) alphabet and
    return two numpy pmfs (each normalized over the union; missing keys -> 0)."""
    keys = sorted(set(p) | set(q))
    pv = np.array([float(p.get(k, 0.0)) for k in keys])
    qv = np.array([float(q.get(k, 0.0)) for k in keys])
    ps, qs = pv.sum(), qv.sum()
    if ps <= 0 or qs <= 0:
        raise ValueError("markov distributions must have positive total mass")
    return pv / ps, qv / qs


def _kl(a, b, eps=1e-12):
    """KL(a || b) in nats over aligned pmfs; b is floored by eps so disjoint
    supports stay finite. Terms where a==0 contribute 0."""
    a = np.asarray(a, float)
    b = np.clip(np.asarray(b, float), eps, 1.0)
    m = a > 0
    return float((a[m] * np.log(a[m] / b[m])).sum())


def markov_js(p, q):
    """Jensen-Shannon divergence (nats) between two next-step state
    distributions given as dicts over a (possibly differing) state alphabet.
    Symmetric, >= 0, and == 0 iff the two distributions are identical. Maxes at
    ln 2 (~0.693) for disjoint supports. Missing keys are treated as 0 mass over
    the union alphabet. Used to test the Markov/aliasing precondition (two
    histories should map to the same canonical next-step state)."""
    pv, qv = _union_pmf(p, q)
    mix = 0.5 * (pv + qv)
    return 0.5 * _kl(pv, mix) + 0.5 * _kl(qv, mix)


def markov_kl(p, q):
    """KL(p || q) in nats between two next-step state distributions given as
    dicts. q is floored by a small eps so missing/zero-mass keys keep the value
    finite. == 0 iff identical, > 0 otherwise. Not symmetric (unlike
    markov_js)."""
    pv, qv = _union_pmf(p, q)
    return _kl(pv, qv)
