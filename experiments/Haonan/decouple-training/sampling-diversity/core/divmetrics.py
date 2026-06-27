# sampling-diversity/core/divmetrics.py
"""Diversity-after-Phi metrics. All operate on a list of state labels (option letters or
'hedge'/'multi'/'none'). Deterministic, model-free."""
import numpy as np

def _valid(states, letters):
    return [s for s in states if s in letters]

def k_eff_distinct(states, letters):
    """Number of DISTINCT option-states committed to (excludes hedge/multi/none)."""
    return len(set(_valid(states, letters)))

def k_eff_entropy(states, letters):
    """Effective number of states = exp(entropy of the committed-option distribution)."""
    vs = _valid(states, letters)
    if not vs:
        return 0.0
    _, c = np.unique(vs, return_counts=True); p = c / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def coverage(states, survivors):
    """Fraction of the j true survivors that appear at least once among the samples."""
    seen = set(s for s in states if s in survivors)
    return len(seen) / len(survivors)

def gen_state_tv(states, target, letters):
    """TV between the empirical committed-option distribution and the true posterior `target`.
    Only committed-option samples enter the empirical distribution; hedge/multi/none are excluded
    (reported separately by abstain_rate). Returns nan if nothing was committed."""
    cnt = {L: 0 for L in letters}; n = 0
    for s in states:
        if s in cnt:
            cnt[s] += 1; n += 1
    if n == 0:
        return float("nan")
    p = np.array([cnt[L] / n for L in letters]); t = np.array([target[L] for L in letters])
    return 0.5 * float(np.abs(p - t).sum())

def eliminated_mass(states, survivors, letters):
    """Fraction of ALL samples that commit to an ELIMINATED option (a real calibration error)."""
    return sum(1 for s in states if s in letters and s not in survivors) / len(states)

def abstain_rate(states):
    """Fraction of samples that hedge or commit to nothing (hedge/none); 'multi' excluded."""
    return sum(1 for s in states if s in ("hedge", "none")) / len(states)

def distinct_text(texts):
    """Raw surface-form distinctness (paraphrase control): distinct strings / token-diversity proxy."""
    return len(set(t.strip().lower() for t in texts))
