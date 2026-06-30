"""A LEARNED, imperfect solvability value V(s) for Countdown (PLAN4 next step).

The whole PLAN4 deflation rests on the verifier being an EXACT oracle (`reachable`). The
only experiment that can change the conclusion is to replace it with a *learned*,
imperfect value and ask whether a soft-λ verifier-weighted decode retains any edge.

This module is that value. It is GENERATOR-FREE and GPU-free: features are cheap state
statistics, labels are the EXACT `reachable` oracle, and the classifier is a small MLP.
Train and eval instances are disjoint by seed so V's error is genuine generalization, not
memorization — i.e. V is a realistic imperfect verifier whose false-negative / false-
positive rates place it on the T1.2 noise curve.

V deliberately uses NO multi-step lookahead (no partial solver baked in): only aggregate
statistics of the value multiset + target. So it is informative but imperfect, which is
exactly the point.
"""

import numpy as np

import core_boot as cb

_MAXK = 6  # multiset is padded to this many sorted values


def _floats(values):
    return [float(v) for v in values]


def state_features(state):
    """Fixed-length cheap feature vector for a Countdown ``State`` (no lookahead)."""
    vals = sorted(_floats(state.values), reverse=True)
    n = len(vals)
    target = float(state.target)
    s = sum(vals)
    mx = max(vals) if vals else 0.0
    mn = min(vals) if vals else 0.0
    mean = s / n if n else 0.0
    var = sum((v - mean) ** 2 for v in vals) / n if n else 0.0
    std = var ** 0.5
    # integer divisibility hints (only for integer-valued entries)
    int_vals = [v for v in state.values if v.denominator == 1]
    tnum = state.target.numerator if state.target.denominator == 1 else None
    n_div = sum(1 for v in int_vals
                if tnum is not None and v.numerator != 0 and tnum % v.numerator == 0)
    n_eq_target = sum(1 for v in vals if abs(v - target) < 1e-9)
    n_gt_target = sum(1 for v in vals if v > target)
    n_one = sum(1 for v in vals if abs(v - 1.0) < 1e-9)
    n_distinct = len(set(vals))
    # bounded log-product (a crude reachability upper-bound proxy)
    logprod = 0.0
    for v in vals:
        logprod += np.log(abs(v) + 1.0)
    feats = [
        float(n), target, s, mx, mn, mean, std,
        target - s, abs(target - s), (target - s) / max(n, 1),
        float(n_div), float(n_eq_target), float(n_gt_target), float(n_one),
        float(n_distinct), float(logprod),
    ]
    # sorted multiset, padded to _MAXK (descending; 0-padded)
    padded = (vals + [0.0] * _MAXK)[:_MAXK]
    feats.extend(padded)
    return np.asarray(feats, dtype=np.float64)


def _walk_states(domain, inst, rng, max_states):
    """Collect states along a random legal-op walk from an instance (covers the trellis
    state distribution: multisets of every size from k down to 1)."""
    s = domain.initial_state(cb.instance_dict(inst))
    out = [s]
    while len(s.values) > 1 and len(out) < max_states:
        ops = domain.legal_ops(s)
        if not ops:
            break
        s = domain.apply(s, ops[rng.randrange(len(ops))])
        out.append(s)
    return out


def make_dataset(seeds, ks, n_per, target_range=(10, 100), max_states=8, seed=0):
    """Return (X, y, states) of exact-labelled states from random walks (generator-free)."""
    import random
    domain = cb.CountdownDomain()
    rng = random.Random(seed)
    X, y, states = [], [], []
    for sd in seeds:
        for k in ks:
            for inst in cb.load_instances({"set": "generated", "seed": sd, "n": n_per,
                                           "k": k, "target_range": list(target_range)}):
                for st in _walk_states(domain, inst, rng, max_states):
                    X.append(state_features(st))
                    y.append(1 if cb.reachable(st) else 0)
                    states.append(st)
    return np.asarray(X), np.asarray(y), states


class ValueModel:
    """Learned V(s) = P(solvable) from cheap features. ``predict_proba(state) -> float``."""

    def __init__(self, hidden=(64, 32), max_iter=300, seed=0, capacity="mlp"):
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        if capacity == "logreg":
            self.clf = LogisticRegression(max_iter=1000, C=1.0)
        else:
            self.clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter,
                                     random_state=seed, early_stopping=True)
        self.capacity = capacity

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        return self

    def proba(self, X):
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)[:, 1]

    def predict_proba(self, state):
        return float(self.proba(state_features(state)[None, :])[0])

    def value_fn(self):
        """A closure ``V(state) -> float in (0,1)`` for the decode arm."""
        return lambda state: self.predict_proba(state)


def evaluate(model, X, y, n_bins=10):
    """Accuracy, AUC, ECE, and the false-neg/false-pos rates that place V on the T1.2
    curve. fn_rate = P(V<0.5 | truly solvable); fp_rate = P(V>=0.5 | truly unsolvable)."""
    from sklearn.metrics import roc_auc_score
    p = model.proba(X)
    pred = (p >= 0.5).astype(int)
    acc = float((pred == y).mean())
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    sol = y == 1
    uns = y == 0
    fn_rate = float((pred[sol] == 0).mean()) if sol.any() else float("nan")
    fp_rate = float((pred[uns] == 1).mean()) if uns.any() else float("nan")
    # ECE
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        if m.any():
            ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return {"acc": acc, "auc": auc, "ece": float(ece),
            "fn_rate": fn_rate, "fp_rate": fp_rate,
            "base_rate_solvable": float(sol.mean()), "n": int(len(y))}
