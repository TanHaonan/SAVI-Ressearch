"""Learned GRADED value V(s)=P(solvable) for the lattice — a realistic imperfect verifier.

Unlike the binary 0.95/0.05 oracle-corruption in ``domain_merge`` (which the adversary
flagged as near-oracle), this is a **capacity-limited** classifier trained on EXACT solvable
labels over instances drawn from DISJOINT seeds. Logistic regression cannot represent the
two-sided feasibility interval ``r <= T-s <= 3r`` (an AND of two half-planes), so it makes
**systematic graded errors near the boundary** — a genuinely imperfect, graded value rather
than a near-oracle. Train/eval seed-disjointness makes its error true generalization;
multiple train seeds give independent noise realizations (kills the "single noise_seed"
caveat). ``value_fn()`` returns a closure ``V(state) -> (0,1)`` for ``savi_value``.
"""

from __future__ import annotations

import numpy as np

import domain_merge as dm


def state_features(state):
    """Partial features (raw + cheap ratios). Deliberately NOT the exact interval
    indicators, so a linear model cannot trivially reconstruct ``r<=T-s<=3r``."""
    s, r, T = float(state.s), float(state.r), float(state.T)
    residual = T - s
    return np.array([
        s, r, T, residual,
        residual / max(r, 1.0),
        s / max(T, 1.0),
        r * r, s * r, residual * residual,
    ], dtype=np.float64)


def make_dataset(seeds, depths, n_per):
    """Labeled lattice states (X, y) = (features, exact solvable) over the reachable grid.

    For each instance (depth D, target T) enumerate every reachable node ``(s, r)`` —
    ``r in [0, D]``, ``s in [0, 3*(D-r)]`` — and label it with the exact oracle. This is the
    state distribution the decoder actually visits.
    """
    dom = dm.MergeLatticeDomain(0)
    X, y = [], []
    for sd in seeds:
        for D in depths:
            for inst in dm.make_lattice_instances(D, n_per, sd):
                T = int(inst.target)
                for r in range(0, D + 1):
                    used = D - r
                    for s in range(0, 3 * used + 1):
                        st = dm.MergeLatticeState(s=s, r=r, T=T)
                        X.append(state_features(st))
                        y.append(1 if dom.solvable(st) else 0)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


class LatticeValueModel:
    """Graded ``V(state)=P(solvable)`` from a capacity-limited classifier.

    ``capacity='logreg'`` (default) cannot represent the two-sided interval -> genuinely
    imperfect, graded. ``capacity='mlp'`` is a stronger (near-oracle) baseline for contrast.
    """

    def __init__(self, capacity="logreg", seed=0):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.capacity = capacity
        if capacity == "logreg":
            from sklearn.linear_model import LogisticRegression
            self.clf = LogisticRegression(max_iter=2000, C=1.0)
        elif capacity == "mlp":
            from sklearn.neural_network import MLPClassifier
            self.clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400,
                                     random_state=seed, early_stopping=True)
        else:
            raise ValueError(f"unknown capacity {capacity!r}")

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        return self

    def proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    def predict_proba(self, state):
        return float(self.proba(state_features(state)[None, :])[0])

    def value_fn(self):
        return lambda state: self.predict_proba(state)


def evaluate(model, X, y, n_bins=10):
    """acc / auc / ece / fn_rate / fp_rate on a (disjoint) eval set."""
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
    fn = float((pred[sol] == 0).mean()) if sol.any() else float("nan")
    fp = float((pred[uns] == 1).mean()) if uns.any() else float("nan")
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"acc": acc, "auc": auc, "ece": float(ece), "fn_rate": fn, "fp_rate": fp,
            "base_rate_solvable": float(sol.mean()), "n": int(len(y)),
            "p_min": float(p.min()), "p_max": float(p.max()),
            "p_frac_midband": float(((p > 0.2) & (p < 0.8)).mean())}
