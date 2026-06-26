"""SemanticGraph: a thin factor-graph wrapper over the verified chain Instance in oracle.py.

A semantic graph is a chain x_0..x_{T-1}, each variable in a domain of size n, with:
  - unary log-potentials theta[t, s]                       (per-position scores)
  - pairwise log-potentials adj[t][s, s'] for edge (t,t+1) (transition constraints)
  - long-range "hub" equality edges (0, j, M)              (i==0, single hub)

Hard constraints are encoded as large negative log-potentials (HARD_PEN), exactly
as in oracle.py, so the verified exact solver and decoders run unchanged on it.
This module only ASSEMBLES the potentials; all solving is delegated to oracle.py.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

# oracle.py (the verified chain core) is vendored alongside this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import oracle  # noqa: E402

HARD_PEN = oracle.HARD_PEN


@dataclass
class SemanticGraph:
    T: int
    n: int
    theta: np.ndarray            # (T, n) unary log-potentials
    adj: np.ndarray              # (T-1, n, n) pairwise log-potentials for edge (t,t+1)
    longrange: list = field(default_factory=list)  # list of (0, j, M), M (n,n)

    def to_instance(self) -> "oracle.Instance":
        """Wrap as a verified chain Instance (validated)."""
        inst = oracle.Instance(
            T=self.T,
            n=self.n,
            theta=np.asarray(self.theta, dtype=float),
            adj=np.asarray(self.adj, dtype=float),
            longrange=[(int(i), int(j), np.asarray(M, dtype=float))
                       for (i, j, M) in self.longrange],
        )
        return inst.validate()

    def violations(self, x) -> bool:
        """True if assignment x (len T) violates ANY hard factor."""
        return not oracle.feasible(self.to_instance(), list(x))

    def violation_count(self, x) -> int:
        """Number of hard factors violated by x (log-potential <= HARD_PEN/2)."""
        x = list(x)
        inst = self.to_instance()
        tol = HARD_PEN / 2
        c = 0
        for t in range(inst.T):
            if inst.theta[t, x[t]] <= tol:
                c += 1
        for t in range(inst.T - 1):
            if inst.adj[t, x[t], x[t + 1]] <= tol:
                c += 1
        for (i, j, M) in inst.longrange:
            if M[x[i], x[j]] <= tol:
                c += 1
        return c


def chain_must_differ(n: int) -> np.ndarray:
    """(n,n) pairwise: 0 off-diagonal, HARD_PEN on diagonal (adjacent must differ)."""
    M = np.zeros((n, n), dtype=float)
    np.fill_diagonal(M, HARD_PEN)
    return M


def equality_M(n: int) -> np.ndarray:
    """(n,n) long-range: 0 on diagonal, HARD_PEN off-diagonal (hub must equal target)."""
    M = np.full((n, n), HARD_PEN, dtype=float)
    np.fill_diagonal(M, 0.0)
    return M


def make_graph(T: int, n: int, theta, mustdiffer: bool = True, hub_targets=()) -> SemanticGraph:
    """Assemble a SemanticGraph.

    adj: must-differ chain if mustdiffer else all-zero (no transition constraint).
    longrange: equality hub edges (0, j) for each j in hub_targets.
    """
    theta = np.asarray(theta, dtype=float)
    assert theta.shape == (T, n), f"theta shape {theta.shape} != {(T, n)}"

    if mustdiffer:
        md = chain_must_differ(n)
        adj = np.tile(md, (T - 1, 1, 1))
    else:
        adj = np.zeros((T - 1, n, n), dtype=float)

    eqM = equality_M(n)
    longrange = []
    for j in hub_targets:
        assert 0 < j < T, f"hub target j must satisfy 0<j<T: got {j}"
        longrange.append((0, int(j), eqM.copy()))

    return SemanticGraph(T=T, n=n, theta=theta, adj=adj, longrange=longrange)
