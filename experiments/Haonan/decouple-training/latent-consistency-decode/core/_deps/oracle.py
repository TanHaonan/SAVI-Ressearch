"""Symbolic exact oracle for long-chain factor graphs.

Factor graph on a chain x_0..x_{T-1}, each x_t in domain of size n:
  E(x) = sum_t theta[t, x_t]                      (unary / emission)
       + sum_t adj[t][x_t, x_{t+1}]               (nearest-neighbour pairwise)
       + sum over longrange edges (i,j,M): M[x_i, x_j]

All "hard" constraints are encoded as large negative log-potentials (HARD_PEN),
not -inf, so energies stay finite and comparable; feasibility = no HARD_PEN paid.

Exactness: we require every long-range edge to have i == 0 (a single hub at the
first position). Then conditioning on x_0 = a turns the problem into a pure
linear chain (the hub edges fold into unaries), which Viterbi / forward-backward
solve exactly. MAP / logZ / marginals are combined over the n hub values.
A brute_force() enumerator cross-checks this for small instances (tests/).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import itertools

HARD_PEN = -1.0e4  # log-potential for a forbidden configuration


def logsumexp(a, axis=None):
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else float(out)


@dataclass
class Instance:
    T: int
    n: int                       # domain size |S|
    theta: np.ndarray            # (T, n) unary log-potentials
    adj: np.ndarray              # (T-1, n, n) pairwise for edge (t, t+1)
    longrange: list              # list of (i, j, M) with i==0, j>0, M shape (n,n)
    meta: dict = None            # generation provenance (pref values, knobs, ...)

    def validate(self):
        assert self.theta.shape == (self.T, self.n)
        assert self.adj.shape == (self.T - 1, self.n, self.n)
        for (i, j, M) in self.longrange:
            assert i == 0 and 0 < j < self.T, f"hub edge must be (0, j): got ({i},{j})"
            assert M.shape == (self.n, self.n)
        return self


# ---------------------------------------------------------------------------
# core energy
# ---------------------------------------------------------------------------
def energy(inst: Instance, x) -> float:
    x = list(x)
    e = float(sum(inst.theta[t, x[t]] for t in range(inst.T)))
    for t in range(inst.T - 1):
        e += float(inst.adj[t, x[t], x[t + 1]])
    for (i, j, M) in inst.longrange:
        e += float(M[x[i], x[j]])
    return e


def feasible(inst: Instance, x, tol=HARD_PEN / 2) -> bool:
    """No hard constraint paid (energy contributions from any factor > tol)."""
    for t in range(inst.T):
        if inst.theta[t, x[t]] <= tol:
            return False
    for t in range(inst.T - 1):
        if inst.adj[t, x[t], x[t + 1]] <= tol:
            return False
    for (i, j, M) in inst.longrange:
        if M[x[i], x[j]] <= tol:
            return False
    return True


# ---------------------------------------------------------------------------
# pure linear-chain primitives (given unaries u (T,n) and adj (T-1,n,n))
# ---------------------------------------------------------------------------
def _chain_viterbi(u, adj):
    T, n = u.shape
    delta = u[0].copy()
    bp = np.zeros((T, n), dtype=int)
    for t in range(1, T):
        # score[s', s] = delta[s'] + adj[t-1][s', s]
        score = delta[:, None] + adj[t - 1]
        bp[t] = np.argmax(score, axis=0)
        delta = u[t] + np.max(score, axis=0)
    x = np.zeros(T, dtype=int)
    x[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        x[t - 1] = bp[t, x[t]]
    return float(np.max(delta)), x


def _chain_fb(u, adj):
    """Returns (logZ, marginals (T,n)) for the linear chain."""
    T, n = u.shape
    alpha = np.zeros((T, n))
    alpha[0] = u[0]
    for t in range(1, T):
        # alpha[t,s] = u[t,s] + logsumexp_{s'} alpha[t-1,s'] + adj[t-1][s',s]
        alpha[t] = u[t] + logsumexp(alpha[t - 1][:, None] + adj[t - 1], axis=0)
    beta = np.zeros((T, n))
    for t in range(T - 2, -1, -1):
        # beta[t,s] = logsumexp_{s'} adj[t][s,s'] + u[t+1,s'] + beta[t+1,s']
        beta[t] = logsumexp(adj[t] + (u[t + 1] + beta[t + 1])[None, :], axis=1)
    logZ = logsumexp(alpha[-1])
    logmarg = alpha + beta - logZ
    return float(logZ), np.exp(logmarg)


# ---------------------------------------------------------------------------
# exact solver via condition-on-hub (x_0)
# ---------------------------------------------------------------------------
def exact(inst: Instance):
    inst.validate()
    T, n = inst.T, inst.n
    best_E, best_x = -np.inf, None
    logZ_a = np.full(n, -np.inf)
    marg_a = np.zeros((n, T, n))
    for a in range(n):
        u = inst.theta.copy()
        pin = np.full(n, -np.inf)
        pin[a] = inst.theta[0, a]
        u[0] = pin
        for (i, j, M) in inst.longrange:
            u[j] = u[j] + M[a, :]
        E_a, x_a = _chain_viterbi(u, inst.adj)
        if E_a > best_E:
            best_E, best_x = E_a, x_a
        logZ_a[a], marg_a[a] = _chain_fb(u, inst.adj)
    logZ = logsumexp(logZ_a)
    w = np.exp(logZ_a - logZ)            # p(x_0 = a)
    marginals = np.tensordot(w, marg_a, axes=(0, 0))  # (T, n)
    map_x = best_x.tolist()
    return {
        "map": map_x,
        "map_energy": float(best_E),
        "logZ": float(logZ),
        "marginals": marginals,
        "maxmarg": [int(np.argmax(marginals[t])) for t in range(T)],
        "feasible_map": feasible(inst, map_x),
    }


# ---------------------------------------------------------------------------
# brute force (tests / ground-truth cross-check; small instances only)
# ---------------------------------------------------------------------------
def brute_force(inst: Instance):
    inst.validate()
    T, n = inst.T, inst.n
    best_E, best_x = -np.inf, None
    energies, assigns = [], []
    for x in itertools.product(range(n), repeat=T):
        e = energy(inst, x)
        energies.append(e)
        assigns.append(x)
        if e > best_E:
            best_E, best_x = e, x
    energies = np.array(energies)
    logZ = logsumexp(energies)
    # marginals
    marg = np.zeros((T, n))
    p = np.exp(energies - logZ)
    for idx, x in enumerate(assigns):
        for t in range(T):
            marg[t, x[t]] += p[idx]
    return {
        "map": list(best_x),
        "map_energy": float(best_E),
        "logZ": float(logZ),
        "marginals": marg,
        "maxmarg": [int(np.argmax(marg[t])) for t in range(T)],
        "feasible_map": feasible(inst, best_x),
    }


# ---------------------------------------------------------------------------
# greedy & beam decoders over the chain (the things DP is meant to beat)
# ---------------------------------------------------------------------------
def greedy(inst: Instance):
    """Left-to-right argmax. At position t includes already-resolved factors:
    the chain edge from t-1 and any long-range edge (i,t) with i<t already set.
    Cannot see future unaries or future long-range partners -> reverse-bite."""
    T = inst.T
    x = []
    for t in range(T):
        score = inst.theta[t].copy()
        if t > 0:
            score = score + inst.adj[t - 1, x[t - 1], :]
        for (i, j, M) in inst.longrange:
            if j == t and i < t:
                score = score + M[x[i], :]
        x.append(int(np.argmax(score)))
    return x


def beam(inst: Instance, width: int):
    """Beam search scoring all factors fully contained in the partial prefix."""
    T, n = inst.T, inst.n
    # each hyp: (partial_score, assignment list)
    beams = [(0.0, [])]
    for t in range(T):
        cand = []
        for sc, asg in beams:
            for s in range(n):
                add = inst.theta[t, s]
                if t > 0:
                    add += inst.adj[t - 1, asg[t - 1], s]
                for (i, j, M) in inst.longrange:
                    if j == t and i < t:
                        add += M[asg[i], s]
                cand.append((sc + add, asg + [s]))
        cand.sort(key=lambda z: z[0], reverse=True)
        beams = cand[:width]
    return beams[0][1]
