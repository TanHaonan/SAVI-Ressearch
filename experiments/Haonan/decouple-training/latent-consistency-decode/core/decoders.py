"""Assemble node emission (unary) + pairwise edges into a factor graph and decode three ways:
  greedy   : per-slot argmax of the unary; ignores edges (the local read-out baseline).
  marginal : sum-product max-posterior-marginal -- per-slot argmax of the exact posterior
             marginal UNDER the hard edges (constrained MPM, NOT an unconstrained argmax).
  viterbi  : max-product / exact MAP under the hard edges.

The carrier's generator only ever produces edges that are adjacent (folded into the chain `adj`)
or hub edges with i==0 (folded into `longrange`), so the verified oracle primitives apply:
  - viterbi   -> oracle.exact  (hub-conditioned chain DP) when every long-range edge is a hub (i==0);
  - marginal  -> exact enumeration via oracle.energy (T,k tiny -> cheap and exact).
For safety, if a non-hub general long-range pair ever appears (i>0, non-adjacent), both decoders
fall back to brute force computed directly from oracle.energy (which does not require validate()),
so they never crash and stay exact. decode_viterbi is pinned == brute-force MAP by the tests.
"""
import itertools, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DEPS = HERE / "_deps"
# oracle.py and graph.py are vendored in _deps/.
if str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))
import oracle                       # Instance, exact, brute_force, energy, HARD_PEN
import graph as SG                  # SemanticGraph


def assemble(theta, edges, k):
    """theta: (T,k) unary log-potentials (= node-emission log-probs). edges: [(i,j,(k,k) M)].
    Adjacent (j==i+1) edges fold into the chain `adj`; all others go to `longrange`."""
    theta = np.asarray(theta, float)
    T = theta.shape[0]
    adj = np.zeros((T - 1, k, k), float)
    longrange = []
    for (i, j, M) in edges:
        i, j = int(i), int(j)
        M = np.asarray(M, float)
        if j == i + 1:
            adj[i] = adj[i] + M
        else:
            longrange.append((i, j, M))
    return SG.SemanticGraph(T=T, n=k, theta=theta, adj=adj, longrange=longrange)


def decode_greedy(graph):
    """Per-slot argmax of the unary; ignores edges entirely (the local read-out baseline)."""
    return [int(v) for v in np.argmax(np.asarray(graph.theta, float), axis=1)]


def _raw_instance(graph):
    """Build an oracle.Instance WITHOUT validate() so general (non-hub) pairs don't raise.
    energy()/feasible() index theta/adj/longrange directly and need no hub constraint."""
    return oracle.Instance(
        T=graph.T,
        n=graph.n,
        theta=np.asarray(graph.theta, float),
        adj=np.asarray(graph.adj, float),
        longrange=[(int(i), int(j), np.asarray(M, float)) for (i, j, M) in graph.longrange],
    )


def _is_chain_hub(inst):
    """True iff every long-range edge is a hub edge (i==0) -> oracle.exact applies."""
    return all(i == 0 for (i, j, _M) in inst.longrange)


def _brute_map(inst):
    """Exact MAP by enumeration via oracle.energy (no validate(); works for general pairs too)."""
    T, k = inst.T, inst.n
    best_E, best_x = -np.inf, None
    for x in itertools.product(range(k), repeat=T):
        e = oracle.energy(inst, list(x))
        if e > best_E:
            best_E, best_x = e, list(x)
    return best_x


def decode_viterbi(graph):
    """Max-product / exact MAP under the hard edges. Returns list[int].
    Hub-chain family -> oracle.exact (verified hub-conditioned DP); else brute force."""
    inst = _raw_instance(graph)
    if _is_chain_hub(inst):
        return [int(v) for v in oracle.exact(inst.validate())["map"]]
    return [int(v) for v in _brute_map(inst)]


def _brute_marginals(inst):
    """Exact per-variable posterior marginals (log domain) by enumeration via oracle.energy.
    Returns (T,k) log posterior marginals. Constrained: the hard pairwise/long-range factors are
    inside energy(), so this is the MPM UNDER the edges, not an unconstrained per-slot argmax."""
    T, k = inst.T, inst.n
    margs = np.full((T, k), -np.inf)
    logZ = -np.inf
    for x in itertools.product(range(k), repeat=T):
        e = oracle.energy(inst, list(x))
        logZ = np.logaddexp(logZ, e)
        for t in range(T):
            margs[t, x[t]] = np.logaddexp(margs[t, x[t]], e)
    return margs - logZ


def decode_marginal(graph):
    """Sum-product max-posterior-marginal: per-slot argmax of the exact posterior marginal
    UNDER the hard edges. Exact enumeration (T,k tiny); this is the test oracle for the MPM."""
    inst = _raw_instance(graph)
    margs = _brute_marginals(inst)
    return [int(np.argmax(margs[t])) for t in range(inst.T)]


def brute_mpm(graph):
    """Independent test oracle for the constrained max-posterior-marginal.

    Deliberately NOT a call to decode_marginal / _brute_marginals: a second,
    structurally-different implementation of the same quantity so the cross-check
    test can catch a future MPM regression. Accumulates posterior mass per
    (slot, value) in the PROBABILITY domain via exp(oracle.energy(inst, x)) over
    all k^T assignments, normalizes per slot, returns the per-slot argmax.
    Uses _raw_instance so non-hub general pairs don't crash."""
    inst = _raw_instance(graph)
    T, k = inst.T, inst.n
    mass = np.zeros((T, k), float)
    for x in itertools.product(range(k), repeat=T):
        w = np.exp(oracle.energy(inst, list(x)))
        for t in range(T):
            mass[t, x[t]] += w
    # normalize per slot (each slot's mass sums to the same Z; argmax is invariant,
    # but normalize anyway for a clean posterior) and take the per-slot argmax.
    row_sums = mass.sum(axis=1, keepdims=True)
    post = mass / np.where(row_sums > 0, row_sums, 1.0)
    return [int(np.argmax(post[t])) for t in range(T)]
