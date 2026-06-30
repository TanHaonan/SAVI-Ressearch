"""Tests for the integer-sum lattice domain, including the ORACLE VALIDATION.

The load-bearing test is ``test_oracle_matches_bruteforce``: for depths 1..7 it
brute-forces ALL move sequences and confirms the O(1) ``solvable((s, r))`` equals exact
goal-reachability for every reachable ``(s, r)``. ``oracle_validated`` in the structured
output is set only if this passes.
"""

import itertools

import pytest

import domain_lattice as dl
from decode_core import decode as dc


# ---------------------------------------------------------------------------
# Basic domain mechanics
# ---------------------------------------------------------------------------

def test_initial_state_and_canon():
    d = dl.LatticeDomain()
    s0 = d.initial_state({"s0": 0, "target": 7, "depth": 4})
    assert (s0.s, s0.r, s0.T) == (0, 4, 7)
    # canon includes T so a generator keyed by canon cannot leak classification across
    # instances that share (s, r) but differ in target (see test_lattice_canon_T_regression).
    assert d.canon(s0) == (0, 4, 7)
    assert s0.target == 7


def test_apply_and_goal():
    d = dl.LatticeDomain()
    s = d.initial_state({"s0": 0, "target": 6, "depth": 3})
    s = d.apply(s, 2)  # (2, 2)
    s = d.apply(s, 3)  # (5, 1)
    s = d.apply(s, 1)  # (6, 0)
    assert d.canon(s) == (6, 0, 6)
    assert d.is_goal(s)
    # Wrong sum at r=0 is not a goal.
    s2 = d.apply(d.apply(d.apply(d.initial_state(
        {"s0": 0, "target": 6, "depth": 3}), 1), 1), 1)  # (3, 0)
    assert not d.is_goal(s2)


def test_parse_move():
    d = dl.LatticeDomain()
    s = d.initial_state({"s0": 0, "target": 6, "depth": 3})
    assert d.parse_move("add 1", s) == 1
    assert d.parse_move("add 3", s) == 3
    assert d.parse_move("add 4", s) is None    # not in M
    assert d.parse_move("add 0", s) is None
    assert d.parse_move("sub 1", s) is None    # wrong op
    assert d.parse_move("garbage", s) is None
    assert d.parse_move("add", s) is None
    # r == 0 -> no legal move
    term = dl.LatticeState(s=6, r=0, T=6)
    assert d.parse_move("add 1", term) is None


def test_parse_chain():
    d = dl.LatticeDomain()
    s = d.initial_state({"s0": 0, "target": 6, "depth": 3})
    assert d.parse_chain("add 2;add 3;add 1", s) == [2, 3, 1]
    assert d.parse_chain("add 2 ; add 3 ; add 1", s) == [2, 3, 1]
    # Too many steps -> illegal (r runs out).
    assert d.parse_chain("add 1;add 1;add 1;add 1", s) is None
    # Bad token -> None.
    assert d.parse_chain("add 2;add 9", s) is None
    # Empty chain.
    assert d.parse_chain("", s) == []


def test_enumerate():
    s = dl.LatticeState(s=0, r=3, T=6)
    assert dl.lattice_enumerate(s) == ["add 1", "add 2", "add 3"]
    term = dl.LatticeState(s=6, r=0, T=6)
    assert dl.lattice_enumerate(term) == []


def test_render():
    d = dl.LatticeDomain()
    s = dl.LatticeState(s=2, r=1, T=6)
    assert "sum=2" in d.render(s) and "remaining=1" in d.render(s)


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def test_make_instances_solvable_and_stable():
    d = dl.LatticeDomain()
    for depth in (1, 4, 8, 16):
        insts = dl.make_lattice_instances(depth, 24, seed=1)
        assert len(insts) == 24
        for inst in insts:
            assert inst.depth == depth
            assert depth <= inst.target <= 3 * depth   # T in [D, 3D]
            s0 = d.initial_state(dl.lattice_inst_dict(inst))
            assert d.solvable(s0)                        # ceiling == 1 by construction
        # Stable ids + deterministic draw.
        again = dl.make_lattice_instances(depth, 24, seed=1)
        assert [i.id for i in insts] == [i.id for i in again]
        assert [i.target for i in insts] == [i.target for i in again]


def test_instances_seed_varies():
    a = dl.make_lattice_instances(8, 24, seed=1)
    b = dl.make_lattice_instances(8, 24, seed=2)
    # Different seed -> (very likely) a different target vector.
    assert [i.target for i in a] != [i.target for i in b]


# ---------------------------------------------------------------------------
# ORACLE VALIDATION: O(1) solvable == exact brute-force reachability
# ---------------------------------------------------------------------------

def _bruteforce_reachable_goals(depth, T):
    """Set of (s, r) states from which the goal is reachable, by exhaustive search.

    Explore the full reachable lattice from (0, D) over all move sequences; a state
    (s, r) is goal-reachable iff some completion of the remaining r moves hits sum T.
    Returns (reachable_states, goal_reachable_states).
    """
    reachable = set()
    # BFS over all reachable (s, r) from the start.
    frontier = {(0, depth)}
    reachable |= frontier
    while frontier:
        nxt = set()
        for (s, r) in frontier:
            if r <= 0:
                continue
            for v in dl.M:
                child = (s + v, r - 1)
                if child not in reachable:
                    nxt.add(child)
        reachable |= nxt
        frontier = nxt

    # goal_reachable: (s, r) can reach (T, 0). Compute backward via DP over r.
    goal_reachable = set()
    # r == 0 states reachable iff s == T.
    for (s, r) in reachable:
        if r == 0 and s == T:
            goal_reachable.add((s, r))
    # ascending r: (s, r) reachable-to-goal iff some (s+v, r-1) is.
    for r in range(1, depth + 1):
        for (s, rr) in reachable:
            if rr != r:
                continue
            if any((s + v, r - 1) in goal_reachable for v in dl.M):
                goal_reachable.add((s, r))
    return reachable, goal_reachable


def test_oracle_matches_bruteforce():
    """For depths 1..7, O(1) solvable((s, r)) == exact goal-reachability everywhere.

    This is the gate for ``oracle_validated``. We sweep several targets per depth
    (covering T in [D, 3D] and out-of-range T) and EVERY reachable (s, r) state.
    """
    d = dl.LatticeDomain()
    for depth in range(1, 8):
        # Cover in-range targets AND a couple out-of-range to test both branches.
        for T in range(0, 3 * depth + 2):
            reachable, goal_reachable = _bruteforce_reachable_goals(depth, T)
            for (s, r) in reachable:
                state = dl.LatticeState(s=s, r=r, T=T)
                oracle = d.solvable(state)
                truth = (s, r) in goal_reachable
                assert oracle == truth, (
                    f"depth={depth} T={T} state=({s},{r}): "
                    f"solvable={oracle} but bruteforce={truth}")


# ---------------------------------------------------------------------------
# The domain plugs into decode_core.savi end-to-end
# ---------------------------------------------------------------------------

def test_savi_solves_lattice_with_oracle_generator():
    """A trivial perfect generator (emits the minimal-step move) lets savi reach goal."""
    d = dl.LatticeDomain()
    insts = dl.make_lattice_instances(4, 6, seed=1)

    def perfect_sample(state, N, tau, seed, mode):
        # Emit only moves whose successor stays solvable (always exists for solvable s).
        legal = dl.lattice_enumerate(state)
        good = [t for t in legal
                if d.solvable(d.apply(state, d.parse_move(t, state)))]
        pool = good or legal
        if mode == "step":
            return [pool[0]] * N
        # chain
        chains = []
        for _ in range(N):
            cur = state
            toks = []
            while not d.is_goal(cur) and cur.r > 0:
                lg = dl.lattice_enumerate(cur)
                g = [t for t in lg
                     if d.solvable(d.apply(cur, d.parse_move(t, cur)))]
                t = (g or lg)[0]
                toks.append(t)
                cur = d.apply(cur, d.parse_move(t, cur))
            chains.append(";".join(toks))
        return chains

    for inst in insts:
        idict = dl.lattice_inst_dict(inst)
        res = dc.savi(d, perfect_sample, idict, 8, 8, "freq", 0.7, 1,
                      verifier=False, max_depth=inst.depth)
        assert res.ok, f"{inst.id} should be solved by the perfect generator"
