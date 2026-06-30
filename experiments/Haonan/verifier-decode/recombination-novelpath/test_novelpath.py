"""Unit tests for the novel-path / recombination metrics."""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPS = os.path.join(_HERE, "_deps")   # vendored domain_merge / domain_lattice / ...
for _p in (_DEPS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import novelpath as NP          # noqa: E402
import domain_merge as dm       # noqa: E402

D0 = dm.MergeLatticeDomain(0)


def S(s, r, T):
    return dm.MergeLatticeState(s=s, r=r, T=T)


def test_pp_probs_mixed_and_uniform():
    # state (0,1,2): add2 -> goal (good), add1/add3 -> unsolvable (bad)
    st = S(0, 1, 2)
    pr7 = NP.pp_probs(D0, st, 0.7)
    assert abs(sum(pr7.values()) - 1.0) < 1e-9
    assert pr7[2] > pr7[1] and pr7[2] > pr7[3]          # good move favored
    assert abs(pr7[2] - 0.7 / 1.3) < 1e-9
    pr5 = NP.pp_probs(D0, st, 0.5)
    assert all(abs(v - 1 / 3) < 1e-9 for v in pr5.values())  # p=0.5 -> uniform


def test_q_of_path_is_product():
    st = S(0, 1, 2)
    q = NP.q_of_path(D0, st, (2,), 0.7)
    assert abs(q - 0.7 / 1.3) < 1e-9
    # two-step joint
    q2 = NP.q_of_path(D0, S(0, 2, 4), (2, 2), 0.6)
    man = NP.pp_probs(D0, S(0, 2, 4), 0.6)[2] * NP.pp_probs(D0, S(2, 1, 4), 0.6)[2]
    assert abs(q2 - man) < 1e-12


def test_path_edges_and_reaches_goal():
    s0 = S(0, 3, 6)
    assert NP.reaches_goal(D0, s0, (1, 2, 3))
    assert not NP.reaches_goal(D0, s0, (1, 1, 1))
    assert NP.path_edges(D0, s0, (1, 2, 3)) == [((0, 3), 1), ((1, 2), 2), ((3, 1), 3)]


def test_stitch_genuine_recombination():
    # (3,1) is reachable by prefix (1,2) AND (2,1); decode path (1,2,3) is the
    # prefix of chain A=(1,2,1) glued to the last edge of chain B=(2,1,3) which
    # reached (3,1) via a DIFFERENT prefix. No single chain is (1,2,3).
    s0 = S(0, 3, 6)
    chains = [(1, 2, 1), (2, 1, 3)]
    info = NP.analyze(D0, s0, (1, 2, 3), chains)
    assert info["novel"] and info["all_edges_covered"] and info["stitch"]


def test_not_stitch_when_an_edge_uncovered():
    s0 = S(0, 3, 6)
    chains = [(1, 2, 1)]               # covers ((0,3),1),((1,2),2) but not ((3,1),3)
    info = NP.analyze(D0, s0, (1, 2, 3), chains)
    assert info["novel"] and not info["all_edges_covered"] and not info["stitch"]


def test_solve_mass_matches_manual():
    # (0,1,2): only add2 reaches goal -> mass = P_p(2) = 0.7/1.3
    assert abs(NP.solve_mass(D0, S(0, 1, 2), 0.7) - 0.7 / 1.3) < 1e-9
    # 2-step (0,2,4): correct paths (1,3),(2,2),(3,1); mass = sum of their joint probs
    s0 = S(0, 2, 4)
    man = (NP.q_of_path(D0, s0, (1, 3), 0.6) + NP.q_of_path(D0, s0, (2, 2), 0.6)
           + NP.q_of_path(D0, s0, (3, 1), 0.6))
    assert abs(NP.solve_mass(D0, s0, 0.6) - man) < 1e-9
    assert 0.0 <= NP.solve_mass(D0, S(0, 24, 48), 0.5) <= 1.0


def test_not_novel_when_path_sampled():
    s0 = S(0, 3, 6)
    chains = [(1, 2, 3), (2, 1, 3)]
    info = NP.analyze(D0, s0, (1, 2, 3), chains)
    assert not info["novel"] and not info["stitch"]
