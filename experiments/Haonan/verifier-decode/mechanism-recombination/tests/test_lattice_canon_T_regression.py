"""Regression: the lattice canon must include T so a generator keyed by canon cannot
leak competence classification across instances with different targets.

This is the bug the PLAN2 Verify agents caught: ``LatticeDomain.canon`` returned
``(s, r)`` (omitting T), and ``gen_fair`` keys both its classification memo AND its RNG
seed by ``canon``. One generator shared across instances with different T but overlapping
``(s, r)`` therefore fixed the weights/seed of the first instance to touch an ``(s, r)``
onto every later instance under the WRONG target. Fixed by ``canon = (s, r, T)``.
"""

import functools
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gen_fair  # noqa: E402
import domain_lattice as L  # noqa: E402


def test_canon_distinguishes_target():
    """Two states equal in (s, r) but differing in T must have different canon."""
    D = L.LatticeDomain()
    a = D.initial_state({"s0": 0, "target": 6, "depth": 2})
    b = D.initial_state({"s0": 0, "target": 2, "depth": 2})
    assert D.canon(a) != D.canon(b)
    assert D.canon(a) == (0, 2, 6)
    assert D.canon(b) == (0, 2, 2)


def test_shared_generator_does_not_leak_classification_across_T():
    """One shared generator, two instances, same (s, r) start, different T.

    At start (0, 2): with T=6 the only solvable-preserving move is ``add 3`` (residual 6
    must land in [1, 3] after one step); with T=2 it is ``add 1``. The argmax (tau=0) move
    must therefore DIFFER per instance. Under the old canon=(s,r) bug, the shared memo/seed
    would force both to the move of whichever instance was sampled first.
    """
    D = L.LatticeDomain()
    gen = gen_fair.make_fair_generator(
        D, functools.partial(L.lattice_enumerate), p=0.99, depth_cap=2)

    a0 = D.initial_state({"s0": 0, "target": 6, "depth": 2})
    b0 = D.initial_state({"s0": 0, "target": 2, "depth": 2})

    # Touch A first (this is what populated the contaminated memo in the bug).
    move_a = gen(a0, 1, 0.0, 0, "step")[0]
    move_b = gen(b0, 1, 0.0, 0, "step")[0]

    assert move_a == "add 3", move_a
    assert move_b == "add 1", move_b
    assert move_a != move_b  # the leak would have made these equal
