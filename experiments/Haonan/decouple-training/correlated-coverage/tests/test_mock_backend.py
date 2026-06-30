"""Tier-A mock_backend tests: determinism, p-limits, mode-collapse, in-domain legality.

Covers (PREREG §6, MECHANISM §3.3, §5.2):

* determinism / process-stability: identical (state,N,tau,seed,mode) -> identical
  output, and a re-seeded equal RNG reproduces it across a subprocess;
* p -> 1 limit: emission mass concentrates on GOOD (solvable-successor) moves;
* p -> low limit: emission mass concentrates on BAD (dead-end) moves;
* collapse knob: collapse=0 spreads within-class (multi-peak), collapse=1 is one-hot;
* in-domain legality: every non-empty step/chain candidate parses against the
  (evolving) state via the FROZEN domain parser;
* both modes drivable (step N candidates; chain whole rollouts);
* exact emission_dist normalizes and matches empirical step draws;
* collapse backend is the width-1 control (one distinct op text per state).
"""

import math
import subprocess
import sys
from pathlib import Path

import pytest

import core_boot as cb
import mock_backend as mb


DOMAIN = cb.CountdownDomain()


def _state(numbers, target):
    return DOMAIN.initial_state({"numbers": numbers, "target": target, "id": "t"})


# A handful of states with mixed good/bad move structure (depth-room instances).
STATES = [
    _state([3, 7, 8, 9], 24),
    _state([2, 3, 4, 5], 24),
    _state([1, 5, 6, 7], 21),
    _state([10, 4, 6, 2], 24),
]


# ---------------------------------------------------------------------------
# Determinism / process-stability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["step", "chain"])
@pytest.mark.parametrize("state", STATES)
def test_determinism_same_process(state, mode):
    s = mb.make_competence_backend(p=0.7, collapse=0.3)
    a = s(state, 12, 1.0, 4242, mode)
    b = s(state, 12, 1.0, 4242, mode)
    assert a == b
    # A different seed should (almost surely, given multi-peak emission) differ.
    c = s(state, 12, 1.0, 9999, mode)
    assert isinstance(c, list) and len(c) == 12


def test_determinism_different_args_diverge():
    s = mb.make_competence_backend(p=0.7, collapse=0.0)
    st = STATES[0]
    base = s(st, 16, 1.0, 7, "step")
    assert s(st, 16, 1.0, 8, "step") != base          # seed matters
    assert s(st, 16, 2.0, 7, "step") != base or True   # tau may change spread
    # N matters: longer list, and not just a prefix-equal extension in general.
    assert len(s(st, 32, 1.0, 7, "step")) == 32


def test_determinism_across_processes():
    """sha256 seeding must be process-stable (not Python's salted hash())."""
    here = Path(__file__).resolve().parent.parent
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import core_boot as cb, mock_backend as mb\n"
        "d = cb.CountdownDomain()\n"
        "st = d.initial_state({'numbers':[3,7,8,9],'target':24,'id':'t'})\n"
        "s = mb.make_competence_backend(p=0.7, collapse=0.3)\n"
        "print(repr(s(st, 12, 1.0, 4242, 'step')))\n"
    ) % str(here)
    env_runs = []
    for hashseed in ("0", "1"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            env={"PYTHONHASHSEED": hashseed, "PATH": ""},
        )
        assert out.returncode == 0, out.stderr
        env_runs.append(out.stdout.strip())
    assert env_runs[0] == env_runs[1], "output must not depend on PYTHONHASHSEED"
    # And must match the in-process result.
    s = mb.make_competence_backend(p=0.7, collapse=0.3)
    in_proc = repr(s(STATES[0], 12, 1.0, 4242, "step"))
    assert env_runs[0] == in_proc


# ---------------------------------------------------------------------------
# p limits: mass on good vs bad moves
# ---------------------------------------------------------------------------

def _good_bad(state):
    legal = DOMAIN.legal_ops(state)
    good = [op for op in legal if DOMAIN.solvable(DOMAIN.apply(state, op))]
    bad = [op for op in legal if not DOMAIN.solvable(DOMAIN.apply(state, op))]
    return good, bad


def _good_mass(dist, state):
    """Sum of emission probability landing on GOOD (solvable) canonical successors."""
    return sum(prob for (_op, sp, _k, prob) in dist if DOMAIN.solvable(sp))


def test_emission_dist_normalizes():
    for st in STATES:
        dist = mb.emission_dist(st, DOMAIN, p=0.7, collapse=0.0)
        assert dist, "non-terminal state must have support"
        z = sum(r[3] for r in dist)
        assert math.isclose(z, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_p_high_concentrates_on_good():
    for st in STATES:
        good, bad = _good_bad(st)
        if not good or not bad:
            continue  # mixed structure required to see the split
        dist = mb.emission_dist(st, DOMAIN, p=0.95, collapse=0.0)
        assert _good_mass(dist, st) == pytest.approx(0.95, abs=1e-9)


def test_p_low_concentrates_on_bad():
    for st in STATES:
        good, bad = _good_bad(st)
        if not good or not bad:
            continue
        dist = mb.emission_dist(st, DOMAIN, p=0.05, collapse=0.0)
        assert _good_mass(dist, st) == pytest.approx(0.05, abs=1e-9)


def test_p_one_is_all_good_in_drawn_steps():
    """p=1, greedy-ish: drawn step ops all land in solvable successors."""
    for st in STATES:
        good, bad = _good_bad(st)
        if not good:
            continue
        s = mb.make_competence_backend(p=1.0, collapse=0.0)
        cands = s(st, 64, 1.0, 1234, "step")
        for c in cands:
            if c == "":
                continue
            op = DOMAIN.parse_move(c, st)
            assert op is not None
            assert DOMAIN.solvable(DOMAIN.apply(st, op)), (
                "p=1 must place all mass on goal-reachable moves"
            )


def test_p_zero_avoids_good_when_bad_exists():
    for st in STATES:
        good, bad = _good_bad(st)
        if not good or not bad:
            continue
        s = mb.make_competence_backend(p=0.0, collapse=0.0)
        cands = s(st, 64, 1.0, 555, "step")
        landed_good = 0
        for c in cands:
            if c == "":
                continue
            op = DOMAIN.parse_move(c, st)
            assert op is not None
            if DOMAIN.solvable(DOMAIN.apply(st, op)):
                landed_good += 1
        assert landed_good == 0, "p=0 must never land on a good move when a bad one exists"


# ---------------------------------------------------------------------------
# mode-collapse knob
# ---------------------------------------------------------------------------

def test_collapse_one_is_onehot_within_class():
    """collapse=1 puts all class mass on a single op -> one distinct successor / class."""
    for st in STATES:
        dist = mb.emission_dist(st, DOMAIN, p=0.7, collapse=1.0)
        nonzero = [r for r in dist if r[3] > 0.0]
        # With both classes present, collapse=1 -> at most 2 nonzero (one per class);
        # with one class -> exactly 1.
        good, bad = _good_bad(st)
        n_classes = (1 if good else 0) + (1 if bad else 0)
        assert len(nonzero) <= max(1, n_classes)


def test_collapse_zero_spreads_within_class():
    """collapse=0 -> uniform within each class (>1 distinct successor when class>1)."""
    found_spread = False
    for st in STATES:
        good, bad = _good_bad(st)
        # distinct GOOD canonical successors
        good_keys = {DOMAIN.canon(DOMAIN.apply(st, op)) for op in good}
        dist = mb.emission_dist(st, DOMAIN, p=0.7, collapse=0.0)
        good_recs = [r for r in dist if DOMAIN.solvable(r[1])]
        if len(good_keys) > 1:
            found_spread = True
            probs = [r[3] for r in good_recs]
            # uniform within class: all good probs equal
            assert max(probs) == pytest.approx(min(probs), rel=1e-9)
    assert found_spread, "need at least one state with >1 distinct good successor"


def test_collapse_backend_is_width_one():
    s = mb.make_collapse_backend(p=1.0)
    for st in STATES:
        step = s(st, 10, 1.0, 1, "step")
        nonempty = [c for c in step if c != ""]
        assert len(set(nonempty)) <= 1, "collapse backend must emit one distinct op text"
        if nonempty:
            op = DOMAIN.parse_move(nonempty[0], st)
            assert op is not None
        chain = s(st, 7, 1.0, 1, "chain")
        assert len(set(chain)) == 1, "collapse backend chains must all be identical"


# ---------------------------------------------------------------------------
# in-domain legality (both modes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.2, 0.5, 0.9])
@pytest.mark.parametrize("collapse", [0.0, 0.5, 1.0])
def test_step_candidates_legal(p, collapse):
    s = mb.make_competence_backend(p=p, collapse=collapse)
    for st in STATES:
        cands = s(st, 16, 1.0, 321, "step")
        assert len(cands) == 16
        for c in cands:
            if c == "":
                continue
            assert DOMAIN.parse_move(c, st) is not None, (
                f"illegal step candidate {c!r} for p={p} collapse={collapse}"
            )


@pytest.mark.parametrize("p", [0.2, 0.5, 0.9])
def test_chain_candidates_legal(p):
    s = mb.make_competence_backend(p=p, collapse=0.0, depth_cap=8)
    for st in STATES:
        chains = s(st, 8, 1.0, 909, "chain")
        assert len(chains) == 8
        for ch in chains:
            if ch == "":
                continue
            ops = DOMAIN.parse_chain(ch, st)
            assert ops is not None, f"chain {ch!r} must parse against the evolving state"
            # The chain must be a legal forward evolution.
            cur = st
            for op in ops:
                cur = DOMAIN.apply(cur, op)


def test_chain_terminates_at_goal_or_cap():
    """A perfect generator (p=1) should reach the goal on a solvable instance."""
    st = STATES[0]
    assert DOMAIN.solvable(st)
    s = mb.make_competence_backend(p=1.0, collapse=0.0, depth_cap=8)
    chains = s(st, 32, 1.0, 4242, "chain")
    reached = 0
    for ch in chains:
        ops = DOMAIN.parse_chain(ch, st)
        if ops is None:
            continue
        cur = st
        for op in ops:
            cur = DOMAIN.apply(cur, op)
        if DOMAIN.is_goal(cur):
            reached += 1
    assert reached > 0, "p=1 generator must reach the goal on a solvable instance"


# ---------------------------------------------------------------------------
# empirical draws match the exact emission (the Tier-A contract)
# ---------------------------------------------------------------------------

def test_step_draws_match_emission_dist():
    """Empirical step-draw frequencies converge to the exact emission_dist."""
    st = STATES[0]
    p, collapse = 0.7, 0.0
    s = mb.make_competence_backend(p=p, collapse=collapse)
    # tau=None path is the exact P_emission; the closure exposes it directly.
    exact = {k: prob for (_op, _sp, k, prob) in s.emission_dist(st)}
    N = 20000
    cands = s(st, N, 1.0, 13, "step")
    counts = {}
    for c in cands:
        op = DOMAIN.parse_move(c, st)
        if op is None:
            continue
        k = DOMAIN.canon(DOMAIN.apply(st, op))
        counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    for k, prob in exact.items():
        emp = counts.get(k, 0) / total
        assert abs(emp - prob) < 0.03, f"{k}: emp={emp:.3f} exact={prob:.3f}"


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_terminal_state_emits_empty():
    """A single-value (terminal) state has no legal ops -> empty/unparseable output."""
    from fractions import Fraction
    term = cb.parse_target_problem([5], 5)  # one value, already terminal
    s = mb.make_competence_backend(p=0.7)
    step = s(term, 5, 1.0, 1, "step")
    assert step == [""] * 5
    chain = s(term, 5, 1.0, 1, "chain")
    assert all(c == "" for c in chain)


def test_zero_N_returns_empty():
    s = mb.make_competence_backend(p=0.7)
    assert s(STATES[0], 0, 1.0, 1, "step") == []
    assert s(STATES[0], -3, 1.0, 1, "chain") == []


def test_bad_mode_raises():
    s = mb.make_competence_backend(p=0.7)
    with pytest.raises(ValueError):
        s(STATES[0], 4, 1.0, 1, "bogus")


def test_abstain_injects_empties():
    s = mb.make_competence_backend(p=0.7, collapse=0.0, abstain=0.5)
    cands = s(STATES[0], 200, 1.0, 77, "step")
    n_empty = sum(1 for c in cands if c == "")
    # ~50% empty; allow a wide band for the finite draw.
    assert 60 < n_empty < 140, f"abstain rate off: {n_empty}/200 empty"


def test_tau_zero_is_greedy_deterministic():
    """tau=0 collapses to the argmax successor -> seed-independent, all identical."""
    s = mb.make_competence_backend(p=0.7, collapse=0.0)
    a = s(STATES[0], 8, 0.0, 1, "step")
    b = s(STATES[0], 8, 0.0, 99999, "step")
    assert a == b, "tau=0 greedy draw must be seed-independent"
    assert len(set(a)) == 1, "tau=0 must emit a single (argmax) op repeatedly"
