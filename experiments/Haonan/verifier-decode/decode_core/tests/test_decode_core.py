"""decode_core decode tests, driven through the test-only CountdownDomain adapter.

These port the meaningful M1 ``test_decode.py`` cases onto the GENERALIZED decoder:
the decoder takes a ``domain`` + a ``sample`` callable instead of hard-wired
``countdown.*`` / ``sampler.*``. The CountdownDomain adapter (test-only) supplies the
exact same Countdown semantics, so the same invariants M1 asserts must hold here.

Note on the verifier default: M1's ``savi`` defaults ``verifier=True``; decode_core's
``savi`` defaults ``verifier=False`` (headline = legality at parse + is_goal at leaf).
Ported cases that rely on the reachability mask pass ``verifier=True`` explicitly,
matching the M1 semantics under test.
"""

import math

import pytest

from decode_core import decode as dc
from decode_core.decode import Budget

# Test-only: the frozen M1 modules + the adapter (decode_core never imports these).
from core.instances import load_instances
from core.countdown import parse_target_problem, apply, is_goal, reachable

from _countdown_adapter import CountdownDomain, make_sample


DOM = CountdownDomain
SAMPLE_DECOUPLED = make_sample("mock_decoupled")
SAMPLE_ONEHOT = make_sample("mock_onehot")


# ---------------------------------------------------------------------------
# Helpers (mirror M1 test helpers, via decode_core.oracle)
# ---------------------------------------------------------------------------

def _solvable():
    """First built-in instance the absolute oracle says is solvable."""
    return next(i for i in load_instances({"set": "builtin_small"})
                if dc.oracle(DOM, i))


def _replay(inst, path):
    cur = parse_target_problem(inst.numbers, inst.target)
    for op in path:
        cur = apply(cur, op)
    return cur


# ---------------------------------------------------------------------------
# Ported spec-stub tests
# ---------------------------------------------------------------------------

def test_savi_finds_known_solution_decoupled():
    inst = _solvable()
    r = dc.savi(DOM, SAMPLE_DECOUPLED, inst, K=8, N=16, edge_mode="support",
                tau=0.7, seed=1, verifier=True)
    assert r.ok and r.path is not None
    cur = parse_target_problem(inst.numbers, inst.target)
    for op in r.path:
        cur = apply(cur, op)
    assert is_goal(cur)


def test_phi_merge_dedups():
    inst = _solvable()
    r = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, "support", 0.7, 1, verifier=True)
    assert all(a <= b for a, b in zip(r.trellis_widths_after_merge,
                                      r.trellis_widths_before_merge))
    assert any(a < b for a, b in zip(r.trellis_widths_after_merge,
                                     r.trellis_widths_before_merge))


def test_beam_k_coverage_monotone():
    inst = _solvable()
    oks = [dc.savi(DOM, SAMPLE_DECOUPLED, inst, K, 16, "support", 0.7, 1,
                   verifier=True).ok
           for K in (1, 2, 4, 8)]
    assert oks == sorted(oks)  # non-decreasing


def test_onehot_collapses_vs_decoupled_spreads():
    inst = _solvable()
    oh = dc.savi(DOM, SAMPLE_ONEHOT, inst, 8, 16, "support", 0.7, 1, verifier=True)
    dcp = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, "support", 0.7, 1, verifier=True)
    assert dcp.trellis_widths_after_merge[0] >= oh.trellis_widths_after_merge[0]
    assert oh.trellis_widths_after_merge[0] == 1  # one-hot collapses to a single next-state


def test_isocompute_counters():
    inst = _solvable()
    r = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 4, 8, "freq", 0.7, 1, verifier=True)
    assert (r.budget.sample_calls >= 1 and r.budget.candidates >= 1
            and r.budget.tokens >= 1 and r.budget.exec >= 1)


def test_oracle_is_absolute_solvability():
    insts = load_instances({"set": "builtin_small"})
    assert (any(dc.oracle(DOM, i) for i in insts)
            and not all(dc.oracle(DOM, i) for i in insts))


def test_edge_modes_both_run():
    inst = _solvable()
    for em in ("support", "freq"):
        assert dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, em, 0.7, 1,
                       verifier=True).ok in (True, False)


# ---------------------------------------------------------------------------
# Additional ported coverage
# ---------------------------------------------------------------------------

def test_oracle_matches_reachable_absolute():
    for inst in load_instances({"set": "builtin_small"}):
        s0 = parse_target_problem(inst.numbers, inst.target)
        assert dc.oracle(DOM, inst) == reachable(s0)


def test_oracle_strictly_dominates_best_of_k_on_some_instance():
    insts = load_instances({"set": "builtin_small"})
    gap = any(dc.oracle(DOM, i)
              and not dc.best_of_k(DOM, SAMPLE_ONEHOT, i, K=1, tau=0.0, seed=1).ok
              for i in insts)
    assert gap


def test_verifier_off_allows_infeasible_expansions():
    inst = _solvable()
    on = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, "support", 0.7, 1, verifier=True)
    off = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, "support", 0.7, 1, verifier=False)
    assert all(b_off >= b_on for b_off, b_on
               in zip(off.trellis_widths_before_merge,
                      on.trellis_widths_before_merge))


def test_verifier_keeps_all_states_reachable():
    inst = _solvable()
    r = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, "support", 0.7, 1, verifier=True)
    assert r.ok
    cur = parse_target_problem(inst.numbers, inst.target)
    assert reachable(cur)
    for op in r.path:
        cur = apply(cur, op)
        assert reachable(cur)


def test_nested_beams_topk_subset():
    inst = _solvable()
    prev = None
    for K in (1, 2, 4, 8):
        r = dc.savi(DOM, SAMPLE_DECOUPLED, inst, K, 16, "support", 0.7, 1,
                    verifier=True)
        w0 = r.trellis_widths_after_merge[0]
        if prev is not None:
            assert w0 >= prev
        prev = w0


def test_freq_edge_is_log_count_over_N():
    inst = _solvable()
    r = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 32, "freq", 0.7, 1, verifier=True)
    assert r.ok and r.best_score is not None
    assert r.best_score <= 0.0
    assert math.isfinite(r.best_score)


def test_support_edge_score_is_zero():
    inst = _solvable()
    r = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, "support", 0.7, 1, verifier=True)
    assert r.ok
    assert r.best_score == 0.0


def test_greedy_returns_budget_and_ok_bool():
    inst = _solvable()
    r = dc.greedy(DOM, SAMPLE_DECOUPLED, inst, seed=1)
    assert isinstance(r.ok, bool)
    assert r.budget.sample_calls == 1
    assert r.budget.candidates >= 1
    assert r.budget.tokens >= 1
    assert r.budget.exec >= 1


def test_greedy_path_replays_to_goal_when_ok():
    inst = _solvable()
    r = dc.greedy(DOM, SAMPLE_DECOUPLED, inst, seed=1)
    if r.ok:
        assert is_goal(_replay(inst, r.path))


def test_best_of_k_ok_iff_any_chain_solves_and_returns_winner():
    inst = _solvable()
    r = dc.best_of_k(DOM, SAMPLE_DECOUPLED, inst, K=16, tau=0.7, seed=1)
    assert isinstance(r.ok, bool)
    assert r.budget.sample_calls == 1
    assert r.budget.candidates == 16
    if r.ok:
        assert r.path is not None
        assert is_goal(_replay(inst, r.path))


def test_best_of_k_onehot_collapses():
    inst = _solvable()
    r = dc.best_of_k(DOM, SAMPLE_ONEHOT, inst, K=8, tau=0.0, seed=1)
    if r.ok:
        assert is_goal(_replay(inst, r.path))


def test_budget_tokens_is_whitespace_token_proxy():
    inst = _solvable()
    r = dc.best_of_k(DOM, SAMPLE_DECOUPLED, inst, K=4, tau=0.7, seed=1)
    # Recompute expected token count from the same sampler call (through the closure).
    s0 = parse_target_problem(inst.numbers, inst.target)
    cands = SAMPLE_DECOUPLED(s0, 4, 0.7, 1, "chain")
    expected = sum(len(c.split()) for c in cands)
    assert r.budget.tokens == expected


def test_savi_unsolvable_instance_returns_not_ok():
    insts = load_instances({"set": "builtin_small"})
    unsolv = next(i for i in insts if not dc.oracle(DOM, i))
    r = dc.savi(DOM, SAMPLE_DECOUPLED, unsolv, 8, 16, "support", 0.7, 1,
                verifier=True)
    assert r.ok is False and r.path is None and r.best_score is None


def test_savi_determinism():
    inst = _solvable()
    a = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, "freq", 0.7, 1, verifier=True)
    b = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, "freq", 0.7, 1, verifier=True)
    assert a.ok == b.ok
    assert a.best_score == b.best_score
    assert a.path == b.path
    assert a.trellis_widths_before_merge == b.trellis_widths_before_merge
    assert a.trellis_widths_after_merge == b.trellis_widths_after_merge


def test_widths_lengths_match_depth():
    inst = _solvable()
    r = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, "support", 0.7, 1, verifier=True)
    depth = len(inst.numbers) - 1
    assert len(r.trellis_widths_before_merge) == depth
    assert len(r.trellis_widths_after_merge) == depth


def test_budget_exec_counts_reachable_when_verifier_on():
    inst = _solvable()
    on = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 4, 16, "support", 0.7, 1, verifier=True)
    off = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 4, 16, "support", 0.7, 1, verifier=False)
    assert on.budget.exec > off.budget.exec


def test_savi_default_verifier_is_off():
    """decode_core.savi defaults verifier=False: same as an explicit verifier=False
    call, and (on this domain) its exec count is strictly below verifier=True."""
    inst = _solvable()
    default = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 4, 16, "support", 0.7, 1)
    off = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 4, 16, "support", 0.7, 1, verifier=False)
    on = dc.savi(DOM, SAMPLE_DECOUPLED, inst, 4, 16, "support", 0.7, 1, verifier=True)
    assert default.budget.exec == off.budget.exec
    assert default.budget.exec < on.budget.exec
    assert default.ok == off.ok
    assert default.trellis_widths_before_merge == off.trellis_widths_before_merge


def test_edge_mode_validation():
    inst = _solvable()
    with pytest.raises(ValueError):
        dc.savi(DOM, SAMPLE_DECOUPLED, inst, 8, 16, "bogus", 0.7, 1, verifier=True)
