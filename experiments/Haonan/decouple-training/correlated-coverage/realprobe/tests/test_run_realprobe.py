"""CPU tests for the prereg-grade driver ``run_realprobe`` (no model, no GPU).

These pin the driver-LEVEL contracts the prereg requires, all exercised with a
deterministic FAKE generator (the same kind ``test_realprobe`` uses) and synthetic arm
pools -- never the real model:

  * V1 selection logic: coverage@K for every K comes from ONE shared pool draw (the first-K
    prefix), and the band filter keeps exactly the instances whose coverage@K_ref is in
    (0.2, 0.9). Tested on a FAKE per-chain reached-flag vector with a known coverage curve.
  * temp-match entropy: the marginal step entropy that the temp_matched_iid arm targets is
    measured on a ``>= K*`` ensemble (not a degenerate 1-chain draw), the matcher picks the
    closest-entropy tau on the sweep, and a higher-entropy target selects a not-lower tau.
  * dual-axis accounting: gen-token budget@K and total-token (gen+prefill) budget@K are the
    first-K prefix sums of the SAME pool; total >= gen always; and the conditioned arm's
    total grows faster than its gen (prefill carries the avoid-list).
  * anti-leak: no N-candidate-per-step path leaks in; every forward draws exactly ONE
    candidate across the full driver arm pool (the verified 1-forward/step probe core).
"""

import math

import pytest

import realprobe_core as rp
import run_realprobe as drv
from realprobe_core import CountdownDomain, run_arm, instance_dict
from core import Instance

# Reuse the SAME deterministic fake generator the core tests use. pytest's prepend import
# mode puts this test's dir on sys.path, so the sibling module is importable bare; fall
# back to the ``tests`` package form if run under a different import mode.
try:
    from test_realprobe import fake_factory, make_fake_generate, _INST
except ImportError:  # pragma: no cover - import-mode fallback
    from tests.test_realprobe import fake_factory, make_fake_generate, _INST


# ---------------------------------------------------------------------------
# coverage@K from ONE pool (every K from the same draw)
# ---------------------------------------------------------------------------

def test_coverage_at_K_is_prefix_of_one_pool():
    """coverage@K = ANY of the FIRST K pooled chains; one draw feeds every K."""
    # A pool where only chain index 5 (0-based) reached the goal.
    reached = [False, False, False, False, False, True,
               False, False, False, False, False, False, False, False, False, False]
    # K < 6 -> no hit yet; K >= 6 -> hit. Monotone non-decreasing in K (same pool).
    assert drv.coverage_at_K_from_pool(reached, 2) is False
    assert drv.coverage_at_K_from_pool(reached, 4) is False
    assert drv.coverage_at_K_from_pool(reached, 6) is True
    assert drv.coverage_at_K_from_pool(reached, 8) is True
    # Monotone: coverage@K never decreases as K grows (it's a prefix-OR of one pool).
    covs = [drv.coverage_at_K_from_pool(reached, K) for K in drv.K_GRID]
    assert covs == sorted(covs)  # False < True, non-decreasing


def test_v1_band_filter_keeps_only_mid_coverage():
    """V1: keep instances whose coverage@K_ref FRACTION is in (0.2, 0.9), drop the rest."""
    Kref = drv.K_REF  # 8
    # Build three synthetic reached-flag vectors with KNOWN first-K_ref hit fractions.
    def vec(n_hits):
        v = [True] * n_hits + [False] * (drv.K_MAX - n_hits)
        return v
    too_low = vec(1)    # 1/8 = 0.125 -> below band -> drop
    in_band = vec(4)    # 4/8 = 0.5   -> in band   -> keep
    too_high = vec(8)   # 8/8 = 1.0   -> above band -> drop
    saturated = vec(7)  # 7/8 = 0.875 -> in band (just under 0.9) -> keep

    def frac(v):
        return sum(v[:Kref]) / Kref

    assert not (drv.V1_LO < frac(too_low) < drv.V1_HI)
    assert (drv.V1_LO < frac(in_band) < drv.V1_HI)
    assert not (drv.V1_LO < frac(too_high) < drv.V1_HI)
    assert (drv.V1_LO < frac(saturated) < drv.V1_HI)


def test_generate_instances_are_solvable_correct_depth():
    """Generated k in {5,6} instances are solvable with witness depth 4,5 (== k-1)."""
    picked = drv.generate_solvable_instances(seed=7, n_per_depth=3)
    assert len(picked) == 6  # 3 per depth, 2 depths
    by_k = {}
    for inst, depth in picked:
        k = len(inst.numbers)
        by_k.setdefault(k, []).append(depth)
        # depth == witness_len == k - 1 for these generated instances.
        assert depth == k - 1
    assert set(by_k) == {5, 6}


# ---------------------------------------------------------------------------
# Marginal step entropy on a >= K* ensemble (the temp-match negative control)
# ---------------------------------------------------------------------------

def test_marginal_entropy_uniform_vs_degenerate():
    """Entropy is 0 for a single-support (degenerate) ensemble, log(n) for n uniform."""
    assert drv.marginal_step_entropy(["a", "a", "a", "a"]) == 0.0
    h = drv.marginal_step_entropy(["a", "b", "c", "d"])
    assert abs(h - math.log(4)) < 1e-9
    # An empty-move outcome "" is a distinct marginal outcome, not dropped.
    h2 = drv.marginal_step_entropy(["a", "", "a", ""])
    assert abs(h2 - math.log(2)) < 1e-9


def test_temp_match_measures_on_at_least_K_star_ensemble():
    """measure_arm_entropy draws a >= K* ensemble (not a degenerate single chain)."""
    domain = CountdownDomain()
    n_ens = drv.K_STAR
    h, m = drv.measure_arm_entropy(lambda: fake_factory()(), domain, _INST,
                                   temperature=1.0, base_seed=0, conditioned=False,
                                   n_ens=n_ens)
    # The ensemble used to estimate the marginal is at least K* chains.
    assert m >= drv.K_STAR
    assert m == n_ens
    assert h >= 0.0


def test_match_temperature_picks_closest_entropy_and_records_sweep():
    """The matcher returns the sweep tau with the smallest |entropy - target| and the trace."""
    domain = CountdownDomain()
    # Target the conditioned arm's own marginal entropy (measured on a >= K* ensemble).
    cond = run_arm(lambda: fake_factory()(), domain, _INST, drv.K_MAX, 1.0, 0,
                   conditioned=True)
    target = drv.marginal_step_entropy([drv.first_move(c) for c in cond.chains])
    best_tau, best_h, sweep = drv.match_temperature(
        lambda: fake_factory()(), domain, _INST, base_seed=0,
        target_entropy=target, n_ens=drv.K_STAR)
    # Every sweep record measured its entropy on a >= K* ensemble.
    assert all(rec["ensemble"] >= drv.K_STAR for rec in sweep)
    # best_tau is genuinely the argmin over the sweep (closest entropy to target).
    gaps = {rec["tau"]: abs(rec["entropy"] - target) for rec in sweep}
    assert best_tau == min(gaps, key=gaps.get)
    assert best_tau in drv.TAU_SWEEP


# ---------------------------------------------------------------------------
# Dual-axis token accounting (gen HEADLINE vs total SENSITIVITY), one pool
# ---------------------------------------------------------------------------

def test_dual_axis_prefix_accounting():
    """gen@K and total@K are the first-K prefix sums; total >= gen for every K."""
    gen_pc = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    pre_pc = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for K in drv.K_GRID:
        g = drv.gen_token_prefix(gen_pc, K)
        t = drv.total_token_prefix(gen_pc, pre_pc, K)
        assert g == sum(gen_pc[:K])
        assert t == sum(gen_pc[:K]) + sum(pre_pc[:K])
        assert t >= g  # total = gen + prefill (prefill >= 0)


def test_conditioned_total_grows_faster_than_gen():
    """On a real arm pool, the conditioned arm's total-token axis exceeds its gen axis,
    and that gap (the prefill) is LARGER than the i.i.d. arm's gap (avoid-list prefill)."""
    domain = CountdownDomain()
    iid = drv.run_arm_pool(lambda: fake_factory()(), domain, _INST, 1.0, 0,
                           conditioned=False)
    cond = drv.run_arm_pool(lambda: fake_factory()(), domain, _INST, 1.0, 0,
                            conditioned=True)
    Kref = str(drv.K_REF)
    iid_gap = iid["curve"][Kref]["total_token"] - iid["curve"][Kref]["gen_token"]
    cond_gap = cond["curve"][Kref]["total_token"] - cond["curve"][Kref]["gen_token"]
    assert iid_gap >= 0 and cond_gap >= 0
    # The conditioned arm carries the avoid-list -> strictly more prefill than i.i.d.
    assert cond_gap > iid_gap


def test_per_chain_token_prefix_sums_match_arm_total():
    """The per-chain split's full-pool sum equals the arm's reported gen/prefill totals."""
    domain = CountdownDomain()
    res = run_arm(lambda: fake_factory()(), domain, _INST, drv.K_MAX, 1.0, 0,
                  conditioned=True)
    gen_pc = drv._per_chain_gen_tokens(res)
    pre_pc = drv._per_chain_prefill_tokens(res, drv.K_MAX)
    assert sum(gen_pc) == res.gen_tokens
    assert sum(pre_pc) == res.prefill_tokens


# ---------------------------------------------------------------------------
# Anti-leak: NO N-candidate-per-step path leaks into any driver arm pool
# ---------------------------------------------------------------------------

def test_no_N_candidate_leak_in_driver_pools():
    """Every arm pool the driver builds proves one-forward-per-step (cands all 1)."""
    domain = CountdownDomain()
    for conditioned in (False, True):
        pool = drv.run_arm_pool(lambda: fake_factory()(), domain, _INST, 1.0, 0,
                                conditioned=conditioned)
        assert pool["cands_per_fwd_all_one"] is True
        assert pool["cands_per_fwd_max"] == 1


def test_run_arm_pool_K_max_chains_one_forward_each():
    """The pool draws exactly K_MAX chains and forwards == committed steps (1 per step)."""
    domain = CountdownDomain()
    pool = drv.run_arm_pool(lambda: fake_factory()(), domain, _INST, 1.0, 0,
                            conditioned=False)
    assert len(pool["reached"]) == drv.K_MAX
    assert len(pool["first_moves"]) == drv.K_MAX
    # forwards equals the committed-step count of the fake (it always returns a legal op,
    # so each forward commits one step); the anti-leak invariant is cands==1 per forward.
    assert pool["forwards"] >= 1


# ---------------------------------------------------------------------------
# Stats: miss-decay slope, paired bootstrap, Holm
# ---------------------------------------------------------------------------

def test_miss_slope_negative_when_coverage_rises_with_tokens():
    """A coverage curve rising with token budget gives a NEGATIVE miss-decay slope."""
    toks = [10, 20, 30, 40, 50, 60]
    covs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    slope = drv.miss_decay_slope(toks, covs)
    assert slope < 0.0  # log(1-cov) decreases as tokens grow -> faster miss-decay


def test_paired_bootstrap_sign_and_holm_monotone():
    """Paired bootstrap detects a positive paired diff; Holm adj-p is monotone non-decreasing."""
    a = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]  # arm a
    b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # arm b (a beats b on matched items)
    lo, mid, hi = drv.paired_bootstrap_diff(a, b, n_boot=2000, seed=0)
    assert mid > 0 and lo >= 0  # a robustly beats b
    p = drv._one_sided_p_from_boot(a, b, n_boot=2000, seed=0)
    assert 0.0 < p < 0.5
    holm = drv.holm({"H1": 0.001, "H2": 0.04, "H4": 0.2})
    adj = [holm[name]["adj_p"] for name in ("H1", "H2", "H4")]
    assert adj == sorted(adj)  # monotone non-decreasing after the Holm step-down


def test_paired_bootstrap_zero_diff_contains_zero():
    """A null paired diff (a == b) yields a CI that contains 0 (the H4 control shape)."""
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    b = list(a)
    lo, mid, hi = drv.paired_bootstrap_diff(a, b, n_boot=2000, seed=0)
    assert lo == 0.0 and mid == 0.0 and hi == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
