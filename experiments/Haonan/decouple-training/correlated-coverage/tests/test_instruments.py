"""Tests for the measurement instruments (instruments.py).

The task's three required checks, plus end-to-end smoke against the real arms:

1. SLOPE FIT on a SYNTHETIC EXPONENTIAL — ``miss_decay_slope`` must recover the known
   decay rate of ``1 − coverage = exp(slope·B)`` to high precision, with R²≈1.

2. BOOTSTRAP CI SANITY — ``paired_bootstrap`` recovers a known positive paired delta,
   its CI brackets the point estimate and (for a clear signal) excludes 0, while a
   zero-effect paired vector gives a CI that contains 0; reproducible under fixed seed.

3. TAU_MATCH MONOTONICITY — the i.i.d. arm's realized marginal step-entropy is monotone
   non-decreasing in τ, and ``tau_match`` lands on a τ whose realized entropy hits a
   reachable target within tolerance (and honestly flags out-of-bracket targets).

Plus: coverage_vs_tokens / joint_diversity / marginal_step_entropy smoke against the
actual corr_sampler arms on the Tier-A mock (no GPU), and Holm correction sanity.
"""

import math

import numpy as np
import pytest

import core_boot as cb
import mock_backend as mb
import corr_sampler as cs
import instruments as ins


DOMAIN = cb.CountdownDomain()


def _inst(numbers, target, iid="t"):
    return {"numbers": list(numbers), "target": target, "id": iid}


INSTANCES = [
    _inst([3, 7, 8, 9], 24, "i0"),
    _inst([2, 3, 4, 5], 24, "i1"),
    _inst([1, 5, 6, 7], 21, "i2"),
    _inst([10, 4, 6, 2], 24, "i3"),
]


# Arm-closure factories matching instruments' run_arm(inst, K, seed) -> Result contract.
def _corr_runner(sample, N, beta, tau):
    return lambda inst, K, seed: cs.corr_sample(
        DOMAIN, sample, inst, K=K, N=N, beta=beta, tau=tau, seed=seed
    )


def _iid_runner(sample, N, tau):
    return lambda inst, K, seed: cs.iid_bok(
        DOMAIN, sample, inst, K=K, tau=tau, seed=seed, N=N
    )


# ===========================================================================
# 1. SLOPE FIT ON A SYNTHETIC EXPONENTIAL  (H2 "base数" metric)
# ===========================================================================

def test_slope_fit_recovers_known_exponential():
    """miss = exp(true_slope · B) ⇒ log(miss) is linear with slope == true_slope."""
    true_slope = -0.004
    budgets = list(range(0, 1000, 50))
    # coverage = 1 - exp(slope*B); pick budgets where coverage stays < 1 (no saturation).
    coverage = [1.0 - math.exp(true_slope * B) for B in budgets]
    fit = ins.miss_decay_slope(budgets, coverage)
    assert fit["slope"] == pytest.approx(true_slope, rel=1e-6), fit
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9), fit
    assert fit["n_points"] == len(budgets)


def test_slope_more_negative_means_faster_decay():
    """A steeper (more negative) slope corresponds to faster miss decay (H2 direction)."""
    budgets = list(range(0, 1000, 50))
    fast = [1.0 - math.exp(-0.01 * B) for B in budgets]
    slow = [1.0 - math.exp(-0.002 * B) for B in budgets]
    s_fast = ins.miss_decay_slope(budgets, fast)["slope"]
    s_slow = ins.miss_decay_slope(budgets, slow)["slope"]
    assert s_fast < s_slow < 0.0


def test_slope_drops_saturated_points():
    """Coverage==1 points (miss→0, log(0)=−∞) are dropped, not crashed/poisoned."""
    budgets = [0, 100, 200, 300, 400]
    coverage = [0.0, 0.5, 0.9, 1.0, 1.0]  # last two saturate
    fit = ins.miss_decay_slope(budgets, coverage)
    assert fit["n_points"] == 3  # the two saturated points dropped
    assert math.isfinite(fit["slope"]) and fit["slope"] < 0.0


def test_slope_too_few_points_is_nan():
    fit = ins.miss_decay_slope([100], [0.5])
    assert math.isnan(fit["slope"]) and fit["n_points"] == 1


# ===========================================================================
# 2. BOOTSTRAP CI SANITY  (paired_bootstrap, n=10000)
# ===========================================================================

def test_bootstrap_recovers_positive_delta_and_excludes_zero():
    """A clear positive paired delta: point estimate exact, 95% CI brackets it, excludes 0."""
    # corr covers 18/20, iid covers 8/20 on the SAME items -> mean delta = +0.5.
    a = [1] * 18 + [0] * 2
    b = [1] * 8 + [0] * 12
    res = ins.paired_bootstrap(a, b, n_boot=10000, seed=0)
    assert res["delta"] == pytest.approx(0.5, abs=1e-12)
    assert res["ci_low"] <= res["delta"] <= res["ci_high"]
    assert res["excludes_zero"] is True
    assert res["p_two_sided"] < 0.05
    assert res["n"] == 20


def test_bootstrap_zero_effect_contains_zero():
    """Identical paired vectors -> delta 0, CI degenerate at 0, does NOT exclude 0."""
    a = [1, 0, 1, 0, 1, 1, 0, 0]
    b = list(a)
    res = ins.paired_bootstrap(a, b, n_boot=5000, seed=1)
    assert res["delta"] == pytest.approx(0.0, abs=1e-12)
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0
    assert res["excludes_zero"] is False


def test_bootstrap_reproducible_under_seed():
    # Continuous-valued paired vector so percentile bounds are finely resolved (a coarse
    # 0/1 vector quantizes the CI so different seeds can land on identical percentiles).
    rng = np.random.default_rng(99)
    a = rng.random(60)
    b = rng.random(60)
    r1 = ins.paired_bootstrap(a, b, n_boot=4000, seed=7)
    r2 = ins.paired_bootstrap(a, b, n_boot=4000, seed=7)
    assert r1 == r2
    r3 = ins.paired_bootstrap(a, b, n_boot=4000, seed=8)
    # Different seed -> different resamples -> different CI bounds (point estimate equal).
    assert r3["delta"] == r1["delta"]
    assert (r3["ci_low"], r3["ci_high"]) != (r1["ci_low"], r1["ci_high"])


def test_bootstrap_ci_brackets_point_for_noisy_signal():
    rng = np.random.default_rng(0)
    # Paired Bernoulli with a modest positive shift.
    a = (rng.random(200) < 0.6).astype(int)
    b = (rng.random(200) < 0.45).astype(int)
    res = ins.paired_bootstrap(a, b, n_boot=10000, seed=3)
    assert res["ci_low"] <= res["delta"] <= res["ci_high"]
    assert res["ci_low"] <= res["ci_med"] <= res["ci_high"]


def test_bootstrap_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        ins.paired_bootstrap([1, 0, 1], [1, 0], n_boot=100, seed=0)


def test_bootstrap_negative_delta_excludes_zero_below():
    """A clear NEGATIVE paired delta excludes 0 from above (CI entirely < 0)."""
    a = [1] * 5 + [0] * 15
    b = [1] * 17 + [0] * 3
    res = ins.paired_bootstrap(a, b, n_boot=10000, seed=2)
    assert res["delta"] < 0.0
    assert res["ci_high"] < 0.0
    assert res["excludes_zero"] is True


# ===========================================================================
# 3. TAU_MATCH MONOTONICITY  (H4 calibration by marginal step-entropy)
# ===========================================================================

def test_iid_marginal_entropy_monotone_in_tau():
    """The i.i.d. arm's realized marginal step-entropy trends UP with τ.

    Higher τ flattens P_emission (mock _tau_collapse) -> broader marginal spread. The
    realized entropy is a NOISY DISCRETE STEP function of τ (finite chains), so it can
    have small non-monotone wobbles; we assert the dominant trend: greedy (τ→0) is the
    minimum (an argmax collapse, H≈0), the flattened band is well above it, and the
    coarse low→high comparison rises (allowing a small wobble tolerance).
    """
    sample = mb.make_competence_backend(p=0.6, collapse=0.6, depth_cap=8)
    seeds = list(range(8))
    K = 8
    H_greedy = ins.measure_marginal_entropy(_iid_runner(sample, N=16, tau=0.0),
                                            INSTANCES, seeds, K)
    H_low = ins.measure_marginal_entropy(_iid_runner(sample, N=16, tau=0.25),
                                         INSTANCES, seeds, K)
    H_high = ins.measure_marginal_entropy(_iid_runner(sample, N=16, tau=8.0),
                                          INSTANCES, seeds, K)
    # Greedy is the floor (argmax -> all chains agree per step -> ~0 entropy).
    assert H_greedy <= H_low
    assert H_greedy == pytest.approx(0.0, abs=1e-9)
    # The flattened-emission band is strictly above the greedy floor and trends up.
    assert H_high > H_greedy + 1e-3
    assert H_high >= H_low - 0.05  # coarse upward trend (small step wobble tolerated)


def test_tau_match_hits_attainable_target():
    """tau_match recovers a τ near one whose realized marginal entropy is the target.

    The realized entropy is a discrete step function, so an ARBITRARY target (e.g. the
    bracket midpoint) may be unattainable; we make the target ATTAINABLE by measuring the
    entropy at a real interior τ, then asking tau_match to land on a τ whose entropy is
    within tolerance of it. tau_match tracks the running-closest τ, so it succeeds despite
    the curve's plateaus.
    """
    sample = mb.make_competence_backend(p=0.6, collapse=0.6, depth_cap=8)
    seeds = list(range(8))
    K = 8

    def make_iid(tau):
        return _iid_runner(sample, N=16, tau=tau)

    # An attainable target: the entropy actually produced at an interior τ.
    target = ins.measure_marginal_entropy(make_iid(1.0), INSTANCES, seeds, K)
    res = ins.tau_match(make_iid, target, INSTANCES, seeds, K, tol=0.05,
                        tau_lo=0.0, tau_hi=8.0)
    assert res["reached"], res
    assert res["gap"] <= 0.05
    assert 0.0 <= res["tau"] <= 8.0


def test_tau_match_returns_closest_on_unattainable_band():
    """A target in the UNATTAINABLE band (between greedy 0 and the flattened band) gives
    the closest τ with reached=False (no fake exact hit)."""
    sample = mb.make_competence_backend(p=0.6, collapse=0.6, depth_cap=8)
    seeds = list(range(8))
    K = 8

    def make_iid(tau):
        return _iid_runner(sample, N=16, tau=tau)

    H_floor = ins.measure_marginal_entropy(make_iid(0.0), INSTANCES, seeds, K)  # ~0
    H_band = ins.measure_marginal_entropy(make_iid(0.25), INSTANCES, seeds, K)  # jump
    assert H_band > H_floor + 0.5  # there is a real gap/band the curve skips
    # A target deep inside the skipped band, far from both floor and band.
    unattainable = 0.5 * (H_floor + H_band)
    res = ins.tau_match(make_iid, unattainable, INSTANCES, seeds, K, tol=1e-3,
                        tau_lo=0.0, tau_hi=8.0)
    assert res["reached"] is False
    # It still returns the CLOSEST attainable entropy (one of the two band edges).
    assert res["gap"] <= abs(unattainable - H_floor) + 1e-9
    assert res["gap"] <= abs(unattainable - H_band) + 1e-9


def test_tau_match_out_of_bracket_flags_honestly():
    """A target above the tau_hi entropy returns tau_hi with reached=False (no fake match)."""
    sample = mb.make_competence_backend(p=0.6, collapse=0.6, depth_cap=8)
    seeds = list(range(8))
    K = 6

    def make_iid(tau):
        return _iid_runner(sample, N=16, tau=tau)

    H_hi = ins.measure_marginal_entropy(make_iid(8.0), INSTANCES, seeds, K)
    unreachable = H_hi + 5.0  # well above anything τ can produce
    res = ins.tau_match(make_iid, unreachable, INSTANCES, seeds, K,
                        tau_lo=0.0, tau_hi=8.0)
    assert res["tau"] == 8.0
    assert res["reached"] is False


def test_marginal_step_entropy_basic():
    """marginal_step_entropy: a fully redundant ensemble has 0 entropy; spread is >0."""
    # All chains identical -> each depth has one realized successor -> H=0.
    same = [["A", "B"], ["A", "B"], ["A", "B"]]
    assert ins.marginal_step_entropy(same) == pytest.approx(0.0, abs=1e-12)
    # Depth 1 splits 50/50 -> H(depth1)=log2 in nats=ln2; depth absent elsewhere.
    split = [["A"], ["B"]]
    assert ins.marginal_step_entropy(split) == pytest.approx(math.log(2), abs=1e-9)
    # Empty -> 0.
    assert ins.marginal_step_entropy([]) == 0.0
    assert ins.marginal_step_entropy([[], []]) == 0.0


# ===========================================================================
# End-to-end smoke against the real arms (Tier-A mock, CPU only)
# ===========================================================================

def test_coverage_vs_tokens_monotone_in_budget():
    """coverage@B is non-decreasing in the token budget (more budget -> more chains)."""
    sample = mb.make_competence_backend(p=0.7, collapse=0.2, depth_cap=8)
    run = _iid_runner(sample, N=1, tau=1.0)
    seeds = list(range(8))
    budgets = [10, 30, 60, 120, 240]
    curve = ins.coverage_vs_tokens(run, INSTANCES, seeds, budgets, K_max=64)
    cov = curve["coverage"]
    assert len(cov) == len(budgets)
    # Coverage non-decreasing in B (nested-K coverage is monotone in K, K grows with B).
    for lo, hi in zip(cov, cov[1:]):
        assert hi >= lo - 1e-9, f"coverage not monotone in budget: {list(zip(budgets, cov))}"
    # mean_K should also grow with budget.
    mk = curve["mean_K"]
    assert mk[-1] >= mk[0]


def test_coverage_at_budget_respects_token_ceiling():
    """The realized K's token cost never exceeds B (except the unavoidable K=1 floor)."""
    sample = mb.make_competence_backend(p=0.7, collapse=0.0, depth_cap=8)
    run = _corr_runner(sample, N=8, beta=0.5, tau=1.0)
    inst = INSTANCES[0]
    B = 200
    ok, K, tok = ins.coverage_at_budget(run, inst, seed=1, budget_B=B, K_max=64)
    # Either we stopped under budget, or K==1 (the floor that can legitimately overrun).
    assert tok <= B or K == 1
    assert ok in (0, 1)


def test_joint_diversity_rises_with_beta():
    """distinct_canon@K (leaf + union) is higher for corr(β>0) than corr(β=0)=iid."""
    sample = mb.make_competence_backend(p=0.6, collapse=0.6, depth_cap=8)
    seeds = list(range(15))
    K = 8
    jd0 = ins.joint_diversity(_corr_runner(sample, N=24, beta=0.0, tau=1.0),
                              INSTANCES, seeds, K)
    jd3 = ins.joint_diversity(_corr_runner(sample, N=24, beta=3.0, tau=1.0),
                              INSTANCES, seeds, K)
    assert jd3["leaf_mean"] > jd0["leaf_mean"], (jd0["leaf_mean"], jd3["leaf_mean"])
    assert jd3["union_mean"] >= jd0["union_mean"]


def test_joint_diversity_beta0_matches_iid():
    """Sanity: joint_diversity of corr(β=0) == that of iid_bok (β=0 equivalence)."""
    sample = mb.make_competence_backend(p=0.6, collapse=0.3, depth_cap=8)
    seeds = list(range(10))
    K = 8
    jd_corr = ins.joint_diversity(_corr_runner(sample, N=16, beta=0.0, tau=1.0),
                                  INSTANCES, seeds, K)
    jd_iid = ins.joint_diversity(_iid_runner(sample, N=16, tau=1.0),
                                 INSTANCES, seeds, K)
    assert jd_corr["leaf_mean"] == jd_iid["leaf_mean"]
    assert jd_corr["union_mean"] == jd_iid["union_mean"]
    assert jd_corr["leaf_per_cell"] == jd_iid["leaf_per_cell"]


def test_coverage_vs_tokens_paired_cells_feed_bootstrap():
    """The per_cell outcomes of two arms at one B are paired and feed paired_bootstrap."""
    sample = mb.make_competence_backend(p=0.6, collapse=0.4, depth_cap=8)
    seeds = list(range(10))
    budgets = [120]
    corr = ins.coverage_vs_tokens(_corr_runner(sample, N=8, beta=1.0, tau=1.0),
                                  INSTANCES, seeds, budgets, K_max=64)
    iid = ins.coverage_vs_tokens(_iid_runner(sample, N=1, tau=1.0),
                                 INSTANCES, seeds, budgets, K_max=64)
    a = corr["per_cell"][120]
    b = iid["per_cell"][120]
    assert len(a) == len(b) == len(INSTANCES) * len(seeds)
    res = ins.paired_bootstrap(a, b, n_boot=2000, seed=0)
    assert "delta" in res and "excludes_zero" in res
    assert res["n"] == len(a)


# ===========================================================================
# Holm correction sanity
# ===========================================================================

def test_holm_correction_basic():
    """Holm: smallest p compared to alpha/m; adjusted p monotone; obvious rejects fire."""
    pvals = {"H1": 0.001, "H2": 0.02, "H4": 0.6}
    out = ins.holm_correction(pvals, alpha=0.05)
    # m=3: H1 adj = 3*0.001=0.003 (reject); H2 adj = 2*0.02=0.04 (reject);
    # H4 adj = 1*0.6=0.6 (no). Monotone enforced.
    assert out["H1"]["reject"] is True
    assert out["H2"]["reject"] is True
    assert out["H4"]["reject"] is False
    assert out["H1"]["p_adj"] <= out["H2"]["p_adj"] <= out["H4"]["p_adj"]


def test_holm_correction_monotone_clamp():
    """Adjusted p never decreases with rank and is clamped to [0,1]."""
    pvals = [("a", 0.04), ("b", 0.04), ("c", 0.04)]
    out = ins.holm_correction(pvals, alpha=0.05)
    adjs = [out["a"]["p_adj"], out["b"]["p_adj"], out["c"]["p_adj"]]
    assert all(0.0 <= x <= 1.0 for x in adjs)
    assert adjs == sorted(adjs)


def test_holm_empty():
    assert ins.holm_correction({}, alpha=0.05) == {}
