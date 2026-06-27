# tests/test_state_metrics.py
# Hand-computed unit tests for the model-free L2 state-level metrics.
import importlib.util as ilu
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bp(n, p):
    s = ilu.spec_from_file_location(n, str(p))
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


M = bp("state_metrics", HERE.parent / "core/state_metrics.py")

# A generic 3-letter alphabet with 2 survivors; "C" is eliminated.
ABC = ["A", "B", "C"]
SURV2 = ["A", "B"]
UNIFORM2 = {"A": 0.5, "B": 0.5, "C": 0.0}


# ---------------------------------------------------------------- calibration_tv
def test_calibration_tv_uniform():
    # empirical {A:.5,B:.5} == target -> TV 0
    states = ["A", "A", "B", "B"]
    target = {"A": 0.5, "B": 0.5}
    assert abs(M.calibration_tv(states, target, ["A", "B"])) < 1e-9


def test_calibration_tv_full_collapse():
    # all mass on A, target uniform over {A,B} -> TV = 0.5
    states = ["A", "A", "A", "A"]
    target = {"A": 0.5, "B": 0.5}
    assert abs(M.calibration_tv(states, target, ["A", "B"]) - 0.5) < 1e-9


def test_calibration_tv_excludes_abstain():
    # abstentions (hedge/none) drop out; committed = {A:1,B:1} matches uniform -> TV 0
    states = ["A", "B", "hedge", "none"]
    target = {"A": 0.5, "B": 0.5}
    assert abs(M.calibration_tv(states, target, ["A", "B"])) < 1e-9


def test_calibration_tv_all_abstain_is_nan():
    states = ["hedge", "none", "multi"]
    target = {"A": 0.5, "B": 0.5}
    assert math.isnan(M.calibration_tv(states, target, ["A", "B"]))


def test_calibration_tv_empty_is_nan():
    assert math.isnan(M.calibration_tv([], {"A": 1.0}, ["A"]))


# ---------------------------------------------------------------- coverage
def test_coverage_partial():
    # only A of survivors {A,B} appears -> 1/2
    assert M.coverage(["A", "A", "A"], ["A", "B"]) == 0.5


def test_coverage_full():
    assert abs(M.coverage(["A", "B", "A", "B"], SURV2) - 1.0) < 1e-9


def test_coverage_zero():
    # no survivor present (all eliminated) -> 0
    assert M.coverage(["C", "C", "hedge"], SURV2) == 0.0


def test_coverage_empty_survivors_is_nan():
    # no true posterior to cover -> nan (not a ZeroDivisionError)
    assert math.isnan(M.coverage(["A", "B"], []))


# ---------------------------------------------------------------- eliminated_mass
def test_eliminated_mass():
    # 1 of 4 samples commits to eliminated C -> 1/4
    states = ["A", "C", "hedge", "B"]
    assert abs(M.eliminated_mass(states, SURV2, ABC) - 0.25) < 1e-9


def test_eliminated_mass_none():
    states = ["A", "B", "hedge", "none"]
    assert M.eliminated_mass(states, SURV2, ABC) == 0.0


# ---------------------------------------------------------------- abstain_rate
def test_abstain_rate():
    # hedge + none count; multi and committed do not -> 2/5
    states = ["A", "hedge", "none", "multi", "B"]
    assert abs(M.abstain_rate(states) - 2 / 5) < 1e-9


def test_abstain_rate_empty_is_zero():
    assert M.abstain_rate([]) == 0.0


# ---------------------------------------------------------------- k_eff
def test_k_eff_distinct():
    # distinct committed states A,B (C eliminated still counts as a state in alphabet);
    # hedge/none excluded
    states = ["A", "A", "B", "hedge", "none"]
    assert M.k_eff_distinct(states, ABC) == 2


def test_k_eff_distinct_eliminated_counts():
    # committing to C (in alphabet) is still a distinct committed state
    assert M.k_eff_distinct(["A", "C"], ABC) == 2


def test_k_eff_entropy_balanced():
    # 3 states each 1/3 -> exp(ln 3) = 3
    states = ["A", "B", "C"] * 4
    assert abs(M.k_eff_entropy(states, ABC) - 3.0) < 1e-6


def test_k_eff_entropy_collapsed():
    # single committed state -> 1.0
    assert abs(M.k_eff_entropy(["A"] * 7, ABC) - 1.0) < 1e-9


def test_k_eff_entropy_no_commit_is_zero():
    assert M.k_eff_entropy(["hedge", "none"], ABC) == 0.0


# ---------------------------------------------------------------- bar / cmr
def test_bar_all_duplicates():
    # 4 candidates all one state -> 1 - 1/4 = 0.75
    assert abs(M.bar(["A", "A", "A", "A"]) - 0.75) < 1e-9


def test_bar_all_distinct():
    assert abs(M.bar(["A", "B", "C"]) - 0.0) < 1e-9


def test_bar_empty_is_zero():
    assert M.bar([]) == 0.0


def test_cmr_equals_bar_prepruning():
    states = ["A", "A", "B", "C"]
    assert abs(M.cmr(states) - M.bar(states)) < 1e-9


def test_bar_cmr_pair():
    before = ["A", "A", "A", "A"]  # bar 0.75
    after = ["A", "B"]             # cmr on after = 0.0
    out = M.bar_cmr(before, after)
    assert abs(out["bar"] - 0.75) < 1e-9
    assert abs(out["cmr"] - 0.0) < 1e-9


def test_bar_cmr_default_after_is_before():
    # when states_after omitted, cmr is measured on the same (pre-pruning) set
    states = ["A", "A", "B", "C"]
    out = M.bar_cmr(states)
    assert abs(out["bar"] - out["cmr"]) < 1e-9


# ---------------------------------------------------------------- markov_js / markov_kl
def test_markov_js_zero_for_identical():
    d = {"A": 0.5, "B": 0.5}
    assert abs(M.markov_js(d, dict(d))) < 1e-9


def test_markov_js_pos_for_divergent():
    # disjoint supports -> JS at its max (ln2 in nats ~0.693)
    assert M.markov_js({"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}) > 0.6


def test_markov_js_symmetric():
    p = {"A": 0.7, "B": 0.2, "C": 0.1}
    q = {"A": 0.1, "B": 0.6, "C": 0.3}
    assert abs(M.markov_js(p, q) - M.markov_js(q, p)) < 1e-12


def test_markov_js_missing_keys_handled():
    # q omits B; treated as 0 mass on B over the union alphabet
    p = {"A": 0.5, "B": 0.5}
    q = {"A": 1.0}
    assert M.markov_js(p, q) > 0.0


def test_markov_js_nonneg():
    p = {"A": 0.3, "B": 0.7}
    q = {"A": 0.9, "B": 0.1}
    assert M.markov_js(p, q) >= 0.0


def test_markov_kl_zero_for_identical():
    d = {"A": 0.25, "B": 0.75}
    assert abs(M.markov_kl(d, dict(d))) < 1e-9


def test_markov_kl_positive_for_divergent():
    p = {"A": 0.9, "B": 0.1}
    q = {"A": 0.1, "B": 0.9}
    assert M.markov_kl(p, q) > 0.0


def test_markov_kl_missing_keys_handled():
    # q missing B -> p has mass on B -> KL well-defined and finite (q smoothed)
    p = {"A": 0.5, "B": 0.5}
    q = {"A": 1.0}
    v = M.markov_kl(p, q)
    assert v > 0.0 and math.isfinite(v)
