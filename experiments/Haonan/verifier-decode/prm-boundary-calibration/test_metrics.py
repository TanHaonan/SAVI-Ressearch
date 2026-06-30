"""Unit tests for the pure calibration metrics (no GPU)."""
import metrics as M


def _rec(cfg, sid, n, label):
    return {"config": cfg, "id": sid, "steps": ["s"] * n, "label": label}


def test_step_table_all_correct():
    recs = [_rec("g", "a", 3, -1)]
    obs = M.build_step_table(recs, {"a": [0.9, 0.8, 0.7]})
    assert len(obs) == 3
    assert all(o.truth == 1 and o.dist is None for o in obs)


def test_step_table_erroneous_excludes_post_error():
    # first error at idx 2 in a 5-step solution -> 2 good + 1 bad, 2 excluded
    recs = [_rec("g", "a", 5, 2)]
    obs = M.build_step_table(recs, {"a": [0.9, 0.8, 0.1, 0.5, 0.5]})
    assert len(obs) == 3
    goods = [o for o in obs if o.truth == 1]
    bads = [o for o in obs if o.truth == 0]
    assert [o.dist for o in goods] == [-2, -1]
    assert len(bads) == 1 and bads[0].dist == 0 and bads[0].idx == 2


def test_perfect_scorer_is_uniform():
    recs = [_rec("g", f"e{i}", 4, 2) for i in range(20)] + \
           [_rec("g", f"c{i}", 4, -1) for i in range(20)]
    rw = {}
    for i in range(20):
        rw[f"e{i}"] = [0.9, 0.9, 0.1, 0.1]   # good high, bad low
        rw[f"c{i}"] = [0.9, 0.9, 0.9, 0.9]
    obs = M.build_step_table(recs, rw)
    rep = M.uniformity_report(obs, theta=0.5, n_boot=500)
    assert rep["fp_rate"][1] == 0.0 and rep["fn_rate"][1] == 0.0
    assert rep["verdict"] == "UNIFORM"


def test_positive_bias_flags_polarity():
    # bad steps scored HIGH (missed) -> FN >> FP -> concentrated polarity.
    # need >= min_n (50) bad steps for the polarity flag to be considered.
    recs = [_rec("g", f"e{i}", 4, 2) for i in range(60)]
    rw = {f"e{i}": [0.9, 0.9, 0.85, 0.0] for i in range(60)}  # step 2 (bad) scored 0.85
    obs = M.build_step_table(recs, rw)
    rep = M.uniformity_report(obs, theta=0.5, n_boot=500)
    assert rep["fn_rate"][1] > rep["fp_rate"][1]
    assert rep["verdict"] == "CONCENTRATED"
    assert any("polarity" in f for f in rep["flags"])


def test_is_error_and_polarity():
    good = M.StepObs("g", "a", 0, 2, -1, 1, 0.2)   # good scored low -> FP
    bad = M.StepObs("g", "a", 1, 2, 0, 0, 0.8)     # bad scored high -> FN
    assert M.is_error(good, 0.5) == 1 and M.polarity(good, 0.5) == "FP"
    assert M.is_error(bad, 0.5) == 1 and M.polarity(bad, 0.5) == "FN"


def test_boot_ci_mean():
    lo, mean, hi = M.boot_ci([0, 0, 1, 1], n_boot=500)
    assert abs(mean - 0.5) < 1e-9 and lo <= mean <= hi


def test_choose_threshold_separates():
    recs = [_rec("g", f"e{i}", 3, 1) for i in range(10)]
    rw = {f"e{i}": [0.9, 0.2, 0.5] for i in range(10)}  # bad step (idx1) = 0.2
    obs = M.build_step_table(recs, rw)
    theta, f1 = M.choose_threshold(obs)
    assert f1 == 1.0 and 0.2 < theta < 0.9
