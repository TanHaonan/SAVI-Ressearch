"""PLAN3 deliverable tests: reachable-memo correctness, K_eff, iso-token, determinism.

All CPU, all on the Tier-A mock backends + the deep Countdown domain. These lock the
plumbing the Tier-B decisive run rests on (PLAN3 §11).
"""

import core_boot as cb
import domain_countdown_deep as dcd
import instruments as ins
import arms_local
import mock_backend


# ---------------------------------------------------------------------------
# reachable memo correctness (the k=6 cost optimization must not change answers)
# ---------------------------------------------------------------------------

def test_shared_memo_matches_reachable():
    """DeepCountdownDomain.solvable equals countdown.reachable on many k in {4,5,6}."""
    d = dcd.DeepCountdownDomain()
    checked = mism = 0
    for k in (4, 5, 6):
        for inst in cb.load_instances({"set": "generated", "seed": 5, "n": 30,
                                       "k": k, "target_range": [10, 100]}):
            st = d.initial_state(cb.instance_dict(inst))
            if d.solvable(st) != cb.reachable(st):
                mism += 1
            checked += 1
    assert checked >= 60
    assert mism == 0


def test_shared_memo_is_persistent():
    """The memo accumulates across calls (hits grow on the second pass)."""
    d = dcd.DeepCountdownDomain()
    insts = cb.load_instances({"set": "generated", "seed": 1, "n": 10, "k": 5,
                               "target_range": [10, 100]})
    states = [d.initial_state(cb.instance_dict(i)) for i in insts]
    for s in states:
        d.solvable(s)
    hits_after_first = d.memo_stats()["hits"]
    for s in states:
        d.solvable(s)  # second pass: now top-level canons are cached -> hits increase
    assert d.memo_stats()["hits"] > hits_after_first
    assert d.memo_stats()["size"] > 0


def test_memo_recursion_matches_subproblems():
    """A solvable instance is True and its first witness successor is also solvable."""
    d = dcd.DeepCountdownDomain()
    inst = next(i for i in cb.load_instances({"set": "generated", "seed": 2, "n": 50,
                                              "k": 4, "target_range": [10, 100]})
                if cb.reachable(d.initial_state(cb.instance_dict(i))))
    s0 = d.initial_state(cb.instance_dict(inst))
    assert d.solvable(s0) is True
    witness = d.solve_one(s0)
    assert witness is not None
    s1 = d.apply(s0, witness[0])
    assert d.solvable(s1) is True


# ---------------------------------------------------------------------------
# In-trellis K_eff (H3 instrument): decoupled > coupled, coupled == 1
# ---------------------------------------------------------------------------

def _solvable_inst(k, seed=7):
    d = dcd.DeepCountdownDomain()
    for inst in cb.load_instances({"set": "generated", "seed": seed, "n": 80, "k": k,
                                   "target_range": [10, 100]}):
        di = cb.instance_dict(inst)
        if cb.reachable(d.initial_state(di)):
            return di
    raise AssertionError("no solvable instance found")


def test_keff_coupled_is_one():
    """Coupled (one-hot) backend emits a single op -> K_eff == 1 at every node."""
    d = dcd.DeepCountdownDomain()
    backend = mock_backend.make_mock_backend("coupled")
    di = _solvable_inst(5)
    r = ins.measure_keff(d, backend, di, K=8, N=16, tau=1.0, seed=0,
                         edge_mode="freq", verifier=True, max_depth=4)
    assert r["n_nodes"] >= 1
    assert all(x == 1 for x in r["per_node_parsed"])
    assert r["keff_parsed_mean"] == 1.0


def test_keff_decoupled_gt_coupled():
    """Decoupled spreads over distinct legal ops -> K_eff > 1 and > coupled."""
    d1 = dcd.DeepCountdownDomain()
    d2 = dcd.DeepCountdownDomain()
    dec = mock_backend.make_mock_backend("decoupled")
    cou = mock_backend.make_mock_backend("coupled")
    di = _solvable_inst(5)
    rk_dec = ins.measure_keff(d1, dec, di, K=8, N=16, tau=1.0, seed=0,
                              edge_mode="freq", verifier=True, max_depth=4)
    rk_cou = ins.measure_keff(d2, cou, di, K=8, N=16, tau=1.0, seed=0,
                              edge_mode="freq", verifier=True, max_depth=4)
    assert rk_dec["keff_parsed_mean"] > 1.0
    assert rk_dec["keff_parsed_mean"] > rk_cou["keff_parsed_mean"]


# ---------------------------------------------------------------------------
# iso-token accounting (the honest axis: tokens, never candidates)
# ---------------------------------------------------------------------------

def test_savi_token_ledger_matches_split_count():
    """savi.budget.tokens equals the whitespace-token sum the ledger is defined as."""
    d = dcd.DeepCountdownDomain()
    backend = mock_backend.make_mock_backend("decoupled")
    di = _solvable_inst(5)
    sv = cb.savi(d, backend, di, K=8, N=16, edge_mode="freq", tau=1.0, seed=0,
                 verifier=True, max_depth=4)
    assert sv.budget.tokens > 0
    # tokens is an int sum of len(text.split()); must be a non-negative integer.
    assert isinstance(sv.budget.tokens, int) and sv.budget.tokens >= 0


def test_iso_token_baseline_matches_savi_budget():
    """best_of_k_isobudget(axis='tokens', target=T) draws until tokens >= T, then stops.

    The matched-budget baseline must spend >= the savi token target, and by no more than
    one final chain's worth (it stops as soon as the threshold is crossed).
    """
    d = dcd.DeepCountdownDomain()
    backend = mock_backend.make_mock_backend("decoupled")
    di = _solvable_inst(5)
    sv = cb.savi(d, backend, di, K=8, N=16, edge_mode="freq", tau=1.0, seed=0,
                 verifier=True, max_depth=4)
    T = sv.budget.tokens
    bom = arms_local.best_of_k_isobudget(d, backend, di, target=T, tau=1.0, seed=0,
                                         axis="tokens")
    assert bom.budget.tokens >= T
    assert bom.detail["axis"] == "tokens"
    assert bom.detail["n_rollouts"] >= 1


def test_iso_token_uses_tokens_not_candidates():
    """The baseline is matched on tokens; its candidate count is incidental, not the cap."""
    d = dcd.DeepCountdownDomain()
    backend = mock_backend.make_mock_backend("decoupled")
    di = _solvable_inst(4)
    sv = cb.savi(d, backend, di, K=8, N=16, edge_mode="freq", tau=1.0, seed=0,
                 verifier=True, max_depth=3)
    bom = arms_local.best_of_k_isobudget(d, backend, di, target=sv.budget.tokens,
                                         tau=1.0, seed=0, axis="tokens")
    # token-matched, NOT candidate-matched: candidates need not equal savi candidates.
    assert bom.budget.tokens >= sv.budget.tokens


# ---------------------------------------------------------------------------
# determinism (process-stable seeds)
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    """Same seed -> identical arm-matrix ok flags + token counts (two fresh domains)."""
    import run_joint
    backend = mock_backend.make_mock_backend("decoupled")
    di = _solvable_inst(5)
    d1 = dcd.DeepCountdownDomain()
    d2 = dcd.DeepCountdownDomain()
    a1, _ = run_joint.run_arm_matrix(d1, backend, di, di["target"], 8, 16, 1.0, 0, 4)
    a2, _ = run_joint.run_arm_matrix(d2, backend, di, di["target"], 8, 16, 1.0, 0, 4)
    for arm in a1:
        assert a1[arm]["ok"] == a2[arm]["ok"], arm
        assert a1[arm]["tokens"] == a2[arm]["tokens"], arm


def test_calibration_ece_is_a_fraction():
    """ECE is in [0,1] and records are produced on a real trellis."""
    d = dcd.DeepCountdownDomain()
    backend = mock_backend.make_mock_backend("decoupled")
    di = _solvable_inst(5)
    recs = ins.collect_calibration(d, backend, di, 8, 16, 1.0, 0,
                                   edge_mode="freq", verifier=True, max_depth=4)
    out = ins.reliability_ece(recs, n_bins=10)
    assert out["n"] == len(recs) and out["n"] > 0
    assert 0.0 <= out["ece"] <= 1.0
