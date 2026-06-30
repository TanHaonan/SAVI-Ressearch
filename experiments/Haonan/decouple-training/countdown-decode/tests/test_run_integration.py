"""L0 integration: the WHOLE arm matrix runs on the CPU mock backends and writes a
well-formed ``results.json`` (PREREG sec 5/8 L0) for COUNTDOWN.

This exercises ``run_decode.run`` end to end on ~5 solvable builtin countdown instances
with BOTH mock emissions (``mock_decoupled`` and ``mock_onehot``) and checks the contract
the sibling stages / the report reader depend on:

  * every arm in the PREREG sec 5 table is present with a ``pass1`` block whose ``mean`` is
    in [0,1], a 3-number bootstrap ``ci``, and ``n`` == the instance count;
  * the ceiling ordering ``oracle >= best_of_k >= greedy`` holds for BOTH backends (the
    sampler-free solvability ceiling can never be below a sampled arm);
  * the mock DECOUPLED backend actually SOLVES (greedy / best_of_k / oracle > 0) AND its
    lambda=0 trellis is NON-DEGENERATE: ``savi`` pass@1 > 0. This is the PIVOT's whole
    point -- countdown is the decoder's native domain where canon STRICTLY SHRINKS each
    step, so the freq trellis reaches the goal through Phi-merged alias paths (unlike the
    algebra version whose canon made productive moves canon-invariant and the trellis
    degenerate). ``savi`` is NOT asserted >= greedy: PREREG sec 6 registers "decoupled+savi
    不胜 greedy" as a live null, and the harness must REPORT that, not be rigged to pass it.

    NOTE on the ONE-HOT floor: the one-hot mock emits the deterministic FIRST legal op at
    every step (``sampler._build_chain`` with ``decoupled=False``), which is myopic and
    dead-ends on every builtin countdown instance -- so the one-hot greedy/best_of_k floor
    is genuinely 0 here (an empirically established fact, not a wiring failure). The one-hot
    backend is therefore only asserted to have a non-zero ORACLE ceiling (sampler-free) and
    to satisfy the ordering; it is NOT asserted to solve.
  * diagnostics are populated (coverage / parse_rate in [0,1]; savi width before/after
    present with after <= before == the Phi merge did something; budgets aligned with the
    decode_core accounting: greedy one chain, best_of_k K chains, savi many calls);
  * ``results.json`` is actually written and round-trips through json.

The metric helpers (``boot_ci`` / ``modal_state`` / ``leaf_check``) get their own focused
unit checks so a regression points at the helper, not the whole run.
"""

import json
import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import core  # noqa: E402
from core import metrics  # noqa: E402
from core.run_decode import (  # noqa: E402
    run, make_emit, load_held_out, main, _ALL_ARMS, _RESULT_ARMS,
)


# A small, fast held-out set: the first 5 builtin countdown instances (bs-00..bs-04, all
# solvable headroom). This is an L0 WIRING/SCHEMA test, so it runs the matrix on a trimmed
# workload (smaller N/K and a shallower savi depth) to stay comfortably under any CI time
# budget. The DOCUMENTED run command (STATUS report / PREREG sec 5) uses the full smoke
# settings (N=16, K=8, best_of_k=16, max_depth=10); nothing asserted here depends on the
# exact values, only that the wiring/schema/sanity hold. Module-scoped so the decode runs
# once per backend.
_N = 8
_K = 4
_BOK = 8
_TAU = 1.0
_SEED = 0
_MAX_DEPTH = 6
_N_INST = 5
_BACKENDS = ("mock_decoupled", "mock_onehot")


@pytest.fixture(scope="module")
def instances():
    insts = load_held_out("builtin")[:_N_INST]
    assert len(insts) == _N_INST
    # all 5 are solvable (the builtin headroom stratum bs-00..bs-04).
    d = core.CountdownDomain()
    for inst in insts:
        di = {"numbers": inst.numbers, "target": inst.target, "id": inst.id}
        assert core.oracle(d, di), f"{inst.id} expected solvable"
    return insts


@pytest.fixture(scope="module")
def results_by_backend(instances):
    """Run the full matrix once per mock backend; return {backend: results dict}."""
    out = {}
    for backend in _BACKENDS:
        emit, meta = make_emit(backend, None, "cpu", lambda *_: None)
        res = run(instances, emit, meta, N=_N, K=_K, tau=_TAU, max_depth=_MAX_DEPTH,
                  seed=_SEED, tag=f"test_{backend}", log=lambda *_: None)
        out[backend] = res
    return out


# ---------------------------------------------------------------------------
# metric helpers (focused units)
# ---------------------------------------------------------------------------

def test_boot_ci_bounds_and_shape():
    ci = metrics.boot_ci([1.0, 1.0, 0.0, 1.0], seed=0)
    assert len(ci) == 3
    assert ci[0] <= ci[1] <= ci[2]
    assert all(0.0 <= q <= 1.0 for q in ci)
    # empty -> all nan
    assert all(math.isnan(q) for q in metrics.boot_ci([]))


def test_modal_state_picks_majority_canon():
    d = core.CountdownDomain()
    # Build terminal states by rolling a couple of instances to single-value goals.
    # Three states canonical-equal to "[24], target 24" (a goal), one to "[20], target 24".
    s_goal = d.initial_state({"numbers": (24,), "target": 24, "id": "g"})
    s_goal_b = d.initial_state({"numbers": (24,), "target": 24, "id": "g2"})
    s_goal_c = d.initial_state({"numbers": (24,), "target": 24, "id": "g3"})
    s_other = d.initial_state({"numbers": (20,), "target": 24, "id": "o"})
    modal, count, share = metrics.modal_state(d, [s_goal, s_goal_b, s_goal_c, s_other])
    assert count == 3
    assert share == pytest.approx(0.75)
    assert d.canon(modal) == d.canon(s_goal)
    # empty bag
    assert metrics.modal_state(d, []) == (None, 0, 0.0)


def test_leaf_check_is_goal_target_match():
    d = core.CountdownDomain()
    target = 24
    # single value == target -> goal -> correct
    goal = d.initial_state({"numbers": (24,), "target": 24, "id": "g"})
    # single value != target -> not a goal -> wrong
    wrong_val = d.initial_state({"numbers": (20,), "target": 24, "id": "w"})
    # multiple values remaining -> not a goal (not yet a single value)
    not_terminal = d.initial_state({"numbers": (4, 6), "target": 24, "id": "n"})
    assert metrics.leaf_check(target, goal, d) is True
    assert metrics.leaf_check(target, wrong_val, d) is False
    assert metrics.leaf_check(target, not_terminal, d) is False
    assert metrics.leaf_check(target, None, d) is False


# ---------------------------------------------------------------------------
# results.json schema + arm presence + value ranges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", _BACKENDS)
def test_every_arm_present_with_pass1_block(results_by_backend, backend, instances):
    res = results_by_backend[backend]
    assert set(res["arms"]) == set(_ALL_ARMS)
    for arm in _ALL_ARMS:
        blk = res["arms"][arm]["pass1"]
        assert blk["n"] == len(instances)
        assert 0.0 <= blk["mean"] <= 1.0
        ci = blk["ci"]
        assert len(ci) == 3 and ci[0] <= ci[1] <= ci[2]
        assert all(0.0 <= q <= 1.0 for q in ci)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ceiling_ordering(results_by_backend, backend):
    """oracle >= best_of_k >= greedy holds for BOTH backends, and oracle is non-trivial."""
    res = results_by_backend[backend]
    m = {a: res["arms"][a]["pass1"]["mean"] for a in _ALL_ARMS}
    assert m["oracle"] >= m["best_of_k"] >= m["greedy"], m
    # the sampler-free solvability ceiling is non-zero on this all-solvable set.
    assert m["oracle"] > 0.0, m


def test_decoupled_solves_and_savi_nondegenerate(results_by_backend):
    """The PIVOT: mock_decoupled solves AND its lambda=0 trellis reaches goals (savi > 0).

    countdown is the decoder's native domain (canon strictly shrinks each step), so unlike
    the degenerate algebra trellis the freq trellis under the decoupled emission reaches the
    goal through Phi-merged alias paths -- savi pass@1 must be strictly positive.
    """
    res = results_by_backend["mock_decoupled"]
    m = {a: res["arms"][a]["pass1"]["mean"] for a in _ALL_ARMS}
    assert m["greedy"] > 0.0, m
    assert m["best_of_k"] > 0.0, m
    assert m["oracle"] > 0.0, m
    assert m["savi"] > 0.0, m  # <-- the non-degeneracy claim, the pivot's whole point


@pytest.mark.parametrize("backend", _BACKENDS)
def test_diagnostics_populated(results_by_backend, backend):
    res = results_by_backend[backend]
    for arm in _ALL_ARMS:
        diag = res["arms"][arm]["diagnostics"]
        # shared emission-side diagnostics on every arm.
        assert 0.0 <= diag["correct_terminal_coverage"] <= 1.0
        assert math.isnan(diag["parse_rate"]) or 0.0 <= diag["parse_rate"] <= 1.0
    # savi-specific aliasing proxy. The before/after keys are always present. They are
    # nan ONLY in the degenerate case where the trellis expanded NO layer at all. When
    # layers DID expand, after-merge width <= before-merge (the Phi merge collapsed alias
    # paths -- in countdown this is a real, non-trivial merge as distinct combine orders
    # reach the same value multiset).
    sd = res["arms"]["savi"]["diagnostics"]
    assert "trellis_width_before_merge" in sd and "trellis_width_after_merge" in sd
    bw, aw = sd["trellis_width_before_merge"], sd["trellis_width_after_merge"]
    assert math.isnan(bw) == math.isnan(aw)  # both present or both degenerate-empty
    if not math.isnan(bw):
        assert aw <= bw + 1e-9
    # self-consistency reports a modal share in (0,1].
    scd = res["arms"]["self_consistency"]["diagnostics"]
    assert 0.0 < scd["modal_share"] <= 1.0


def test_decoupled_trellis_merges_aliases(results_by_backend):
    """In countdown the decoupled trellis genuinely Phi-merges: after-width STRICTLY below
    before-width on average (different combine orders reach the same value multiset)."""
    sd = results_by_backend["mock_decoupled"]["arms"]["savi"]["diagnostics"]
    bw, aw = sd["trellis_width_before_merge"], sd["trellis_width_after_merge"]
    assert not math.isnan(bw) and not math.isnan(aw)
    assert aw < bw  # real aliasing collapse (the countdown pivot vs the degenerate algebra)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_budget_aligned_with_decode_core(results_by_backend, backend):
    """The iso-compute ledger matches decode_core accounting per arm."""
    res = results_by_backend[backend]
    for arm in _RESULT_ARMS:
        bud = res["arms"][arm]["diagnostics"]["budget"]
        assert set(bud) == {"sample_calls", "candidates", "tokens", "exec"}
        assert all(v >= 0 for v in bud.values())
    # greedy = exactly one chain draw (one sample call, one candidate).
    gb = res["arms"]["greedy"]["diagnostics"]["budget"]
    assert gb["sample_calls"] == pytest.approx(1.0)
    assert gb["candidates"] == pytest.approx(1.0)
    # best_of_k = one call returning K chains (default K=_K inside run()).
    bb = res["arms"]["best_of_k"]["diagnostics"]["budget"]
    assert bb["sample_calls"] == pytest.approx(1.0)
    assert bb["candidates"] == pytest.approx(float(_K))
    # savi draws many step-candidates: more sample calls than greedy.
    sb = res["arms"]["savi"]["diagnostics"]["budget"]
    assert sb["sample_calls"] >= 1.0
    assert sb["candidates"] >= sb["sample_calls"]


# ---------------------------------------------------------------------------
# full CLI path: writes results.json + run.log, round-trips
# ---------------------------------------------------------------------------

def test_main_writes_results_json(tmp_path):
    """``main`` (CLI entry) runs the matrix and writes a valid results.json + log."""
    out = tmp_path / "results.json"
    argv = [
        "--backend", "mock_decoupled",
        "--instances", "builtin",
        "--n", str(_N), "--k", str(_K), "--best_of_k", str(_BOK),
        "--max_depth", str(_MAX_DEPTH), "--seed", str(_SEED),
        "--tag", "ci_smoke",
        "--out", str(out),
    ]
    path = main(argv)
    assert Path(path) == out and out.exists()
    res = json.loads(out.read_text())
    assert set(res["arms"]) == set(_ALL_ARMS)
    assert res["tag"] == "ci_smoke"
    assert res["config"]["best_of_k_K"] == _BOK
    assert res["config"]["n_instances"] == 28  # the full builtin set
    # best_of_k arm carries its dedicated K in the diagnostics (re-run path: _BOK != _K).
    assert res["arms"]["best_of_k"]["diagnostics"]["K"] == _BOK
    # the tagged run log was written BESIDE the --out results (not in the shared outputs
    # dir), so a test never pollutes outputs/.
    log = out.parent / "run_ci_smoke.log"
    assert log.exists() and log.read_text().strip() != ""


# ---------------------------------------------------------------------------
# instance-spec parsing
# ---------------------------------------------------------------------------

def test_load_held_out_specs(tmp_path):
    builtin = load_held_out("builtin")
    assert len(builtin) == 28 and builtin[0].id == "bs-00"
    assert builtin[0].numbers == (1, 2, 4, 12) and builtin[0].target == 24
    gen = load_held_out("generated:seed=1,n=4,k=4")
    assert len(gen) == 4 and gen[0].id == "gen-1-000"
    gen_tr = load_held_out("generated:seed=2,n=3,k=3,target_range=20-30")
    assert len(gen_tr) == 3
    assert all(20 <= i.target <= 30 for i in gen_tr)
    assert all(len(i.numbers) == 3 for i in gen_tr)
    # a jsonl held-out file ({numbers, target, id})
    p = tmp_path / "held.jsonl"
    p.write_text('{"numbers": [3, 5, 6, 8], "target": 24, "id": "h-0"}\n'
                 '{"numbers": [2, 9], "target": 18, "id": "h-1"}\n')
    fromfile = load_held_out(str(p))
    assert [i.id for i in fromfile] == ["h-0", "h-1"]
    assert fromfile[0].numbers == (3, 5, 6, 8) and fromfile[0].target == 24
