"""L0 scaffold smoke: the vendored decode path runs end-to-end on the COUNTDOWN domain
with the GPU-free MOCK emission backend, and PROVES the canon-merged step-trellis is
non-degenerate here -- the thing the algebra version failed.

Why this matters (the pivot)
----------------------------
The algebra capstone's Phi (canon = lhs-rhs scalar class) made productive moves
canon-invariant, so the merged step-trellis was degenerate and savi could not assemble a
solving path through it. Countdown is the decoder's NATIVE domain: a state is a multiset
of values + a target; a move combines two values into one; the Phi key
``canon = (sorted values, target)`` STRICTLY SHRINKS each step (one fewer value). So the
step-trellis is finite, bounded, never recurs, and different combine orders that reach
the same multiset are genuinely Phi-merged (real aliasing). This file demonstrates all
three: (a) savi REACHES a goal, (b) the trellis is BOUNDED + fast, (c) the arm ordering
oracle >= best_of_k >= greedy holds.

Setup
-----
Build ``CountdownDomain``; load SOLVABLE builtin countdown instances; run
greedy / best_of_k(K=16) / savi(K=8, N=16, edge_mode="freq", verifier=False,
max_depth=10) / oracle with the mock backends (the decoupled treatment + the one-hot
control). The decoupled mock spreads candidates over distinct legal combines, so its
trellis branches (width ~30 at the middle layer) and savi can find the goal; the one-hot
mock collapses to one op per step (width 1) -- a well-formed DEGENERATE trellis (no
exploration), which is the contrast the experiment is about, not a wiring failure.
"""

import sys
import time
from pathlib import Path

import pytest

# The experiment root (the dir containing the ``core`` package) on sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import core  # noqa: E402  (path setup must precede the import)


# decode hyperparameters (the PREREG smoke settings).
_K = 8            # savi beam width
_BOK_K = 16       # best_of_k draws (the ceiling ruler)
_N = 16           # samples per state
_TAU = 1.0
_SEED = 0
_MAX_DEPTH = 10
_TIME_BUDGET_S = 10.0   # per-decode hard ceiling (countdown decode is ~0.02s)
_BACKENDS = ("mock_decoupled", "mock_onehot")
# The treatment backend whose calibrated multi-peak emission makes the trellis branch;
# the non-degeneracy claim (savi reaches a goal) is asserted on this one. The one-hot
# control collapses each step to a single op (width-1 trellis) by design.
_TREATMENT = "mock_decoupled"


def _sampler(backend):
    """Adapt the vendored ``sample`` to the decoder's ``sample(state,N,temp,seed,mode)``
    signature by baking in ``backend`` (the emission closure the decoder consumes)."""
    return lambda state, N, temperature, seed, mode: core.sample(
        state, N, temperature, seed, backend, mode
    )


@pytest.fixture(scope="module")
def domain():
    return core.CountdownDomain()


@pytest.fixture(scope="module")
def instances(domain):
    """5 SOLVABLE builtin countdown instances (stable ids, deterministic).

    Taken from the curated greedy-solvable block (bs-16..) so each is oracle-solvable AND
    at least one is reachable by savi at the smoke beam width K=8 -- the slice over which
    the non-degeneracy assertion is meaningful (the headroom block bs-00.. is solvable but
    needs a wider beam, which the harness sweeps; the scaffold only proves the path EXISTS
    and is found at all). Every returned instance is asserted oracle-solvable below.
    """
    block_ids = ("bs-16", "bs-17", "bs-18", "bs-19", "bs-20", "bs-21", "bs-22", "bs-23")
    builtin = core.load_instances({"set": "builtin"})
    by_id = {i.id: i for i in builtin}
    solvable = [by_id[bid] for bid in block_ids
                if core.oracle(domain, core.instance_dict(by_id[bid]))]
    insts = solvable[:5]
    assert len(insts) == 5
    return insts


def _inst_dict(inst):
    return core.instance_dict(inst)


def test_instances_are_solvable_countdown(domain, instances):
    """Sanity on the smoke set: each builds a state, renders, and is oracle-solvable
    (the absolute ceiling is True) -- otherwise the smoke would prove nothing."""
    for inst in instances:
        di = _inst_dict(inst)
        s0 = domain.initial_state(di)
        # canon is a hashable structural key (sorted values tuple, target).
        key = domain.canon(s0)
        assert isinstance(key, tuple) and hash(key) is not None
        # render is a non-empty prompt string in the documented format.
        r = domain.render(s0)
        assert isinstance(r, str) and "target:" in r and "propose ONE" in r
        # absolute solvability ceiling + an explicit witness path of combines.
        assert core.oracle(domain, di) is True
        assert domain.solve_one(s0) is not None


def _assert_result(r):
    """Every arm Result must carry the documented fields with a populated Budget."""
    assert isinstance(r, core.Result)
    assert isinstance(r.ok, bool)
    assert isinstance(r.budget, core.Budget)
    assert r.budget.sample_calls >= 1
    assert r.budget.candidates >= 1
    assert isinstance(r.budget.tokens, int)
    assert isinstance(r.budget.exec, int)
    assert r.path is None or isinstance(r.path, list)
    assert isinstance(r.trellis_widths_before_merge, list)
    assert isinstance(r.trellis_widths_after_merge, list)
    assert r.best_score is None or isinstance(r.best_score, float)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_greedy_arm(domain, instances, backend):
    sample = _sampler(backend)
    for inst in instances:
        r = core.greedy(domain, sample, _inst_dict(inst), seed=_SEED)
        _assert_result(r)
        assert r.budget.sample_calls == 1  # greedy = exactly one chain draw


@pytest.mark.parametrize("backend", _BACKENDS)
def test_best_of_k_arm(domain, instances, backend):
    sample = _sampler(backend)
    for inst in instances:
        r = core.best_of_k(domain, sample, _inst_dict(inst), K=_BOK_K, tau=_TAU, seed=_SEED)
        _assert_result(r)
        assert r.budget.sample_calls == 1       # one call returning K chains
        assert r.budget.candidates == _BOK_K    # K chains drawn


@pytest.mark.parametrize("backend", _BACKENDS)
def test_savi_runs_and_is_bounded(domain, instances, backend):
    """savi(verifier=False, edge_mode="freq") = the lambda=0 emission-weighted Viterbi.

    CRITICAL ASSERTION (b) THE TRELLIS IS BOUNDED: because canon = (sorted values,
    target) strictly shrinks each step, the per-layer width list is SHORT and FINITE
    (no recurrence, no explosion) and each decode finishes far under the time budget.
    For a 4-number countdown instance the trellis is at most 3 informative layers
    (4 -> 3 -> 2 -> 1 values) regardless of beam width.
    """
    sample = _sampler(backend)
    for inst in instances:
        di = _inst_dict(inst)
        t0 = time.time()
        r = core.savi(
            domain, sample, di,
            K=_K, N=_N, edge_mode="freq", tau=_TAU, seed=_SEED,
            verifier=False, max_depth=_MAX_DEPTH,
        )
        dt = time.time() - t0
        _assert_result(r)
        # before/after merge width lists agree in length and honor the depth cap.
        assert len(r.trellis_widths_after_merge) == len(r.trellis_widths_before_merge)
        assert len(r.trellis_widths_after_merge) <= _MAX_DEPTH
        # BOUNDED: state shrinks each step -> a short finite list of positive widths,
        # never exceeding (#values - 1) informative layers (4-number instance -> <= 3).
        n_values = len(domain.initial_state(di).values)
        assert 0 < len(r.trellis_widths_after_merge) <= n_values - 1
        assert all(isinstance(w, int) and w >= 1 for w in r.trellis_widths_after_merge)
        # after-merge width <= before-merge width at every layer (Phi merged aliases).
        for wa, wb in zip(r.trellis_widths_after_merge, r.trellis_widths_before_merge):
            assert wa <= wb
        # FAST: decode finishes well under the per-instance time budget.
        assert dt < _TIME_BUDGET_S, f"savi took {dt:.3f}s (> {_TIME_BUDGET_S}s)"
        # In freq mode a reached goal carries a real (<=0) log-frequency path score.
        if r.ok:
            assert isinstance(r.best_score, float) and r.best_score <= 0.0


def test_savi_reaches_goal_non_degenerate(domain, instances):
    """CRITICAL ASSERTION (a): savi REACHES a goal on >=1 instance under the decoupled
    (treatment) emission -- proving the canon-merged step-trellis is NON-DEGENERATE here,
    unlike the algebra version (where canon-invariant productive moves made savi unable to
    assemble any solving path).

    The decoupled mock spreads its N candidates over distinct legal combines, so the
    trellis genuinely BRANCHES (middle-layer width well above 1 after Phi-merge); savi
    then finds a path whose terminal is the target single-value state. We assert both the
    branch (a wide merged layer somewhere) and the reach (>=1 ok).
    """
    sample = _sampler(_TREATMENT)
    oks = []
    branched = False
    for inst in instances:
        r = core.savi(
            domain, sample, _inst_dict(inst),
            K=_K, N=_N, edge_mode="freq", tau=_TAU, seed=_SEED,
            verifier=False, max_depth=_MAX_DEPTH,
        )
        oks.append(r.ok)
        if any(w > 1 for w in r.trellis_widths_after_merge):
            branched = True
    # Non-degenerate: the merged trellis is wider than 1 (real aliasing / exploration)
    # AND savi assembles a goal-reaching path on at least one instance.
    assert branched, "decoupled trellis never branched (width-1 everywhere): degenerate"
    assert sum(oks) >= 1, (
        f"savi reached no goal on any smoke instance under {_TREATMENT!r}: "
        f"the canon-merged step-trellis would be degenerate (oks={oks})"
    )


def test_onehot_trellis_is_collapsed(domain, instances):
    """Contrast: the ONE-HOT mock collapses each step to a single legal op, so the
    merged trellis is width-1 at every layer -- a well-formed DEGENERATE trellis (no
    exploration). This is the control the decoupled treatment is measured against, and it
    confirms the width-1 case is reachable and handled (not a crash)."""
    sample = _sampler("mock_onehot")
    for inst in instances:
        r = core.savi(
            domain, sample, _inst_dict(inst),
            K=_K, N=_N, edge_mode="freq", tau=_TAU, seed=_SEED,
            verifier=False, max_depth=_MAX_DEPTH,
        )
        _assert_result(r)
        # Collapsed emission -> every merged layer has exactly one canonical state.
        assert all(w == 1 for w in r.trellis_widths_after_merge)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_oracle_arm(domain, instances, backend):
    """oracle = absolute solvability ceiling (sampler-independent); True on this set."""
    for inst in instances:
        assert core.oracle(domain, _inst_dict(inst)) is True


@pytest.mark.parametrize("backend", _BACKENDS)
def test_arm_ordering_sanity(domain, instances, backend):
    """CRITICAL ASSERTION (c): oracle >= best_of_k >= greedy on mean pass@1 (Result.ok).

    pass@1 here is the decoder's ``Result.ok`` (reached a goal) for the Result arms and
    the solvability bool for oracle; an exact leaf-value check is the harness's job, not
    the scaffold's. The ordering is the structural sanity that the ceiling dominates the
    selection ruler dominates the floor.
    """
    sample = _sampler(backend)
    greedy_oks, bok_oks, oracle_oks = [], [], []
    for inst in instances:
        di = _inst_dict(inst)
        greedy_oks.append(core.greedy(domain, sample, di, seed=_SEED).ok)
        bok_oks.append(core.best_of_k(domain, sample, di, K=_BOK_K, tau=_TAU, seed=_SEED).ok)
        oracle_oks.append(core.oracle(domain, di))
    g = sum(greedy_oks) / len(greedy_oks)
    b = sum(bok_oks) / len(bok_oks)
    o = sum(oracle_oks) / len(oracle_oks)
    assert o >= b >= g, f"arm ordering violated: oracle={o} best_of_k={b} greedy={g}"


@pytest.mark.parametrize("backend", _BACKENDS)
def test_all_arms_run_together(domain, instances, backend):
    """The full arm matrix runs back to back on each instance with one backend, fast."""
    sample = _sampler(backend)
    for inst in instances:
        di = _inst_dict(inst)
        t0 = time.time()
        g = core.greedy(domain, sample, di, seed=_SEED)
        bk = core.best_of_k(domain, sample, di, K=_BOK_K, tau=_TAU, seed=_SEED)
        sv = core.savi(
            domain, sample, di, K=_K, N=_N, edge_mode="freq", tau=_TAU,
            seed=_SEED, verifier=False, max_depth=_MAX_DEPTH,
        )
        oc = core.oracle(domain, di)
        dt = time.time() - t0
        _assert_result(g)
        _assert_result(bk)
        _assert_result(sv)
        assert isinstance(oc, bool)
        assert dt < _TIME_BUDGET_S
