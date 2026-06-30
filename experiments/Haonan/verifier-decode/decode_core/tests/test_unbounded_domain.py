"""Regression tests: savi must terminate on UNBOUNDED and CYCLIC domains.

These use tiny, deterministic, dependency-free synthetic Domains (state = an integer)
to exercise the two termination guards added to ``decode_core.decode.savi``:

  * ``max_depth`` — a finite depth cap, so an INC-only domain whose layer is never
    empty still terminates within ``max_depth`` layers;
  * the cross-layer visited-set — so a CYCLIC domain (inc/dec returning to a seen
    canon) cannot grow the frontier without bound and terminates with work bounded by
    the number of DISTINCT canonical states.

The Countdown adapter is NOT used here; these domains stand alone. Determinism: the
fake ``sample`` returns a fixed list of move texts for every call (no RNG).
"""

import pytest

from decode_core import decode as dc


# ---------------------------------------------------------------------------
# Synthetic Domain: state is an int. Move "inc" -> state+1 (always legal).
# canon(state) = state. is_goal(state) = (state == target). No target on the inc-only
# path is reachable only if target > start; we choose target far enough that WITHOUT a
# depth cap the layer is never empty (inc is always legal), so the loop would not stop
# on its own.
# ---------------------------------------------------------------------------

class IncOnlyDomain:
    """Unbounded: from any integer, the only move ``inc`` (+1) is always legal."""

    def __init__(self, target):
        self.target = target
        # Instrument total apply() calls to bound work in assertions.
        self.apply_calls = 0

    def initial_state(self, instance):
        return int(instance)

    def canon(self, state):
        return state

    def is_goal(self, state):
        return state == self.target

    def apply(self, state, move):
        self.apply_calls += 1
        assert move == "inc"
        return state + 1

    def parse_move(self, text, state):
        return "inc" if text == "inc" else None

    def parse_chain(self, text, state):
        moves = [p for p in text.split(";") if p]
        return moves if all(m == "inc" for m in moves) else None

    def solvable(self, state):
        # A goal is reachable by repeated inc iff target >= state.
        return state <= self.target

    def render(self, state):
        return f"n={state}"


# ---------------------------------------------------------------------------
# Synthetic Domain: state is an int in {0, 1}. Moves "inc"/"dec" toggle it, so canons
# recur across layers (0 -> 1 -> 0 -> ...). Unbounded in DEPTH (always a legal move),
# but only TWO distinct canonical states exist, so the visited-set must bound total
# expansions to ~2 even with no depth cap.
# ---------------------------------------------------------------------------

class ToggleDomain:
    """Cyclic: state toggles between two canonical values via inc/dec."""

    def __init__(self):
        self.apply_calls = 0

    def initial_state(self, instance):
        return int(instance)  # 0

    def canon(self, state):
        return state

    def is_goal(self, state):
        return False  # never a goal -> forces the loop to rely on the guards

    def apply(self, state, move):
        self.apply_calls += 1
        if move == "inc":
            return 1 - state if state == 1 else state + 1  # 0->1, 1->1 (clamp)
        # "dec": 1 -> 0, 0 -> 0
        return state - 1 if state == 1 else 0

    def parse_move(self, text, state):
        return text if text in ("inc", "dec") else None

    def parse_chain(self, text, state):
        moves = [p for p in text.split(";") if p]
        return moves if all(m in ("inc", "dec") for m in moves) else None

    def solvable(self, state):
        return True

    def render(self, state):
        return f"t={state}"


def _fixed_sample(texts):
    """A deterministic ``sample(state, N, t, s, mode)`` returning N copies of ``texts``
    (cycled to length N). No RNG."""
    def sample(state, N, temperature, seed, mode):
        if N <= 0:
            return []
        return [texts[i % len(texts)] for i in range(N)]
    return sample


# ---------------------------------------------------------------------------
# (a) max_depth bounds an inc-only (never-empty-layer) domain.
# ---------------------------------------------------------------------------

def test_inc_only_without_cap_would_not_self_terminate_but_cap_bounds_it():
    # Target far above any depth we run, so a goal is never collected and every layer
    # always has the legal "inc" successor -> the empty-layer break never fires. Only
    # max_depth can stop it.
    dom = IncOnlyDomain(target=10**9)
    sample = _fixed_sample(["inc"])

    K = 4
    for cap in (1, 3, 5):
        dom.apply_calls = 0
        r = dc.savi(dom, sample, 0, K=K, N=4, edge_mode="support", tau=0.0, seed=1,
                    verifier=False, max_depth=cap)
        # Exactly `cap` layers recorded (each layer is non-empty: inc always legal and
        # each step's single successor is a NEW canon, so it is never skipped).
        assert len(r.trellis_widths_after_merge) == cap
        assert len(r.trellis_widths_before_merge) == cap
        # No goal in range -> ok False, path None.
        assert r.ok is False and r.path is None
        # The frontier is a single chain node at every layer (inc strictly increases the
        # canon, so each layer has exactly one distinct successor), so each layer does
        # N=4 applies (one per sampled candidate, all "inc"): bounded, finite,
        # deterministic. Total = cap * N regardless of K.
        assert dom.apply_calls == cap * 4


def test_inc_only_finds_goal_within_cap_when_reachable():
    # Goal at depth 3 from start 0 (0->1->2->3). A cap >= 3 must find it; the cross-layer
    # visited-set does not interfere (each inc yields a fresh canon).
    dom = IncOnlyDomain(target=3)
    sample = _fixed_sample(["inc"])
    r = dc.savi(dom, sample, 0, K=2, N=2, edge_mode="support", tau=0.0, seed=1,
                verifier=False, max_depth=5)
    assert r.ok is True
    assert r.path == ["inc", "inc", "inc"]


def test_max_depth_default_none_is_uncapped_and_self_terminates_when_layers_empty():
    # An inc-only domain whose ONLY successor is immediately a visited canon cannot
    # happen (inc strictly grows), so to show default-None self-termination we use a
    # target reachable then unreachable: start ABOVE target so solvable() is False under
    # verifier -> first layer prunes everything -> empty-layer break stops it with no cap.
    dom = IncOnlyDomain(target=0)
    sample = _fixed_sample(["inc"])
    r = dc.savi(dom, sample, 5, K=2, N=2, edge_mode="support", tau=0.0, seed=1,
                verifier=True, max_depth=None)  # start 5 > target 0 -> all pruned
    assert r.ok is False and r.path is None
    assert r.trellis_widths_after_merge == []  # empty-layer break at depth 0


# ---------------------------------------------------------------------------
# (b) Cross-layer visited-set bounds a cyclic domain even with NO depth cap.
# ---------------------------------------------------------------------------

def test_cyclic_domain_terminates_via_visited_set_without_cap():
    # Two distinct canons {0,1}; inc/dec cycle between them. With NO max_depth the only
    # thing that can stop the loop is the visited-set emptying the next layer once both
    # canons have been seen.
    dom = ToggleDomain()
    sample = _fixed_sample(["inc", "dec"])

    r = dc.savi(dom, sample, 0, K=8, N=8, edge_mode="support", tau=0.0, seed=1,
                verifier=False, max_depth=None)
    # Terminates (did not hang). Never a goal.
    assert r.ok is False and r.path is None
    # Only canon 1 is a NEW successor at depth 0 (canon 0 is the root, already seen);
    # at depth 1 every successor (0 and 1) is already seen -> empty layer -> stop.
    # So exactly ONE informative layer is recorded.
    assert r.trellis_widths_after_merge == [1]
    # Total expansions are bounded by O(distinct states * K * N), NOT unbounded.
    assert dom.apply_calls <= 2 * 8 * 8


def test_cyclic_domain_apply_calls_bounded_independent_of_large_N():
    # Even with a large N (many duplicate candidates per call), the visited-set keeps
    # the number of LAYERS bounded (2), so total work stays finite and proportional.
    dom = ToggleDomain()
    sample = _fixed_sample(["inc", "dec"])
    r = dc.savi(dom, sample, 0, K=4, N=64, edge_mode="support", tau=0.0, seed=1,
                verifier=False, max_depth=None)
    assert r.trellis_widths_after_merge == [1]
    # At most 2 layers actually expand (depth 0 then the empty depth-1), each layer
    # does K*N applies at most on the single-node frontier.
    assert dom.apply_calls <= 2 * 4 * 64


def test_visited_set_and_cap_compose():
    # Cyclic domain WITH a cap: terminates by whichever guard fires first; result is the
    # same as uncapped here because the visited-set empties the layer at depth 1 < cap.
    dom = ToggleDomain()
    sample = _fixed_sample(["inc", "dec"])
    capped = dc.savi(dom, sample, 0, K=8, N=8, edge_mode="support", tau=0.0, seed=1,
                     verifier=False, max_depth=10)
    assert capped.trellis_widths_after_merge == [1]
    assert capped.ok is False


def test_max_depth_zero_runs_no_layers():
    dom = IncOnlyDomain(target=3)
    sample = _fixed_sample(["inc"])
    r = dc.savi(dom, sample, 0, K=2, N=2, edge_mode="support", tau=0.0, seed=1,
                verifier=False, max_depth=0)
    assert r.trellis_widths_after_merge == []
    assert r.trellis_widths_before_merge == []
    assert r.ok is False and r.path is None
    assert dom.apply_calls == 0
