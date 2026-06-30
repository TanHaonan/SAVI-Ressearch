"""Behavior-preservation proof: decode_core (via adapter) == M1's own decode.savi.

The generalized ``decode_core.savi`` is supposed to reproduce M1's Countdown decode
EXACTLY through the thin CountdownDomain adapter + a backend-baked ``sample`` closure.
For a grid of fixed builtin_small instances, seeds, (K, N), edge modes, backends, and
BOTH verifier settings, we assert decode_core's result matches M1's own decoder on:

  * ``ok``                                  (did the arm reach a goal),
  * whether the returned ``path`` replays from the start to a goal,
  * the final ``trellis_widths_after_merge`` list.

M1's code is imported and run UNMODIFIED. decode_core never imports M1; the adapter
lives only in the test tree.
"""

import itertools

import pytest

# decode_core (the thing under test).
from decode_core import decode as dc

# Test-only: the FROZEN M1 modules + the adapter.
from core.instances import load_instances
from core.countdown import parse_target_problem, apply, is_goal
from core import decode as m1_decode

from _countdown_adapter import CountdownDomain, make_sample


DOM = CountdownDomain

# A handful of fixed builtin_small instances spanning solvable + unsolvable.
_ALL = load_instances({"set": "builtin_small"})
_INSTANCES = [
    next(i for i in _ALL if m1_decode.oracle(i)),                 # first solvable
    next(i for i in _ALL if not m1_decode.oracle(i)),             # first unsolvable
    _ALL[5],                                                       # mid headroom
    _ALL[20],                                                     # a control
]

_BACKENDS = ["mock_decoupled", "mock_onehot"]
_KN = [(1, 8), (4, 16), (8, 16)]
_EDGE_MODES = ["support", "freq"]
_SEEDS = [1, 7]
_VERIFIER = [True, False]


def _replays_to_goal(inst, path):
    if path is None:
        return False
    cur = parse_target_problem(inst.numbers, inst.target)
    for op in path:
        cur = apply(cur, op)
    return is_goal(cur)


def _strip_trailing_zeros(widths):
    """Drop trailing all-zero layers.

    M1's decoder runs a FIXED ``len(s0.values) - 1`` layers, so once the verifier mask
    empties the frontier it keeps appending empty (width-0) layers up to that fixed
    count. The generalized decode_core decoder has no per-domain depth and instead stops
    at the first empty layer, so it records no trailing zeros. Trailing all-zero layers
    contain no trellis information (no states, no goals, no path), so they are stripped
    from both sides before the width lists are compared — the informative prefix must
    match exactly.
    """
    out = list(widths)
    while out and out[-1] == 0:
        out.pop()
    return out


def _grid():
    for inst, backend, (K, N), em, seed, verifier in itertools.product(
        _INSTANCES, _BACKENDS, _KN, _EDGE_MODES, _SEEDS, _VERIFIER
    ):
        yield inst, backend, K, N, em, seed, verifier


@pytest.mark.parametrize("inst,backend,K,N,em,seed,verifier", list(_grid()))
def test_savi_parity_with_m1(inst, backend, K, N, em, seed, verifier):
    sample = make_sample(backend)

    core_r = dc.savi(DOM, sample, inst, K, N, em, tau=0.7, seed=seed,
                     verifier=verifier)
    m1_r = m1_decode.savi(inst, K, N, em, backend, τ=0.7, seed=seed,
                          verifier=verifier)

    # 1. ok must match.
    assert core_r.ok == m1_r.ok, (
        f"ok mismatch inst={inst.id} backend={backend} K={K} N={N} "
        f"em={em} seed={seed} verifier={verifier}"
    )

    # 2. whether the returned path replays to a goal must match.
    assert _replays_to_goal(inst, core_r.path) == _replays_to_goal(inst, m1_r.path)
    # And when ok, both must actually replay to a goal.
    if core_r.ok:
        assert _replays_to_goal(inst, core_r.path)

    # 3. trellis_widths_after_merge must match exactly on the informative prefix
    #    (trailing all-zero layers stripped — see _strip_trailing_zeros).
    assert (_strip_trailing_zeros(core_r.trellis_widths_after_merge)
            == _strip_trailing_zeros(m1_r.trellis_widths_after_merge)), (
        f"after-merge widths mismatch inst={inst.id} backend={backend} K={K} "
        f"N={N} em={em} seed={seed} verifier={verifier}: "
        f"{core_r.trellis_widths_after_merge} vs {m1_r.trellis_widths_after_merge}"
    )


def test_parity_grid_nonempty_and_exercises_both_verifier_and_ok_branches():
    """Guard: the parity grid is non-trivial — it actually contains both verifier
    settings AND produces at least one ok=True and one ok=False outcome under M1, so
    the parity above is not vacuously over only one branch."""
    oks = set()
    verifiers = set()
    for inst, backend, K, N, em, seed, verifier in _grid():
        verifiers.add(verifier)
        m1_r = m1_decode.savi(inst, K, N, em, backend, τ=0.7, seed=seed,
                              verifier=verifier)
        oks.add(m1_r.ok)
    assert verifiers == {True, False}
    assert oks == {True, False}
