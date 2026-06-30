"""Instance set, oracle/greedy labels, and headroom stratification for M1 Countdown.

This module turns the Countdown domain (`core.countdown`) into a labelled set of
problem instances and defines the **headroom stratum**: instances that are solvable
by the exact backward oracle yet missed by a parameter-free, model-free greedy-myopic
proxy. The headroom stratum is the denominator for validity gate V1 (selectable
headroom is non-empty) and is fixed here, before any decisive run.

Determinism contract
--------------------
`load_instances({"set": "builtin_small"})` returns a FIXED, curated, deterministically
ordered list with stable ids. The curated tuple of (numbers, target) below was selected
once by `_curate_builtin_small` (a seeded RNG candidate sweep filtered to guarantee a
non-empty headroom stratum plus a couple of unsolvable instances) and is then hard-coded
so the set never drifts. `load_instances({"set": "generated", ...})` performs seeded
generation and is reproducible for a fixed spec.

The greedy-myopic proxy
-----------------------
`greedy_myopic_outcome` is NOT an experiment arm and uses no model. It is a structural
difficulty label: repeatedly take the single legal op whose resulting value is
numerically closest to the target, breaking ties by the smallest canonical op tuple,
until one value remains; report whether that value equals the target. Because a myopic
"closest to target" chain can be lured into a dead end on instances a global search
still solves, instances where the oracle succeeds but this proxy fails are exactly the
places a global trellis decode can beat a myopic forward chain.
"""

from dataclasses import dataclass
from fractions import Fraction
import random

from _deps.countdown import (
    parse_target_problem,
    legal_ops,
    apply,
    reachable,
    solve_one,
)


@dataclass(frozen=True)
class Instance:
    numbers: tuple  # tuple[int, ...]
    target: int
    id: str


# ---------------------------------------------------------------------------
# Curated built-in set (28 instances, all target 24).
#
# Each entry is (numbers, target). This list was produced ONCE, deterministically,
# by `_select_builtin_small` (a seeded sweep + seeded diverse subset pick over the
# `_curate_builtin_small` candidate buckets; see the bottom of this file) and then
# frozen here so the set never drifts. It is curated so that:
#   * the headroom stratum (solvable AND greedy-myopic fails) is non-empty (V1) —
#     16 headroom instances;
#   * greedy-solvable instances exist (so headroom is a STRICT subset) — 8 of them;
#   * unsolvable instances are present (so labels exercise both branches) — 4 of them.
# All 28 number-multisets are distinct and span a range of magnitudes.
#
# Ids are assigned positionally as "bs-00", "bs-01", ... by load_instances, so the
# ordering of this tuple defines the stable ids. To reproduce / re-derive this exact
# tuple, run `_select_builtin_small()` (deterministic).
# ---------------------------------------------------------------------------
_BUILTIN_SMALL = (
    # --- headroom: oracle-solvable, greedy-myopic proxy FAILS (16) -------------
    ((1, 2, 4, 12), 24),
    ((1, 2, 11, 13), 24),
    ((2, 6, 8, 12), 24),
    ((2, 8, 8, 11), 24),
    ((3, 3, 9, 11), 24),
    ((3, 5, 6, 8), 24),
    ((3, 5, 11, 12), 24),
    ((4, 4, 12, 13), 24),
    ((4, 5, 5, 8), 24),
    ((5, 6, 7, 9), 24),
    ((5, 8, 9, 11), 24),
    ((5, 8, 9, 12), 24),
    ((6, 7, 11, 13), 24),
    ((6, 8, 8, 12), 24),
    ((8, 10, 11, 11), 24),
    ((10, 13, 13, 13), 24),
    # --- control: oracle-solvable AND greedy-myopic proxy SUCCEEDS (8) ---------
    ((1, 3, 3, 7), 24),
    ((1, 3, 5, 6), 24),
    ((1, 4, 8, 8), 24),
    ((1, 4, 9, 12), 24),
    ((1, 7, 8, 9), 24),
    ((2, 2, 4, 6), 24),
    ((2, 7, 7, 12), 24),
    ((3, 4, 4, 11), 24),
    # --- unsolvable (oracle says no path reaches target) (4) ------------------
    ((1, 6, 11, 11), 24),
    ((2, 5, 5, 10), 24),
    ((2, 6, 11, 11), 24),
    ((7, 12, 12, 12), 24),
)


def _make_state(inst):
    return parse_target_problem(inst.numbers, inst.target)


def load_instances(spec):
    """Return a deterministic, ordered list of `Instance` for `spec`.

    spec={"set":"builtin_small"} -> the frozen curated set (stable ids bs-00..).
    spec={"set":"generated","seed":S,"n":N,"k":4,"target_range":[a,b]}
        -> N seeded k-number instances; reproducible for a fixed spec.
    """
    kind = spec.get("set")
    if kind == "builtin_small":
        return [
            Instance(numbers=tuple(nums), target=int(tgt), id=f"bs-{idx:02d}")
            for idx, (nums, tgt) in enumerate(_BUILTIN_SMALL)
        ]
    if kind == "generated":
        return _load_generated(spec)
    raise ValueError(f"unknown instance spec set: {kind!r}")


def _load_generated(spec):
    seed = int(spec["seed"])
    n = int(spec["n"])
    k = int(spec.get("k", 4))
    a, b = spec.get("target_range", [10, 100])
    a, b = int(a), int(b)
    rng = random.Random(seed)
    insts = []
    for idx in range(n):
        numbers = tuple(rng.randint(1, 13) for _ in range(k))
        target = rng.randint(a, b)
        insts.append(
            Instance(numbers=numbers, target=target, id=f"gen-{seed}-{idx:03d}")
        )
    return insts


def label_instance(inst):
    """Oracle label: {"solvable": bool, "witness_len": int|None}.

    `witness_len` is len(solve_one) when solvable, else None.
    """
    state = _make_state(inst)
    if not reachable(state):
        return {"solvable": False, "witness_len": None}
    witness = solve_one(state)
    # reachable() True implies solve_one() returns a path; guard defensively.
    witness_len = len(witness) if witness is not None else None
    solvable = witness_len is not None
    return {"solvable": solvable, "witness_len": witness_len}


def greedy_myopic_outcome(inst):
    """Parameter-free, model-free greedy-myopic proxy (NOT an experiment arm).

    Repeatedly take the legal op whose resulting single value is numerically closest
    to the target, breaking ties by the smallest canonical op tuple, until one value
    remains. Return whether that final value equals the target. Used ONLY to label
    structural difficulty for the headroom stratum.
    """
    state = _make_state(inst)
    target = Fraction(inst.target)
    while len(state.values) > 1:
        ops = legal_ops(state)
        if not ops:
            return False
        # Score each op by how close its PRODUCED value is to the target; break
        # ties by the smallest canonical op tuple (i, j, symbol). The sort key
        # (distance, op_tuple) makes the choice fully deterministic.
        best_op = min(
            ops,
            key=lambda op: (abs(_produced_value(state, op) - target), op),
        )
        state = apply(state, best_op)
    return state.values[0] == target


def _produced_value(state, op):
    """Return the single value produced by applying `op` to `state` (a Fraction).

    `op = (i, j, symbol)` indexes the sorted values; the produced value is a∘b.
    """
    i, j, symbol = op
    vals = tuple(sorted(state.values))
    a = vals[i]
    b = vals[j]
    if symbol == "+":
        return a + b
    if symbol == "*":
        return a * b
    if symbol == "-":
        return b - a
    if symbol == "/":
        return b / a
    raise ValueError(f"unknown operation symbol: {symbol!r}")


def headroom_stratum(instances):
    """Ids of instances that are oracle-solvable AND greedy_myopic_outcome is False.

    These are the instances where a global decode over Φ-merged canonical states can
    beat a myopic forward chain. Establishes validity gate V1 (non-empty headroom).
    """
    out = []
    for inst in instances:
        if label_instance(inst)["solvable"] and not greedy_myopic_outcome(inst):
            out.append(inst.id)
    return out


# ---------------------------------------------------------------------------
# Curation helper (offline use only; not exercised by the harness at runtime).
# Kept so the frozen _BUILTIN_SMALL set is reproducible / auditable.
# ---------------------------------------------------------------------------
def _curate_builtin_small(seed=0, n_candidates=4000, target=24):
    """Sweep seeded four-number candidates; return three labelled buckets.

    Returns (headroom, greedy_solved, unsolvable) lists of (numbers, target). Used
    once to choose _BUILTIN_SMALL; deterministic for a fixed seed so anyone can
    reproduce the curation and confirm the headroom stratum is non-empty.
    """
    rng = random.Random(seed)
    headroom, greedy_solved, unsolvable = [], [], []
    seen = set()
    for _ in range(n_candidates):
        numbers = tuple(sorted(rng.randint(1, 13) for _ in range(4)))
        key = (numbers, target)
        if key in seen:
            continue
        seen.add(key)
        inst = Instance(numbers=numbers, target=target, id="probe")
        solvable = label_instance(inst)["solvable"]
        greedy = greedy_myopic_outcome(inst)
        if solvable and not greedy:
            headroom.append((numbers, target))
        elif solvable and greedy:
            greedy_solved.append((numbers, target))
        elif not solvable:
            unsolvable.append((numbers, target))
    return headroom, greedy_solved, unsolvable


def _select_builtin_small():
    """Re-derive the exact frozen `_BUILTIN_SMALL` tuple, deterministically.

    Sweeps candidates with `_curate_builtin_small(seed=0)`, then picks a diverse but
    reproducible subset from each bucket via a seeded shuffle (so the set is not a
    monotonous lexicographic slice). Returns the ordered tuple of (numbers, target).
    Running this and comparing to `_BUILTIN_SMALL` audits that the frozen set is the
    genuine output of the documented procedure.
    """
    headroom, greedy_solved, unsolvable = _curate_builtin_small(
        seed=0, n_candidates=4000, target=24
    )

    def pick(bucket, k, seed):
        ordered = sorted(bucket)  # deterministic base order
        rng = random.Random(seed)
        rng.shuffle(ordered)  # seeded shuffle -> diverse but reproducible
        return sorted(ordered[:k])  # sort the chosen subset for a stable final order

    chosen = pick(headroom, 16, 101) + pick(greedy_solved, 8, 202) + pick(unsolvable, 4, 303)
    return tuple(chosen)
