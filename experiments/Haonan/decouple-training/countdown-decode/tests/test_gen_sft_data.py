"""L0 datagen tests for ``core.gen_sft_data`` (PREREG sec 3 data-level decoupling, CPU).

Asserts the contract the train/decode siblings rely on, adapted from the algebra version
to COUNTDOWN semantics (state = multiset of values + target; a move combines two values
into one; canon = (sorted values, target)):

  * G(S) is non-empty, DISTINCT-by-successor, all-legal (parses via
    ``CountdownDomain.parse_move`` on S), and goal-PRESERVING (``reachable(apply(S,op))``)
    on every non-goal state along each instance's witness solve path;
  * some labelled states are genuinely AMBIGUOUS (|G(S)| > 1), else decoupled collapses
    to coupled and proves nothing;
  * coupled emits exactly ONE row per labelled state; decoupled emits ``min(m, |G(S)|)``
    rows per state, > 1 whenever |G(S)| > 1, and <= ``m`` always;
  * EVERY completion is ``"MOVE: <text>"`` whose ``<text>`` is a legal move on the row's
    own state (re-derived from the prompt's numbers + target) and whose canon matches the
    row's ``state_key``;
  * the coupled pick is itself a member of G(S);
  * the writer is DETERMINISTIC: same seed -> byte-identical jsonl files, and a different
    seed leaves the coupled (deterministic) file unchanged.
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

# The experiment root (the dir containing the ``core`` package) on sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import core  # noqa: E402  (path setup must precede the import)
from gen_sft_data import (  # noqa: E402
    goal_preserving_moves,
    _path_states,
    _rows_for_instance,
    _COMPLETION_PREFIX,
    _state_key,
)


_M = 4
_SEED = 0


@pytest.fixture(scope="module")
def domain():
    return core.CountdownDomain()


@pytest.fixture(scope="module")
def instances():
    """A spread of SOLVABLE countdown instances: curated builtin (headroom + greedy) plus
    seeded generated. Unsolvable builtin ids (no witness) yield no path states and are
    naturally skipped; we keep a few greedy + headroom shapes that have multi-step paths."""
    builtin = core.load_instances({"set": "builtin"})
    picked = [i for i in builtin
              if i.id in {"bs-00", "bs-05", "bs-09", "bs-16", "bs-20"}]
    gen = core.load_instances({"set": "generated", "seed": 0, "n": 8, "k": 4})
    # Keep only solvable instances (a witness path exists -> labelled states exist).
    return [i for i in picked + gen
            if _path_states(i)]


def _state_from_prompt(domain, prompt, inst_id):
    """Recover the countdown state from a render()-based prompt.

    render(S) == "numbers: <v1>, <v2>, ... | target: <t> | propose ONE next operation".
    Parse the numbers list and target back into an instance dict and build the state.
    """
    nums_part = prompt.split("numbers:", 1)[1].split("|", 1)[0].strip()
    target_part = prompt.split("target:", 1)[1].split("|", 1)[0].strip()
    numbers = tuple(_parse_num(tok.strip()) for tok in nums_part.split(","))
    target = _parse_num(target_part)
    return domain.initial_state(
        {"numbers": numbers, "target": target, "id": inst_id}
    )


def _parse_num(tok):
    """Parse an integer-or-'p/q' rendered value back into an int or Fraction."""
    if "/" in tok:
        return Fraction(tok)
    return int(tok)


# --------------------------------------------------------------------------- G(S)


def test_G_nonempty_distinct_legal_goal_preserving(domain, instances):
    """G(S) is non-empty and every member parses legal, is goal-preserving
    (reachable successor), and leads to a DISTINCT successor, on every non-goal state
    along each instance's witness path."""
    saw_state = False
    for inst in instances:
        states = _path_states(inst)
        assert states, f"no witness-path states for {inst.id}"
        for s in states:
            assert not domain.is_goal(s)  # _path_states excludes the solved leaf
            G = goal_preserving_moves(s)
            assert G, f"empty G(S) at {inst.id}: {domain.render(s)}"
            seen_keys = set()
            for text in G:
                op = domain.parse_move(text, s)
                assert op is not None, f"illegal move {text!r} in G(S)"
                nxt = domain.apply(s, op)
                assert domain.reachable(nxt)  # goal-preserving
                key = domain.canon(nxt)
                assert key not in seen_keys  # distinct by successor multiset
                seen_keys.add(key)
            saw_state = True
    assert saw_state


def test_G_witness_first_op_is_member(domain, instances):
    """The witness's own next op at S is goal-preserving and appears in G(S) (so G(S) is
    provably non-empty on a solvable state and the coupled pick is well defined)."""
    for inst in instances:
        for s in _path_states(inst):
            witness = domain.solve_one(s)
            assert witness, f"no witness at non-goal state of {inst.id}"
            first_text = domain.render_op(s, witness[0])
            assert first_text in goal_preserving_moves(s)


def test_G_has_multipeak_states(domain, instances):
    """At least some labelled states are genuinely AMBIGUOUS (|G(S)| > 1) -- otherwise
    the decoupled target would collapse to the coupled one and prove nothing."""
    multipeak = 0
    for inst in instances:
        for s in _path_states(inst):
            if len(goal_preserving_moves(s)) > 1:
                multipeak += 1
    assert multipeak > 0


# --------------------------------------------------------------- row-level contract


def test_coupled_one_row_per_state_decoupled_more(domain, instances):
    """coupled: exactly 1 row/state. decoupled: min(m,|G|) rows/state, > 1 when |G|>1."""
    for inst in instances:
        coupled, decoupled = _rows_for_instance(inst, _M, _SEED)
        states = [s for s in _path_states(inst) if goal_preserving_moves(s)]
        # coupled is one-to-one with labelled states.
        assert len(coupled) == len(states)
        # decoupled grouped by state_key matches min(m, |G(S)|) and exceeds 1 when |G|>1.
        by_key = {}
        for row in decoupled:
            by_key.setdefault(row["state_key"], []).append(row)
        for s in states:
            G = goal_preserving_moves(s)
            key = _state_key(s)
            rows = by_key[key]
            assert len(rows) == min(_M, len(G))
            assert len(rows) <= _M
            if len(G) > 1:
                assert len(rows) > 1
            # decoupled moves at a state are DISTINCT (sampled without replacement).
            comps = [r["completion"] for r in rows]
            assert len(set(comps)) == len(comps)


def test_every_completion_parses_on_its_state(domain, instances):
    """Each row's completion is 'MOVE: <text>' and <text> is a legal move on the state
    recovered from the row's own prompt (and that state's canon matches state_key)."""
    for inst in instances:
        coupled, decoupled = _rows_for_instance(inst, _M, _SEED)
        for row in coupled + decoupled:
            assert row["instance_id"] == inst.id
            assert row["completion"].startswith(_COMPLETION_PREFIX)
            # prompt is the RAW render(S) (no baked-in format suffix); the MOVE framing is
            # added identically at train/inference by the real backend's prompt builder.
            s = _state_from_prompt(domain, row["prompt"], inst.id)
            assert row["prompt"] == domain.render(s)
            assert _state_key(s) == row["state_key"]  # prompt <-> state_key agree
            move_text = row["completion"][len(_COMPLETION_PREFIX):]
            assert domain.parse_move(move_text, s) is not None  # legal on THIS state


def test_coupled_is_a_member_of_G(domain, instances):
    """The coupled pick at each state is itself a member of G(S) (the deterministic
    one-hot is drawn from the same goal-preserving set, not invented)."""
    for inst in instances:
        coupled, _ = _rows_for_instance(inst, _M, _SEED)
        by_key = {_state_key(s): goal_preserving_moves(s) for s in _path_states(inst)}
        for row in coupled:
            move_text = row["completion"][len(_COMPLETION_PREFIX):]
            assert move_text in by_key[row["state_key"]]


# ------------------------------------------------------------------- file writer


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def test_gen_writes_both_files_and_summary(tmp_path, instances):
    out = core.gen_sft_data(instances, tmp_path, m_decoupled=_M, seed=_SEED)
    coupled_path = tmp_path / "data" / "sft_coupled.jsonl"
    decoupled_path = tmp_path / "data" / "sft_decoupled.jsonl"
    assert coupled_path.exists() and decoupled_path.exists()
    coupled = [json.loads(l) for l in _read(coupled_path).splitlines()]
    decoupled = [json.loads(l) for l in _read(decoupled_path).splitlines()]
    # summary counts match the files.
    assert out["coupled_rows"] == len(coupled) == out["states"]
    assert out["decoupled_rows"] == len(decoupled)
    assert out["instances"] == len(instances)
    # decoupled has strictly more rows than coupled (multi-peak states exist).
    assert len(decoupled) > len(coupled)
    # every row carries exactly the four schema fields.
    for row in coupled + decoupled:
        assert set(row) == {"prompt", "completion", "state_key", "instance_id"}


def test_determinism_same_seed_identical_files(tmp_path, instances):
    """Same seed -> byte-identical jsonl on two independent runs (to fresh dirs)."""
    a, b = tmp_path / "a", tmp_path / "b"
    core.gen_sft_data(instances, a, m_decoupled=_M, seed=_SEED)
    core.gen_sft_data(instances, b, m_decoupled=_M, seed=_SEED)
    for name in ("sft_coupled.jsonl", "sft_decoupled.jsonl"):
        assert _read(a / "data" / name) == _read(b / "data" / name)


def test_seed_changes_decoupled_only(tmp_path, instances):
    """A different seed leaves coupled (deterministic pick) UNCHANGED but may reorder /
    re-sample the decoupled draw -- the one-hot control must not move with the seed."""
    a, b = tmp_path / "s0", tmp_path / "s1"
    core.gen_sft_data(instances, a, m_decoupled=_M, seed=0)
    core.gen_sft_data(instances, b, m_decoupled=_M, seed=1)
    # coupled is seed-independent.
    assert _read(a / "data" / "sft_coupled.jsonl") == _read(b / "data" / "sft_coupled.jsonl")
