"""Data-level decoupling SFT generator for COUNTDOWN (PREREG sec 3): roll a CORRECT
solve path, and at each intermediate state S emit the goal-preserving move set G(S) as
SFT rows.

This is the countdown analogue of ``algebra-decode/core/gen_sft_data.py``. Countdown is
the decoder's NATIVE domain: a state is a multiset of values + a target; a move combines
two values into one; the Phi key ``canon = (sorted values, target)`` STRICTLY SHRINKS
each step (one fewer value), so the step-trellis is finite, bounded, never recurs, and
distinct combine orders reaching the same multiset are genuinely Phi-merged (real
aliasing) -- unlike the algebra version whose canon (lhs-rhs scalar class) made
productive moves canon-invariant and the trellis degenerate.

At a non-goal state S of a SOLVABLE instance::

    G(S) = goal-preserving DISTINCT moves: every op in ``legal_ops(S)`` (already
           de-duplicated by RESULTING multiset) whose successor ``apply(S, op)``
           keeps a path to the target (``reachable(apply(S, op))``), rendered to its
           move TEXT via ``render_op(S, op)`` ("<a> <symbol> <b>", a <= b canonical).

``legal_ops`` is de-duped by outcome and ``reachable`` keeps only the ops that preserve a
path to the target, so each text in G(S) leads to a DISTINCT goal-preserving successor.
G(S) is non-empty on every non-goal state of a solvable instance: the witness's own next
op from ``solve_one`` is itself a goal-preserving legal op, so it (or its canonical
representative) is in G(S).

Two SFT targets, SAME prompt/format, differing ONLY in how many moves per S:

    coupled (= one-hot control): exactly ONE move per S -- the deterministic pick
        (prefer the witness op at S; else the first of G(S)). One jsonl row per S.

    decoupled (= treatment): up to ``m`` DISTINCT moves drawn ~uniform over G(S)
        (seeded, without replacement). ``min(m, |G(S)|)`` jsonl rows per S. This is the
        calibrated multi-peak target landing at the ``MOVE:`` slot.

Row schema (one JSON object per line)::

    {"prompt": render(S),                  # RAW render, NO format suffix -- the chat
                                           # framing is added identically at train +
                                           # inference by the real backend builder.
     "completion": "MOVE: <move text>",    # MOVE_LINE_PREFIX + " " + text
     "state_key": <canon(S)-as-str>,       # Phi key string; equal across coupled/
                                           # decoupled for the same S
     "instance_id": <inst.id>}

``gen(instances, out_dir, m_decoupled=4, seed=0)`` writes ``data/sft_coupled.jsonl`` and
``data/sft_decoupled.jsonl`` under ``out_dir`` and is DETERMINISTIC given ``seed``
(``random.Random`` seeded per S from a stable string of seed + the state's Phi key). It
is CPU-only and never loads a model.
"""

import json
import random
from pathlib import Path

from _deps.countdown_domain import CountdownDomain

# The supervised completion is ``MOVE: <move>``; the ``MOVE:`` literal is the single
# source of truth shared with the real emission backend (train and inference must agree
# on the exact slot). Source it from ``real_backend`` when present; fall back to the
# coordinated literal ``"MOVE:"`` so this module imports standalone (CPU, no model).
try:  # pragma: no cover - exercised both ways depending on backend availability
    from real_backend import MOVE_LINE_PREFIX
except Exception:  # real_backend not yet vendored / importable
    MOVE_LINE_PREFIX = "MOVE:"


_D = CountdownDomain()

# completion == MOVE_LINE_PREFIX + " " + <move text>.
_COMPLETION_PREFIX = MOVE_LINE_PREFIX + " "


def _state_key(state):
    """Stringified Phi key ``canon(S)`` for the row's ``state_key`` field.

    ``canon`` returns ``(tuple[Fraction, ...], Fraction)`` which is hashable but not JSON
    serialisable; we render it to a stable string so the row schema is plain JSON and the
    key is equal across coupled/decoupled rows for the SAME canonical state.
    """
    return str(_D.canon(state))


def _prompt(state):
    """Stable prompt for a state: the RAW domain render.

    The MOVE-format framing (chat system + user template) is added IDENTICALLY at both
    training and inference by the real backend's prompt builder -- it must NOT be baked
    in differently here (that would be a train/inference prompt mismatch). So the stored
    ``prompt`` is exactly ``render(S)``.
    """
    return _D.render(state)


def _completion(move_text):
    """The MOVE-slot completion for a move text."""
    return _COMPLETION_PREFIX + move_text


def goal_preserving_moves(state):
    """G(S): goal-preserving DISTINCT move texts legal at ``state`` (PREREG sec 3).

    Every op in ``legal_ops(state)`` (de-duped by resulting multiset) whose successor
    ``apply(state, op)`` is still ``reachable`` (keeps a path to the target), rendered to
    its move text via ``render_op``. Order follows ``legal_ops`` (deterministic). The
    texts are distinct because the surviving ops have distinct successors. Non-empty on a
    non-goal state of a solvable instance (the witness's next op is goal-preserving).
    """
    keep = []
    for op in _D.legal_ops(state):
        if _D.reachable(_D.apply(state, op)):
            keep.append(_D.render_op(state, op))
    return keep


def _witness_first_text(state):
    """Move text of the witness's first op at ``state`` (``solve_one``), or None.

    The witness path's leading op is itself a goal-preserving legal op, so its text is a
    member of G(S) -- used as the deterministic coupled pick when available.
    """
    witness = _D.solve_one(state)
    if not witness:
        return None
    return _D.render_op(state, witness[0])


def _coupled_pick(state, moves):
    """Deterministic ONE-move pick over G(S): prefer the witness op's text at S; else the
    first member of G(S) (which follows the deterministic ``legal_ops`` order)."""
    witness_text = _witness_first_text(state)
    if witness_text is not None and witness_text in moves:
        return witness_text
    return moves[0]


def _decoupled_pick(state, moves, m, seed):
    """Up to ``m`` DISTINCT moves drawn ~uniform (without replacement) over G(S).

    Seeded per state from a stable string of ``seed`` and the state's Phi key, so the
    draw is reproducible across processes (``canon`` is rendered to a stable string).
    Returns a deterministic ordering of the ``min(m, |G(S)|)`` chosen texts.
    """
    k = min(int(m), len(moves))
    rng = random.Random(f"sft-decoupled|{seed}|{_state_key(state)}")
    return rng.sample(list(moves), k)


def _path_states(inst):
    """Intermediate NON-GOAL states along the witness solve path of ``inst``, de-duped by
    their Phi key.

    Rolls the exact-oracle witness chain (``solve_one`` on the initial state) through the
    domain, yielding each state visited BEFORE ``is_goal`` (the root and every
    intermediate; the solved single-value leaf is excluded -- no move to label there).
    Canon-equivalent states are collapsed to their FIRST occurrence: they are the SAME
    node in the Phi-merged trellis, so ``canon(S)`` is a unique row-group key and
    supervision is not duplicated. Returns [] if the instance is unsolvable (no witness).
    """
    state0 = _D.initial_state(
        {"numbers": inst.numbers, "target": inst.target, "id": inst.id}
    )
    witness = _D.solve_one(state0)
    if witness is None:
        return []
    states = []
    seen = set()
    state = state0
    for op in [None] + list(witness):
        if op is not None:
            state = _D.apply(state, op)
        if _D.is_goal(state):
            break
        key = _D.canon(state)
        if key in seen:
            continue
        seen.add(key)
        states.append(state)
    return states


def _rows_for_instance(inst, m_decoupled, seed):
    """(coupled_rows, decoupled_rows) for one instance. Each row is a dict.

    For every non-goal state S on the witness solve path with non-empty G(S): coupled gets
    the single deterministic pick; decoupled gets up to ``m_decoupled`` uniform distinct
    moves. Skips a (defensively impossible on a solvable instance) empty G(S).
    """
    coupled, decoupled = [], []
    for state in _path_states(inst):
        moves = goal_preserving_moves(state)
        if not moves:
            continue
        prompt = _prompt(state)
        state_key = _state_key(state)

        pick = _coupled_pick(state, moves)
        coupled.append({
            "prompt": prompt,
            "completion": _completion(pick),
            "state_key": state_key,
            "instance_id": inst.id,
        })

        for text in _decoupled_pick(state, moves, m_decoupled, seed):
            decoupled.append({
                "prompt": prompt,
                "completion": _completion(text),
                "state_key": state_key,
                "instance_id": inst.id,
            })
    return coupled, decoupled


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def gen(instances, out_dir, m_decoupled=4, seed=0):
    """Write ``data/sft_coupled.jsonl`` and ``data/sft_decoupled.jsonl`` under ``out_dir``.

    For each instance, roll its correct witness solve path and emit, per non-goal state S,
    the coupled one-hot target and the decoupled multi-peak target (PREREG sec 3).
    Deterministic given ``seed`` (and the fixed instance order). Returns a small summary
    dict ``{instances, states, coupled_rows, decoupled_rows, coupled_path,
    decoupled_path}``.
    """
    out_dir = Path(out_dir)
    coupled_all, decoupled_all = [], []
    n_states = 0
    for inst in instances:
        coupled, decoupled = _rows_for_instance(inst, m_decoupled, seed)
        coupled_all.extend(coupled)
        decoupled_all.extend(decoupled)
        n_states += len(coupled)  # exactly one coupled row per labelled state

    coupled_path = out_dir / "data" / "sft_coupled.jsonl"
    decoupled_path = out_dir / "data" / "sft_decoupled.jsonl"
    _write_jsonl(coupled_path, coupled_all)
    _write_jsonl(decoupled_path, decoupled_all)

    return {
        "instances": len(instances),
        "states": n_states,
        "coupled_rows": len(coupled_all),
        "decoupled_rows": len(decoupled_all),
        "coupled_path": str(coupled_path),
        "decoupled_path": str(decoupled_path),
    }
