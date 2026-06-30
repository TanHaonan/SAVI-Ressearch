"""Sampler interface (`sample`) + deterministic GPU-free mock generators + parser.

This module implements the per-state emission primitive the decoder consumes:

    sample(state, N, temperature, seed, backend, mode="step") -> list[str]

The decoder reads candidate *texts* only and never logits. Two mock backends let
the whole harness be built and validated with zero GPU dependency; the real
generators (one-hot / decoupled, trained on the emission line) plug into the
identical interface later (the `real_*` backends raise until then).

Text formats (stable, documented)
---------------------------------
ONE operation is rendered as ``"<a> <symbol> <b>"`` (three SPACE-delimited tokens)
where ``a <= b`` are the two operand VALUES in CANONICAL (sorted) order and
``symbol in {'+','-','*','/'}``. Values are formatted as an integer when whole,
else as ``"p/q"`` (exact rationals, no floats), matching ``core.countdown._fmt``.
The space delimiter is deliberate: it keeps the op text unambiguous even when an
operand is itself a rational containing '/', e.g. ``"5/4 / 9"`` means (5/4) divided
into 9 -> 9 / (5/4). Examples (state values {3,7,8,9}):

    "3 + 7", "8 * 9", "3 - 9" meaning 9-3 (larger minus smaller),
    "2 / 7" for a state containing 2 and 7 meaning 7/2.

A CHAIN (full operation sequence) is the ';'-joined concatenation of single-op
texts, each parsed against the EVOLVING state, e.g. ``"3 + 7;8 * 9;10 - 72"``.

Determinism (process-stable)
----------------------------
Python's builtin ``hash()`` of strings is salted per process, so it cannot seed an
RNG that must be reproducible ACROSS processes. We instead derive every RNG seed
from a SHA-256 digest of a stable string built from the call arguments
(``seed``, ``canon(state)`` rendered exactly, ``backend``, ``mode``, ``N``,
``temperature``). Same arguments -> identical digest -> identical output list, in
any process / with any ``PYTHONHASHSEED``.
"""

import hashlib
import random

from _deps.countdown import canon, legal_ops, apply, solve_one, _fmt


# ---------------------------------------------------------------------------
# Op-text rendering and parsing
# ---------------------------------------------------------------------------

_SYMBOLS = ("+", "-", "*", "/")
_CHAIN_SEP = ";"


def render_op(state, op):
    """Render a single Op (i, j, symbol) over canon(state) as ``"<a> <symbol> <b>"``.

    ``a <= b`` are the operand VALUES at the canonical (sorted) indices i < j.
    Three space-delimited tokens (see module docstring for the format rationale).
    """
    i, j, symbol = op
    vals = tuple(sorted(state.values))
    a = vals[i]
    b = vals[j]
    return f"{_fmt(a)} {symbol} {_fmt(b)}"


def _parse_value(token):
    """Parse a value token ('int' or 'p/q') into a Fraction, or None if malformed.

    Rejects tokens carrying an arithmetic operator other than the single '/' of a
    rational (so a stray operator cannot sneak through as an operand).
    """
    from fractions import Fraction
    token = token.strip()
    if not token:
        return None
    if token.count("/") > 1:
        return None
    if any(s in token for s in ("+", "-", "*", " ")):
        return None  # no embedded operators / spaces inside an operand token
    try:
        # Fraction accepts both "5" and "7/2"; reject anything else (floats, words).
        return Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None


def _split_op(text):
    """Split a single-op text ``"<a> <symbol> <b>"`` into (value_a, symbol, value_b).

    Returns None on any malformation (wrong token count, unknown symbol, missing or
    non-numeric operand). The space delimiter makes parsing unambiguous even when an
    operand is a rational like ``5/4``.
    """
    if text is None:
        return None
    tokens = text.strip().split()
    if len(tokens) != 3:
        return None
    left, sym, right = tokens
    if sym not in _SYMBOLS:
        return None
    a = _parse_value(left)
    b = _parse_value(right)
    if a is None or b is None:
        return None
    return (a, sym, b)


def parse(text, state):
    """Map a single-op text back to a legal Op (i, j, symbol) over canon(state).

    Returns None if: malformed, unknown symbol, either operand value is not
    present in the current values, or the resulting op is not in legal_ops(state).
    Duplicate values: the first matching index pair is chosen. Round-trips with
    `render_op`.
    """
    parsed = _split_op(text)
    if parsed is None:
        return None
    a, symbol, b = parsed
    if symbol not in _SYMBOLS:
        return None
    # Canonical order: operand a must be <= operand b (render uses sorted operands).
    if a > b:
        a, b = b, a
    vals = tuple(sorted(state.values))
    # Find first index i with vals[i] == a, then first j > i with vals[j] == b.
    i = None
    for idx, v in enumerate(vals):
        if v == a:
            i = idx
            break
    if i is None:
        return None
    j = None
    for idx in range(i + 1, len(vals)):
        if vals[idx] == b:
            j = idx
            break
    if j is None:
        return None
    op = (i, j, symbol)
    # The op must be legal in this state. legal_ops de-dups by outcome, so an op
    # may be valid yet collapsed to a different representative index pair; accept
    # if the resulting state matches any legal op's outcome.
    legal = legal_ops(state)
    if op in legal:
        return op
    target_key = None
    try:
        target_key = canon(apply(state, op))
    except (ValueError, ZeroDivisionError):
        return None
    for lop in legal:
        if canon(apply(state, lop)) == target_key:
            # Equivalent legal op with the same outcome; return our op (it is a
            # valid (i,j,symbol) producing exactly this legal outcome).
            return op
    return None


def parse_chain(text, state):
    """Split a chain text on ';' and parse each op against the EVOLVING state.

    Returns the list of Ops, or None if the chain is empty/malformed or any step
    is illegal in the state reached so far.
    """
    if text is None:
        return None
    parts = [p for p in text.split(_CHAIN_SEP)]
    parts = [p for p in parts if p.strip() != ""]
    if not parts:
        return None
    ops = []
    cur = state
    for part in parts:
        op = parse(part, cur)
        if op is None:
            return None
        ops.append(op)
        cur = apply(cur, op)
    return ops


# ---------------------------------------------------------------------------
# Deterministic, process-stable RNG seeding
# ---------------------------------------------------------------------------

def _digest_seed(seed, state, backend, mode, N, temperature):
    """Derive a process-stable integer RNG seed from a SHA-256 digest.

    Built from a stable string of all arguments. canon(state) is rendered with
    `_fmt` so equivalent states produce the same digest regardless of value order.
    """
    vals, target = canon(state)
    state_str = ",".join(_fmt(v) for v in vals) + "|" + _fmt(target)
    # temperature is rounded to a stable decimal string to avoid float repr drift.
    temp_str = f"{float(temperature):.6f}"
    payload = "::".join([
        str(seed),
        state_str,
        str(backend),
        str(mode),
        str(int(N)),
        temp_str,
    ])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(seed, state, backend, mode, N, temperature):
    return random.Random(_digest_seed(seed, state, backend, mode, N, temperature))


# ---------------------------------------------------------------------------
# Chain construction (shared by both mocks)
# ---------------------------------------------------------------------------

def _build_chain(state, rng, decoupled, follow_oracle=False):
    """Build one full chain by repeatedly choosing a legal op until one value remains.

    Returns (chain_text, ops). For the one-hot mock, the choice at each step is the
    fixed first legal op (deterministic, collapsed). For the decoupled mock, ops are
    sampled (uniformly, varied) from the legal set so distinct draws yield distinct
    chains.

    `follow_oracle` (decoupled only): when True, from the current state we take the
    exact-oracle witness path (`solve_one`) for the remaining steps. The oracle's ops
    are themselves legal ops, so the chain stays strictly in-domain; this guarantees
    that on a solvable instance a *correct* chain appears among the decoupled draws
    (a property the downstream decoder relies on). When the state is unsolvable the
    oracle returns None and we fall back to uniform legal choices.
    """
    if decoupled and follow_oracle:
        witness = solve_one(state)
        if witness is not None:
            texts = []
            cur = state
            for op in witness:
                texts.append(render_op(cur, op))
                cur = apply(cur, op)
            return (_CHAIN_SEP.join(texts), list(witness))

    ops = []
    texts = []
    cur = state
    while len(cur.values) > 1:
        legal = legal_ops(cur)
        if not legal:
            break
        if decoupled:
            op = rng.choice(legal)
        else:
            op = legal[0]  # fixed collapsed choice
        texts.append(render_op(cur, op))
        ops.append(op)
        cur = apply(cur, op)
    return (_CHAIN_SEP.join(texts), ops)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_BACKENDS = {"mock_onehot", "mock_decoupled", "real_onehot", "real_decoupled"}


def sample(state, N, temperature, seed, backend, mode="step"):
    """Return N candidate TEXTS for `state` (see module docstring for formats).

    mode="step":  each candidate is ONE next operation rendered as text.
    mode="chain": each candidate is a FULL operation sequence (to one remaining
                  value), rendered as text.

    backend in {"mock_onehot","mock_decoupled","real_onehot","real_decoupled"}.
    The real_* backends raise NotImplementedError until the emission line is wired.

    Deterministic given (state, N, temperature, seed, backend, mode): identical
    args -> identical output list, across processes.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend: {backend!r}")
    if backend.startswith("real_"):
        raise NotImplementedError("emission line dependency")
    if mode not in ("step", "chain"):
        raise ValueError(f"unknown mode: {mode!r}")
    if N <= 0:
        return []

    decoupled = backend == "mock_decoupled"
    rng = _rng(seed, state, backend, mode, N, temperature)

    if mode == "step":
        return _sample_step(state, N, temperature, rng, decoupled)
    return _sample_chain(state, N, rng, decoupled)


def _sample_step(state, N, temperature, rng, decoupled):
    """N single-op texts. One-hot: N copies of one op. Decoupled: spread over ops."""
    legal = legal_ops(state)
    if not legal:
        return []  # terminal state: nothing to emit

    if not decoupled:
        # Collapsed emission: deterministically pick ONE legal op, return N copies.
        op = legal[0]
        text = render_op(state, op)
        return [text] * N

    # Decoupled / calibrated multimodality: spread N candidates across SEVERAL
    # distinct legal ops. Temperature modulates spread: higher temperature -> more
    # uniform across the legal set (more distinct ops represented); lower
    # temperature -> mass concentrated on a leading subset (but still >= the
    # number guaranteed by the spec when enough legal ops exist).
    #
    # We sample WITHOUT collapsing: ensure >= min(3, len(legal)) distinct legal ops
    # appear whenever >= 3 legal ops exist, then fill the remaining slots by a
    # temperature-shaped weighted draw over the legal ops.
    texts = []
    # Guarantee distinct coverage first.
    guaranteed = min(len(legal), max(3, _temp_breadth(temperature, len(legal))))
    guaranteed = min(guaranteed, N)
    # Deterministic but seed-shuffled order for which ops get the guaranteed slots.
    order = list(range(len(legal)))
    rng.shuffle(order)
    chosen_ops = [legal[order[k]] for k in range(guaranteed)]
    for op in chosen_ops:
        texts.append(render_op(state, op))

    # Fill the rest with a temperature-shaped weighted draw over ALL legal ops.
    remaining = N - len(texts)
    if remaining > 0:
        weights = _temp_weights(temperature, len(legal))
        # Reorder weights to follow `order` so the leading (guaranteed) ops keep
        # higher mass at low temperature -> stable, intuitive shape.
        pop = [legal[i] for i in order]
        fill = rng.choices(pop, weights=weights, k=remaining)
        for op in fill:
            texts.append(render_op(state, op))

    # Shuffle final list deterministically so guaranteed ops aren't always first.
    rng.shuffle(texts)
    return texts


def _temp_breadth(temperature, n_legal):
    """How many distinct ops to guarantee, growing with temperature (>= 3 floor).

    At temperature ~0 we still guarantee 3 (the spec floor for >=3 legal ops); at
    higher temperature we guarantee more, up to the full legal set.
    """
    t = max(0.0, float(temperature))
    # Map temperature in [0, ~2] roughly onto [3, n_legal].
    extra = int(round(t * 3))
    return 3 + extra


def _temp_weights(temperature, n):
    """Weights over n ops shaped by temperature.

    Higher temperature -> flatter (more uniform) weights; lower temperature ->
    steeper (mass on leading ops). Always strictly positive. Deterministic.
    """
    t = max(1e-3, float(temperature))
    # Geometric decay whose ratio -> 1 (uniform) as t grows.
    ratio = min(0.999, max(0.05, 1.0 - 1.0 / (1.0 + t)))
    w = []
    cur = 1.0
    for _ in range(n):
        w.append(cur)
        cur *= ratio
    return w


def _sample_chain(state, N, rng, decoupled):
    """N full-chain texts. One-hot: N copies of one chain. Decoupled: N varied chains."""
    if len(state.values) <= 1:
        return [""] * N  # already terminal: empty chain

    if not decoupled:
        # Collapsed: one deterministically-built chain, repeated N times.
        text, _ = _build_chain(state, rng, decoupled=False)
        return [text] * N

    # Decoupled: build N varied chains (varied first ops etc.). To mimic a
    # calibrated generator whose mass includes the correct completion, a fraction
    # of the draws follow the exact-oracle witness (still strictly in-domain),
    # guaranteeing that for a SOLVABLE instance at least one correct chain appears
    # among the N draws (a property the downstream decoder relies on). The rest are
    # varied uniform-legal chains. The oracle-follow slots are placed at
    # deterministic positions and the whole list is order-shuffled by the seeded RNG.
    n_oracle = max(1, N // 8) if solve_one(state) is not None else 0
    chains = []
    for k in range(N):
        follow = k < n_oracle
        text, _ = _build_chain(state, rng, decoupled=True, follow_oracle=follow)
        chains.append(text)
    rng.shuffle(chains)
    return chains
