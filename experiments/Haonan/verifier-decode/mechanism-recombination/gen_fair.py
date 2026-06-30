"""The fair, competence-parameterized per-step generator ``P_p(move | state)``.

A single emission policy consumed IDENTICALLY by every arm (no arm ever gets oracle
injection). See ``PLAN.md`` "Core design".

Pipeline
--------
1. ``enumerate_legal_texts(domain, step_sampler, state)`` lists the legal move TEXTS at
   a state, dedup'd by successor canon (the SAME enumeration the M2 Markov / false-merge
   audits use: sample many step texts at high temperature, ``parse_move``, keep one text
   per distinct successor canon, deterministic order).
2. ``make_fair_generator(domain, enumerate_moves_fn, p, depth_cap)`` builds a
   ``sample(state, N, tau, seed, mode)`` closure:
   * classify each legal move GOOD (``domain.solvable(apply)``) vs BAD; weights
     good=p, bad=1-p, normalized within the enumerated legal set; all-same-class ->
     uniform.
   * ``mode="step"``: N i.i.d. weighted draws from ``P_p`` (move texts).
   * ``mode="chain"``: N autoregressive rollouts of ``P_p`` (draw 1 move, apply, repeat
     to ``is_goal`` / terminal / ``depth_cap``), ';'-joined; NO oracle injection.
   * ``tau == 0`` -> deterministic argmax-weight (greedy): the single highest-weight
     move each step (ties broken by successor canon then text).
   * process-stable sha256 seeding (seed, canon(state), p, mode, N, temp).

Determinism is process-stable: identical args -> identical output list across processes
(no Python ``hash()`` of strings; sha256 of a stable payload, mirroring the frozen
samplers).
"""

from __future__ import annotations

import hashlib
import random


# ---------------------------------------------------------------------------
# Legal-move-text enumeration (dedup by successor canon)
# ---------------------------------------------------------------------------

# Enumeration parameters. The mock step sampler caps its per-call breadth (a
# temperature-shaped subset of the legal set), so a SINGLE call does not surface the
# COMPLETE distinct-successor set the fair P_p needs. We therefore drive the SAME
# step-sampler enumeration to a FIXPOINT: draw a high-temperature batch under a sequence
# of varied seeds and accumulate distinct successor canons until ``_ENUM_PATIENCE``
# consecutive batches add nothing new (or a hard round cap is hit). This uses only the
# step sampler (no peeking at the domain's internal legal-op list), is deterministic
# (fixed seed sequence), and saturates reliably across Countdown / algebra branching.
_ENUM_N = 64
_ENUM_TAU = 2.0
_ENUM_PATIENCE = 4
_ENUM_MAX_ROUNDS = 64


def enumerate_legal_texts(domain, step_sampler, state, enum_n=_ENUM_N,
                          enum_tau=_ENUM_TAU, patience=_ENUM_PATIENCE,
                          max_rounds=_ENUM_MAX_ROUNDS):
    """Deterministic, COMPLETE list of legal move TEXTS at ``state``, dedup by successor.

    Mirrors the M2 ``run_m2._find_merge_points`` ``legal_moves`` enumeration (sample
    step texts at high temperature via ``step_sampler``, ``parse_move``, keep one text
    per distinct ``canon(apply(state, move))``) but DRIVES IT TO A FIXPOINT so the full
    legal set is recovered despite the mock's per-call breadth cap: successive
    high-temperature batches under varied (deterministic) seeds accumulate distinct
    successor canons until ``patience`` consecutive batches add nothing (or ``max_rounds``
    is reached). The first text reaching each new successor canon wins, so the order is
    the deterministic order the batches surfaced them.

    Returns ``[]`` at a terminal state (no legal move emitted).
    """
    seen_keys = set()
    texts = []
    stale = 0
    round_idx = 0
    while round_idx < max_rounds and stale < patience:
        # Vary the seed per round so the mock's randomized spread covers new ops; the
        # seed SEQUENCE is fixed, so the whole enumeration is deterministic.
        cands = step_sampler(state, enum_n, enum_tau, round_idx, "step")
        added = 0
        for c in cands:
            move = domain.parse_move(c, state)
            if move is None:
                continue
            sp = domain.apply(state, move)
            key = domain.canon(sp)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            texts.append(c)
            added += 1
        if added == 0:
            stale += 1
        else:
            stale = 0
        round_idx += 1
    return texts


# ---------------------------------------------------------------------------
# Process-stable RNG seeding (sha256 of seed, canon(state), p, mode, N, temp)
# ---------------------------------------------------------------------------

def _digest_seed(domain, seed, state, p, mode, N, temperature):
    """Process-stable integer RNG seed from a sha256 of a stable argument payload.

    ``canon(state)`` is rendered with ``repr`` so equivalent states (equal canon)
    produce equal digests regardless of value order. ``p`` and ``temperature`` are
    rounded to stable decimal strings to avoid float-repr drift.
    """
    state_str = repr(domain.canon(state))
    payload = "::".join([
        str(seed),
        state_str,
        f"{float(p):.6f}",
        str(mode),
        str(int(N)),
        f"{float(temperature):.6f}",
    ])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(domain, seed, state, p, mode, N, temperature):
    return random.Random(_digest_seed(domain, seed, state, p, mode, N, temperature))


# ---------------------------------------------------------------------------
# P_p over the legal set at a state
# ---------------------------------------------------------------------------

def _classify_weights(domain, texts, state, p):
    """Return parallel (texts, weights) under P_p over the enumerated legal ``texts``.

    Each text is classified GOOD if ``domain.solvable(apply(state, move))`` else BAD.
    good -> weight p, bad -> weight (1 - p), normalized. If all texts are the same
    class (all good or all bad), weights are uniform. Texts that fail to parse here are
    dropped (they were enumerated as legal, so this is defensive).
    """
    kept = []
    is_good = []
    for c in texts:
        move = domain.parse_move(c, state)
        if move is None:
            continue
        sp = domain.apply(state, move)
        kept.append((c, move, sp))
        is_good.append(bool(domain.solvable(sp)))
    if not kept:
        return [], []
    n_good = sum(1 for g in is_good if g)
    if n_good == 0 or n_good == len(kept):
        # all-same-class -> uniform
        weights = [1.0] * len(kept)
    else:
        weights = [float(p) if g else float(1.0 - p) for g in is_good]
    total = sum(weights)
    if total <= 0:
        weights = [1.0] * len(kept)
        total = float(len(kept))
    weights = [w / total for w in weights]
    return kept, weights


def _argmax_index(domain, kept, weights):
    """Deterministic argmax-weight index; ties broken by (canon(successor), text)."""
    best = None
    for idx, ((c, move, sp), w) in enumerate(zip(kept, weights)):
        key = (-w, repr(domain.canon(sp)), c)
        if best is None or key < best[0]:
            best = (key, idx)
    return best[1] if best is not None else None


# ---------------------------------------------------------------------------
# The fair generator
# ---------------------------------------------------------------------------

def make_fair_generator(domain, enumerate_moves_fn, p, depth_cap):
    """Build ``sample(state, N, tau, seed, mode)`` implementing P_p (see module doc).

    ``enumerate_moves_fn(state) -> [text]`` lists the legal move texts at a state
    (typically ``functools.partial(enumerate_legal_texts, domain, step_sampler)``).
    ``p`` is the competence parameter; ``depth_cap`` bounds chain rollouts.

    The returned closure never injects an oracle move: chains are pure autoregressive
    rollouts of P_p. ``solvable`` is used ONLY to CLASSIFY move competence (it is in the
    emission, identically for every arm, never in a decoder).
    """
    p = float(p)
    depth_cap = int(depth_cap)

    # The (state -> legal-texts) enumeration and the (state -> (texts, weights))
    # classification are PURE functions of canon(state), so both are memoized by canon
    # for the lifetime of this generator. The memo changes NO output (identical results,
    # just cached); it makes a savi decode -- which revisits each canon across the N step
    # candidates at a node -- two orders of magnitude cheaper. A fresh generator (fresh
    # empty memo) is built per (domain, p) cell, keeping cells independent + deterministic.
    _enum_memo = {}
    _class_memo = {}

    def _enumerate(state):
        key = domain.canon(state)
        cached = _enum_memo.get(key)
        if cached is None:
            cached = enumerate_moves_fn(state)
            _enum_memo[key] = cached
        return cached

    def _weights(state):
        key = domain.canon(state)
        cached = _class_memo.get(key)
        if cached is None:
            texts = _enumerate(state)
            cached = _classify_weights(domain, texts, state, p)
            _class_memo[key] = cached
        return cached

    def sample(state, N, tau, seed, mode):
        if mode not in ("step", "chain"):
            raise ValueError(f"unknown mode: {mode!r}")
        if N is None or int(N) <= 0:
            return []
        N = int(N)

        if mode == "step":
            return _sample_step(state, N, tau, seed)
        return _sample_chain(state, N, tau, seed)

    def _draw_one(state, rng, tau):
        """Draw ONE move text from P_p at ``state``; return (text, move, successor).

        ``tau == 0`` -> deterministic argmax-weight move; else a weighted draw. Returns
        ``None`` at a terminal state (no legal move).
        """
        kept, weights = _weights(state)
        if not kept:
            return None
        if float(tau) == 0.0:
            idx = _argmax_index(domain, kept, weights)
        else:
            idx = rng.choices(range(len(kept)), weights=weights, k=1)[0]
        return kept[idx]  # (text, move, successor)

    def _sample_step(state, N, tau, seed):
        # N i.i.d. weighted draws (texts). tau==0 -> N copies of the argmax move.
        rng = _rng(domain, seed, state, p, "step", N, tau)
        out = []
        for _ in range(N):
            drawn = _draw_one(state, rng, tau)
            if drawn is None:
                break  # terminal: nothing legal to emit
            out.append(drawn[0])
        return out

    def _sample_chain(state, N, tau, seed):
        # N autoregressive rollouts of P_p; ';'-joined move texts. No oracle injection.
        chains = []
        sep = ";"
        for k in range(N):
            # Per-rollout RNG: vary by rollout index k so the N rollouts differ while
            # staying process-stable. tau==0 makes every rollout the argmax chain.
            rng = _rng(domain, (seed, k), state, p, "chain", N, tau)
            cur = state
            texts = []
            for _depth in range(depth_cap):
                if domain.is_goal(cur):
                    break
                drawn = _draw_one(cur, rng, tau)
                if drawn is None:
                    break  # terminal / no legal move
                text, move, sp = drawn
                texts.append(text)
                cur = sp
            chains.append(sep.join(texts))
        return chains

    return sample
