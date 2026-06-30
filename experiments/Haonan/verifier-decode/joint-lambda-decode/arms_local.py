"""PLAN3-local copies of the two extra arms (``decode_core`` stays FROZEN).

These are verbatim ports of ``mechanism-recombination/arms_ext.py`` so PLAN3's harness
is self-contained in this dir and binds the SAME (vendored) ``decode_core`` that
``core_boot`` loads — avoiding two divergent ``decode_core`` module copies on the path.
Both reuse ``decode_core``'s ``Budget`` / ``Node`` / ``Result`` / ``_backtrack`` so the
iso-compute ledger is shared with every other arm.

* ``best_of_k_isobudget`` — the iso-compute selection baseline. Draws chain rollouts one
  at a time (each a distinct seed) accumulating a chosen budget axis until it reaches a
  target. PLAN3's headline baseline is this with ``axis="tokens"`` and
  ``target = savi(lambda>0,freq).budget.tokens`` (the token-matched best-of-many).

* ``beam_no_merge`` — the Phi-merge ablation (H4 / E6c). Same layer-by-layer loop as
  ``savi`` but nodes are keyed PER-PATH (no canonical collapse), so the only difference
  from ``savi`` is the merge. ``savi - beam_no_merge`` isolates Phi.
"""

import math
import random

# Resolve the SAME vendored decode_core that core_boot/core loaded.
import core_boot as _cb  # noqa: F401  (ensures decode_core is importable + single-sourced)
from decode_core import decode as _dc  # noqa: E402

Budget = _dc.Budget
Node = _dc.Node
Result = _dc.Result


# ---------------------------------------------------------------------------
# masked_best_of_many — verifier-IN-THE-LOOP selection, NO beam / merge / DP (PLAN4 T1.1)
# ---------------------------------------------------------------------------

def masked_best_of_many(domain, sample, inst, target, N, tau, seed, max_depth,
                        axis="tokens", max_rollouts=1000000):
    """Independent masked rollouts, selected at iso-token. Isolates the per-step mask.

    Each rollout walks forward: at each step draw N step-candidates, parse, apply, and KEEP
    only successors with ``domain.solvable`` (the per-step exact mask savi uses); pick ONE
    surviving canonical successor stochastically, weighted by its emission frequency;
    advance to goal / no-solvable-emitted / depth. Draw rollouts until the token budget
    matches savi(K=8)'s, ok iff ANY rollout reaches the goal.

    This is exactly savi's verifier WITHOUT the beam, the Φ-merge, or any cross-path DP:
    one masked path per rollout, classic best-of-many over them. ``savi(K=8) −
    masked_best_of_many`` therefore isolates whatever the global decode adds beyond the
    per-step mask. Token + exec accounting mirror savi (exec += 2 per candidate: apply +
    solvable), so the comparison is iso-token AND iso-exec.
    """
    if axis not in ("candidates", "exec", "tokens"):
        raise ValueError(f"unknown axis: {axis!r}")
    budget = Budget()
    s0 = domain.initial_state(inst)
    target = int(target)
    win_path = None
    n_rollouts = 0
    while True:
        roll_seed = (int(seed) * 2_000_003 + n_rollouts) & 0x7FFFFFFF
        rng = random.Random(roll_seed)
        cur = s0
        path = []
        for d in range(max_depth):
            if domain.is_goal(cur):
                break
            step_seed = (roll_seed * 7919 + d) & 0x7FFFFFFF
            cands = sample(cur, N, tau, step_seed, "step")
            budget.record_sample(cands)
            tally = {}  # canon(sp) -> [count, move, sp], over SOLVABLE successors only
            for c in cands:
                mv = domain.parse_move(c, cur)
                if mv is None:
                    continue
                sp = domain.apply(cur, mv)
                budget.exec += 1
                budget.exec += 1  # solvable() call (mirrors savi verifier accounting)
                if not domain.solvable(sp):
                    continue
                key = domain.canon(sp)
                if key not in tally:
                    tally[key] = [0, mv, sp]
                tally[key][0] += 1
            if not tally:
                break  # no solvable successor emitted this step -> rollout dies
            keys = list(tally.keys())
            weights = [tally[k][0] for k in keys]
            chosen = rng.choices(keys, weights=weights, k=1)[0]
            _cnt, mv, sp = tally[chosen]
            path.append(mv)
            cur = sp
        if win_path is None and domain.is_goal(cur):
            win_path = path
        n_rollouts += 1
        if getattr(budget, axis) >= target or n_rollouts >= max_rollouts:
            break
    return Result(ok=win_path is not None, path=win_path, budget=budget, best_score=None,
                  detail={"n_rollouts": n_rollouts, "axis": axis, "target": target,
                          "tokens": budget.tokens})


# ---------------------------------------------------------------------------
# best_of_k_isobudget — compute-matched selection (headline baseline)
# ---------------------------------------------------------------------------

def best_of_k_isobudget(domain, sample, inst, target, tau, seed,
                        axis="tokens", max_rollouts=1000000):
    """Compute-matched selection: draw chain rollouts until ``budget[axis] >= target``.

    Each rollout is ONE ``sample(state, 1, tau, rollout_seed, "chain")`` call (a distinct
    seed per rollout) accounted into a single shared ``Budget``. ok iff ANY rollout
    reached the goal. ``budget.exec`` (and tokens/candidates) are accumulated for EVERY
    rollout, never stopped at the first win — a win only sets ``win_path`` (the critical
    fix from arms_ext). ``axis in {"candidates","exec","tokens"}``; PLAN3 uses
    ``"tokens"`` (the honest iso-compute axis: tokens, never candidates).
    """
    if axis not in ("candidates", "exec", "tokens"):
        raise ValueError(f"unknown axis: {axis!r} (valid: candidates, exec, tokens)")

    budget = Budget()
    s0 = domain.initial_state(inst)
    target = int(target)

    win_path = None
    n_rollouts = 0
    while True:
        # INT roll seed (varied per rollout). The real backend does int() arithmetic on
        # the seed, so a tuple seed (as in the mock-only arms_ext) would crash Tier B;
        # this stays a plain int and is process-stable for both backends.
        roll_seed = (int(seed) * 1_000_003 + n_rollouts) & 0x7FFFFFFF
        cands = sample(s0, 1, tau, roll_seed, "chain")
        budget.record_sample(cands)
        n_rollouts += 1

        for text in cands:
            moves = domain.parse_chain(text, s0)
            if moves is None:
                continue
            cur = s0
            for move in moves:
                cur = domain.apply(cur, move)
                budget.exec += 1
            if win_path is None and domain.is_goal(cur):
                win_path = moves

        spent = getattr(budget, axis)
        if spent >= target or n_rollouts >= max_rollouts:
            break

    ok = win_path is not None
    return Result(ok=ok, path=win_path, budget=budget, best_score=None,
                  detail={"n_rollouts": n_rollouts, "axis": axis, "target": target,
                          "candidates": budget.candidates, "exec": budget.exec,
                          "tokens": budget.tokens})


# ---------------------------------------------------------------------------
# beam_no_merge — Phi-ablation (per-path nodes, NO canonical collapse)
# ---------------------------------------------------------------------------

def _stable_topk(domain, nodes, K):
    return sorted(nodes, key=lambda n: (-n.score, domain.canon(n.state)))[:K]


def strict_savi(domain, sample, inst, K, N, tau, seed, verifier=True, max_depth=None,
                edge_mode="freq"):
    """savi but goals are collected ONLY from the BEAM-KEPT frontier (PLAN4 T3.1).

    The frozen ``savi`` collects goals over the full merged layer BEFORE beam pruning, so
    a goal that the beam drops still counts — which is exactly why Φ-merge shows no pass@1
    effect (H4=0): merge frees beam slots, but pre-prune collection already counted the
    goal. This STRICT variant counts a goal only if it survives into the top-K beam, so
    merge (which frees slots for goal-bearing canons) can bind. Compared against a
    strict-collection beam_no_merge it gives the honest E6c at narrow K. Otherwise
    identical to ``savi`` (freq/support edges, verifier mask, cross-layer seen guard).
    """
    if edge_mode not in ("support", "freq"):
        raise ValueError(f"unknown edge_mode: {edge_mode!r}")
    budget = Budget()
    s0 = domain.initial_state(inst)
    root_key = domain.canon(s0)
    frontier = {root_key: Node(score=0.0, bp=None, state=s0)}
    seen = {root_key}
    goals = []
    depth = 0
    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        nxt = {}
        for node in _stable_topk(domain, list(frontier.values()), K):
            cands = sample(node.state, N, tau, seed, "step")
            budget.record_sample(cands)
            tally, cache = {}, {}
            for c in cands:
                move = domain.parse_move(c, node.state)
                if move is None:
                    continue
                sp = domain.apply(node.state, move)
                budget.exec += 1
                if verifier:
                    budget.exec += 1
                    if not domain.solvable(sp):
                        continue
                sp_key = domain.canon(sp)
                if sp_key in seen:
                    continue
                tally[(sp_key, move)] = tally.get((sp_key, move), 0) + 1
                cache[(sp_key, move)] = sp
            for (sp_key, move), cnt in tally.items():
                edge = 0.0 if edge_mode == "support" else math.log(cnt / N)
                sp = cache[(sp_key, move)]
                cand_score = node.score + edge
                if sp_key not in nxt or cand_score > nxt[sp_key].score:
                    nxt[sp_key] = Node(score=cand_score, bp=(node, move), state=sp)
        if not nxt:
            break
        kept = _stable_topk(domain, list(nxt.values()), K)
        # STRICT: collect goals only among the beam-kept nodes (not the full nxt).
        for n in kept:
            if domain.is_goal(n.state):
                goals.append(n)
        frontier = {domain.canon(n.state): n for n in kept}
        seen.update(frontier.keys())
        depth += 1
    best = max(goals, key=lambda n: n.score) if goals else None
    return Result(ok=best is not None, path=_dc._backtrack(best), budget=budget,
                  best_score=(best.score if best is not None else None))


def savi_value(domain, sample, inst, value_fn, lam, K, N, tau, seed, max_depth,
               edge_mode="freq", hard_thresh=None, value_floor=1e-3):
    """Verifier-weighted decode with a LEARNED value V (PLAN4 soft-λ; the decisive test).

    The exact mask is OFF. Each successor's edge is the emission term plus a λ-weighted
    value term: ``edge = log(count/N) + λ·log(clip(V(s'), floor, 1))``. With λ=0 this is
    the freq-only emission decode (the loss); as λ grows the beam is steered toward
    high-V (likely-solvable) states — a SOFT verifier that, unlike a hard mask, only
    down-weights a wrongly-low-V solvable successor instead of deleting it (so it should
    tolerate V's false-negatives). ``hard_thresh`` (optional) instead PRUNES V<thresh —
    the 'learned hard mask' that should sit on the T1.2 noise curve at V's fn/fp rates.

    Goals are collected over the full pre-prune merged layer (mirrors ``savi``) so this is
    directly comparable to the headline. Token cost matches ``savi`` (same K·N·depth
    sampling); ``exec`` counts apply + one value eval per successor.
    """
    if edge_mode not in ("support", "freq"):
        raise ValueError(f"unknown edge_mode: {edge_mode!r}")
    budget = Budget()
    s0 = domain.initial_state(inst)
    root_key = domain.canon(s0)
    frontier = {root_key: Node(score=0.0, bp=None, state=s0)}
    seen = {root_key}
    goals = []
    depth = 0
    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        nxt = {}
        for node in _stable_topk(domain, list(frontier.values()), K):
            cands = sample(node.state, N, tau, seed, "step")
            budget.record_sample(cands)
            tally, cache = {}, {}
            for c in cands:
                move = domain.parse_move(c, node.state)
                if move is None:
                    continue
                sp = domain.apply(node.state, move)
                budget.exec += 1
                sp_key = domain.canon(sp)
                if sp_key in seen:
                    continue
                tally[(sp_key, move)] = tally.get((sp_key, move), 0) + 1
                cache[(sp_key, move)] = sp
            for (sp_key, move), cnt in tally.items():
                sp = cache[(sp_key, move)]
                v = value_fn(sp)
                budget.exec += 1
                if hard_thresh is not None and v < hard_thresh:
                    continue
                freq_edge = 0.0 if edge_mode == "support" else math.log(cnt / N)
                val_edge = lam * math.log(max(v, value_floor))
                cand_score = node.score + freq_edge + val_edge
                if sp_key not in nxt or cand_score > nxt[sp_key].score:
                    nxt[sp_key] = Node(score=cand_score, bp=(node, move), state=sp)
        if not nxt:
            break
        for n in nxt.values():
            if domain.is_goal(n.state):
                goals.append(n)
        frontier = {domain.canon(n.state): n
                    for n in _stable_topk(domain, list(nxt.values()), K)}
        seen.update(frontier.keys())
        depth += 1
    best = max(goals, key=lambda n: n.score) if goals else None
    return Result(ok=best is not None, path=_dc._backtrack(best), budget=budget,
                  best_score=(best.score if best is not None else None))


def _stable_topk_seq(domain, nodes_with_seq, K):
    """Top-K over (node, seq) by the stable order ``(-score, canon, seq)`` (seq makes the
    order total even when two distinct per-path nodes share a canon)."""
    ordered = sorted(
        nodes_with_seq,
        key=lambda ns: (-ns[0].score, domain.canon(ns[0].state), ns[1]),
    )
    return [ns for ns in ordered[:K]]


def beam_no_merge(domain, sample, inst, K, N, tau, seed, verifier=False,
                  max_depth=None, edge_mode="support", strict=False):
    """Width-K beam over the generator with NO Phi-merge (nodes keyed per-path).

    Identical layer-by-layer structure to ``decode_core.savi`` EXCEPT successors are not
    collapsed by canon: every distinct ``(parent, move)`` expansion is its own node, so
    two histories reaching the same canonical state occupy two beam slots (the merge
    ``savi`` would do is ablated). The stable top-K frontier, the cross-layer ``seen``
    visited guard, goal collection over the full expanded layer, edge weights, and the
    ``Budget`` accounting all mirror ``savi``.
    """
    if edge_mode not in ("support", "freq"):
        raise ValueError(f"unknown edge_mode: {edge_mode!r}")

    budget = Budget()
    s0 = domain.initial_state(inst)
    root = Node(score=0.0, bp=None, state=s0)
    seq_counter = 0
    frontier = [(root, seq_counter)]
    seq_counter += 1

    root_key = domain.canon(s0)
    seen = {root_key}

    before = []
    after = []
    goals = []

    depth = 0
    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        nxt = []
        raw = 0
        for node, _node_seq in _stable_topk_seq(domain, frontier, K):
            cands = sample(node.state, N, tau, seed, "step")
            budget.record_sample(cands)

            tally = {}
            move_cache = {}
            for c in cands:
                move = domain.parse_move(c, node.state)
                if move is None:
                    continue
                sp = domain.apply(node.state, move)
                budget.exec += 1
                if verifier:
                    budget.exec += 1
                    if not domain.solvable(sp):
                        continue
                sp_key = domain.canon(sp)
                if sp_key in seen:
                    continue
                raw += 1
                key = (sp_key, move)
                tally[key] = tally.get(key, 0) + 1
                move_cache[key] = (move, sp)

            for (sp_key, move), cnt in tally.items():
                edge = 0.0 if edge_mode == "support" else math.log(cnt / N)
                _move, sp = move_cache[(sp_key, move)]
                child = Node(score=node.score + edge, bp=(node, move), state=sp)
                nxt.append((child, seq_counter))
                seq_counter += 1

        if not nxt:
            break

        before.append(raw)
        after.append(len(nxt))

        frontier = _stable_topk_seq(domain, nxt, K)
        # Goal collection: full expanded layer (default, mirrors savi) or beam-kept only
        # (strict, PLAN4 T3.1 fair comparison vs strict_savi).
        for child, _s in (frontier if strict else nxt):
            if domain.is_goal(child.state):
                goals.append(child)

        seen.update(domain.canon(node.state) for node, _s in frontier)
        depth += 1

    best = max(goals, key=lambda n: n.score) if goals else None
    return Result(
        ok=best is not None,
        path=_dc._backtrack(best),
        budget=budget,
        trellis_widths_before_merge=before,
        trellis_widths_after_merge=after,
        best_score=(best.score if best is not None else None),
    )
