"""Extra arms for the mechanism check, reusing FROZEN ``decode_core`` internals.

Two arms live here (``decode_core`` is never edited):

* ``best_of_k_isobudget`` — the compute-matched selection baseline. Draws chain
  rollouts one at a time (each a separate ``sample(...,"chain")`` call with a varied
  seed) accumulating ``Budget.candidates`` until the cumulative candidate count reaches
  a target (the trellis's candidate budget for the SAME instance). ok iff ANY rollout
  reached the goal. Returns a ``decode_core.Result`` with the real ``Budget``.

* ``beam_no_merge`` — the Φ-ablation. The SAME layer-by-layer loop as
  ``decode_core.savi`` BUT nodes are keyed PER-PATH: every distinct ``(parent, move)``
  successor is its own node, with NO canonical collapse. Top-K is by the stable order
  ``(-score, canon, seq)``. Goals are collected over the full merged-out layer (before
  beam pruning), and the Budget accounting is identical to ``savi``. A test asserts
  ``beam_no_merge == savi`` on an instance whose successor canons never coincide within
  the search.

Both reuse ``decode_core``'s ``Budget`` / ``Node`` / ``Result`` / ``_backtrack`` /
``_stable_order`` so iso-compute holds across arms.
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../recombination-novelpath/_deps/
# decode_core ships separately at the verifier-decode/ level: _deps/ -> experiment/ -> verifier-decode/
_VD = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _VD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decode_core import decode as _dc  # noqa: E402

Budget = _dc.Budget
Node = _dc.Node
Result = _dc.Result


# ---------------------------------------------------------------------------
# best_of_k_isobudget — compute-matched selection (headline baseline)
# ---------------------------------------------------------------------------

def best_of_k_isobudget(domain, sample, inst, target, tau, seed,
                        axis="candidates", max_rollouts=1000000):
    """Compute-matched selection: draw chain rollouts until ``budget[axis] >= target``.

    Each rollout is ONE ``sample(state, 1, tau, rollout_seed, "chain")`` call (a
    distinct seed per rollout so the rollouts vary), accounted into a single shared
    ``Budget``. ok iff ANY rollout reached the goal. Always runs at least one rollout;
    ``max_rollouts`` is a defensive cap against a degenerate generator that returns
    empty chains forever.

    ``axis`` selects the iso-compute axis the budget is matched on:

    * ``"candidates"`` (DEFAULT, back-compat with ``run_mechanism.py``): stop when the
      cumulative CANDIDATE count reaches ``target``. This prices one full multi-step
      rollout the same as one single decoder step — the PLAN1 artifact (hands selection
      ~depth x more complete attempts).
    * ``"exec"`` (the honest moves axis, PLAN2 headline baseline): stop when the
      cumulative MOVE-APPLY count (``budget.exec``) reaches ``target``. A chain of depth
      ``D`` costs ``D`` exec, so this draws ~``D`` x fewer rollouts than candidate
      parity — the correct compute match against the trellis's per-instance exec budget.
    * ``"tokens"``: stop on cumulative token count.

    CRITICAL FIX (vs the PLAN1 probe): ``budget.exec`` is incremented for EVERY rollout,
    not only up to the first win. The PLAN1 probe stopped accumulating exec after the
    first winning chain, which under-counted exec and broke the exec-parity loop (it
    would terminate early on a win). Here ``win_path`` is tracked SEPARATELY from the
    budget: a win never stops exec accounting, only the budget threshold (or
    ``max_rollouts``) does.

    Returns a ``decode_core.Result`` with the real ``Budget`` (so the iso-compute audit
    can compare its exec/candidate total to the trellis's).
    """
    if axis not in ("candidates", "exec", "tokens"):
        raise ValueError(f"unknown axis: {axis!r} (valid: candidates, exec, tokens)")

    budget = Budget()
    s0 = domain.initial_state(inst)
    target = int(target)

    win_path = None
    n_rollouts = 0
    while True:
        # Vary the seed per rollout so successive rollouts are distinct draws.
        roll_seed = (seed, n_rollouts)
        cands = sample(s0, 1, tau, roll_seed, "chain")
        budget.record_sample(cands)
        n_rollouts += 1

        # Execute EVERY rollout's chain in full and account ALL move-applies into the
        # budget (the critical fix). A win is recorded in ``win_path`` but does NOT
        # stop exec accounting; only the budget threshold / max_rollouts stops the loop.
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

        # Stop once we have matched the requested axis (>= target). Always run at least
        # one rollout; a defensive cap guards against empty-chain loops.
        spent = getattr(budget, axis)
        if spent >= target or n_rollouts >= max_rollouts:
            break

    ok = win_path is not None
    return Result(ok=ok, path=win_path, budget=budget, best_score=None,
                  detail={"n_rollouts": n_rollouts,
                          "axis": axis,
                          "target": target,
                          "candidates": budget.candidates,
                          "exec": budget.exec})


# ---------------------------------------------------------------------------
# beam_no_merge — Φ-ablation (per-path nodes, NO canonical collapse)
# ---------------------------------------------------------------------------

def _stable_topk_seq(domain, nodes_with_seq, K):
    """Top-K over (node, seq) tuples by the stable order ``(-score, canon, seq)``.

    ``seq`` is a per-node monotone integer that makes the order total even when two
    distinct per-path nodes share the same canon (which is exactly what merging would
    have collapsed and this ablation keeps apart). Returns the first K nodes.
    """
    ordered = sorted(
        nodes_with_seq,
        key=lambda ns: (-ns[0].score, domain.canon(ns[0].state), ns[1]),
    )
    return [ns for ns in ordered[:K]]


def beam_no_merge(domain, sample, inst, K, N, tau, seed, verifier=False,
                  max_depth=None, edge_mode="support"):
    """Width-K beam over the generator with NO Φ-merge (nodes keyed per-path).

    Identical layer-by-layer structure to ``decode_core.savi`` EXCEPT that successors
    are NOT collapsed by ``canon``: every distinct ``(parent, move)`` expansion becomes
    its own node. Two histories reaching the same canonical state therefore occupy TWO
    beam slots (the merge that ``savi`` would have done is ablated). Everything else --
    the stable top-K frontier, the cross-layer ``seen`` visited guard semantics, goal
    collection over the full merged-out layer, edge weights, and the ``Budget``
    accounting -- mirrors ``savi``.

    On an instance whose successor canons never coincide within the search, no merge
    ever fires in ``savi``, so this is identical to ``savi`` (the equivalence test).
    """
    if edge_mode not in ("support", "freq"):
        raise ValueError(f"unknown edge_mode: {edge_mode!r}")

    budget = Budget()
    s0 = domain.initial_state(inst)
    root = Node(score=0.0, bp=None, state=s0)
    seq_counter = 0
    # Frontier is a list of (node, seq) -- NO canon keying.
    frontier = [(root, seq_counter)]
    seq_counter += 1

    # Cross-layer visited guard, MIRRORING savi: a set of canonical keys that have
    # appeared in the BEAM-KEPT frontier so far (root included). savi marks only the
    # beam-kept (expanded) frontier as seen; we do the same so behavior matches when
    # canons never coincide.
    root_key = domain.canon(s0)
    seen = {root_key}

    before = []
    after = []
    goals = []

    depth = 0
    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        nxt = []  # list of (node, seq) -- every distinct expansion, no collapse
        raw = 0
        for node, _node_seq in _stable_topk_seq(domain, frontier, K):
            cands = sample(node.state, N, tau, seed, "step")
            budget.record_sample(cands)

            # Tally (canon(s'), move) multiplicities over feasible candidates -- the
            # tally drives the freq edge weight exactly as in savi; the NODE creation,
            # however, is per distinct (move) WITHOUT canonical collapse across the
            # whole layer.
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
                # Same cross-layer guard as savi (skip already-seen canon).
                if sp_key in seen:
                    continue
                raw += 1
                key = (sp_key, move)
                tally[key] = tally.get(key, 0) + 1
                move_cache[key] = (move, sp)

            # Emit one node PER distinct (canon(s'), move) at THIS parent -- per-path,
            # no merge against other parents' successors.
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

        # Collect goals over the FULL expanded layer (before beam pruning).
        for child, _s in nxt:
            if domain.is_goal(child.state):
                goals.append(child)

        # Advance to the stable top-K of the expanded layer.
        frontier = _stable_topk_seq(domain, nxt, K)

        # Mirror savi: mark only the BEAM-KEPT frontier's canons as seen.
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
