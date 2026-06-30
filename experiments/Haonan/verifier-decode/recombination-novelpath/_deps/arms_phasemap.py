"""Arms for the merge×noise phase map. ``decode_core`` stays FROZEN.

Re-exports the existing iso-compute selection baseline and the Phi-ablation from the
mechanism-recombination harness (same vendored ``decode_core``), and provides a
domain-generic ``savi_value`` (the soft-λ learned-value decode, arm A4) ported verbatim
from ``../joint-lambda-decode/arms_local.py`` but binding ``decode_core`` directly (no
Countdown ``core_boot`` dependency).
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../recombination-novelpath/_deps/
# decode_core ships separately at the verifier-decode/ level: _deps/ -> experiment/ -> verifier-decode/.
# arms_ext is vendored alongside this module in _deps/ (_HERE).
_VD = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _VD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decode_core import decode as _dc  # noqa: E402
import arms_ext as _arms_ext  # noqa: E402  (best_of_k_isobudget, beam_no_merge)

Budget = _dc.Budget
Node = _dc.Node
Result = _dc.Result

# Re-export the baseline + ablation unchanged (same decode_core).
best_of_k_isobudget = _arms_ext.best_of_k_isobudget
beam_no_merge = _arms_ext.beam_no_merge


def _stable_topk(domain, nodes, K):
    return sorted(nodes, key=lambda n: (-n.score, domain.canon(n.state)))[:K]


def savi_value(domain, sample, inst, value_fn, lam, K, N, tau, seed, max_depth,
               edge_mode="freq", hard_thresh=None, value_floor=1e-3):
    """Verifier-weighted decode with a (corrupted/learned) value V — arm A4.

    Edge = ``log(count/N) + lam·log(clip(V(s'), floor, 1))``. ``lam=0`` is the freq-only
    floor; growing ``lam`` steers the beam toward high-V states (soft verifier: it
    down-weights a wrongly-low-V solvable successor instead of deleting it). ``hard_thresh``
    instead PRUNES V<thresh (the learned hard mask). Goals collected over the full
    pre-prune merged layer (mirrors ``savi``); ``exec`` counts apply + one value eval per
    successor (iso-exec parity with the verifier arm). Verbatim port of
    ``arms_local.savi_value``.
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
