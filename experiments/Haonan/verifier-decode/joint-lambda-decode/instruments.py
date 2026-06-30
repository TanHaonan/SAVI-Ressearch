"""New-this-milestone instruments (PLAN3 §6). All read-only over the shared substrate.

1. ``measure_keff`` — in-trellis K_eff (H3, the marginal-vs-path probe). At each node the
   trellis actually EXPANDS, count distinct ``canon(s')`` among the N parsed successors.
   This mirrors ``decode_core.savi``'s expansion EXACTLY (freq-edge scoring, verifier
   mask, cross-layer visited guard, stable top-K) so the nodes measured are the nodes the
   headline arm decodes over — condition iii at the actual decode nodes, not the
   answer-position logit. Reports K_eff among PARSED successors (the generator's live
   branching) and among FEASIBLE successors (after the verifier/visited filter).

2. ``collect_calibration`` / ``reliability_ece`` — value calibration (H5, condition v).
   The freq edge treats emission frequency ``p = count/N`` of a successor as a graded
   value. Calibration asks: does that p track the EXACT backward oracle ``solvable``?
   We pair every distinct (canon(s'), move) with predicted ``p`` and label
   ``solvable(s')``, bin by p, and report the reliability curve + ECE (per layer).

3. ``merge_ratio_from_result`` — Phi compression (Sum before / Sum after merge).

4. ``paired_delta`` — paired bootstrap of a per-instance 0/1 difference (Holm done in the
   harness), reusing ``core.metrics.boot_ci``.
"""

import math

import core_boot as cb


# ---------------------------------------------------------------------------
# 1. In-trellis K_eff (mirrors decode_core.savi's expansion; measure-only)
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("score", "state")

    def __init__(self, score, state):
        self.score = score
        self.state = state


def _stable_topk(domain, nodes, K):
    return sorted(nodes, key=lambda n: (-n.score, domain.canon(n.state)))[:K]


def measure_keff(domain, sample, inst, K, N, tau, seed, edge_mode="freq",
                 verifier=True, max_depth=None, with_calibration=False):
    """In-trellis K_eff over the nodes the headline arm expands (see module docstring).

    Returns a dict with mean K_eff among parsed and feasible successors, per-node detail,
    the per-depth K_eff, total parsed/raw candidate counts (-> parse rate), and node
    count. Defaults to the headline config (freq edge, verifier=True). With
    ``with_calibration`` it ALSO returns ``calib_records`` (one ``{p, label, depth}`` per
    distinct (canon,move) among PARSED successors) from the SAME walk, so K_eff (H3) and
    the freq-edge calibration (H5) are measured in a single trellis pass.
    """
    s0 = domain.initial_state(inst)
    root_key = domain.canon(s0)
    frontier = {root_key: _Node(0.0, s0)}
    seen = {root_key}

    per_node_parsed = []     # distinct canon among parsed-legal successors, per node
    per_node_feasible = []   # distinct canon among feasible (verifier+visited) successors
    per_depth_parsed = []    # mean parsed-K_eff at each depth
    n_emitted = 0            # total candidate texts drawn
    n_parsed = 0             # total candidates that parsed to a legal move
    calib_records = []       # (p, label, depth) over distinct parsed (canon,move)

    depth = 0
    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        nxt = {}
        depth_keffs = []
        for node in _stable_topk(domain, list(frontier.values()), K):
            cands = sample(node.state, N, tau, seed, "step")
            n_emitted += len(cands)

            parsed_canons = set()       # distinct canon(s') among parsed-legal successors
            parsed_tally = {}           # (canon, move) -> count, among ALL parsed (pre-mask)
            sp_by_key = {}
            for c in cands:
                move = domain.parse_move(c, node.state)
                if move is None:
                    continue
                n_parsed += 1
                sp = domain.apply(node.state, move)
                sp_key = domain.canon(sp)
                parsed_canons.add(sp_key)
                parsed_tally[(sp_key, move)] = parsed_tally.get((sp_key, move), 0) + 1
                sp_by_key[(sp_key, move)] = sp

            # Feasible (verifier + visited) subset drives advancement + feasible K_eff.
            feasible_canons = set()
            for (sp_key, move), cnt in parsed_tally.items():
                sp = sp_by_key[(sp_key, move)]
                solvable = domain.solvable(sp)
                if with_calibration:
                    calib_records.append({"p": cnt / N, "label": bool(solvable),
                                          "depth": depth})
                if verifier and not solvable:
                    continue
                if sp_key in seen:
                    continue
                feasible_canons.add(sp_key)
                edge = 0.0 if edge_mode == "support" else math.log(cnt / N)
                cand_score = node.score + edge
                if sp_key not in nxt or cand_score > nxt[sp_key].score:
                    nxt[sp_key] = _Node(cand_score, sp)

            per_node_parsed.append(len(parsed_canons))
            per_node_feasible.append(len(feasible_canons))
            depth_keffs.append(len(parsed_canons))

        if not nxt:
            break
        per_depth_parsed.append(_mean(depth_keffs))
        frontier = {domain.canon(n.state): n
                    for n in _stable_topk(domain, list(nxt.values()), K)}
        seen.update(frontier.keys())
        depth += 1

    out = {
        "n_nodes": len(per_node_parsed),
        "keff_parsed_mean": _mean(per_node_parsed),
        "keff_feasible_mean": _mean(per_node_feasible),
        "keff_per_depth": per_depth_parsed,
        "parse_rate": (n_parsed / n_emitted) if n_emitted else float("nan"),
        "per_node_parsed": per_node_parsed,
    }
    if with_calibration:
        out["calib_records"] = calib_records
    return out


# ---------------------------------------------------------------------------
# 2. Value calibration: freq edge p=count/N vs exact reachable (H5 / condition v)
# ---------------------------------------------------------------------------

def collect_calibration(domain, sample, inst, K, N, tau, seed, edge_mode="freq",
                        verifier=True, max_depth=None):
    """Pairs ``(p=count/N, label=solvable(s'), depth)`` over the expanded trellis.

    One record per distinct (canon(s'), move) at each expanded node: the predicted value
    is the emission frequency p; the label is the exact backward-oracle solvability of
    the successor. These feed :func:`reliability_ece`. Mirrors savi's expansion so the
    distribution of p matches the headline arm's edges.
    """
    s0 = domain.initial_state(inst)
    root_key = domain.canon(s0)
    frontier = {root_key: _Node(0.0, s0)}
    seen = {root_key}
    records = []

    depth = 0
    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        nxt = {}
        for node in _stable_topk(domain, list(frontier.values()), K):
            cands = sample(node.state, N, tau, seed, "step")
            tally = {}
            cache = {}
            for c in cands:
                move = domain.parse_move(c, node.state)
                if move is None:
                    continue
                sp = domain.apply(node.state, move)
                sp_key = domain.canon(sp)
                tally[(sp_key, move)] = tally.get((sp_key, move), 0) + 1
                cache[(sp_key, move)] = sp
            for (sp_key, move), cnt in tally.items():
                sp = cache[(sp_key, move)]
                records.append({
                    "p": cnt / N,
                    "label": bool(domain.solvable(sp)),
                    "depth": depth,
                })
                # advance only feasible+unseen (savi-faithful)
                if verifier and not domain.solvable(sp):
                    continue
                if sp_key in seen:
                    continue
                edge = 0.0 if edge_mode == "support" else math.log(cnt / N)
                cand_score = node.score + edge
                if sp_key not in nxt or cand_score > nxt[sp_key].score:
                    nxt[sp_key] = _Node(cand_score, sp)
        if not nxt:
            break
        frontier = {domain.canon(n.state): n
                    for n in _stable_topk(domain, list(nxt.values()), K)}
        seen.update(frontier.keys())
        depth += 1
    return records


def reliability_ece(records, n_bins=10):
    """Reliability curve + ECE of predicted p vs empirical solvability over ``records``.

    ECE = sum over bins of (|bin| / total) * |mean_p - frac_label|. Returns the scalar
    ECE, per-bin detail, and the support count.
    """
    if not records:
        return {"ece": float("nan"), "n": 0, "bins": []}
    bins = [[] for _ in range(n_bins)]
    for r in records:
        idx = min(n_bins - 1, int(r["p"] * n_bins))
        bins[idx].append(r)
    total = len(records)
    ece = 0.0
    detail = []
    for b_idx, b in enumerate(bins):
        if not b:
            detail.append({"bin": b_idx, "n": 0, "mean_p": None, "frac_solvable": None})
            continue
        mean_p = sum(r["p"] for r in b) / len(b)
        frac = sum(1 for r in b if r["label"]) / len(b)
        ece += (len(b) / total) * abs(mean_p - frac)
        detail.append({"bin": b_idx, "n": len(b), "mean_p": mean_p, "frac_solvable": frac})
    return {"ece": ece, "n": total, "bins": detail}


# ---------------------------------------------------------------------------
# 3. Phi-merge compression ratio
# ---------------------------------------------------------------------------

def merge_ratio_from_result(result):
    """Sum(before) / Sum(after) over a Result's per-depth trellis widths (>= 1)."""
    before = sum(result.trellis_widths_before_merge or [])
    after = sum(result.trellis_widths_after_merge or [])
    if after == 0:
        return float("nan")
    return before / after


# ---------------------------------------------------------------------------
# 4. Paired bootstrap delta (reuses core.metrics.boot_ci)
# ---------------------------------------------------------------------------

def paired_delta(a_flags, b_flags, n_boot=10000, seed=0):
    """Paired bootstrap of mean(a_i - b_i) over aligned per-instance 0/1 outcomes.

    Returns {delta, lo, hi, n}. ``boot_ci`` returns [lo, mean, hi]; we bootstrap the
    difference vector so the CI is the paired delta's CI (excludes 0 iff significant).
    """
    if len(a_flags) != len(b_flags):
        raise ValueError("paired_delta requires aligned, equal-length flag lists")
    diffs = [int(bool(a)) - int(bool(b)) for a, b in zip(a_flags, b_flags)]
    if not diffs:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    lo, mean, hi = cb.metrics.boot_ci(diffs, n_boot=n_boot, seed=seed)
    return {"delta": mean, "lo": lo, "hi": hi, "n": len(diffs)}


def holm_reject(pvals_named, alpha=0.05):
    """Holm-Bonferroni: given {name: pseudo_p}, return {name: reject_bool}.

    Used over H1/H4 (the family PLAN3 §9 corrects across). We approximate a one-sided
    'CI excludes 0' decision elsewhere; this helper is provided for explicit p-value
    families if the harness computes them.
    """
    items = sorted(pvals_named.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev_reject = True
    for rank, (name, p) in enumerate(items):
        thresh = alpha / (m - rank)
        rej = prev_reject and (p <= thresh)
        out[name] = rej
        prev_reject = rej
    return out


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")
