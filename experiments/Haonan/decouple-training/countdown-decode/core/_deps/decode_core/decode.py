"""Domain-agnostic trellis decoder + experiment arms (generalized from M1).

This is M1's ``core/decode.py`` lifted off Countdown: every hard ``countdown.*`` /
``sampler.*`` call is replaced by a method on an injected ``domain`` (see
``decode_core.domain.Domain``) and an injected ``sample`` callable. The trellis
mechanics are otherwise IDENTICAL to M1, which is what the Countdown parity test
proves (``decode_core/tests/test_countdown_parity.py``).

Load-bearing semantics (carried over verbatim from M1)
-----------------------------------------------------
* **Verifier = the exact constraint** (when ``verifier=True``): the per-step
  feasibility mask is ``domain.solvable(s')`` — prune any successor from which a goal
  is no longer reachable. ``verifier=False`` (the DEFAULT here) removes the mask: the
  headline arm is legality at parse + ``is_goal`` at the leaf only, so the generator is
  load-bearing and the solvability oracle is the ceiling/ablation only.
* **Edge modes.** ``"support"``: every edge weight is 0.0 (emission contributes only
  its support). ``"freq"``: edge weight is ``log(count / N)`` where ``count`` is the
  number of sampled candidates that produced this ``(canon(s'), move)`` and ``N`` is
  the FULL sample count.
* **Φ-merge.** Next-layer nodes are keyed by ``domain.canon(s')``. If two histories
  reach the same key, the higher-cumulative-score one is kept (its back-pointer stored).
* **Nested beams / stable tie-break.** Top-K uses the deterministic, K-independent
  total order ``(-cumulative_score, canon_key)`` and takes the first K. In support mode
  all scores tie, so the canon_key tie-break makes top-K(1) ⊆ top-K(2) ⊆ ... and
  coverage monotone in K.
* **Oracle = ABSOLUTE solvability.** ``oracle(domain, inst) =
  domain.solvable(domain.initial_state(inst))`` — the ceiling / headroom numerator.

Layer count (the one generalization beyond a textual rename)
-----------------------------------------------------------
M1 ran a fixed ``len(s0.values) - 1`` layers (a Countdown fact). The Domain interface
exposes no depth, so the generalized decoder instead expands LAYER BY LAYER until the
frontier can no longer be expanded — i.e. until the merged next layer is empty (every
surviving state is terminal, or the verifier pruned them all). For Countdown this
produces exactly M1's per-depth width lists in every non-degenerate case; the only
difference is that M1 keeps padding all-zero layers up to its fixed step count once the
verifier has emptied the frontier, whereas this decoder stops at the first empty layer.
Those trailing all-zero layers carry no trellis information (no states, no goals), so
the decode OUTCOME (ok / path / best_score) is identical; the parity test compares the
informative (trailing-zero-stripped) width lists.

Iso-compute Budget
------------------
``Budget`` tracks ``sample_calls``, ``candidates``, ``tokens`` (Σ ``len(text.split())``
over returned texts), and ``exec`` (``apply`` + ``solvable`` evaluations during decode).
Counters increment identically to M1 so downstream iso-compute holds across arms.
"""

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Iso-compute accounting
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Per-arm compute accounting.

    sample_calls : number of ``sample()`` invocations.
    candidates   : total candidate texts returned by those calls.
    tokens       : Σ over all returned candidate texts of ``len(text.split())``.
    exec         : count of ``domain.apply()`` + ``domain.solvable()`` evaluations.
    """

    sample_calls: int = 0
    candidates: int = 0
    tokens: int = 0
    exec: int = 0

    def record_sample(self, cands):
        """Account for one ``sample()`` call that returned the list ``cands``."""
        self.sample_calls += 1
        self.candidates += len(cands)
        self.tokens += sum(len(c.split()) for c in cands)


# ---------------------------------------------------------------------------
# Trellis node + result
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """One trellis node: a canonical state with its best history to it.

    score : cumulative path score (sum of edge weights) of the best history reaching
            this canonical state.
    bp    : backpointer ``(parent_node, move)`` or ``None`` at the root.
    state : the domain state (its ``domain.canon`` is the merge key).
    """

    score: float
    bp: object  # (Node, move) | None
    state: object


@dataclass
class Result:
    """Outcome of an arm.

    ok                          : did the arm reach the goal?
    path                        : forward-order list of moves to the goal, or None.
    budget                      : iso-compute Budget for this arm.
    trellis_widths_before_merge : per-depth feasible expansions (duplicates counted).
    trellis_widths_after_merge  : per-depth distinct canonical states after Φ-merge.
    best_score                  : score of the winning goal node, or None.
    detail                      : optional free-form dict.
    """

    ok: bool
    path: object = None
    budget: Budget = field(default_factory=Budget)
    trellis_widths_before_merge: list = field(default_factory=list)
    trellis_widths_after_merge: list = field(default_factory=list)
    best_score: object = None
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stable, K-independent beam selection (nested beams)
# ---------------------------------------------------------------------------

def _stable_order(domain, nodes):
    """Return nodes in the deterministic, K-independent total order ``(-score, key)``.

    Because the key is independent of K, the first K elements for any K form a nested
    chain: top-K(1) ⊆ top-K(2) ⊆ ... (coverage monotone in K). In support mode all
    scores tie, so the canon_key tie-break alone decides, giving fully nested beams.
    """
    return sorted(nodes, key=lambda n: (-n.score, domain.canon(n.state)))


def _stable_topk(domain, nodes, K):
    """First K nodes under the stable total order (a nested prefix)."""
    return _stable_order(domain, nodes)[:K]


# ---------------------------------------------------------------------------
# Backtracking
# ---------------------------------------------------------------------------

def _backtrack(node):
    """Follow ``bp`` to the root; return the move list in FORWARD order (or None)."""
    if node is None:
        return None
    moves = []
    cur = node
    while cur.bp is not None:
        parent, move = cur.bp
        moves.append(move)
        cur = parent
    moves.reverse()
    return moves


# ---------------------------------------------------------------------------
# Arm: oracle (ABSOLUTE solvability — the ceiling)
# ---------------------------------------------------------------------------

def oracle(domain, inst):
    """Absolute solvability ceiling: ``domain.solvable(domain.initial_state(inst))``.

    The headroom numerator / absolute ceiling, NOT "a correct chain was sampled".
    """
    s0 = domain.initial_state(inst)
    return domain.solvable(s0)


# ---------------------------------------------------------------------------
# Arm: greedy (one temperature-0 chain)
# ---------------------------------------------------------------------------

def greedy(domain, sample, inst, seed):
    """Sample ONE chain at temperature 0; execute it; ok iff it reaches the goal.

    Consumes the sampler only. Exactly one ``sample()`` call (mode="chain").
    """
    budget = Budget()
    s0 = domain.initial_state(inst)
    cands = sample(s0, 1, 0.0, seed, "chain")
    budget.record_sample(cands)

    path = None
    ok = False
    if cands:
        moves = domain.parse_chain(cands[0], s0)
        if moves is not None:
            cur = s0
            for move in moves:
                cur = domain.apply(cur, move)
                budget.exec += 1
            if domain.is_goal(cur):
                ok = True
                path = moves
    return Result(ok=ok, path=path, budget=budget, best_score=None,
                  detail={"chain": cands[0] if cands else None})


# ---------------------------------------------------------------------------
# Arm: best_of_k (K whole chains; ok iff ANY reaches the goal)
# ---------------------------------------------------------------------------

def best_of_k(domain, sample, inst, K, tau, seed):
    """Sample K whole chains; ok iff ANY chain reaches the goal; return the winner.

    Consumes the sampler only. One ``sample()`` call returning K chains.
    """
    budget = Budget()
    s0 = domain.initial_state(inst)
    cands = sample(s0, K, tau, seed, "chain")
    budget.record_sample(cands)

    win_path = None
    win_idx = None
    for idx, text in enumerate(cands):
        moves = domain.parse_chain(text, s0)
        if moves is None:
            continue
        cur = s0
        for move in moves:
            cur = domain.apply(cur, move)
            budget.exec += 1
        if domain.is_goal(cur):
            win_path = moves
            win_idx = idx
            break

    ok = win_path is not None
    return Result(ok=ok, path=win_path, budget=budget, best_score=None,
                  detail={"winning_index": win_idx,
                          "winning_chain": cands[win_idx] if win_idx is not None else None})


# ---------------------------------------------------------------------------
# Arm: savi (the trellis decode)
# ---------------------------------------------------------------------------

def savi(domain, sample, inst, K, N, edge_mode, tau, seed, verifier=False,
         max_depth=None):
    """Trellis decode over Φ-merged canonical states (optionally verifier-masked).

    Builds one trellis layer per step. At each layer, expand the stable top-K frontier:
    draw N step-candidates per node, parse each to a move, execute it, (optionally) mask
    with ``domain.solvable``, tally ``(canon(s'), move)`` multiplicities, add edges
    (weight 0 in support mode, ``log(count/N)`` in freq mode), Φ-merging by
    ``domain.canon(s')`` keeping the higher-scoring history. Goal nodes are collected;
    the highest-scoring goal node yields the path via backtracking.

    edge_mode in {"support","freq"}; ``verifier`` toggles the exact feasibility mask
    (DEFAULT False: legality at parse + is_goal at the leaf only).

    Termination on UNBOUNDED domains
    --------------------------------
    Countdown's state shrinks every step, so the trellis self-terminates (a layer
    eventually expands to nothing) and canonical keys never recur. Other domains may be
    unbounded (a state that can always emit a legal move) and/or cyclic (move chains
    that revisit a canonical state, e.g. ``inc`` then ``dec``). Two guards bound the
    work WITHOUT changing Countdown behavior:

    * ``max_depth`` (default ``None`` = no cap): stop after expanding ``max_depth``
      layers. With ``None`` the loop runs exactly as before, so Countdown / parity is
      unchanged; the harness passes a finite ``max_depth`` for unbounded domains.
    * a CROSS-LAYER visited-set ``seen`` of canonical keys that have appeared in ANY
      frontier (including the root). A candidate successor whose canon is already in
      ``seen`` is skipped before it can enter the next layer — this kills cycles and
      bounds total expansions to the number of DISTINCT canonical states. This is a
      no-op for Countdown: the value-multiset strictly shrinks each step, so a canon
      can never reappear at a later depth, so nothing is ever skipped (parity holds
      byte-for-byte).
    """
    if edge_mode not in ("support", "freq"):
        raise ValueError(f"unknown edge_mode: {edge_mode!r}")

    budget = Budget()
    s0 = domain.initial_state(inst)
    root_key = domain.canon(s0)
    frontier = {root_key: Node(score=0.0, bp=None, state=s0)}
    # Canonical keys that have appeared in any frontier so far (root included). Used to
    # skip already-visited successors across layers (cycle / unbounded-growth guard).
    seen = {root_key}

    before = []  # per-depth feasible expansions, duplicates counted
    after = []   # per-depth distinct canonical states after Φ-merge
    goals = []

    # Expand layer by layer. Termination is the FIRST of: the merged next layer is
    # empty (all surviving states terminal / pruned / already-visited), or the depth cap
    # ``max_depth`` is reached. For Countdown both guards are inert (layers self-empty
    # before any cap, canons never recur), reproducing M1's per-depth widths exactly
    # (modulo trailing all-zero layers M1 padded up to its fixed step count).
    depth = 0
    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        nxt = {}
        raw = 0
        for node in _stable_topk(domain, list(frontier.values()), K):
            cands = sample(node.state, N, tau, seed, "step")
            budget.record_sample(cands)

            # Tally (canon(s'), move) multiplicities over feasible candidates.
            tally = {}
            sp_cache = {}
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
                # Cross-layer cycle / unbounded-growth guard: skip a successor whose
                # canon already appeared in some earlier (or the current) frontier.
                # No-op for Countdown (canons never recur as the multiset shrinks).
                if sp_key in seen:
                    continue
                raw += 1
                key = (sp_key, move)
                tally[key] = tally.get(key, 0) + 1
                sp_cache[key] = sp

            # Emit edges; Φ-merge by canon(s') keeping the higher cumulative score.
            for (sp_key, move), cnt in tally.items():
                edge = 0.0 if edge_mode == "support" else math.log(cnt / N)
                sp = sp_cache[(sp_key, move)]
                cand_score = node.score + edge
                if sp_key not in nxt or cand_score > nxt[sp_key].score:
                    nxt[sp_key] = Node(score=cand_score, bp=(node, move), state=sp)

        # Stop once the frontier produced no successors (terminal / fully pruned /
        # all-visited): no states, no goals, no information — exactly M1's trailing
        # zero layers.
        if not nxt:
            break

        before.append(raw)
        after.append(len(nxt))

        # Collect goals among the FULL merged next layer (before beam pruning) — a goal
        # that is beam-pruned out of the top-K still counts as reached.
        for _key, n in nxt.items():
            if domain.is_goal(n.state):
                goals.append(n)

        # Advance the frontier to the stable top-K of the merged layer.
        frontier = {domain.canon(n.state): n
                    for n in _stable_topk(domain, list(nxt.values()), K)}

        # Mark only the BEAM-KEPT (expanded) frontier as seen — NOT the full merged
        # layer. A canon that is beam-pruned out of the top-K is deliberately left out
        # of ``seen`` so it can re-enter a later layer via a higher-scoring path (recall
        # correctness); a kept canon goes into ``seen`` so cycles among expanded states
        # are still blocked. For Countdown this is identical to marking all of ``nxt``
        # (canons never recur across depths), so parity is unaffected.
        seen.update(frontier.keys())
        depth += 1

    best = max(goals, key=lambda n: n.score) if goals else None
    return Result(
        ok=best is not None,
        path=_backtrack(best),
        budget=budget,
        trellis_widths_before_merge=before,
        trellis_widths_after_merge=after,
        best_score=(best.score if best is not None else None),
    )
