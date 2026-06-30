"""Regression tests: the cross-layer visited-set marks only the BEAM-KEPT frontier.

A canon that is beam-pruned out of the top-K must NOT be marked ``seen``, so it can
re-enter a later layer via a higher-scoring path (recall correctness). Only the kept
(expanded) frontier goes into ``seen`` (so cycles among expanded states stay blocked).

The decisive test builds a tiny deterministic DAG where, at beam width K=1, a canon is
beam-pruned at depth 0 (lower freq score) yet is the ONLY route to a goal reachable at
depth 2 via a higher-scoring re-entry from the kept node at depth 1. Under the OLD
"mark all of nxt as seen" behavior that canon would be blocked from re-entry and the
goal would be missed (ok=False); under the fix it is found (ok=True).

Fully deterministic: the fake ``sample`` returns a fixed multiset of move texts per
state (controlling freq-mode edge scores via candidate multiplicity); no RNG.
"""

from decode_core import decode as dc


# ---------------------------------------------------------------------------
# A tiny labelled-state DAG domain.
#
# States are short string labels: "S" (start), "A", "B", "G" (goal).
# Moves are the literal strings "toA", "toB", "BtoA", "AtoG".
# Transition table (state -> {move: next_state}); a move is legal iff present here.
#
#   S --toA--> A        (depth 0)
#   S --toB--> B        (depth 0)
#   B --BtoA--> A       (depth 1)  re-enters canon A from the kept node B
#   A --AtoG--> G       (depth 2)  the goal, reachable ONLY through A
#
# canon(state) = the label itself. is_goal = (label == "G").
#
# Beam K=1, freq edge mode: at depth 0 we make "toB" out-number "toA" in the sample, so
# B scores higher (log(count/N) is larger) and is the single kept beam node; A is pruned
# out of the top-1. The ONLY path to G goes S->...->A->G, and A can only re-appear at
# depth 1 (via B) if it was NOT marked seen when it was pruned at depth 0.
# ---------------------------------------------------------------------------

_TRANSITIONS = {
    "S": {"toA": "A", "toB": "B"},
    "B": {"BtoA": "A"},
    "A": {"AtoG": "G"},
    "G": {},  # terminal: no moves
}


class DagDomain:
    """Deterministic labelled DAG (see module docstring)."""

    def __init__(self):
        self.apply_calls = 0

    def initial_state(self, instance):
        return "S"

    def canon(self, state):
        return state

    def is_goal(self, state):
        return state == "G"

    def apply(self, state, move):
        self.apply_calls += 1
        return _TRANSITIONS[state][move]

    def parse_move(self, text, state):
        return text if text in _TRANSITIONS.get(state, {}) else None

    def parse_chain(self, text, state):
        moves = []
        cur = state
        for part in text.split(";"):
            if not part:
                continue
            mv = self.parse_move(part, cur)
            if mv is None:
                return None
            moves.append(mv)
            cur = self.apply(cur, mv)
        return moves or None

    def solvable(self, state):
        # Reachability of "G" within this finite DAG.
        seen = set()
        stack = [state]
        while stack:
            s = stack.pop()
            if s == "G":
                return True
            if s in seen:
                continue
            seen.add(s)
            stack.extend(_TRANSITIONS.get(s, {}).values())
        return False

    def render(self, state):
        return state


def _dag_sample():
    """Deterministic sample: per state, a fixed multiset of move texts.

    At "S" we emit "toB" more often than "toA" so that in freq mode B's edge
    (log(count/N)) outscores A's, making B the single top-1 beam node and pruning A.
    """
    table = {
        "S": ["toB", "toB", "toB", "toA"],  # B (3/4) outscores A (1/4)
        "B": ["BtoA"],
        "A": ["AtoG"],
        "G": [],
    }

    def sample(state, N, temperature, seed, mode):
        moves = table.get(state, [])
        if not moves or N <= 0:
            return []
        return [moves[i % len(moves)] for i in range(N)]

    return sample


# ---------------------------------------------------------------------------
# Re-implementation of the OLD (buggy) seen-update to PROVE the fix changes outcome.
# This mirrors savi exactly but marks the FULL merged layer as seen before pruning,
# which is the behavior the fix removed. It lives only in this test so we can assert
# "old would have failed, fix succeeds" on the very same domain/inputs.
# ---------------------------------------------------------------------------

def _savi_old_full_nxt_seen(domain, sample, inst, K, N, edge_mode, tau, seed):
    import math
    from decode_core.decode import Node, _stable_topk, _backtrack

    s0 = domain.initial_state(inst)
    root_key = domain.canon(s0)
    frontier = {root_key: Node(score=0.0, bp=None, state=s0)}
    seen = {root_key}
    goals = []
    depth = 0
    while frontier:
        if depth >= 50:  # hard safety bound for the test
            break
        nxt = {}
        for node in _stable_topk(domain, list(frontier.values()), K):
            cands = sample(node.state, N, tau, seed, "step")
            tally, sp_cache = {}, {}
            for c in cands:
                move = domain.parse_move(c, node.state)
                if move is None:
                    continue
                sp = domain.apply(node.state, move)
                sp_key = domain.canon(sp)
                if sp_key in seen:
                    continue
                key = (sp_key, move)
                tally[key] = tally.get(key, 0) + 1
                sp_cache[key] = sp
            for (sp_key, move), cnt in tally.items():
                edge = 0.0 if edge_mode == "support" else math.log(cnt / N)
                cand_score = node.score + edge
                if sp_key not in nxt or cand_score > nxt[sp_key].score:
                    nxt[sp_key] = Node(score=cand_score, bp=(node, move),
                                       state=sp_cache[(sp_key, move)])
        if not nxt:
            break
        seen.update(nxt.keys())  # <-- the OLD behavior: mark FULL layer before pruning
        for _k, n in nxt.items():
            if domain.is_goal(n.state):
                goals.append(n)
        frontier = {domain.canon(n.state): n
                    for n in _stable_topk(domain, list(nxt.values()), K)}
        depth += 1
    best = max(goals, key=lambda n: n.score) if goals else None
    return best is not None, _backtrack(best)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_old_full_nxt_seen_would_block_the_goal():
    """The OLD behavior (mark full nxt as seen) blocks A's re-entry -> goal missed."""
    dom = DagDomain()
    sample = _dag_sample()
    ok, path = _savi_old_full_nxt_seen(dom, sample, None, K=1, N=4,
                                       edge_mode="freq", tau=0.0, seed=1)
    assert ok is False and path is None


def test_fix_finds_goal_via_beam_pruned_canon_reentry():
    """The fix (mark only beam-kept frontier) lets A re-enter at depth 1 -> goal found."""
    dom = DagDomain()
    sample = _dag_sample()
    r = dc.savi(dom, sample, None, K=1, N=4, edge_mode="freq", tau=0.0, seed=1,
                verifier=False)
    assert r.ok is True
    # The only route to G: S -> B -> A -> G (A was beam-pruned at depth 0, re-entered
    # at depth 1 from the kept node B).
    assert r.path == ["toB", "BtoA", "AtoG"]


def test_beam_pruned_canon_is_not_in_seen_so_it_can_reenter():
    """Behavioral proxy for the invariant: at K=1 the lower-score canon A is pruned at
    depth 0, yet a later layer still reaches a goal that is ONLY reachable through A —
    which is impossible unless A was left out of ``seen`` after being pruned."""
    dom = DagDomain()
    sample = _dag_sample()
    r = dc.savi(dom, sample, None, K=1, N=4, edge_mode="freq", tau=0.0, seed=1)
    assert r.ok is True
    assert r.path is not None and r.path[0] == "toB"  # B was the kept (higher-score) node


def test_fix_still_terminates_on_this_dag():
    """The kept-only seen-set still blocks cycles among EXPANDED states: this DAG has no
    cycle, but the loop must still terminate (no max_depth needed) because G is terminal
    and all reachable canons are eventually kept->seen."""
    dom = DagDomain()
    sample = _dag_sample()
    r = dc.savi(dom, sample, None, K=1, N=4, edge_mode="freq", tau=0.0, seed=1,
                max_depth=None)
    assert r.ok is True
    # Bounded, finite work (4 distinct canons; small K, N).
    assert dom.apply_calls < 100
