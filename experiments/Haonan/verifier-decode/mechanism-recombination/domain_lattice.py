"""Integer-sum Viterbi lattice substrate (PLAN2 "Substrate").

A deliberately minimal domain that is the trellis algorithm's literal home turf: a
position x time lattice. It has a CHEAP O(1) exact solvability oracle, so depth is
unbounded-cheap (unlike ``countdown.reachable``, which is exponential and caps depth
~5). If global decode can ever beat compute-matched selection it must be able to here.

State / moves
-------------
* **State** ``(s, r)``: current sum ``s``, remaining steps ``r``. Start ``(0, D)``.
* **Moves** ``add v`` for ``v in M = {1, 2, 3}``. ``apply((s, r), v) = (s + v, r - 1)``.
* **Goal** ``is_goal((s, r)) = (s == T and r == 0)``.
* **canon** ``(s, r)`` — the classic position x time lattice. Different move-orders that
  reach the same ``(s, r)`` MERGE (this is where Phi has teeth, by construction).
* **solvable** ``(s, r) = (r <= (T - s) <= 3*r)`` — exact O(1). Since ``1 in M`` every
  integer in ``[r, 3r]`` is a sum of ``r`` moves from ``{1, 2, 3}``, so a goal is
  reachable from ``(s, r)`` iff the residual ``T - s`` lies in ``[r, 3r]``.

The domain carries the per-instance target ``T``: ``LatticeState`` stores ``T`` so that
``is_goal`` / ``solvable`` / ``apply`` are pure state methods (the Domain interface takes
no instance handle past ``initial_state``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


M = (1, 2, 3)  # the move set; 1 in M is what makes [r, 3r] fully covered
_MAX_V = max(M)
_MIN_V = min(M)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatticeState:
    """A lattice node ``(s, r)`` carrying its instance target ``T``.

    ``s`` current sum, ``r`` remaining steps, ``T`` the goal sum (closed over per
    instance so ``is_goal`` / ``solvable`` need no external handle). ``canon`` is
    ``(s, r)`` only — two states with the same ``(s, r)`` under the SAME ``T`` are the
    same lattice node (and within one decode ``T`` is fixed, so this is exact).

    Exposes ``.lhs`` / ``.rhs`` style attrs some harness code probes generically:
    ``.s`` and ``.r`` are the primary fields; ``.target`` mirrors ``T``.
    """

    s: int
    r: int
    T: int

    @property
    def target(self) -> int:
        return self.T


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

class LatticeDomain:
    """``decode_core.domain.Domain`` over the integer-sum lattice (see module doc)."""

    # -- construction -------------------------------------------------------
    def initial_state(self, instance) -> LatticeState:
        """Start state ``(s=0, r=D)`` with target ``T`` from the instance dict.

        ``instance`` keys: ``s0`` (start sum, default 0), ``target`` (T), ``depth`` (D).
        """
        s0 = int(instance.get("s0", 0))
        T = int(instance["target"])
        D = int(instance["depth"])
        return LatticeState(s=s0, r=D, T=T)

    # -- merge key ----------------------------------------------------------
    def canon(self, state: LatticeState):
        # Include T: canon must fully identify the problem instance, else a generator
        # whose memo/seed is keyed by canon (gen_fair) leaks competence classification
        # across instances that share (s, r) but differ in T. Within ONE decode T is
        # constant, so (s, r, T) keys nodes identically to (s, r) -> intra-decode
        # Phi-merging is unchanged; only cross-instance generator sharing is fixed.
        return (state.s, state.r, state.T)

    # -- goal ---------------------------------------------------------------
    def is_goal(self, state: LatticeState) -> bool:
        return state.s == state.T and state.r == 0

    # -- transition ---------------------------------------------------------
    def apply(self, state: LatticeState, move) -> LatticeState:
        v = int(move)
        return LatticeState(s=state.s + v, r=state.r - 1, T=state.T)

    # -- parsing ------------------------------------------------------------
    def parse_move(self, text: str, state: LatticeState):
        """Parse one ``"add v"`` text into a legal move ``v`` at ``state``, else None.

        Legal iff ``v in M`` AND there is a remaining step (``r > 0``).
        """
        if text is None:
            return None
        parts = str(text).strip().split()
        if len(parts) != 2 or parts[0] != "add":
            return None
        try:
            v = int(parts[1])
        except ValueError:
            return None
        if v not in M:
            return None
        if state.r <= 0:
            return None
        return v

    def parse_chain(self, text: str, state: LatticeState):
        """Parse a ';'-joined chain of ``"add v"`` moves applied to the evolving state.

        Returns the list of moves, or None if any step is malformed/illegal. An empty
        chain text yields ``[]`` (a zero-move chain — only at a state already terminal).
        """
        if text is None:
            return None
        text = str(text).strip()
        if text == "":
            return []
        moves = []
        cur = state
        for tok in text.split(";"):
            tok = tok.strip()
            if tok == "":
                continue
            move = self.parse_move(tok, cur)
            if move is None:
                return None
            moves.append(move)
            cur = self.apply(cur, move)
        return moves

    # -- exact O(1) oracle --------------------------------------------------
    def solvable(self, state: LatticeState) -> bool:
        """``r <= (T - s) <= 3r`` — exact, O(1). Goal reachable from ``(s, r)``."""
        residual = state.T - state.s
        return (_MIN_V * state.r) <= residual <= (_MAX_V * state.r)

    # -- rendering ----------------------------------------------------------
    def render(self, state: LatticeState) -> str:
        return f"sum={state.s} remaining={state.r} target={state.T}"


# ---------------------------------------------------------------------------
# Enumeration (the finite known move set; no mock sampler needed)
# ---------------------------------------------------------------------------

def lattice_enumerate(state: LatticeState):
    """Return ``["add 1", "add 2", "add 3"]`` filtered by ``parse_move`` (r > 0).

    A pure, complete legal-move-text enumeration: at a terminal state (``r <= 0``) it
    returns ``[]``. ``gen_fair.make_fair_generator`` consumes this directly as its
    ``enumerate_moves_fn`` (no fixpoint mock enumeration needed).
    """
    _dom = _SHARED_DOMAIN
    out = []
    for v in M:
        text = f"add {v}"
        if _dom.parse_move(text, state) is not None:
            out.append(text)
    return out


# A module-level domain instance used purely by ``lattice_enumerate`` for its legality
# check (parse_move is stateless w.r.t. T, so any domain instance suffices).
_SHARED_DOMAIN = LatticeDomain()


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatticeInstance:
    """One lattice problem: depth ``D``, target ``T``, stable ``id``.

    ``id`` is a deterministic string id; ``depth`` and ``target`` define the instance.
    """

    id: str
    depth: int
    target: int
    s0: int = 0


def _det_rng_int(seed, depth, idx, lo, hi):
    """Deterministic integer in ``[lo, hi]`` from a stable sha256 of the payload.

    Process-stable (no Python ``hash()``); identical args -> identical draw.
    """
    payload = f"lattice::{int(seed)}::{int(depth)}::{int(idx)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    span = hi - lo + 1
    return lo + (int.from_bytes(digest[:8], "big") % span)


def make_lattice_instances(depth, n, seed):
    """``n`` solvable lattice instances at ``depth`` D, ``T`` drawn in ``[D, 3D]``.

    Every instance is solvable from ``(0, D)`` (ceiling = 1) because ``T in [D, 3D]``
    means residual ``T - 0 = T in [r, 3r]`` at the start ``r = D``. The minimal-step
    solution length is exactly ``D``, so "depth" is honest. Ids are stable
    (``lat_d{D}_s{seed}_{idx}``), the draw is deterministic in ``(seed, depth, idx)``.
    """
    D = int(depth)
    out = []
    for idx in range(int(n)):
        T = _det_rng_int(seed, D, idx, D * _MIN_V, D * _MAX_V)
        out.append(LatticeInstance(
            id=f"lat_d{D}_s{int(seed)}_{idx}",
            depth=D,
            target=int(T),
            s0=0,
        ))
    return out


def lattice_inst_dict(inst: LatticeInstance):
    """The mapping ``LatticeDomain.initial_state`` consumes for an instance."""
    return {"s0": int(inst.s0), "target": int(inst.target), "depth": int(inst.depth),
            "id": inst.id}
