"""``CountdownDomain``: the Countdown adapter satisfying ``decode_core.domain.Domain``.

This is the countdown analogue of ``algebra-decode``'s ``AlgebraDomain``: it wraps the
vendored ``countdown`` (Phi + executor + exact oracle) and ``sampler`` (op-text
parser/renderer) so the domain-agnostic decoder (``decode_core.decode``) runs over
Countdown semantics. It mirrors the test-only adapter at
``verifier-decode/decode_core/tests/_countdown_adapter.py``, with one difference: the
decoder/harness pass an instance as a DICT ``{"numbers", "target", "id"}`` (not an
``Instance`` object), so ``initial_state`` reads those keys.

Countdown is the decoder's NATIVE domain. A state is a multiset of available exact
values plus a fixed target; a move combines two values into one (consuming both); the
Phi key ``canon = (sorted values tuple, target)`` STRICTLY SHRINKS each step (one fewer
value), so the step-trellis is finite, never recurs, and is bounded by construction —
unlike the algebra version whose canon (lhs-rhs scalar class) made productive moves
canon-invariant and the trellis degenerate.

Domain method contract (decode_core.domain.Domain)
--------------------------------------------------
  initial_state(instance dict) = countdown.parse_target_problem(numbers, target)
  canon         = countdown.canon          # (sorted values, target) — the Phi merge key
  is_goal       = countdown.is_goal        # one value remaining == target
  apply         = countdown.apply          # combine two values -> sorted-values successor
  parse_move    = sampler.parse            # single op text -> legal Op or None
  parse_chain   = sampler.parse_chain      # chain text -> list[Op] over the evolving state
  solvable      = countdown.reachable      # exact backward oracle / per-step mask
  render        = countdown.render         # "numbers: ... | target: ... | propose ONE ..."

Extra helpers (exposed for datagen / backend reuse, NOT part of the Domain Protocol):
  render_op(state, op)  = sampler.render_op    # Op -> "<a> <symbol> <b>" text
  legal_ops(state)      = countdown.legal_ops  # de-duped legal combines at a state
  reachable(state)      = countdown.reachable  # exact goal-reachability oracle
  solve_one(state)      = countdown.solve_one  # a witness Op path to the goal, or None
"""

from _deps import countdown as _countdown
from _deps import sampler as _sampler


class CountdownDomain:
    """Adapter satisfying ``decode_core.domain.Domain`` over the vendored Countdown."""

    # --- Domain Protocol --------------------------------------------------

    def initial_state(self, instance):
        """Start state from an instance dict ``{"numbers", "target", "id"}``.

        ``id`` is carried by the instance for bookkeeping but is not needed to build the
        state; only ``numbers`` and ``target`` define the Countdown start state.
        """
        return _countdown.parse_target_problem(instance["numbers"], instance["target"])

    def canon(self, state):
        return _countdown.canon(state)

    def is_goal(self, state):
        return _countdown.is_goal(state)

    def apply(self, state, move):
        return _countdown.apply(state, move)

    def parse_move(self, text, state):
        return _sampler.parse(text, state)

    def parse_chain(self, text, state):
        return _sampler.parse_chain(text, state)

    def solvable(self, state):
        return _countdown.reachable(state)

    def render(self, state):
        return _countdown.render(state)

    # --- Extra helpers (datagen / backend reuse) --------------------------

    def render_op(self, state, op):
        """Render a single Op ``(i, j, symbol)`` over ``canon(state)`` as op text."""
        return _sampler.render_op(state, op)

    def legal_ops(self, state):
        """De-duped (by outcome) legal combine Ops at ``state``."""
        return _countdown.legal_ops(state)

    def reachable(self, state):
        """Exact backward reachability oracle: True iff a goal is reachable."""
        return _countdown.reachable(state)

    def solve_one(self, state):
        """A witness Op path to a goal from ``state``, or ``None`` if unsolvable."""
        return _countdown.solve_one(state)
