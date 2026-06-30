"""The ``Domain`` interface: everything the generalized decoder needs from a domain.

A ``Domain`` is the small plug-in surface that lets ``decode_core.decode`` run over an
arbitrary problem domain (Countdown, algebra, ...). The decoder never imports any
domain module; instead a caller passes a concrete object satisfying this Protocol plus
a ``sample`` callable (the generator backend baked in). All semantics that used to be
hard-wired to ``countdown.*`` in M1 are expressed here as method calls.

Method contract (mirrors how M1's decoder used the Countdown functions)
----------------------------------------------------------------------
* ``initial_state(instance)`` — build the start state for a problem instance.
* ``canon(state)``            — the Φ merge key: a *hashable* canonical identity. Two
                                states with equal ``canon`` are merged in the trellis.
* ``is_goal(state)``          — True iff ``state`` is an accepting (solved) state.
* ``apply(state, move)``      — execute one ``move`` (as returned by ``parse_move`` /
                                ``parse_chain``) producing the successor state.
* ``parse_move(text, state)`` — parse ONE candidate text into a legal move at ``state``
                                (parse + legality in one step), or ``None`` if the text
                                is malformed or the move is illegal here.
* ``parse_chain(text, state)``— parse a FULL chain text into a list of moves applied to
                                the evolving state, or ``None`` if any step is
                                illegal/malformed.
* ``solvable(state)``         — exact ceiling / oracle: True iff a goal is reachable
                                from ``state``. Used as the per-step feasibility mask
                                when the decoder runs with ``verifier=True``, and as the
                                absolute solvability oracle for the ``oracle`` arm.
* ``render(state)``           — a prompt rendering of ``state`` (used by the GENERATOR,
                                not by the decoder; included so a domain is self-contained).
"""

from typing import Any, Hashable, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Domain(Protocol):
    """Plug-in surface for the generalized trellis decoder (see module docstring)."""

    def initial_state(self, instance: Any) -> Any: ...

    def canon(self, state: Any) -> Hashable: ...  # merge key (Φ)

    def is_goal(self, state: Any) -> bool: ...

    def apply(self, state: Any, move: Any) -> Any: ...  # move from parse_move(.,state)

    def parse_move(self, text: str, state: Any) -> Optional[Any]: ...  # single move or None

    def parse_chain(self, text: str, state: Any) -> Optional[List[Any]]: ...  # chain or None

    def solvable(self, state: Any) -> bool: ...  # ceiling/oracle; per-step mask when verifier=True

    def render(self, state: Any) -> str: ...  # prompt rendering (generator, not decoder)
