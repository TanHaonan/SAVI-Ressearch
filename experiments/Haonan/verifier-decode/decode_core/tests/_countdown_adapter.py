"""Test-only Countdown adapter: wraps the FROZEN M1 modules to the Domain interface.

This adapter exists ONLY in the test tree. ``decode_core`` never imports it, and
never imports the M1 modules. It is the bridge that lets us drive the generalized
``decode_core`` decoder over the exact same Countdown semantics M1 used, so the
parity test can prove behavior-preservation.

The Domain methods map straight onto the M1 functions:
  initial_state = parse_target_problem(inst.numbers, inst.target)
  canon         = countdown.canon
  is_goal       = countdown.is_goal
  apply         = countdown.apply
  parse_move    = sampler.parse           (single op text -> legal Op or None)
  parse_chain   = sampler.parse_chain     (chain text -> list[Op] applied to evolving state)
  solvable      = countdown.reachable     (exact backward oracle; per-step mask)
  render        = countdown.render

The ``sample`` closure bakes the chosen mock backend into the
``sample(state, N, temperature, seed, mode) -> list[str]`` signature the decoder
expects (the M1 sampler additionally takes ``backend`` and a keyword ``mode``).
"""

# Imported via conftest sys.path (m1 line root on path); decode_core never does this.
from core import countdown as _countdown
from core import sampler as _sampler


class CountdownDomain:
    """Adapter satisfying decode_core.domain.Domain over the frozen M1 Countdown."""

    @staticmethod
    def initial_state(inst):
        return _countdown.parse_target_problem(inst.numbers, inst.target)

    @staticmethod
    def canon(state):
        return _countdown.canon(state)

    @staticmethod
    def is_goal(state):
        return _countdown.is_goal(state)

    @staticmethod
    def apply(state, move):
        return _countdown.apply(state, move)

    @staticmethod
    def parse_move(text, state):
        return _sampler.parse(text, state)

    @staticmethod
    def parse_chain(text, state):
        return _sampler.parse_chain(text, state)

    @staticmethod
    def solvable(state):
        return _countdown.reachable(state)

    @staticmethod
    def render(state):
        return _countdown.render(state)


def make_sample(backend):
    """Return a ``sample(state, N, temperature, seed, mode)`` closure for ``backend``.

    Bakes the M1 sampler backend into the decoder's expected 5-arg signature.
    """
    def sample(state, N, temperature, seed, mode):
        return _sampler.sample(state, N, temperature, seed, backend, mode)

    return sample
