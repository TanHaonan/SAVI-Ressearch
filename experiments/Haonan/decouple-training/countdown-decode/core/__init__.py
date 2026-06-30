"""Public surface of the countdown-decode experiment (self-contained, GPU-free).

This is the COUNTDOWN capstone of the decoupled-emission line, pivoting off the algebra
version whose Phi (canon = lhs-rhs scalar class) made productive moves canon-invariant
and the trellis degenerate. Countdown is the decoder's NATIVE domain: a state is a
multiset of values + a target; a move combines two values into one; the Phi key
``canon = (sorted values, target)`` STRICTLY SHRINKS each step (one fewer value), so the
step-trellis is finite, bounded, never recurs, and real aliasing (different combine
orders reaching the same multiset) is merged.

Re-exports the vendored decoder + countdown domain so sibling modules import from one
place and never reach into ``_deps`` directly::

    from core import (
        CountdownDomain,          # the Countdown Domain (Phi = (sorted values, target))
        load_instances,           # countdown instance loader (builtin / generated)
        Instance,                 # instance dataclass (numbers, target, id)
        sample,                   # the mock emission backend (sampler.sample)
        render_op, legal_ops, reachable, solve_one,  # domain helpers (datagen/backend)
        savi, greedy, best_of_k, oracle,             # decode_core arms
        Result, Budget, Node,     # decode_core result/accounting dataclasses
    )

The vendored tree under ``_deps/`` is importable as the top-level package ``_deps``; its
modules import each other as ``from _deps.countdown import ...`` and the decoder as
``from decode_core import decode``. Both roots (this package's dir, for ``_deps``; and
``_deps/`` itself, for ``decode_core``) are placed on ``sys.path`` here so the imports
resolve regardless of the caller's cwd, mirroring how ``algebra-decode`` boots its own
vendored modules.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEPS = _HERE / "_deps"
# ``_deps`` is a top-level package -> its PARENT (this dir) on the path.
# ``decode_core`` lives inside ``_deps`` and is imported bare -> ``_deps`` on the path.
for _p in (str(_HERE), str(_DEPS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _deps.countdown_domain import CountdownDomain
from _deps import countdown as _countdown
from _deps import sampler as _sampler
from _deps import instances as _instances
from _deps.instances import Instance
from _deps.sampler import sample, render_op
from real_backend import (
    make_real_sampler,
    load_countdown_model,
    build_move_prompt,
    extract_move_text,
    MOVE_PROMPT_SYSTEM,
    MOVE_PROMPT_USER,
    MOVE_LINE_PREFIX,
    STEP_CAP,
    MAX_NEW_TOKENS,
)
from decode_core.decode import (
    savi,
    greedy,
    best_of_k,
    oracle,
    Result,
    Budget,
    Node,
)
from decode_core import domain as decode_domain  # the Domain Protocol
from gen_sft_data import gen as gen_sft_data, goal_preserving_moves
from . import metrics  # aggregation helpers (boot_ci / modal_state / leaf_check)


# ---------------------------------------------------------------------------
# Instance loading (countdown): {"set":"builtin"} | {"set":"generated", ...}
# ---------------------------------------------------------------------------

def load_instances(spec):
    """Return a deterministic, ordered list of countdown ``Instance`` for ``spec``.

    Wraps the vendored ``instances.load_instances`` and accepts:

      ``{"set": "builtin"}``
          the frozen curated 28-instance set (stable ids ``bs-00``..``bs-27``):
          16 headroom (oracle-solvable, greedy-myopic fails), 8 greedy-solvable,
          4 unsolvable. ``"builtin_small"`` is also accepted (the vendored name).

      ``{"set": "generated", "seed": S, "n": N, "k": 4, "target_range": [a, b]}``
          N seeded k-number instances (ids ``gen-S-000``..); reproducible for a fixed
          spec. ``k`` (numbers per instance, default 4) and ``target_range`` (default
          [10, 100]) are optional.

    Each ``Instance`` carries the fields ``numbers`` (tuple[int, ...]), ``target``
    (int), and ``id`` (str). To run the decoder over an instance, pass the dict
    ``{"numbers": inst.numbers, "target": inst.target, "id": inst.id}`` to
    ``CountdownDomain.initial_state`` (the arms take that dict as their ``inst``).
    """
    kind = spec.get("set")
    if kind == "builtin":
        return _instances.load_instances({"set": "builtin_small"})
    return _instances.load_instances(spec)


def instance_dict(inst):
    """The decoder's instance dict for an ``Instance`` (what the arms / ``initial_state``
    consume): ``{"numbers", "target", "id"}``."""
    return {"numbers": inst.numbers, "target": inst.target, "id": inst.id}


# ---------------------------------------------------------------------------
# Domain helpers re-exported at the package level (datagen / backend reuse)
# ---------------------------------------------------------------------------

# State-level functions (operate on a ``State``, not a ``CountdownDomain``); exposed so
# datagen / a real backend can reuse the exact oracle + witness path + legal-op set
# without reaching into ``_deps`` directly.
legal_ops = _countdown.legal_ops
reachable = _countdown.reachable
solve_one = _countdown.solve_one
parse_target_problem = _countdown.parse_target_problem

__all__ = [
    "metrics",
    "CountdownDomain",
    "load_instances",
    "instance_dict",
    "Instance",
    "sample",
    "render_op",
    "make_real_sampler",
    "load_countdown_model",
    "build_move_prompt",
    "extract_move_text",
    "MOVE_PROMPT_SYSTEM",
    "MOVE_PROMPT_USER",
    "MOVE_LINE_PREFIX",
    "STEP_CAP",
    "MAX_NEW_TOKENS",
    "legal_ops",
    "reachable",
    "solve_one",
    "parse_target_problem",
    "savi",
    "greedy",
    "best_of_k",
    "oracle",
    "Result",
    "Budget",
    "Node",
    "decode_domain",
    "gen_sft_data",
    "goal_preserving_moves",
]
