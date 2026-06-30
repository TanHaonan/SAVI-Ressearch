"""Bootstrap the shared decode substrate for correlated-coverage (PREREG §13).

The correlated-coverage experiment consumes the SAME ``sample()`` interface,
``CountdownDomain`` (Phi = (sorted values, target)), and FROZEN ``decode_core``
arms as the emission line's ``countdown-decode/core`` package. Rather than reach
into ``_deps`` directly, this module locates that package (env
``COUNTDOWN_DECODE_DIR`` overrides the default absolute path), places it on
``sys.path``, imports it, and re-exports the names this experiment needs.

This mirrors the verifier line's ``joint-lambda-decode/core_boot.py`` (the line
seam at ``sample()``), trimmed to the surface correlated-coverage actually uses.
Import is CPU-only: the real backend defers all torch imports inside
``load_countdown_model``, so this is safe to import in Tier-A CPU tests.
"""

import os
import sys
from pathlib import Path

# Default location of the emission line's countdown-decode package (the substrate).
# Override with COUNTDOWN_DECODE_DIR if the worktree layout differs.
_DEFAULT_COUNTDOWN_DECODE = str(
    Path(__file__).resolve().parent.parent / "countdown-decode"
)


def _resolve_dir():
    d = os.environ.get("COUNTDOWN_DECODE_DIR", _DEFAULT_COUNTDOWN_DECODE)
    p = Path(d).resolve()
    if not (p / "core" / "__init__.py").is_file():
        raise RuntimeError(
            f"countdown-decode substrate not found at {p}.\n"
            "Set COUNTDOWN_DECODE_DIR to the emission line's countdown-decode dir "
            "(the one holding core/__init__.py)."
        )
    return p


COUNTDOWN_DECODE_DIR = _resolve_dir()
# The package PARENT must be on sys.path so ``import core`` resolves; core/__init__
# then puts core/ and core/_deps on the path itself.
if str(COUNTDOWN_DECODE_DIR) not in sys.path:
    sys.path.insert(0, str(COUNTDOWN_DECODE_DIR))

import core  # noqa: E402  (emission-line package; self-bootstraps _deps + decode_core)

# Re-export the surface correlated-coverage consumes. Everything below is the SAME
# object the emission line uses, so domain/decoder/ledger semantics are shared.
CountdownDomain = core.CountdownDomain
load_instances = core.load_instances
instance_dict = core.instance_dict
Instance = core.Instance

# Mock + real emission backends (swapped between Tier A and Tier B).
mock_sample = core.sample                       # sample(state,N,tau,seed,backend,mode)
make_real_sampler = core.make_real_sampler      # (model,tok,domain,device) -> closure
load_countdown_model = core.load_countdown_model

# FROZEN decode_core arms (vendored; byte-identical to verifier-decode/decode_core).
savi = core.savi
greedy = core.greedy
best_of_k = core.best_of_k
oracle = core.oracle
Result = core.Result
Budget = core.Budget
Node = core.Node

# Domain-level Countdown helpers (operate on a ``State``), for datagen / instruments.
legal_ops = core.legal_ops
reachable = core.reachable
solve_one = core.solve_one
parse_target_problem = core.parse_target_problem
render_op = core.render_op

# Aggregation helpers + decode_core's Domain Protocol module.
metrics = core.metrics                          # boot_ci / modal_state / leaf_check
decode_domain = core.decode_domain

__all__ = [
    "COUNTDOWN_DECODE_DIR",
    "core",
    "CountdownDomain",
    "load_instances",
    "instance_dict",
    "Instance",
    "mock_sample",
    "make_real_sampler",
    "load_countdown_model",
    "savi",
    "greedy",
    "best_of_k",
    "oracle",
    "Result",
    "Budget",
    "Node",
    "legal_ops",
    "reachable",
    "solve_one",
    "parse_target_problem",
    "render_op",
    "metrics",
    "decode_domain",
]
