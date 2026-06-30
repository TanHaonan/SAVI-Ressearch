"""Bootstrap the shared decode substrate at the line seam (PLAN3 §10).

PLAN3 is "the joint milestone where the two lines meet at the ``sample()`` interface".
The emission line's ``countdown-decode/core`` package is exactly that seam: a
self-contained, GPU-free-importable layer that vendors

  * ``decode_core`` (the FROZEN trellis decoder + arms; the vendored copy is
    byte-identical to ``verifier-decode/decode_core`` — verified 2026-06-29),
  * the Countdown domain (``CountdownDomain``, Phi = (sorted values, target)),
  * the mock emission sampler (``sample(state, N, tau, seed, backend, mode)``), and
  * the real backend (``make_real_sampler`` / ``load_countdown_model``) matched to the
    trained Qwen3-4B+LoRA adapters.

Importing that one package keeps Tier A (mock sampler) and Tier B (real sampler) on a
SINGLE, identical domain + decoder + token-ledger, so the only thing that changes
between tiers is the ``sample`` closure — which is the whole point of the seam.

This module locates that package (env ``COUNTDOWN_DECODE_DIR`` overrides the default
absolute path), puts it on ``sys.path``, imports it, and re-exports the names PLAN3's
harness needs. Import is CPU-only (the real backend defers all torch imports inside
``load_countdown_model``), so this is safe to import in CPU tests / Tier A.
"""

import os
import sys
from pathlib import Path

# Default location of the emission line's countdown-decode package (the seam),
# resolved repo-relative to this file
# (contrib-savi/experiments/Haonan/verifier-decode/mass-harness/_deps/core_boot.py
#  -> contrib-savi/experiments/Haonan/decouple-training/countdown-decode). Override
# with COUNTDOWN_DECODE_DIR if the worktree layout differs.
_DEFAULT_COUNTDOWN_DECODE = str(
    Path(__file__).resolve().parents[3]
    / "decouple-training" / "countdown-decode"
)


def _resolve_dir():
    d = os.environ.get("COUNTDOWN_DECODE_DIR", _DEFAULT_COUNTDOWN_DECODE)
    p = Path(d).resolve()
    if not (p / "core" / "__init__.py").is_file():
        raise RuntimeError(
            f"countdown-decode substrate not found at {p}.\n"
            "Set COUNTDOWN_DECODE_DIR to the emission line's countdown-decode dir "
            "(the one holding core/__init__.py + core/real_backend.py)."
        )
    return p


COUNTDOWN_DECODE_DIR = _resolve_dir()
# The package PARENT must be on sys.path so ``import core`` resolves; core/__init__ then
# puts core/ and core/_deps on the path itself.
if str(COUNTDOWN_DECODE_DIR) not in sys.path:
    sys.path.insert(0, str(COUNTDOWN_DECODE_DIR))

import core  # noqa: E402  (the emission-line package; self-bootstraps _deps + decode_core)

# Re-export the surface PLAN3's harness consumes. Everything below is the SAME object
# the emission line uses, so domain/decoder/ledger semantics are shared across tiers.
CountdownDomain = core.CountdownDomain
load_instances = core.load_instances
instance_dict = core.instance_dict
Instance = core.Instance

# Mock + real emission backends (the two things that get swapped between Tier A and B).
mock_sample = core.sample                      # sample(state,N,tau,seed,backend,mode)
make_real_sampler = core.make_real_sampler     # (model,tok,domain,device) -> closure
load_countdown_model = core.load_countdown_model

# FROZEN decode_core arms (vendored; identical to verifier-decode/decode_core).
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

# Aggregation helpers.
metrics = core.metrics                         # boot_ci / modal_state / leaf_check
from core.run_decode import _result_pass1, _budget_dict  # noqa: E402

# decode_core's domain Protocol module (for isinstance / structural checks in tests).
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
    "_result_pass1",
    "_budget_dict",
    "decode_domain",
]
