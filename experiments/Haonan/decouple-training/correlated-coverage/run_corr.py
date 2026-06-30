"""Arm matrix + sweep harness for correlated-coverage (PREREG §5/§6/§7/§8/§9).

This is the Tier-A driver. It runs the four arms of PREREG §5 over the
``p × β × seed`` grid of PREREG §6, at the iso-token budget grid of PREREG §7,
on instances chosen so the i.i.d. coverage lands in the head-cover-room band
(V1), and emits the validity gates (V1–V4) and the H2/H3/H4/H5 instruments to
``outputs/tierA.json``. ``decode_core`` stays FROZEN; everything here consumes
only the experiment-local arms (``corr_sampler``), the Tier-A mock emission
(``mock_backend``), the measurement layer (``instruments``), and the substrate's
``CountdownDomain`` / ``oracle`` (read through ``core_boot``).

Arms (PREREG §5; all consume the SAME mock emission, only the sampling process
differs):

* ``iid_bok``           — K i.i.d. stepwise chains (the Erdős–Rényi / random-code
  baseline). N=1 candidate per committed step, so a depth-d chain costs ≈3d tokens.
* ``corr(β)``           — the constructive S' (canonical-state repulsion). N>1
  step-mode candidates per committed step (the ~N× tax is in its token ledger).
* ``temp_matched_iid``  — i.i.d. at the temperature ``instruments.tau_match`` chose
  so its realized MARGINAL step-entropy equals ``corr(β*)``'s (the H4 negative
  control: marginal spread ≠ joint anti-correlation).
* ``oracle``            — ``domain.solvable(s0)`` ceiling (the head-cover-room
  numerator; PREREG §5, V4).

The headline axis is generated tokens (PREREG §7): every arm's coverage is read at
a matched token budget B (the fast ``arm_curve`` here, which runs the arm ONCE at K_max
and derives coverage at every budget from the per-chain ``cum_tokens`` prefix — exactly
matching the slow ``instruments.coverage_at_budget``, verified by a test). corr reaches B
at a smaller K than iid because of its step-mode tax — the hard iso-token test, MECHANISM
§3.2 MAIN. For each ``(p)`` we pick a HEADLINE budget ``B*`` whose i.i.d. coverage lands
closest to the centre of the V1 band (0.2,0.9); the iso-token H1 / joint-diversity are
adjudicated at ``B*`` and ``K*`` (the K iid realizes at ``B*``). The H3 inverted-U is
measured CHAIN-MATCHED (fixed K, mechanism-cost-free) per PREREG §6 — Tier A's job is to
prove the PRINCIPLE separately from the step-mode token tax.

Validity gates (PREREG §9):

* V1 (head-cover-room): i.i.d. coverage@B* ∈ (0.2, 0.9) — checked per p-cell.
* V2 (well-posed β=0 ≡ i.i.d.): corr(β=0) and iid_bok produce byte-identical K-chain
  path lists on a sample of instances/seeds (MECHANISM §2 route A+B).
* V3 (iso-token): the per-arm token ledgers at B* are within tolerance of B*, and
  corr's step-mode N× overhead is present (corr's mean K < iid's mean K at the same B).
* V4 (clean eval oracle): the ``is_goal``/``apply`` evaluation path is identical
  across arms (the arms share ``_draw_one_chain``/``domain``), and the oracle ceiling
  strictly exceeds i.i.d. coverage@B* (real head-cover-room).

Instruments (PREREG §8):

* H2 (base数 / miss-decay): slope of log(1−coverage) vs token budget; corr's slope
  more negative than iid's (faster miss decay), with a paired bootstrap on the per-B
  coverage cells feeding the slope-difference CI.
* H3 (inverted-U): coverage_corr(β) over the β grid at B*; locate β* and check the
  non-monotone rise-then-fall (β* > 0 with coverage_corr(β*) > coverage_corr(0)+δ and
  a fall at large β).
* H4 (相关 ≠ 边际散开): temp_matched_iid's coverage@B* and joint distinctness vs
  corr(β*)'s, at matched marginal step-entropy.
* H5 (regime dependence): the gain coverage_corr(β*)−coverage_iid regressed on the
  i.i.d. baseline coverage across p — the slope should be < 0 (gain shrinks as the
  baseline saturates).

Determinism + resumability: the sweep is keyed by ``(p, β)`` cells; partial results
in ``outputs/tierA.json`` are reloaded and only missing cells are recomputed. All RNG
is the process-stable sha256-seeded mock + numpy ``default_rng(seed)`` bootstrap.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import core_boot as cb
import mock_backend as mb
import corr_sampler as cs
import instruments as ins


# ---------------------------------------------------------------------------
# Fixed experiment configuration (the prereg-frozen grid; PREREG §6, §10)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
OUTPUTS_DIR = HERE / "outputs"
DEFAULT_OUT = OUTPUTS_DIR / "tierA.json"
SMOKE_OUT = OUTPUTS_DIR / "tierA_smoke.json"

DOMAIN = cb.CountdownDomain()

# RETUNE (this correction; PREREG §6 + the round-3 findings). competence p × correlation β
# × mode-collapse × seeds. The prior frozen knobs (single collapse=0.5, N_corr=8, coarse β,
# low budgets) STARVED corr to K=1 and put it on a narrow funnel; we widen all four axes:
#
# * COLLAPSE_GRID is now a SWEEP AXIS {0.5, 0.8, 0.95} — the off-protocol probe recovered a
#   genuine inverted-U at collapse=0.8 (a redundant/broad good region, ~ real LLM mode
#   collapse), which the single collapse=0.5 funnel missed. Each (p, collapse) is a CELL.
# * BETA_GRID is FINE and LOW {0,0.1,0.25,0.5,0.75,1,2,4} — the lift lives at small β
#   (probe: 0.625->0.688 at β=0.5), so the coarse old grid under-sampled the rise.
# * N_CORR = 3 (was 8): the step-mode N× tax at N=8 (~72 tok/chain) starved corr to K=1 at
#   every in-band budget; N=3 (~9 tok/op·chain) lets corr afford K>=4 in-band (instance
#   filter below enforces this), so β finally has a >=2-chain ensemble to act on.
# * BUDGET_GRID pushed UP so corr can afford K>=4 in-band at N_corr=3.
P_GRID = [0.5, 0.6, 0.7, 0.8]
COLLAPSE_GRID = [0.5, 0.8, 0.95]
BETA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0]
SEEDS = list(range(1, 9))

DEPTH_CAP = 8
# Step-mode candidate count for corr's P_emission reconstruction (the N× tax knob).
N_CORR = 3         # RETUNE: was 8 (the K=1-starving tax); 3 lets corr afford K>=4 in-band.
N_IID = 1          # i.i.d. baseline: one candidate per committed step (≈3d tokens/chain)
TAU = 1.0          # base sampling temperature for iid/corr (the un-flattened emission)

# Cheap-arm prompt-overhead knobs (MECHANISM §3.2 BACKUP; corr_sampler.corr_cheap). The
# conditioning context that lists prior canonical states to avoid is CHARGED into Budget:
#   overhead(chain i) = PROMPT_OVERHEAD_BASE + PROMPT_OVERHEAD_PER_STATE · #prior-canon-states
# Both >0 so conditioning is NOT free; tunable here.
PROMPT_OVERHEAD_PER_STATE = 1
PROMPT_OVERHEAD_BASE = 0

# Token budget grid (generated tokens; PREREG §7). RETUNE: pushed UP so corr at N_corr=3
# can afford K>=4 in-band (a depth-3 corr chain costs ≈3·N·d≈27 tok, so K=4 needs ≈108+).
BUDGET_GRID = [24, 48, 96, 192, 384, 768]
K_MAX = 128        # cap on chains when growing K to a token budget

# Prereg thresholds (DRAFT; PREREG §10). Frozen on Tier A before any Tier B run.
DELTA_H1 = 0.05            # δ: coverage_corr(β*) − coverage_iid ≥ +0.05 (iso-token)
V1_BAND = (0.2, 0.9)      # head-cover-room band for iid coverage@B*
ISO_TOKEN_TOL = 0.0       # corr/iid ledgers must be ≤ B* exactly (the budget is a ceiling)
CORR_MIN_REALIZED_K = 4   # instance filter: corr must realize K>=4 chains at B* (else starved)
BOOT_N = 10000            # paired bootstrap resamples (PREREG §8, §10)

# Instance set: solvable depth-3 Countdown instances (target 24) curated so iid
# coverage spans the V1 band across the budget grid. Drawn from the substrate's
# curated headroom + a couple extras; all are oracle-solvable (V4 head-cover-room).
_RAW_INSTANCES = [
    (3, 7, 8, 9), (2, 3, 4, 5), (1, 5, 6, 7), (10, 4, 6, 2),
    (3, 5, 6, 8), (5, 6, 7, 9), (2, 6, 8, 12), (4, 5, 5, 8),
    (5, 8, 9, 11), (6, 7, 11, 13), (8, 10, 11, 11), (3, 3, 9, 11),
]
TARGET = 24


def make_instances():
    """The Tier-A instance set as decoder-arm instance dicts (PREREG §6)."""
    return [
        {"numbers": list(n), "target": TARGET, "id": f"ta-{i:02d}"}
        for i, n in enumerate(_RAW_INSTANCES)
    ]


# ---------------------------------------------------------------------------
# Arm closures (run_arm(inst, K, seed) -> Result; the instruments' contract)
# ---------------------------------------------------------------------------

def _safe_mean(xs):
    """Mean that returns ``nan`` (quietly) for an empty list (e.g. an empty filtered set)."""
    return float(np.mean(xs)) if len(xs) else float("nan")


def _argmax_ignore_nan(vals):
    """Index of the max, ignoring nan; returns 0 if every value is nan (degenerate cell)."""
    finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
    if not finite:
        return 0
    return max(finite, key=lambda iv: iv[1])[0]


def make_backend(p, collapse=0.5):
    """The Tier-A competence-p × mode-collapse mock emission closure (RETUNE: collapse axis)."""
    return mb.make_competence_backend(p=p, collapse=collapse, depth_cap=DEPTH_CAP,
                                      domain=DOMAIN)


def corr_runner(sample, beta, N=N_CORR, tau=TAU):
    """``corr_step(β)`` arm closure: step-mode canonical-state repulsion (the N× claim, S')."""
    return lambda inst, K, seed: cs.corr_step(
        DOMAIN, sample, inst, K=K, N=N, beta=beta, tau=tau, seed=seed
    )


def corr_cheap_runner(sample, beta, N=N_CORR, tau=TAU,
                      per_state=PROMPT_OVERHEAD_PER_STATE, base=PROMPT_OVERHEAD_BASE):
    """``corr_cheap(β)`` arm closure: prompt-conditioning repulsion at ~1× generation cost
    plus an EXPLICIT, charged prompt-overhead for the avoid-list (MECHANISM §3.2 BACKUP).

    The decisive arm of this correction: does the lift survive INDUCING the correlation at
    ~1× generation (the only thing a real, cheap LLM mechanism could afford), once the
    conditioning-context prompt tokens are honestly charged into the SAME Budget ledger?
    """
    return lambda inst, K, seed: cs.corr_cheap(
        DOMAIN, sample, inst, K=K, N=N, beta=beta, tau=tau, seed=seed,
        prompt_overhead_per_state=per_state, prompt_overhead_base=base,
    )


def iid_runner(sample, N=N_IID, tau=TAU):
    """``iid_bok`` arm closure: K i.i.d. stepwise chains (the headline baseline)."""
    return lambda inst, K, seed: cs.iid_bok(
        DOMAIN, sample, inst, K=K, tau=tau, seed=seed, N=N
    )


def oracle_ceiling(instances):
    """``oracle`` arm: ``domain.solvable(s0)`` mean over instances (the ceiling, V4)."""
    vals = [int(bool(cb.oracle(DOMAIN, inst))) for inst in instances]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Fast iso-token curve from a SINGLE run at K_max (the budget-prefix trick)
# ---------------------------------------------------------------------------
#
# Both arms draw the K chains STRICTLY SEQUENTIALLY with per-chain seeds
# ``seed_i = f(base_seed, inst, i)`` that do not depend on K, and the per-(canon,depth)
# counter is built incrementally, so the first K' chains of a K_max run are BYTE-IDENTICAL
# to a standalone K=K' run (verified: prefix equivalence). Each arm now records
# ``detail["cum_tokens"]`` = the accumulated ``Budget.tokens`` after each chain. Hence ONE
# run at K_max yields coverage and tokens at EVERY budget via a prefix scan — replacing the
# O(K_max) re-draws of the generic ``instruments.coverage_vs_tokens`` with O(1) arm calls
# per (inst, seed). The numbers are identical to the slow path; only the cost changes.


def _coverage_token_at_budget_from_run(res, budget_B):
    """Coverage (0/1), realized K, and tokens of ONE arm run at token budget ``budget_B``.

    ``res`` is the arm Result at K_max. ``cum_tokens[k-1]`` is the accumulated tokens after
    drawing k chains; ``reached[:k]`` are those chains' goal flags. We take the LARGEST k
    whose ``cum_tokens[k-1] <= budget_B`` and report coverage = any(reached[:k]). If even
    one chain overruns ``budget_B`` we keep the k=1 floor (an arm must draw >=1 chain;
    recorded honestly, matching ``instruments.coverage_at_budget``).
    """
    cum = res.detail["cum_tokens"]
    reached = res.detail["reached"]
    if not cum:
        return 0, 0, 0
    k_fit = 0
    for k in range(1, len(cum) + 1):
        if cum[k - 1] <= budget_B:
            k_fit = k
        else:
            break
    if k_fit == 0:
        # Even the first chain overruns B: keep the k=1 floor (cannot draw fewer).
        return int(bool(reached[0])), 1, int(cum[0])
    return int(any(reached[:k_fit])), k_fit, int(cum[k_fit - 1])


def _k_for_budget(budgets, min_chain_tokens=3, cap=K_MAX, margin=2):
    """Smallest K_max that can exhaust the largest budget at the floor per-chain cost.

    The cheapest possible chain is a single legal op (≈3 space-delimited tokens), so
    ``ceil(max_budget / 3)`` chains suffice to overrun any budget for ANY arm; we add a
    small margin and cap at ``K_MAX`` so one run at this K covers the whole budget grid
    (prefix equivalence) without the wasteful fixed K=128 (which dominated runtime).
    """
    max_B = max(budgets) if budgets else 0
    need = int(max_B // max(1, min_chain_tokens)) + margin
    return max(2, min(cap, need))


def arm_curve(run_arm, instances, seeds, budgets, K_max=None):
    """Fast coverage-vs-token curve for one arm (the iso-token headline; PREREG §7).

    Runs ``run_arm(inst, K_max, seed)`` ONCE per (inst, seed) and derives coverage/K at
    every budget via the prefix scan above. ``K_max`` defaults to the smallest count that
    can exhaust the largest budget at the floor per-chain cost (``_k_for_budget``), so one
    run covers the whole budget grid cheaply. Returns the same shape as
    ``instruments.coverage_vs_tokens`` (``budgets``/``coverage``/``per_cell``/``mean_K``)
    PLUS the cached per-(inst,seed) Result list in ``runs`` (so callers can reuse it for
    joint-diversity / entropy at a chosen K without re-running). Coverage values match the
    slow instrument exactly (prefix equivalence).
    """
    if K_max is None:
        # Probe ONE chain per (inst, seed) — cheap, K=1 — to learn this arm's realized
        # per-chain token cost, then size K_max from the MINIMUM (cheapest) per-chain cost
        # so the largest budget is exhausted for EVERY cell (the cheapest chains need the
        # most chains to reach B). This avoids running the expensive step-mode corr arm out
        # to the iid-floor K (e.g. 64) when only a few of its costly chains fit, while
        # guaranteeing no cell under-counts coverage at the largest budget.
        max_B = max(budgets) if budgets else 0
        per_chain_costs = []
        for inst in instances:
            for seed in seeds:
                probe = run_arm(inst, 1, seed)
                ct = probe.detail.get("cum_tokens")
                if ct:
                    per_chain_costs.append(ct[0])
        min_pc = min(per_chain_costs) if per_chain_costs else 0
        if min_pc > 0:
            K_max = max(2, min(K_MAX, int(max_B // min_pc) + 3))
        else:
            K_max = _k_for_budget(budgets)
    runs = [[run_arm(inst, K_max, seed) for seed in seeds] for inst in instances]
    coverage, mean_K, per_cell = [], [], {}
    for B in budgets:
        cell, ks = [], []
        for i_inst in range(len(instances)):
            for i_seed in range(len(seeds)):
                ok, K, _tok = _coverage_token_at_budget_from_run(runs[i_inst][i_seed], B)
                cell.append(int(ok))
                ks.append(K)
        per_cell[B] = cell
        coverage.append(float(np.mean(cell)) if cell else float("nan"))
        mean_K.append(float(np.mean(ks)) if ks else float("nan"))
    return {
        "budgets": list(budgets),
        "coverage": coverage,
        "per_cell": per_cell,
        "mean_K": mean_K,
        "runs": runs,
    }


def _mean_tokens_at_B_from_runs(runs, budget_B):
    """Mean realized ``Budget.tokens`` at ``budget_B`` over a cached ``runs`` grid (V3)."""
    toks = []
    for row in runs:
        for res in row:
            _ok, _K, tok = _coverage_token_at_budget_from_run(res, budget_B)
            toks.append(tok)
    return float(np.mean(toks)) if toks else float("nan")


def _jd_at_K_from_runs(runs, K, keep_mask=None):
    """Mean ``distinct_canon@K`` over a cached ``runs`` grid, using the first K chains.

    ``keep_mask`` (optional, per-instance) restricts to the filtered instance set so the
    joint-diversity read matches the H1/H4 adjudication set (RETUNE instance filter).
    """
    leaf_vals, union_vals = [], []
    for i_inst, row in enumerate(runs):
        if keep_mask is not None and not keep_mask[i_inst]:
            continue
        for res in row:
            chains = res.detail["chains"][:K]
            d = ins.distinct_canon_at_K(chains)
            leaf_vals.append(d["leaf"])
            union_vals.append(d["union"])
    return {
        "leaf_mean": float(np.mean(leaf_vals)) if leaf_vals else float("nan"),
        "union_mean": float(np.mean(union_vals)) if union_vals else float("nan"),
    }


def _entropy_at_K_from_runs(runs, K, keep_mask=None):
    """Mean realized marginal step-entropy over a cached ``runs`` grid (first K chains).

    ``keep_mask`` (optional) restricts to the filtered instance set (RETUNE), so the
    τ-match TARGET entropy is measured on the same instances H4 is adjudicated on.
    """
    vals = []
    for i_inst, row in enumerate(runs):
        if keep_mask is not None and not keep_mask[i_inst]:
            continue
        for res in row:
            vals.append(ins.marginal_step_entropy(res.detail["chains"][:K]))
    return float(np.mean(vals)) if vals else 0.0


def _coverage_cells_at_K_from_runs(runs, K, keep_mask=None):
    """Paired per-(inst,seed) coverage 0/1 over a cached ``runs`` grid at EXACTLY K chains.

    Coverage = any of the first K chains reached the goal (prefix equivalence makes this
    identical to a standalone K-chain run). This is the CHAIN-MATCHED (mechanism-cost-free)
    axis PREREG §6 prescribes for the H3 inverted-U: it isolates the correlation effect of
    β from the step-mode token tax (which the iso-token H1/V3 charge separately).
    ``keep_mask`` (optional) restricts to the filtered instance set (RETUNE).
    """
    cells = []
    for i_inst, row in enumerate(runs):
        if keep_mask is not None and not keep_mask[i_inst]:
            continue
        for res in row:
            reached = res.detail["reached"][:K]
            cells.append(int(any(reached)))
    return cells


# ---------------------------------------------------------------------------
# Validity gates (PREREG §9)
# ---------------------------------------------------------------------------

def gate_v1(iid_cov_at_B, band=V1_BAND):
    """V1: i.i.d. coverage@B* ∈ (lo, hi) — head-cover-room non-empty / not saturated."""
    lo, hi = band
    return {
        "name": "V1_head_cover_room",
        "iid_cov_at_B": float(iid_cov_at_B),
        "band": list(band),
        "pass": bool(lo < iid_cov_at_B < hi),
    }


def gate_v2(sample, instances, seeds, beta_grid_unused=None, K=6, N=N_CORR, tau=TAU):
    """V2: BOTH corr_step(β=0) AND corr_cheap(β=0) ≡ iid_bok byte-for-byte (MECHANISM §2 A+B).

    Asserts the realized K-chain path lists are byte-identical to ``iid_bok`` at matched N
    for BOTH repulsion arms. The β=0 reweight is the identity and (for the cheap arm) β=0
    short-circuits the conditioning entirely (no avoid-list, no prompt overhead), so both
    arms reduce EXACTLY to i.i.d. best-of-K. If either arm mismatches, this gate fails.
    """
    mismatches_step = 0
    mismatches_cheap = 0
    checked = 0
    for inst in instances:
        for seed in seeds:
            ri = cs.iid_bok(DOMAIN, sample, inst, K=K, tau=tau, seed=seed, N=N)
            rs = cs.corr_step(DOMAIN, sample, inst, K=K, N=N, beta=0.0, tau=tau,
                              seed=seed)
            rch = cs.corr_cheap(DOMAIN, sample, inst, K=K, N=N, beta=0.0, tau=tau,
                                seed=seed)
            checked += 1
            if rs.detail["chains"] != ri.detail["chains"] \
                    or rs.detail["reached"] != ri.detail["reached"]:
                mismatches_step += 1
            if rch.detail["chains"] != ri.detail["chains"] \
                    or rch.detail["reached"] != ri.detail["reached"]:
                mismatches_cheap += 1
    return {
        "name": "V2_beta0_equiv",
        "checked_cells": int(checked),
        "mismatches": int(mismatches_step + mismatches_cheap),
        "mismatches_step": int(mismatches_step),
        "mismatches_cheap": int(mismatches_cheap),
        "K": int(K),
        "N": int(N),
        "pass": bool(mismatches_step == 0 and mismatches_cheap == 0 and checked > 0),
    }


def gate_v3(iid_tokens, corr_tokens, iid_meanK, corr_meanK, budget_B,
            corr_per_chain_tokens, tol=ISO_TOKEN_TOL):
    """V3: iso-token — per-arm realized tokens ≤ B* (within tol), and corr's N× tax is
    present (corr realizes fewer chains than iid at the same B).

    The token-ceiling check honours the documented K=1 floor: an arm MUST draw at least
    one chain, so if a single corr chain (``corr_per_chain_tokens``) already exceeds B*,
    the overrun is the unavoidable floor (matching ``instruments.coverage_at_budget``),
    not a ledger bug — we then require corr's realized tokens to be ≤ that one-chain cost.
    The headline iso-token fairness is still binding: both arms are compared at the SAME
    B*, and corr's step-mode N× tax shows up as ``corr_mean_K < iid_mean_K`` (the tax).
    """
    iid_ok = iid_tokens <= budget_B * (1.0 + tol) + 1e-9
    one_chain_floor = corr_per_chain_tokens > budget_B
    if one_chain_floor:
        # B* too small for even one corr chain: the K=1 floor legitimately overruns.
        corr_ok = corr_tokens <= corr_per_chain_tokens + 1e-9
    else:
        corr_ok = corr_tokens <= budget_B * (1.0 + tol) + 1e-9
    tax_present = corr_meanK < iid_meanK  # corr pays N× more per chain -> fewer chains
    return {
        "name": "V3_iso_token",
        "budget_B": float(budget_B),
        "iid_tokens_at_B": float(iid_tokens),
        "corr_tokens_at_B": float(corr_tokens),
        "corr_per_chain_tokens": float(corr_per_chain_tokens),
        "k1_floor_overrun": bool(one_chain_floor),
        "iid_mean_K": float(iid_meanK),
        "corr_mean_K": float(corr_meanK),
        "tax_present": bool(tax_present),
        "pass": bool(iid_ok and corr_ok and tax_present),
    }


def gate_v4(oracle_cov, iid_cov_at_B):
    """V4: clean eval oracle — ceiling > i.i.d. coverage@B* (real head-cover-room).

    The is_goal/apply evaluation path is identical across arms by construction (they
    share ``_draw_one_chain`` + the same ``domain``), so this gate reduces to the
    head-cover-room check: the absolute solvability ceiling must exceed the realized
    i.i.d. coverage at the headline budget (there is space to push coverage up).
    """
    return {
        "name": "V4_clean_oracle",
        "oracle_ceiling": float(oracle_cov),
        "iid_cov_at_B": float(iid_cov_at_B),
        "shared_eval_path": True,  # arms share _draw_one_chain + domain (structural)
        "pass": bool(oracle_cov > iid_cov_at_B),
    }


# ---------------------------------------------------------------------------
# Per-p cell computation (one competence level; full β sweep + instruments)
# ---------------------------------------------------------------------------

def _pick_headline_budget(iid_curve, band=V1_BAND, corr_curve=None,
                          min_K=CORR_MIN_REALIZED_K):
    """Pick the headline budget B* (RETUNE: corr-aware so corr can afford K>=4 in-band).

    B* must satisfy TWO things for a non-degenerate iso-token H1: (i) iid coverage@B* lands
    inside the V1 band (lo,hi) — the head-cover-room constraint — AND (ii) corr realizes at
    least ``min_K`` chains at B* so β has a >=4-chain ensemble to act on (the exact
    degeneracy this correction removes; the old picker optimized only (i) and let corr starve
    to K=1). Among budgets satisfying BOTH we take the LARGEST (most corr chains, most room
    for the repulsion to spread); if NONE satisfies both we relax to the in-band B with the
    most corr chains, then to the iid-band-centre fallback (V1/the filter then fail honestly).
    """
    lo, hi = band
    centre = 0.5 * (lo + hi)
    budgets = iid_curve["budgets"]
    cov = iid_curve["coverage"]
    corr_K = corr_curve["mean_K"] if corr_curve is not None else None
    inside = [i for i, c in enumerate(cov) if lo < c < hi]
    if corr_K is not None and inside:
        both = [i for i in inside if corr_K[i] >= min_K]
        if both:
            best_i = max(both, key=lambda i: budgets[i])  # largest in-band B with K>=min_K
            return budgets[best_i], best_i
        # No in-band B affords K>=min_K: take the in-band B with the most corr chains.
        best_i = max(inside, key=lambda i: (corr_K[i], budgets[i]))
        return budgets[best_i], best_i
    pool = inside if inside else list(range(len(cov)))
    best_i = min(pool, key=lambda i: abs(cov[i] - centre))
    return budgets[best_i], best_i


# ---------------------------------------------------------------------------
# Instance filter (RETUNE): keep instances that are oracle-solvable AND in the iid V1 band
# AND on which corr realizes K>=4 at B* (else corr is STARVED and β acts on nothing).
# ---------------------------------------------------------------------------

def _per_instance_iid_cov_at_B(iid_runs, seeds, B_star):
    """Mean iid coverage@B* PER INSTANCE (averaged over seeds) — for the V1-band filter."""
    out = []
    for row in iid_runs:  # one row per instance
        cell = []
        for res in row:
            ok, _K, _tok = _coverage_token_at_budget_from_run(res, B_star)
            cell.append(ok)
        out.append(float(np.mean(cell)) if cell else float("nan"))
    return out


def _per_instance_corr_realizedK_at_B(corr_runs, B_star):
    """Mean realized corr K@B* PER INSTANCE (averaged over seeds) — for the K>=4 filter."""
    out = []
    for row in corr_runs:
        ks = []
        for res in row:
            _ok, K, _tok = _coverage_token_at_budget_from_run(res, B_star)
            ks.append(K)
        out.append(float(np.mean(ks)) if ks else 0.0)
    return out


def _instance_filter_mask(instances, iid_runs, corr_runs, seeds, B_star, band=V1_BAND,
                          min_K=CORR_MIN_REALIZED_K):
    """Boolean keep-mask over ``instances`` for the RETUNE filter (PREREG §9 V1 + this fix).

    An instance is KEPT iff: (i) oracle-solvable (head-cover-room numerator nonzero),
    (ii) its per-instance iid coverage@B* lands strictly inside the V1 band ``(lo,hi)`` (so
    the instance is neither saturated nor unreachable), and (iii) corr realizes at least
    ``min_K`` chains at B* on it (else corr is starved to <4 chains and β has no >=2-chain
    ensemble to repel — the exact degeneracy this correction removes).
    """
    lo, hi = band
    iid_pc = _per_instance_iid_cov_at_B(iid_runs, seeds, B_star)
    corr_pc = _per_instance_corr_realizedK_at_B(corr_runs, B_star)
    mask = []
    for idx, inst in enumerate(instances):
        solvable = bool(cb.oracle(DOMAIN, inst))
        in_band = lo < iid_pc[idx] < hi
        enough_K = corr_pc[idx] >= min_K
        mask.append(bool(solvable and in_band and enough_K))
    return mask, iid_pc, corr_pc


def _cells_from_runs_filtered(runs, B_star, seeds, keep_mask):
    """Paired per-(inst,seed) coverage 0/1 at B*, keeping only instances in ``keep_mask``."""
    cells = []
    for i_inst, row in enumerate(runs):
        if not keep_mask[i_inst]:
            continue
        for res in row:
            ok, _K, _tok = _coverage_token_at_budget_from_run(res, B_star)
            cells.append(int(ok))
    return cells


def run_p_cell(p, beta_grid, seeds, instances, budgets, verbose=False,
               boot_n=BOOT_N, n_corr=N_CORR, collapse=0.5):
    """Run the full β sweep + 3-arm matrix + instruments for one ``(p, collapse)`` cell.

    RETUNE: a CELL is now ``(p, collapse)`` (PREREG §6 + the round-3 findings). Both
    repulsion arms are run over the fine β grid: ``corr_step`` (the N× step-mode tax) and
    ``corr_cheap`` (prompt-conditioning at ~1× generation + a charged prompt-overhead). The
    headline H1 is adjudicated for BOTH arms at the iso-token B*, on the FILTERED instance
    set (oracle-solvable AND iid cov@B* in band AND corr realizes K>=4 at B*). H4's τ-match
    target entropy is measured at K*>=4 (the fix for the K=1-degenerate control). Returns a
    JSON-serializable cell dict. ``boot_n`` / ``n_corr`` are exposed so the smoke can shrink
    them; H5 is assembled across cells by the caller.
    """
    t0 = time.time()
    sample = make_backend(p, collapse=collapse)
    iid_run = iid_runner(sample)

    # --- iid baseline + both repulsion arms' coverage-vs-token curves over β -----
    # All curves are computed FIRST (each is a single run at K_max with a prefix scan), so
    # B* can be picked CORR-AWARE (the budget where iid is in-band AND corr affords K>=4).
    iid_curve = arm_curve(iid_run, instances, seeds, budgets)
    iid_runs = iid_curve["runs"]

    corr_curves, corr_runs = {}, {}
    cheap_curves, cheap_runs = {}, {}
    for beta in beta_grid:
        corr_curves[beta] = arm_curve(corr_runner(sample, beta, N=n_corr),
                                      instances, seeds, budgets)
        corr_runs[beta] = corr_curves[beta]["runs"]
        cheap_curves[beta] = arm_curve(corr_cheap_runner(sample, beta, N=n_corr),
                                       instances, seeds, budgets)
        cheap_runs[beta] = cheap_curves[beta]["runs"]

    # B* is corr-aware: the largest in-band budget where corr(β=0) realizes K>=4 (so β has a
    # >=4-chain ensemble to act on). The corr(β=0) curve is the reference (cost is β-invariant
    # at fixed B, so any β realizes ~the same K at B).
    corr_ref_curve = corr_curves[0.0] if 0.0 in corr_curves else corr_curves[beta_grid[0]]
    B_star, B_idx = _pick_headline_budget(iid_curve, V1_BAND, corr_curve=corr_ref_curve,
                                          min_K=CORR_MIN_REALIZED_K)
    iid_cov_at_B = iid_curve["coverage"][B_idx]
    iid_meanK_at_B = iid_curve["mean_K"][B_idx]

    # --- B*-dependent reads for both repulsion arms ----------------------------
    corr_cov_at_B, corr_meanK_at_B, corr_tokens_at_B = {}, {}, {}
    cheap_cov_at_B, cheap_meanK_at_B, cheap_tokens_at_B, cheap_overhead = {}, {}, {}, {}
    for beta in beta_grid:
        curve = corr_curves[beta]
        corr_cov_at_B[beta] = curve["coverage"][B_idx]
        corr_meanK_at_B[beta] = curve["mean_K"][B_idx]
        corr_tokens_at_B[beta] = _mean_tokens_at_B_from_runs(curve["runs"], B_star)

        ccurve = cheap_curves[beta]
        cheap_cov_at_B[beta] = ccurve["coverage"][B_idx]
        cheap_meanK_at_B[beta] = ccurve["mean_K"][B_idx]
        cheap_tokens_at_B[beta] = _mean_tokens_at_B_from_runs(ccurve["runs"], B_star)
        cheap_overhead[beta] = float(np.mean([
            ccurve["runs"][i][j].detail.get("prompt_overhead_tokens", 0)
            for i in range(len(ccurve["runs"]))
            for j in range(len(ccurve["runs"][i]))
        ])) if ccurve["runs"] else 0.0

    betas_sorted = sorted(beta_grid)

    # --- INSTANCE FILTER (RETUNE): keep oracle-solvable + iid-in-band + corr-K>=4 -----
    # The filter mask is built from the corr_step runs at B* (the step arm is the one that
    # can starve to K<4); the same kept instances are used for ALL paired adjudications so
    # the arms are compared on the identical, non-degenerate instance set.
    # Use the β with the best iso-token corr coverage to choose the filter reference run
    # (any β realizes the same K at fixed B since cost ~ chains; β=0 is the cleanest ref).
    filter_ref_runs = corr_runs[0.0] if 0.0 in corr_runs else corr_runs[betas_sorted[0]]
    keep_mask, iid_pc, corr_pc = _instance_filter_mask(
        instances, iid_runs, filter_ref_runs, seeds, B_star,
        band=V1_BAND, min_K=CORR_MIN_REALIZED_K)
    n_kept = int(sum(keep_mask))

    # Paired 0/1 cells at B* on the FILTERED instance set (the H1 legs).
    iid_cell = _cells_from_runs_filtered(iid_runs, B_star, seeds, keep_mask)
    corr_cells_at_B = {b: _cells_from_runs_filtered(corr_runs[b], B_star, seeds, keep_mask)
                       for b in betas_sorted}
    cheap_cells_at_B = {b: _cells_from_runs_filtered(cheap_runs[b], B_star, seeds, keep_mask)
                        for b in betas_sorted}
    # Filtered iid coverage@B* (the headline baseline on the adjudication set).
    iid_cov_at_B_filt = float(np.mean(iid_cell)) if iid_cell else float("nan")

    # --- H3: inverted-U over β (CHAIN-MATCHED, mechanism-cost-free) — corr_step ---
    K_H3 = max(CORR_MIN_REALIZED_K, int(round(iid_meanK_at_B))) \
        if np.isfinite(iid_meanK_at_B) else CORR_MIN_REALIZED_K
    covK_over_beta = [
        _safe_mean(_coverage_cells_at_K_from_runs(corr_runs[b], K_H3, keep_mask))
        for b in betas_sorted
    ]
    cheap_covK_over_beta = [
        _safe_mean(_coverage_cells_at_K_from_runs(cheap_runs[b], K_H3, keep_mask))
        for b in betas_sorted
    ]
    beta_star_K = betas_sorted[_argmax_ignore_nan(covK_over_beta)]
    covK0 = covK_over_beta[0]
    covK_star = covK_over_beta[_argmax_ignore_nan(covK_over_beta)]
    covK_max = covK_over_beta[-1]
    h3 = {
        "betas": betas_sorted,
        "K_H3": int(K_H3),
        "coverageK_over_beta": [float(c) for c in covK_over_beta],   # corr_step (primary)
        "cheap_coverageK_over_beta": [float(c) for c in cheap_covK_over_beta],
        "coverage_isotoken_over_beta": [float(corr_cov_at_B[b]) for b in betas_sorted],
        "beta_star_chain_matched": float(beta_star_K),
        "covK_at_beta0": float(covK0),
        "covK_at_beta_star": float(covK_star),
        "covK_at_beta_max": float(covK_max),
        "interior_peak": bool(beta_star_K > 0.0),
        "rise_over_delta": bool(covK_star > covK0 + DELTA_H1),
        "falls_at_large_beta": bool(covK_max < covK_star),
    }
    h3["pass"] = bool(h3["interior_peak"] and h3["rise_over_delta"]
                      and h3["falls_at_large_beta"])

    # β* for the HEADLINE (iso-token) H1: the iso-token argmax of corr_step coverage on the
    # filtered set. Cheap arm picks its OWN iso-token β* (its budget profile differs).
    corr_cov_filt = {b: _safe_mean(corr_cells_at_B[b]) for b in betas_sorted}
    cheap_cov_filt = {b: _safe_mean(cheap_cells_at_B[b]) for b in betas_sorted}
    beta_star = betas_sorted[_argmax_ignore_nan([corr_cov_filt[b] for b in betas_sorted])]
    cheap_beta_star = betas_sorted[
        _argmax_ignore_nan([cheap_cov_filt[b] for b in betas_sorted])]
    cov_at_beta_star = corr_cov_filt[beta_star]

    # --- H1: corr_step(β*) − iid AND corr_cheap(β*) − iid at B* (paired bootstrap) ---
    h1_boot = ins.paired_bootstrap(corr_cells_at_B[beta_star], iid_cell,
                                   n_boot=boot_n, seed=int(round(p * 1000)))
    h1 = {
        "arm": "corr_step",
        "beta_star": float(beta_star),
        "delta": h1_boot["delta"],
        "ci_low": h1_boot["ci_low"],
        "ci_high": h1_boot["ci_high"],
        "excludes_zero": h1_boot["excludes_zero"],
        "p_two_sided": h1_boot["p_two_sided"],
        "meets_delta": bool(h1_boot["delta"] >= DELTA_H1),
        "pass": bool(h1_boot["excludes_zero"] and h1_boot["delta"] >= DELTA_H1),
    }
    h1c_boot = ins.paired_bootstrap(cheap_cells_at_B[cheap_beta_star], iid_cell,
                                    n_boot=boot_n, seed=int(round(p * 1000)) + 7)
    h1_cheap = {
        "arm": "corr_cheap",
        "beta_star": float(cheap_beta_star),
        "delta": h1c_boot["delta"],
        "ci_low": h1c_boot["ci_low"],
        "ci_high": h1c_boot["ci_high"],
        "excludes_zero": h1c_boot["excludes_zero"],
        "p_two_sided": h1c_boot["p_two_sided"],
        "meets_delta": bool(h1c_boot["delta"] >= DELTA_H1),
        "mean_prompt_overhead_tokens": float(cheap_overhead[cheap_beta_star]),
        "pass": bool(h1c_boot["excludes_zero"] and h1c_boot["delta"] >= DELTA_H1),
    }

    # --- H2: miss-decay slope (base数) — corr_step(β*) vs iid -------------------
    iid_slope = ins.miss_decay_slope(budgets, iid_curve["coverage"])
    corr_slope = ins.miss_decay_slope(budgets, corr_curves[beta_star]["coverage"])
    h2 = {
        "iid_slope": iid_slope["slope"],
        "corr_slope": corr_slope["slope"],
        "iid_r2": iid_slope["r2"],
        "corr_r2": corr_slope["r2"],
        "slope_diff": (float(corr_slope["slope"] - iid_slope["slope"])
                       if np.isfinite(corr_slope["slope"])
                       and np.isfinite(iid_slope["slope"]) else float("nan")),
        "pass": bool(np.isfinite(corr_slope["slope"]) and np.isfinite(iid_slope["slope"])
                     and corr_slope["slope"] < iid_slope["slope"]),
    }

    # --- H4: temp_matched_iid (marginal-entropy matched negative control) ------
    # FIX (H4): measure the τ-match TARGET marginal entropy at K*>=4 — NOT the K=1 that made
    # the old control degenerate (a 1-chain ensemble has 0 marginal step-entropy by
    # construction, so any τ "matched" it and the control was vacuous). K* is the corr arm's
    # realized chains at B*, floored at CORR_MIN_REALIZED_K so the marginal entropy the
    # temperature arm must match is the entropy of a >=4-chain ensemble (non-degenerate).
    K_star_realized = int(round(corr_meanK_at_B[beta_star])) \
        if np.isfinite(corr_meanK_at_B[beta_star]) else CORR_MIN_REALIZED_K
    K_star = max(CORR_MIN_REALIZED_K, K_star_realized)
    corr_star_runs = corr_runs[beta_star]
    # Target entropy at K*>=4 on the FILTERED instance set (the H4 adjudication set).
    target_H = _entropy_at_K_from_runs(corr_star_runs, K_star, keep_mask)
    tau_fit = ins.tau_match(lambda tau: iid_runner(sample, N=N_IID, tau=tau),
                            target_H, instances, seeds, K_star,
                            tau_lo=0.0, tau_hi=8.0, tol=0.05, max_iter=12)
    temp_run = iid_runner(sample, N=N_IID, tau=tau_fit["tau"])
    temp_curve = arm_curve(temp_run, instances, seeds, budgets)
    temp_runs = temp_curve["runs"]
    temp_cell = _cells_from_runs_filtered(temp_runs, B_star, seeds, keep_mask)
    temp_cov_at_B = float(np.mean(temp_cell)) if temp_cell else float("nan")
    jd_corr = _jd_at_K_from_runs(corr_star_runs, K_star, keep_mask)
    jd_iid = _jd_at_K_from_runs(iid_runs, K_star, keep_mask)
    jd_temp = _jd_at_K_from_runs(temp_runs, K_star, keep_mask)
    h4_boot = ins.paired_bootstrap(temp_cell, iid_cell, n_boot=boot_n,
                                   seed=int(round(p * 1000)) + 1) if temp_cell else \
        {"delta": float("nan"), "excludes_zero": False}
    h4 = {
        "tau_matched": tau_fit["tau"],
        "tau_match_target_entropy": float(target_H),
        "tau_match_target_K": int(K_star),  # FIX: target entropy measured at this K>=4
        "tau_match_reached": bool(tau_fit["reached"]),
        "tau_match_gap": float(tau_fit["gap"]),
        "K_star": int(K_star),
        "K_star_realized": int(K_star_realized),
        "temp_cov_at_B": float(temp_cov_at_B),
        "corr_cov_at_B": float(cov_at_beta_star),
        "iid_cov_at_B": float(iid_cov_at_B_filt),
        "temp_minus_iid_delta": h4_boot["delta"],
        "temp_minus_iid_excludes_zero": h4_boot["excludes_zero"],
        "jd_corr_leaf": jd_corr["leaf_mean"],
        "jd_iid_leaf": jd_iid["leaf_mean"],
        "jd_temp_leaf": jd_temp["leaf_mean"],
        "jd_corr_union": jd_corr["union_mean"],
        "jd_temp_union": jd_temp["union_mean"],
        "temp_gain_is_null": bool(not h4_boot["excludes_zero"]),
        "corr_beats_temp_jointly": bool(jd_corr["leaf_mean"] > jd_temp["leaf_mean"]),
    }
    h4["pass"] = bool(h4["temp_gain_is_null"] and h4["corr_beats_temp_jointly"])

    # --- validity gates --------------------------------------------------------
    oracle_cov = oracle_ceiling(instances)
    iid_tokens_at_B = _mean_tokens_at_B_from_runs(iid_runs, B_star)
    corr_one_chain = float(np.mean([
        corr_star_runs[i][j].detail["cum_tokens"][0]
        for i in range(len(corr_star_runs))
        for j in range(len(corr_star_runs[i]))
        if corr_star_runs[i][j].detail["cum_tokens"]
    ])) if corr_star_runs else float("nan")
    v1 = gate_v1(iid_cov_at_B)
    v2 = gate_v2(sample, instances[:4], seeds[:4], K=6, N=n_corr, tau=TAU)
    v3 = gate_v3(iid_tokens_at_B, corr_tokens_at_B[beta_star], iid_meanK_at_B,
                 corr_meanK_at_B[beta_star], B_star, corr_one_chain)
    v4 = gate_v4(oracle_cov, iid_cov_at_B)

    cell = {
        "p": float(p),
        "collapse": float(collapse),
        "B_star": float(B_star),
        "B_idx": int(B_idx),
        "iid_cov_at_B": float(iid_cov_at_B),
        "iid_cov_at_B_filtered": float(iid_cov_at_B_filt),
        "oracle_ceiling": float(oracle_cov),
        "n_instances": int(len(instances)),
        "n_instances_kept": n_kept,
        "instance_filter": {
            "keep_mask": [bool(m) for m in keep_mask],
            "iid_cov_per_instance": [float(x) for x in iid_pc],
            "corr_realizedK_per_instance": [float(x) for x in corr_pc],
            "min_realized_K": int(CORR_MIN_REALIZED_K),
            "band": list(V1_BAND),
        },
        "iid_curve": {
            "budgets": iid_curve["budgets"],
            "coverage": [float(c) for c in iid_curve["coverage"]],
            "mean_K": [float(k) for k in iid_curve["mean_K"]],
        },
        "corr_curves": {
            str(b): {
                "budgets": corr_curves[b]["budgets"],
                "coverage": [float(c) for c in corr_curves[b]["coverage"]],
                "mean_K": [float(k) for k in corr_curves[b]["mean_K"]],
                "cov_at_B": float(corr_cov_at_B[b]),
                "cov_at_B_filtered": float(corr_cov_filt[b]),
            }
            for b in betas_sorted
        },
        "cheap_curves": {
            str(b): {
                "budgets": cheap_curves[b]["budgets"],
                "coverage": [float(c) for c in cheap_curves[b]["coverage"]],
                "mean_K": [float(k) for k in cheap_curves[b]["mean_K"]],
                "cov_at_B": float(cheap_cov_at_B[b]),
                "cov_at_B_filtered": float(cheap_cov_filt[b]),
                "mean_prompt_overhead_tokens": float(cheap_overhead[b]),
            }
            for b in betas_sorted
        },
        "gates": {"V1": v1, "V2": v2, "V3": v3, "V4": v4},
        "H1": h1,
        "H1_cheap": h1_cheap,
        "H2": h2,
        "H3": h3,
        "H4": h4,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    if verbose:
        print(f"[p={p} c={collapse}] B*={B_star} iid@B={iid_cov_at_B:.3f} "
              f"kept={n_kept}/{len(instances)} "
              f"step:β*={beta_star} Δ={h1['delta']:+.3f}({'Y' if h1['pass'] else 'n'}) "
              f"cheap:β*={cheap_beta_star} Δ={h1_cheap['delta']:+.3f}"
              f"({'Y' if h1_cheap['pass'] else 'n'}) "
              f"H3={'U' if h3['pass'] else '-'} H4={'sep' if h4['pass'] else '-'} "
              f"[{cell['elapsed_sec']}s]")
    return cell


# ---------------------------------------------------------------------------
# H5: cross-p regression of the gain on the i.i.d. baseline coverage
# ---------------------------------------------------------------------------

def compute_h5(cells):
    """H5: regress gain = corr(β*)−iid coverage@B* on iid baseline coverage across p.

    PREREG §3 H5 / §10: the gain should DECREASE as the i.i.d. baseline coverage rises
    (head-cover-room shrinks toward saturation). A negative regression slope confirms
    the regime dependence (Tier-A p-axis is the competence knob, same axis as the
    verifier line). Returns slope/intercept/r2 and the per-p (baseline, gain) points.
    """
    xs = []  # iid baseline coverage@B* (on the filtered adjudication set)
    ys = []  # gain = corr_step(β*) − iid at B*
    pts = []
    for c in cells:
        baseline = c.get("iid_cov_at_B_filtered", c["iid_cov_at_B"])
        if not np.isfinite(baseline):
            baseline = c["iid_cov_at_B"]
        gain = c["H1"]["delta"]
        xs.append(baseline)
        ys.append(gain)
        pts.append({"p": c["p"], "collapse": c.get("collapse"),
                    "baseline": baseline, "gain": gain})
    if len(xs) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "points": pts, "pass": False, "n": len(xs)}
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "points": pts,
        "n": int(len(xs)),
        "pass": bool(slope < 0.0),  # gain shrinks as baseline saturates
    }


# ---------------------------------------------------------------------------
# Holm correction across the cross-p H1/H2/H4 family (PREREG §10)
# ---------------------------------------------------------------------------

def compute_holm(cells):
    """Holm-correct the pooled-direction p-values for H1/H2/H4 across the grid.

    We pool each hypothesis to ONE family member by taking the worst (largest) per-p
    bootstrap p-value for H1 and H4 (a conservative family summary), and a sign-test
    p for H2's slope-difference direction across p-cells. Returns the Holm-adjusted
    family decision (PREREG §10: "H1/H2/H4 间 Holm 校正").
    """
    # H1 (corr_step): largest per-cell two-sided bootstrap p (conservative worst case).
    h1_ps = [c["H1"]["p_two_sided"] for c in cells if np.isfinite(c["H1"]["p_two_sided"])]
    h1_p = max(h1_ps) if h1_ps else 1.0
    # H1_cheap (corr_cheap): the decisive cheap-arm family member (does the lift survive the
    # ~1× generation + charged prompt-overhead mechanism?).
    h1c_ps = [c["H1_cheap"]["p_two_sided"] for c in cells
              if "H1_cheap" in c and np.isfinite(c["H1_cheap"]["p_two_sided"])]
    h1c_p = max(h1c_ps) if h1c_ps else 1.0
    # H4: we WANT the temp-vs-iid gain to be null, so the family member is the FRACTION
    # of cells where corr beats temp jointly (a direction summary); convert to a sign-test
    # style p over the per-cell joint-distinctness sign.
    h4_signs = [1 if c["H4"]["corr_beats_temp_jointly"] else 0 for c in cells]
    h4_p = _sign_test_p(h4_signs)
    # H2: sign test that corr's slope is more negative than iid's across cells.
    h2_signs = [1 if (np.isfinite(c["H2"]["slope_diff"]) and c["H2"]["slope_diff"] < 0)
                else 0 for c in cells]
    h2_p = _sign_test_p(h2_signs)
    return ins.holm_correction(
        {"H1": h1_p, "H1_cheap": h1c_p, "H2": h2_p, "H4": h4_p}, alpha=0.05)


def _sign_test_p(signs):
    """Two-sided exact-binomial-ish sign-test p for ``#successes`` out of ``n`` at q=0.5.

    A simple, dependency-light sign test: under H0 each cell is a fair coin; the p-value
    is the two-sided tail of the binomial(n, 0.5) at the observed success count. Used as
    a conservative family summary for the cross-p direction of H2/H4.
    """
    n = len(signs)
    if n == 0:
        return 1.0
    k = sum(signs)
    # two-sided: 2 * min(P(X<=k), P(X>=k)) under Binom(n,0.5), clamped to 1.
    from math import comb
    def cdf_le(x):
        return sum(comb(n, j) for j in range(0, x + 1)) / (2 ** n)
    p_le = cdf_le(k)
    p_ge = 1.0 - cdf_le(k - 1) if k > 0 else 1.0
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


# ---------------------------------------------------------------------------
# Sweep driver (deterministic + resumable)
# ---------------------------------------------------------------------------

def _cell_key(p, collapse):
    """The composite, JSON-stable string key for a ``(p, collapse)`` cell."""
    return f"p{float(p):.6f}|c{float(collapse):.6f}"


def _config_dict(p_grid, collapse_grid, beta_grid, seeds, budgets, instances,
                 boot_n, n_corr):
    """The resume-key config. NOTE: ``p_grid`` and ``collapse_grid`` are the iteration axes
    (each ``(p, collapse)`` is a resumable CELL), recorded for provenance but EXCLUDED from
    the resume key: extending either grid must reuse already-computed cells, not invalidate
    them. The resume key is everything that determines a CELL's numbers (β/seeds/budgets/
    instances/knobs); changing any of those forces a recompute (correctly).
    """
    return {
        "_p_grid_provenance": list(p_grid),          # recorded only; NOT in the resume key
        "_collapse_grid_provenance": list(collapse_grid),  # recorded only; NOT in resume key
        "beta_grid": list(beta_grid),
        "seeds": list(seeds),
        "budget_grid": list(budgets),
        "depth_cap": DEPTH_CAP,
        "N_corr": int(n_corr),
        "N_iid": N_IID,
        "tau": TAU,
        "prompt_overhead_per_state": PROMPT_OVERHEAD_PER_STATE,
        "prompt_overhead_base": PROMPT_OVERHEAD_BASE,
        "delta_H1": DELTA_H1,
        "V1_band": list(V1_BAND),
        "corr_min_realized_K": CORR_MIN_REALIZED_K,
        "boot_n": int(boot_n),
        "K_max": K_MAX,
        "instances": [i["id"] + ":" + str(tuple(i["numbers"])) + "->" + str(i["target"])
                      for i in instances],
    }


def _resume_key(config):
    """The subset of ``config`` that determines a cell's numbers (excludes provenance)."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


def _load_resume(out_path, config):
    """Load any compatible partial results from ``out_path`` for resumption.

    Returns ``{cell_key: cell}`` for cells whose config matches; an incompatible or
    missing file yields ``{}`` (a fresh run). Resumption is keyed on the frozen config so
    a config change forces a recompute.
    """
    if not Path(out_path).is_file():
        return {}
    try:
        with open(out_path) as f:
            prev = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if _resume_key(prev.get("config", {})) != _resume_key(config):
        return {}
    return {_cell_key(c["p"], c.get("collapse", 0.5)): c for c in prev.get("cells", [])}


def run_sweep(p_grid=P_GRID, collapse_grid=COLLAPSE_GRID, beta_grid=BETA_GRID,
              seeds=SEEDS, budgets=BUDGET_GRID, out_path=DEFAULT_OUT, resume=True,
              verbose=True, instances=None, boot_n=BOOT_N, n_corr=N_CORR):
    """Run the full (p × collapse) × β × seed Tier-A sweep and write ``outputs/tierA.json``.

    RETUNE: a CELL is now ``(p, collapse)`` — the collapse axis {0.5,0.8,0.95} is swept so
    the broad/redundant good region (collapse≈0.8) where repulsion helps is sampled, not
    just the narrow collapse=0.5 funnel. Deterministic + resumable (matching cells reloaded,
    only missing ``(p, collapse)`` cells recomputed, file rewritten after each cell).
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if instances is None:
        instances = make_instances()
    config = _config_dict(p_grid, collapse_grid, beta_grid, seeds, budgets, instances,
                          boot_n, n_corr)
    done = _load_resume(out_path, config) if resume else {}
    cells_by_key = dict(done)

    cell_specs = [(p, c) for c in collapse_grid for p in p_grid]
    if verbose:
        print(f"Tier-A sweep: (p×collapse)×β×seed = "
              f"({len(p_grid)}×{len(collapse_grid)})×{len(beta_grid)}×{len(seeds)} "
              f"on {len(instances)} instances, budgets={budgets}, N_corr={n_corr}")
        if done:
            print(f"  resuming: {len(done)} cell(s) already done")

    t_start = time.time()
    for (p, collapse) in cell_specs:
        key = _cell_key(p, collapse)
        if key in cells_by_key:
            if verbose:
                print(f"[p={p} c={collapse}] cached")
            continue
        cell = run_p_cell(p, beta_grid, seeds, instances, budgets, verbose=verbose,
                          boot_n=boot_n, n_corr=n_corr, collapse=collapse)
        cells_by_key[key] = cell
        _write_results(out_path, config, cells_by_key, instances, partial=True)

    cells = [cells_by_key[_cell_key(p, c)] for (p, c) in cell_specs
             if _cell_key(p, c) in cells_by_key]
    _write_results(out_path, config, cells_by_key, instances, partial=False,
                   elapsed=round(time.time() - t_start, 2))
    if verbose:
        summary = _summarize(cells)
        print(f"Tier-A done in {round(time.time() - t_start, 1)}s on {len(cells)} cells. "
              f"H1(step) {summary['H1_pass']}, H1(cheap) {summary['H1_cheap_pass']}, "
              f"H3 {summary['H3_pass']}, H4 {summary['H4_pass']}; "
              f"H5 slope={summary['H5_slope']}; gates "
              f"V1 {summary['V1_pass']} V2 {summary['V2_pass']} "
              f"V3 {summary['V3_pass']} V4 {summary['V4_pass']}")
    return json.loads(Path(out_path).read_text())


def _summarize(cells):
    h5 = compute_h5(cells)
    return {
        "n_cells": len(cells),
        "H1_pass": sum(1 for c in cells if c["H1"]["pass"]),
        "H1_cheap_pass": sum(1 for c in cells if c.get("H1_cheap", {}).get("pass")),
        "H3_pass": sum(1 for c in cells if c["H3"]["pass"]),
        "H4_pass": sum(1 for c in cells if c["H4"]["pass"]),
        "V1_pass": sum(1 for c in cells if c["gates"]["V1"]["pass"]),
        "V2_pass": sum(1 for c in cells if c["gates"]["V2"]["pass"]),
        "V3_pass": sum(1 for c in cells if c["gates"]["V3"]["pass"]),
        "V4_pass": sum(1 for c in cells if c["gates"]["V4"]["pass"]),
        "H5_slope": round(h5["slope"], 4) if np.isfinite(h5["slope"]) else None,
    }


def _write_results(out_path, config, cells_by_key, instances, partial, elapsed=None):
    """Write the results JSON (atomic). Assembles cross-cell instruments (H5, Holm)."""
    cells = [cells_by_key[k] for k in sorted(cells_by_key.keys())]
    doc = {
        "schema": "correlated-coverage/tierA/v2",
        "tier": "A-mock",
        "frozen_decode_core": True,
        "partial": bool(partial),
        "generated_unix": time.time(),
        "config": config,
        "cells": cells,
    }
    if not partial:
        doc["H5"] = compute_h5(cells)
        doc["holm"] = compute_holm(cells)
        doc["summary"] = _summarize(cells)
        if elapsed is not None:
            doc["elapsed_sec"] = elapsed
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True))
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# TINY smoke: one cell — fast CI sanity
# ---------------------------------------------------------------------------

def run_smoke(out_path=SMOKE_OUT, verbose=True):
    """One-cell smoke: a single ``(p, collapse)`` over a reduced β/seed/budget grid.

    Exercises the WHOLE pipeline — the iid baseline, BOTH repulsion arms (``corr_step`` and
    ``corr_cheap``), the iso-token curve, the instance filter, gates V1–V4, and H1/H1_cheap/
    H2/H3/H4 (with the K*>=4 τ-match fix) — on ONE cell with a small instance/seed/budget
    grid, a reduced ``n_corr`` (smaller step tax so corr affords >=4 chains at the smoke
    budgets), and a small ``boot_n``, finishing in a few seconds. NOT prereg-grade numbers;
    proves the pipeline + schema. Used by ``tests/test_run_corr.py``.
    """
    instances = make_instances()
    seeds = [1, 2, 3]
    betas = [0.0, 0.25, 1.0, 4.0]
    # p=0.5, collapse=0.8: iid stays in the V1 band out to B≈192–384 while corr (N=3) affords
    # K>=4 there, so the corr-aware B* lands a non-empty filtered set (the regime the RETUNE
    # targets). Larger budgets than the old smoke so corr is not starved to K<4.
    budgets = [96, 192, 384]
    return run_sweep(
        p_grid=[0.5], collapse_grid=[0.8], beta_grid=betas, seeds=seeds, budgets=budgets,
        out_path=out_path, resume=False, verbose=verbose,
        instances=instances, boot_n=1000, n_corr=3,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier-A correlated-coverage sweep harness")
    ap.add_argument("--smoke", action="store_true",
                    help="run the tiny one-cell smoke (fast) instead of the full sweep")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing partial results and recompute all cells")
    ap.add_argument("--quiet", action="store_true", help="suppress progress prints")
    args = ap.parse_args(argv)

    verbose = not args.quiet
    if args.smoke:
        out = args.out or SMOKE_OUT
        run_smoke(out_path=out, verbose=verbose)
    else:
        out = args.out or DEFAULT_OUT
        run_sweep(out_path=out, resume=not args.no_resume, verbose=verbose)


if __name__ == "__main__":
    main()
