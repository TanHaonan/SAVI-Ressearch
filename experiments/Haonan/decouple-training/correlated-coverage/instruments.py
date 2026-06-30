"""Instruments for the correlated-coverage experiment (PREREG §8, MECHANISM §3/§5.3).

This module is the measurement layer over the two headline arms in ``corr_sampler``
(``corr_sample`` = the constructive S', ``iid_bok`` = the i.i.d. best-of-K baseline /
β=0 reference). It is PURE measurement: nothing here touches the FROZEN ``decode_core``,
and every number is read off the arms' ``Result`` objects (coverage = ``Result.ok``,
token ledger = ``Result.budget.tokens``, realized chains = ``Result.detail["chains"]``).

Five deliverables (PREREG §8, the task brief):

(a) ``coverage_vs_tokens`` — the HEADLINE curve. For each token budget ``B`` on a grid,
    grow each arm to the largest K whose accumulated ``Budget.tokens ≤ B`` and report the
    paired per-instance coverage. This is the iso-token comparison (MECHANISM §3): both
    arms are compared at EQUAL B, and the step-mode N× tax is inside corr's ledger by
    construction, so a smaller K_corr at a given B is the honest cost of the mechanism.

(b) ``miss_decay_slope`` — the "base数" metric (PREREG §7, H2). Fit the slope of
    ``log(1 − coverage)`` against the token budget. A more negative slope = miss
    probability decaying faster = the Erdős→sphere-graph "底数 +ε" direction. Returns the
    slope, intercept, and R².

(c) ``joint_diversity`` — ``distinct_canon@K`` (MECHANISM §5.3). The ENSEMBLE statistic
    that distinguishes joint anti-correlation from marginal spread: distinct canonical
    states across the K chains, per-depth and at the leaf. Marginal entropy cannot control
    this; coverage depends on it directly.

(d) ``tau_match`` — the H4 negative-control calibrator. Given a corr arm's realized
    MARGINAL per-step entropy, find the i.i.d. temperature whose realized marginal entropy
    equals it (so the ``temp_matched_iid`` arm spreads MARGINALLY like corr but, the claim
    goes, stays jointly redundant). Uses a monotone bisection over τ on the empirically
    measured marginal step-entropy of the i.i.d. arm.

(e) ``paired_bootstrap`` — n=10000 paired bootstrap for coverage deltas. Resamples
    INSTANCES with replacement (the item-level convention of ``core.metrics.boot_ci``) so
    the CI reflects instance-to-instance variation in the paired (corr − iid) coverage.

Determinism: all bootstraps take an explicit ``seed`` (numpy ``default_rng``); ``tau_match``
is a deterministic bisection. No global RNG state is touched.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _coverage_of(chains_reached):
    """coverage = ANY chain reached the goal, from a list of per-chain reached flags."""
    return 1 if any(chains_reached) else 0


def _entropy(probs):
    """Shannon entropy (nats) of a probability list; 0·log0 := 0. Empty -> 0.0."""
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log(p)
    return h


# ---------------------------------------------------------------------------
# (a) coverage-vs-tokens curve  (the headline, iso-token; MECHANISM §3)
# ---------------------------------------------------------------------------

def coverage_at_budget(run_arm, inst, seed, budget_B, K_max=128):
    """Coverage of ``run_arm`` on ``inst`` at token budget ``budget_B`` (a single 0/1).

    ``run_arm(inst, K, seed) -> Result`` is a closure that runs the arm at chain count
    ``K`` (the caller binds β/N/τ/domain/sample). We grow ``K`` from 1 upward and stop at
    the LARGEST K whose accumulated ``Result.budget.tokens ≤ budget_B``; coverage is that
    arm's ``Result.ok`` at that K. If even K=1 already exceeds ``budget_B`` we return that
    K=1 run's coverage anyway (an arm cannot draw fewer than one chain; the budget is then
    simply too small for this arm — recorded honestly, not silently zeroed). ``K_max``
    caps the search.

    Because both arms record EVERY ``sample`` call through the frozen ``Budget``, this is
    an exact iso-token read: corr's step-mode N× tax means it reaches budget_B at a smaller
    K than iid, which is the whole point of the hard test (MECHANISM §3.2 MAIN).
    """
    last_ok = None
    last_tokens = 0
    for K in range(1, K_max + 1):
        res = run_arm(inst, K, seed)
        tok = res.budget.tokens
        if K == 1:
            # Always keep K=1 as the floor (an arm must draw >=1 chain).
            last_ok = int(res.ok)
            last_tokens = tok
            if tok > budget_B:
                # Even one chain overruns B; honestly report this floor coverage.
                return last_ok, K, tok
            continue
        if tok > budget_B:
            return last_ok, K - 1, last_tokens
        last_ok = int(res.ok)
        last_tokens = tok
    return last_ok, K_max, last_tokens


def coverage_vs_tokens(run_arm, instances, seeds, budgets, K_max=128):
    """Coverage@B over a grid of token budgets ``B`` for ONE arm (PREREG §8.1).

    For every budget ``B`` in ``budgets``, average ``coverage_at_budget`` over the
    ``instances × seeds`` grid. Returns a dict::

        {
          "budgets": [B0, B1, ...],
          "coverage": [cov@B0, cov@B1, ...],            # mean over instance×seed
          "per_cell": {B: [0/1 per (inst,seed)]},        # the raw paired outcomes
          "mean_K":   [meanK@B0, ...],                    # mean realized K at each B
        }

    ``run_arm(inst, K, seed) -> Result`` is the same closure as ``coverage_at_budget``.
    The ``per_cell`` outcomes are the input to ``paired_bootstrap`` (one arm's leg).
    """
    budgets = list(budgets)
    coverage = []
    mean_K = []
    per_cell = {}
    for B in budgets:
        cell = []
        ks = []
        for inst in instances:
            for seed in seeds:
                ok, K, _tok = coverage_at_budget(run_arm, inst, seed, B, K_max=K_max)
                cell.append(int(ok))
                ks.append(K)
        per_cell[B] = cell
        coverage.append(float(np.mean(cell)) if cell else float("nan"))
        mean_K.append(float(np.mean(ks)) if ks else float("nan"))
    return {
        "budgets": budgets,
        "coverage": coverage,
        "per_cell": per_cell,
        "mean_K": mean_K,
    }


# ---------------------------------------------------------------------------
# (b) miss-decay slope  (the "base数" metric; PREREG §7, H2)
# ---------------------------------------------------------------------------

def miss_decay_slope(budgets, coverage, eps=1e-9):
    """Fit slope of ``log(1 − coverage)`` vs token budget (the H2 "底数" metric).

    ``budgets`` and ``coverage`` are paired lists (e.g. from ``coverage_vs_tokens``).
    We regress ``y = log(1 − coverage)`` on ``x = budget`` by ordinary least squares.
    A more NEGATIVE slope means the joint-miss probability ``P(all K miss)`` decays
    faster with tokens — the constructive "顶破 i.i.d. 天花板 / base数 +ε" direction
    (PREREG §1, §7). Points with ``coverage ≥ 1 − eps`` (miss underflow → log(0)=−∞)
    are dropped from the fit (they carry no slope information). Coverage is clamped into
    ``[eps, 1−eps]`` first so ``log(1−c)`` is always finite for the kept points.

    Returns ``{"slope", "intercept", "r2", "n_points", "x", "y"}``. With fewer than two
    usable points the slope/intercept/r2 are ``nan`` (an honest "cannot fit").
    """
    x_all = np.asarray(budgets, float)
    c_all = np.clip(np.asarray(coverage, float), eps, 1.0 - eps)
    miss = 1.0 - c_all
    y_all = np.log(miss)
    # Drop saturated points (coverage essentially 1 ⇒ miss ~ eps ⇒ y is a clamp artifact).
    keep = (np.asarray(coverage, float) < 1.0 - eps) & np.isfinite(y_all)
    x = x_all[keep]
    y = y_all[keep]
    if len(x) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "n_points": int(len(x)),
            "x": x.tolist(),
            "y": y.tolist(),
        }
    # OLS via numpy.polyfit (degree 1).
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n_points": int(len(x)),
        "x": x.tolist(),
        "y": y.tolist(),
    }


# ---------------------------------------------------------------------------
# (c) joint diversity  (distinct_canon@K; MECHANISM §5.3)
# ---------------------------------------------------------------------------

def distinct_canon_at_K(chains):
    """``distinct_canon@K`` over a list of per-chain ``path_keys`` (MECHANISM §5.3).

    Returns ``{"per_depth": {t: #distinct}, "leaf": #distinct leaf canon, "union":
    #distinct over all depths}``. A JOINT (ensemble-level) statistic: it counts the
    non-redundancy of the K committed samples TOGETHER. (Re-exported here so instruments
    is self-contained; identical to ``corr_sampler.distinct_canon_at_K``.)
    """
    per_depth = {}
    for path in chains:
        for t, key in enumerate(path, start=1):
            per_depth.setdefault(t, set()).add(key)
    leaf = {path[-1] for path in chains if path}
    union = set()
    for s in per_depth.values():
        union |= s
    return {
        "per_depth": {t: len(s) for t, s in sorted(per_depth.items())},
        "leaf": len(leaf),
        "union": len(union),
    }


def joint_diversity(run_arm, instances, seeds, K):
    """Mean ``distinct_canon@K`` for ONE arm over the instance×seed grid (PREREG §8.3).

    ``run_arm(inst, K, seed) -> Result`` is the arm closure; we read its realized
    ``Result.detail["chains"]`` (the per-chain canonical path lists) and aggregate
    ``distinct_canon_at_K``. Returns::

        {
          "K": K,
          "leaf_mean": mean distinct leaf canon per ensemble,
          "union_mean": mean distinct canon over all depths per ensemble,
          "per_depth_mean": {t: mean distinct canon at depth t},
          "leaf_per_cell": [distinct leaf per (inst,seed)],   # for bootstrap / KS
        }

    This is the operational separator for H4: ``corr(β*)`` should beat both ``iid_bok``
    and the marginally-matched ``temp_matched_iid`` on ``leaf_mean`` / ``union_mean``,
    even when the temperature arm matches marginal step-entropy (MECHANISM §5.3 use 2).
    """
    leaf_vals = []
    union_vals = []
    depth_acc = {}  # depth -> list of distinct counts
    for inst in instances:
        for seed in seeds:
            res = run_arm(inst, K, seed)
            d = distinct_canon_at_K(res.detail["chains"])
            leaf_vals.append(d["leaf"])
            union_vals.append(d["union"])
            for t, v in d["per_depth"].items():
                depth_acc.setdefault(t, []).append(v)
    per_depth_mean = {t: float(np.mean(v)) for t, v in sorted(depth_acc.items())}
    return {
        "K": int(K),
        "leaf_mean": float(np.mean(leaf_vals)) if leaf_vals else float("nan"),
        "union_mean": float(np.mean(union_vals)) if union_vals else float("nan"),
        "per_depth_mean": per_depth_mean,
        "leaf_per_cell": [int(v) for v in leaf_vals],
    }


# ---------------------------------------------------------------------------
# (d) tau-match  (H4 calibration by MARGINAL per-step entropy)
# ---------------------------------------------------------------------------

def marginal_step_entropy(chains):
    """Mean per-step MARGINAL entropy (nats) of the canonical-successor distribution.

    Given the realized per-chain ``path_keys``, the MARGINAL step distribution at depth t
    is the empirical distribution over canonical successors actually realized across the K
    chains at that depth. We compute the Shannon entropy at each depth and average over
    depths (equal weight per depth). This is the ensemble's REALIZED marginal step-spread
    — what temperature controls — as opposed to the joint distinctness ``distinct_canon@K``
    (MECHANISM §5.3): two ensembles can match here yet differ jointly.

    Empty (no realized steps) -> 0.0.
    """
    per_depth = {}
    for path in chains:
        for t, key in enumerate(path, start=1):
            per_depth.setdefault(t, {})
            per_depth[t][key] = per_depth[t].get(key, 0) + 1
    if not per_depth:
        return 0.0
    hs = []
    for t, counts in per_depth.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        probs = [c / total for c in counts.values()]
        hs.append(_entropy(probs))
    return float(np.mean(hs)) if hs else 0.0


def measure_marginal_entropy(run_arm, instances, seeds, K):
    """Mean realized ``marginal_step_entropy`` of ONE arm over the instance×seed grid.

    ``run_arm(inst, K, seed) -> Result``. Averages ``marginal_step_entropy`` of each
    ensemble's realized chains. This is the scalar ``tau_match`` drives the i.i.d. arm to.
    """
    vals = []
    for inst in instances:
        for seed in seeds:
            res = run_arm(inst, K, seed)
            vals.append(marginal_step_entropy(res.detail["chains"]))
    return float(np.mean(vals)) if vals else 0.0


def tau_match(make_iid_run_arm, target_entropy, instances, seeds, K,
              tau_lo=0.0, tau_hi=8.0, tol=1e-3, max_iter=40):
    """Find the i.i.d. temperature whose MARGINAL step-entropy equals ``target_entropy``.

    ``make_iid_run_arm(tau) -> run_arm`` is a factory: given a temperature it returns an
    arm closure ``run_arm(inst, K, seed) -> Result`` for the i.i.d. baseline at that τ
    (the caller binds domain/sample/N). ``target_entropy`` is the corr arm's realized
    marginal step-entropy (from ``measure_marginal_entropy`` on ``corr(β*)``).

    The i.i.d. arm's realized marginal step-entropy is APPROXIMATELY MONOTONE
    NON-DECREASING in τ (higher τ flattens ``P_emission`` ⇒ broader marginal spread;
    MECHANISM mock ``_tau_collapse``). It is, however, a NOISY DISCRETE STEP FUNCTION of τ
    — it is reconstructed from finitely many realized chains, so (i) it can have flat
    plateaus and small non-monotone wobbles, and (ii) whole entropy bands are UNATTAINABLE
    (e.g. the mock jumps from H=0 at τ=0/greedy straight to the flattened-emission band).
    So this is an APPROXIMATE matcher: we bisect on ``H(τ) − target`` to drive τ toward the
    target band and RETURN THE CLOSEST τ ever evaluated (the one minimizing ``|H(τ) −
    target|``), not blindly the last midpoint. ``reached`` is True iff that closest gap is
    within ``tol`` (so an unattainable target honestly reports ``reached=False`` with the
    nearest τ, rather than a fake exact hit). Tracking the running best makes the result
    robust to the curve's plateaus/wobbles.

    Out-of-bracket: if the target is ≤ ``H(tau_lo)`` we return ``tau_lo``; if ≥
    ``H(tau_hi)`` we return ``tau_hi`` — the nearest reachable endpoint, ``reached`` set by
    the same ``tol`` test.

    Returns ``{"tau": τ*, "entropy": H(τ*), "target", "reached", "iters", "gap":
    |H(τ*)−target|, "bracket": (H(tau_lo), H(tau_hi))}``.
    """
    def H(tau):
        return measure_marginal_entropy(make_iid_run_arm(tau), instances, seeds, K)

    lo, hi = float(tau_lo), float(tau_hi)
    H_lo = H(lo)
    H_hi = H(hi)

    # Track the closest (tau, entropy) ever seen so a noisy/step curve still yields the
    # best attainable match (not just the final bisection midpoint).
    best_tau, best_H = lo, H_lo
    best_gap = abs(H_lo - target_entropy)

    def _consider(tau, h):
        nonlocal best_tau, best_H, best_gap
        g = abs(h - target_entropy)
        if g < best_gap:
            best_tau, best_H, best_gap = tau, h, g

    _consider(hi, H_hi)

    # Out-of-bracket: nearest endpoint (still passed through _consider above).
    if target_entropy <= H_lo or target_entropy >= H_hi:
        endpoint = lo if target_entropy <= H_lo else hi
        endpoint_H = H_lo if target_entropy <= H_lo else H_hi
        return {
            "tau": float(endpoint), "entropy": float(endpoint_H),
            "target": float(target_entropy),
            "reached": abs(endpoint_H - target_entropy) <= tol, "iters": 0,
            "gap": abs(endpoint_H - target_entropy), "bracket": (H_lo, H_hi),
        }

    # Bisection on the (approximately) increasing H; keep the running-closest τ.
    iters = 0
    for _ in range(max_iter):
        iters += 1
        mid = 0.5 * (lo + hi)
        H_mid = H(mid)
        _consider(mid, H_mid)
        if best_gap <= tol:
            break
        if H_mid < target_entropy:
            lo = mid
        else:
            hi = mid
    return {
        "tau": float(best_tau),
        "entropy": float(best_H),
        "target": float(target_entropy),
        "reached": best_gap <= tol,
        "iters": iters,
        "gap": float(best_gap),
        "bracket": (H_lo, H_hi),
    }


# ---------------------------------------------------------------------------
# (e) paired bootstrap  (n=10000; coverage deltas; PREREG §8, §10 Holm)
# ---------------------------------------------------------------------------

def paired_bootstrap(a_vals, b_vals, n_boot=10000, seed=0, ci=(2.5, 50, 97.5)):
    """Paired bootstrap of the mean difference ``mean(a − b)`` (n=10000 default).

    ``a_vals`` and ``b_vals`` are PAIRED per-instance(×seed) outcome vectors (e.g. the
    coverage 0/1 cells of corr and iid at one budget B). We resample the PAIRED INDICES
    with replacement (the item-level convention of ``core.metrics.boot_ci``), so each
    resample keeps the corr/iid pairing intact and the CI reflects instance-to-instance
    variation in the paired delta. Returns::

        {
          "delta": mean(a) − mean(b),          # the point estimate
          "ci_low", "ci_med", "ci_high",       # bootstrap percentiles of the delta
          "p_two_sided": fraction of resamples on the opposite side of 0 (×2, clamped),
          "excludes_zero": bool,               # 95% CI excludes 0 (H1/H2 gate)
          "n": number of paired items,
        }

    ``excludes_zero`` is the PREREG H1 gate (CI排除 0). ``p_two_sided`` is a bootstrap
    two-sided p-value for delta==0 (the proportion of resamples whose delta has the sign
    opposite the point estimate, doubled and clamped to [0,1]).
    """
    a = np.asarray(a_vals, float)
    b = np.asarray(b_vals, float)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors must match: {a.shape} vs {b.shape}")
    n = len(a)
    if n == 0:
        return {
            "delta": float("nan"), "ci_low": float("nan"), "ci_med": float("nan"),
            "ci_high": float("nan"), "p_two_sided": float("nan"),
            "excludes_zero": False, "n": 0,
        }
    diff = a - b
    point = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_deltas = diff[idx].mean(axis=1)
    lo, med, hi = (float(np.percentile(boot_deltas, q)) for q in ci)
    # Two-sided bootstrap p: proportion on the far side of 0 from the point estimate.
    if point >= 0:
        tail = float(np.mean(boot_deltas <= 0.0))
    else:
        tail = float(np.mean(boot_deltas >= 0.0))
    p_two = min(1.0, 2.0 * tail)
    return {
        "delta": point,
        "ci_low": lo,
        "ci_med": med,
        "ci_high": hi,
        "p_two_sided": p_two,
        "excludes_zero": (lo > 0.0) or (hi < 0.0),
        "n": int(n),
    }


def holm_correction(pvalues, alpha=0.05):
    """Holm step-down correction over a dict/list of p-values (PREREG §10).

    ``pvalues`` is a dict ``{name: p}`` (or list of ``(name, p)``). Returns a dict
    ``{name: {"p", "p_adj", "reject"}}`` where ``p_adj`` is the Holm-adjusted p (the
    running max of ``(m − rank) ⋅ p`` so adjusted p is monotone in rank) and ``reject``
    is ``p_adj ≤ alpha``. Used to control the family-wise error across H1/H2/H4
    (PREREG §10: "H1/H2/H4 间 Holm 校正").
    """
    if isinstance(pvalues, dict):
        items = list(pvalues.items())
    else:
        items = list(pvalues)
    m = len(items)
    if m == 0:
        return {}
    order = sorted(range(m), key=lambda i: items[i][1])
    out = {}
    running = 0.0
    for rank, i in enumerate(order):
        name, p = items[i]
        adj = (m - rank) * p
        running = max(running, adj)  # enforce monotone non-decreasing adjusted p
        p_adj = min(1.0, running)
        out[name] = {"p": float(p), "p_adj": float(p_adj), "reject": p_adj <= alpha}
    return out
