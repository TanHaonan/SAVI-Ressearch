"""PLAN2 depth x K x p sweep — the CI-grade, exec-parity, deep-substrate mechanism run.

Corrects PLAN1's three artifacts (see ``results.json -> axis_correction_addendum``):
1. compute axis = moves/exec, never candidate count (the iso baseline is
   ``best_of_k_isobudget(axis="exec")`` matched to the trellis's per-instance exec);
2. headline arms = ``savi_freq`` / ``savi_verifier_on``, not the support-only floor;
3. deep substrate = the integer-sum lattice (``domain_lattice``) with an O(1) oracle, so
   depth is unbounded-cheap.

Arms
----
    greedy                 decode_core.greedy (one tau=0 chain)
    iso_exec               best_of_k_isobudget(axis="exec")   [HEADLINE BASELINE]
    savi_support           savi(..., "support", verifier=False)   (floor)
    savi_freq              savi(..., "freq",    verifier=False)   [HEADLINE]
    savi_verifier_on       savi(..., "freq",    verifier=True)    [HEADLINE, oracle mask]
    beam_no_merge          beam_no_merge(edge="freq")             (Phi ablation)
    ceiling                oracle (== 1 by construction; sanity)

Headline metrics per (depth, K, p) — paired bootstrap (n=10000) on per-instance flags:
    D1_exec = savi_freq        - iso_exec
    D_verif = savi_verifier_on - iso_exec
    D_merge = savi_freq        - beam_no_merge

At ONE (depth, K) cell also compute ``best_of_k_isobudget(axis="candidates")`` to exhibit
the axis artifact directly (candidate parity hands selection ~depth x more attempts).

Exec parity: ``iso_exec``'s target is the SAME instance's ``savi_freq`` ``budget.exec``;
``max_depth`` = the instance depth. We assert ``iso exec >= savi exec`` per instance.

Deterministic + RESUMABLE: every (arm, instance) result is cached by a cell key in
``<outdir>/cache.json``, so ``--depth D`` can be run separately into a shared ``--outdir``
and resumed; ``results2.json`` is rebuilt from the union of cached cells each run.

CLI
---
    python -m run_mechanism2 --domain lattice --depths 4,8,16,24 \
        --K-grid 8,16,32 --p-grid 0.7 --N 16 --seeds 1,2,3,4,5,6,7,8 \
        --inst-per-depth 24 --outdir <dir> [--depth 8]
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.dirname(_HERE)  # verifier-decode/ (holds decode_core/)
for _p in (_HERE, _PACKAGE_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import domain_lattice as _lattice  # noqa: E402
import gen_fair as _gen_fair  # noqa: E402
import arms_ext as _arms_ext  # noqa: E402

from decode_core import decode as _dc  # noqa: E402
from decode_core._deps.metrics.passk import pass_at_1, paired_bootstrap  # noqa: E402


ARMS = (
    "greedy",
    "iso_exec",
    "savi_support",
    "savi_freq",
    "savi_verifier_on",
    "beam_no_merge",
    "ceiling",
)

DEFAULT_TAU = 0.7
BOOTSTRAP_N = 10000


# ---------------------------------------------------------------------------
# Domain registry (lattice for now; extensible)
# ---------------------------------------------------------------------------

def _make_lattice_bundle(inst_per_depth):
    """Bundle for the integer-sum lattice substrate."""
    domain = _lattice.LatticeDomain()

    def make_instances(depth, seed):
        return _lattice.make_lattice_instances(depth, inst_per_depth, seed)

    return {
        "name": "lattice",
        "domain": domain,
        "enumerate": _lattice.lattice_enumerate,
        "make_instances": make_instances,
        "inst_dict": _lattice.lattice_inst_dict,
    }


def _make_bundle(domain_name, inst_per_depth):
    if domain_name == "lattice":
        return _make_lattice_bundle(inst_per_depth)
    raise ValueError(f"unknown domain: {domain_name!r} (valid: lattice)")


# ---------------------------------------------------------------------------
# Cache (resumable; keyed by the full cell + arm + instance content)
# ---------------------------------------------------------------------------

def _cache_key(domain_name, depth, K, p, seed, arm, inst_id, N, tau, extra=""):
    return "|".join([
        str(domain_name), str(int(depth)), str(int(K)), f"{float(p):.6f}",
        str(int(seed)), str(arm), str(inst_id), str(int(N)), f"{float(tau):.6f}",
        str(extra),
    ])


def _load_cache(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"WARN: cache {path} unreadable ({exc!r}); rebuilding", file=sys.stderr)
        return {}


def _save_cache(path, cache):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cache, fh)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------

def _budget_dict(b):
    return {"sample_calls": int(b.sample_calls), "candidates": int(b.candidates),
            "tokens": int(b.tokens), "exec": int(b.exec)}


def _zero_budget():
    return {"sample_calls": 0, "candidates": 0, "tokens": 0, "exec": 0}


# ---------------------------------------------------------------------------
# One cell: build the fair generator, run all arms over all instances
# ---------------------------------------------------------------------------

def _run_cell(bundle, depth, K, p, seed, N, tau, cache, cache_path,
              axis_artifact=False):
    """Run all arms over every instance for ONE (depth, K, p, seed) cell.

    Returns a dict with per-instance ok flags + budgets per arm. A single fair
    generator P_p is built here and shared by every arm (identical emission).
    Exec parity: ``iso_exec``'s per-instance target is that instance's ``savi_freq``
    ``budget.exec``; ``max_depth`` = ``depth``. Resumable: per (arm, instance) cached.

    When ``axis_artifact`` is True, ALSO runs ``iso_candidates`` (the candidate-parity
    baseline) matched on the trellis's CANDIDATE budget, to exhibit the artifact at this
    one cell (candidate parity hands selection ~depth x more complete attempts).
    """
    domain = bundle["domain"]
    name = bundle["name"]
    inst_dict = bundle["inst_dict"]
    insts = bundle["make_instances"](depth, seed)
    max_depth = int(depth)

    gen = _gen_fair.make_fair_generator(domain, bundle["enumerate"], p, max_depth)

    ok = {arm: {} for arm in ARMS}
    budgets = {arm: {} for arm in ARMS}
    cache_dirty = False

    # savi_freq budget cached per instance (the exec/candidate iso targets).
    freq_exec_target = {}
    freq_cand_target = {}

    def _replay_ok(res, idict):
        """Leaf-success: reached goal (lattice: is_goal IS the exact criterion)."""
        if not res.ok:
            return False
        if res.path is None:
            return bool(res.ok)  # ceiling: ok is the criterion
        s = domain.initial_state(idict)
        for move in res.path:
            s = domain.apply(s, move)
        return bool(domain.is_goal(s))

    def _cached(arm, inst, compute, extra=""):
        nonlocal cache_dirty
        key = _cache_key(name, depth, K, p, seed, arm, inst.id, N, tau, extra)
        entry = cache.get(key)
        if entry is None:
            res = compute()
            entry = {"ok": bool(_replay_ok(res, inst_dict(inst))),
                     "budget": _budget_dict(res.budget),
                     "exec": int(res.budget.exec),
                     "candidates": int(res.budget.candidates)}
            cache[key] = entry
            cache_dirty = True
        return entry

    # ----- 1) savi_freq FIRST: it defines the exec/candidate iso targets. -----
    for inst in insts:
        idict = inst_dict(inst)
        entry = _cached("savi_freq", inst,
                        lambda i=idict: _dc.savi(domain, gen, i, K, N, "freq",
                                                 tau, seed, verifier=False,
                                                 max_depth=max_depth))
        ok["savi_freq"][inst.id] = bool(entry["ok"])
        budgets["savi_freq"][inst.id] = entry["budget"]
        freq_exec_target[inst.id] = max(1, int(entry["exec"]))
        freq_cand_target[inst.id] = max(1, int(entry["candidates"]))

    # ----- 2) the remaining savi-family + ablation + ceiling + greedy. -----
    for inst in insts:
        idict = inst_dict(inst)

        entry = _cached("greedy", inst,
                        lambda i=idict: _dc.greedy(domain, gen, i, seed))
        ok["greedy"][inst.id] = bool(entry["ok"])
        budgets["greedy"][inst.id] = entry["budget"]

        entry = _cached("savi_support", inst,
                        lambda i=idict: _dc.savi(domain, gen, i, K, N, "support",
                                                 tau, seed, verifier=False,
                                                 max_depth=max_depth))
        ok["savi_support"][inst.id] = bool(entry["ok"])
        budgets["savi_support"][inst.id] = entry["budget"]

        entry = _cached("savi_verifier_on", inst,
                        lambda i=idict: _dc.savi(domain, gen, i, K, N, "freq",
                                                 tau, seed, verifier=True,
                                                 max_depth=max_depth))
        ok["savi_verifier_on"][inst.id] = bool(entry["ok"])
        budgets["savi_verifier_on"][inst.id] = entry["budget"]

        entry = _cached("beam_no_merge", inst,
                        lambda i=idict: _arms_ext.beam_no_merge(
                            domain, gen, i, K, N, tau, seed, verifier=False,
                            max_depth=max_depth, edge_mode="freq"))
        ok["beam_no_merge"][inst.id] = bool(entry["ok"])
        budgets["beam_no_merge"][inst.id] = entry["budget"]

        def _ceil(i=idict):
            okc = _dc.oracle(domain, i)
            return _dc.Result(ok=bool(okc), path=None, budget=_dc.Budget())
        entry = _cached("ceiling", inst, _ceil)
        ok["ceiling"][inst.id] = bool(entry["ok"])
        budgets["ceiling"][inst.id] = entry["budget"]

    # ----- 3) iso_exec: matched to THIS instance's savi_freq exec. -----
    for inst in insts:
        idict = inst_dict(inst)
        t = freq_exec_target[inst.id]
        entry = _cached(
            "iso_exec", inst,
            lambda i=idict, tt=t: _arms_ext.best_of_k_isobudget(
                domain, gen, i, tt, tau, seed, axis="exec"),
            extra=f"exec={t}")
        ok["iso_exec"][inst.id] = bool(entry["ok"])
        budgets["iso_exec"][inst.id] = entry["budget"]

    # iso exec >= savi exec per instance (exec parity check).
    iso_exec_geq_savi = all(
        budgets["iso_exec"][inst.id]["exec"] >= budgets["savi_freq"][inst.id]["exec"]
        for inst in insts)

    # ----- 4) optional axis-artifact: iso_candidates matched to savi_freq cands. -----
    artifact = None
    if axis_artifact:
        iso_cand_ok = {}
        iso_cand_budget = {}
        for inst in insts:
            idict = inst_dict(inst)
            t = freq_cand_target[inst.id]
            entry = _cached(
                "iso_candidates", inst,
                lambda i=idict, tt=t: _arms_ext.best_of_k_isobudget(
                    domain, gen, i, tt, tau, seed, axis="candidates"),
                extra=f"cand={t}")
            iso_cand_ok[inst.id] = bool(entry["ok"])
            iso_cand_budget[inst.id] = entry["budget"]
        artifact = {
            "iso_candidates_ok": iso_cand_ok,
            "iso_candidates_budget": iso_cand_budget,
        }

    if cache_dirty:
        _save_cache(cache_path, cache)

    return {
        "depth": depth, "K": K, "p": float(p), "seed": int(seed),
        "inst_ids": [i.id for i in insts],
        "ok": ok,
        "budgets": budgets,
        "iso_exec_geq_savi": bool(iso_exec_geq_savi),
        "artifact": artifact,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _pass_at_1_pooled(cells, arm):
    flags = []
    for cell in cells:
        for i in cell["inst_ids"]:
            flags.append(bool(cell["ok"][arm][i]))
    return float(pass_at_1(flags)), flags


def _budget_totals_pooled(cells, arm):
    agg = _zero_budget()
    for cell in cells:
        for i in cell["inst_ids"]:
            b = cell["budgets"][arm][i]
            for f in agg:
                agg[f] += int(b[f])
    return agg


def _delta_pooled(cells, a_arm, b_arm, boot_seed):
    a = []
    b = []
    for cell in cells:
        for i in cell["inst_ids"]:
            a.append(bool(cell["ok"][a_arm][i]))
            b.append(bool(cell["ok"][b_arm][i]))
    if not a:
        return {"status": "skipped", "note": "empty stratum"}
    delta, lo, hi = paired_bootstrap(a, b, n=BOOTSTRAP_N, seed=boot_seed)
    return {"delta": float(delta), "lo": float(lo), "hi": float(hi),
            "n": len(a), "status": "ok"}


# ---------------------------------------------------------------------------
# Finiteness guard
# ---------------------------------------------------------------------------

def _check_finite_tree(obj, ctx="root"):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite_tree(v, f"{ctx}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            _check_finite_tree(v, f"{ctx}[{idx}]")
    elif isinstance(obj, (int, float)):
        if not math.isfinite(float(obj)):
            raise ValueError(f"non-finite metric at {ctx}: {obj!r}")


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def _artifact_cell_choice(depths, K_grid):
    """The ONE (depth, K) cell at which to compute the candidate-parity artifact.

    A mid depth with the smallest K (so the artifact — selection winning under candidate
    parity but not exec parity — is most visible). Deterministic.
    """
    sorted_d = sorted(depths)
    mid_depth = sorted_d[len(sorted_d) // 2] if sorted_d else None
    min_K = min(K_grid) if K_grid else None
    return mid_depth, min_K


def run(domain_name, depths, K_grid, p_grid, seeds, N, inst_per_depth, outdir,
        tau=DEFAULT_TAU, only_depth=None):
    """Run the depth x K x p grid; write results2.json; return it.

    ``only_depth`` (the ``--depth`` CLI) restricts THIS invocation to one depth (still
    sharing ``outdir`` / cache), so depths can be run sequentially and resumed. The
    written ``results2.json`` always reflects every depth whose cells are present in the
    cache (the union across invocations).
    """
    os.makedirs(outdir, exist_ok=True)
    results_path = os.path.join(outdir, "results2.json")
    cache_path = os.path.join(outdir, "cache.json")
    log_path = os.path.join(outdir, "run2.log")

    bundle = _make_bundle(domain_name, inst_per_depth)
    cache = _load_cache(cache_path)
    log = open(log_path, "a")
    t_start = time.time()

    depths = [int(d) for d in depths]
    K_grid = [int(k) for k in K_grid]
    p_grid = [float(x) for x in p_grid]
    seeds = [int(s) for s in seeds]

    config = {
        "domain": domain_name, "depths": depths, "K_grid": K_grid,
        "p_grid": p_grid, "seeds": seeds, "N": int(N),
        "inst_per_depth": int(inst_per_depth), "tau": float(tau),
        "bootstrap_n": BOOTSTRAP_N, "arms": list(ARMS),
        "only_depth": (int(only_depth) if only_depth is not None else None),
    }

    art_depth, art_K = _artifact_cell_choice(depths, K_grid)

    run_depths = [int(only_depth)] if only_depth is not None else depths

    # Run the requested depth(s); cells are accumulated in the cache.
    for depth in run_depths:
        for K in K_grid:
            for p in p_grid:
                for seed in seeds:
                    is_art = (depth == art_depth and K == art_K)
                    t0 = time.time()
                    _run_cell(bundle, depth, K, p, seed, N, tau, cache, cache_path,
                              axis_artifact=is_art)
                    dt = time.time() - t0
                    line = (f"cell domain={domain_name} depth={depth} K={K} p={p} "
                            f"seed={seed} {dt:.1f}s")
                    print(line, flush=True)
                    log.write(line + "\n"); log.flush()

    # ---- Rebuild results2.json from the FULL cache (union of all depths run). ----
    result = _build_results(bundle, config, depths, K_grid, p_grid, seeds, N, tau,
                            cache, art_depth, art_K)
    _check_finite_tree(result, "results2")
    with open(results_path, "w") as fh:
        json.dump(result, fh, indent=2)

    elapsed = time.time() - t_start
    _emit_summary(result, elapsed)
    log.write(f"DONE elapsed={elapsed:.1f}s wrote {results_path}\n")
    log.close()
    return result


def _recover_cell(bundle, domain_name, depth, K, p, seed, N, tau, cache, want_artifact):
    """Reconstruct a cell summary from the cache only (no recompute). None if missing."""
    name = bundle["name"]
    insts = bundle["make_instances"](depth, seed)
    ok = {arm: {} for arm in ARMS}
    budgets = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for inst in insts:
            # iso_exec / savi_freq use deterministic targets; recompute the extra tag.
            extra = ""
            if arm == "iso_exec":
                fk = _cache_key(name, depth, K, p, seed, "savi_freq", inst.id, N, tau)
                fe = cache.get(fk)
                if fe is None:
                    return None
                extra = f"exec={max(1, int(fe['exec']))}"
            key = _cache_key(name, depth, K, p, seed, arm, inst.id, N, tau, extra)
            entry = cache.get(key)
            if entry is None:
                return None
            ok[arm][inst.id] = bool(entry["ok"])
            budgets[arm][inst.id] = entry["budget"]

    iso_exec_geq_savi = all(
        budgets["iso_exec"][inst.id]["exec"] >= budgets["savi_freq"][inst.id]["exec"]
        for inst in insts)

    artifact = None
    if want_artifact:
        iso_cand_ok = {}
        iso_cand_budget = {}
        for inst in insts:
            fk = _cache_key(name, depth, K, p, seed, "savi_freq", inst.id, N, tau)
            fe = cache.get(fk)
            if fe is None:
                return None
            extra = f"cand={max(1, int(fe['candidates']))}"
            key = _cache_key(name, depth, K, p, seed, "iso_candidates", inst.id, N, tau,
                             extra)
            entry = cache.get(key)
            if entry is None:
                return None
            iso_cand_ok[inst.id] = bool(entry["ok"])
            iso_cand_budget[inst.id] = entry["budget"]
        artifact = {"iso_candidates_ok": iso_cand_ok,
                    "iso_candidates_budget": iso_cand_budget}

    return {"depth": depth, "K": K, "p": float(p), "seed": int(seed),
            "inst_ids": [i.id for i in insts], "ok": ok, "budgets": budgets,
            "iso_exec_geq_savi": bool(iso_exec_geq_savi), "artifact": artifact}


def _build_results(bundle, config, depths, K_grid, p_grid, seeds, N, tau, cache,
                   art_depth, art_K):
    """Assemble results2.json from cached cells (union over depths present)."""
    domain_name = config["domain"]
    per_cell = []
    axis_artifact_cell = None

    for depth in depths:
        for K in K_grid:
            for p in p_grid:
                want_art = (depth == art_depth and K == art_K)
                # Recover the per-seed cells for this (depth, K, p).
                seed_cells = []
                for seed in seeds:
                    sc = _recover_cell(bundle, domain_name, depth, K, p, seed, N, tau,
                                       cache, want_art)
                    if sc is not None:
                        seed_cells.append(sc)
                if not seed_cells:
                    continue  # this depth not yet run

                pass_at_1_arm = {}
                for arm in ARMS:
                    val, _ = _pass_at_1_pooled(seed_cells, arm)
                    pass_at_1_arm[arm] = val

                boot_seed = sum(seeds) if seeds else 0
                D1_exec = _delta_pooled(seed_cells, "savi_freq", "iso_exec", boot_seed)
                D_verif = _delta_pooled(seed_cells, "savi_verifier_on", "iso_exec",
                                        boot_seed)
                D_merge = _delta_pooled(seed_cells, "savi_freq", "beam_no_merge",
                                        boot_seed)

                budgets = {arm: _budget_totals_pooled(seed_cells, arm) for arm in ARMS}
                iso_exec_geq_savi = all(c["iso_exec_geq_savi"] for c in seed_cells)

                cell = {
                    "depth": depth, "K": K, "p": float(p),
                    "n_instances": sum(len(c["inst_ids"]) for c in seed_cells),
                    "pass_at_1": pass_at_1_arm,
                    "D1_exec": D1_exec,
                    "D_verif": D_verif,
                    "D_merge": D_merge,
                    "budgets": budgets,
                    "iso_exec_geq_savi": bool(iso_exec_geq_savi),
                }
                per_cell.append(cell)

                # axis artifact cell: candidate-parity vs exec-parity selection pass@1.
                if want_art and seed_cells[0].get("artifact"):
                    cand_flags = []
                    exec_flags = []
                    for c in seed_cells:
                        art = c.get("artifact")
                        if not art:
                            continue
                        for i in c["inst_ids"]:
                            cand_flags.append(bool(art["iso_candidates_ok"][i]))
                            exec_flags.append(bool(c["ok"]["iso_exec"][i]))
                    if cand_flags:
                        axis_artifact_cell = {
                            "depth": depth, "K": K, "p": float(p),
                            "iso_candidates_pass": float(pass_at_1(cand_flags)),
                            "iso_exec_pass": float(pass_at_1(exec_flags)),
                            "savi_freq_pass": pass_at_1_arm["savi_freq"],
                            "n": len(cand_flags),
                            "note": ("candidate parity hands selection ~depth x more "
                                     "complete attempts than exec parity"),
                        }

    controls = _controls(per_cell, config)

    return {
        "config": config,
        "per_cell": per_cell,
        "axis_artifact_cell": axis_artifact_cell,
        "controls": controls,
    }


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def _controls(per_cell, config):
    """C1/C3 + depth monotonicity + axis cross-check, read off ``per_cell``."""
    # C3: ceiling == 1.0 on every cell (lattice solvable by construction). And no arm
    # should beat the ceiling (would be a bug solving a non-goal state).
    ceiling_all_one = all(c["pass_at_1"]["ceiling"] == 1.0 for c in per_cell) \
        if per_cell else None

    # Depth monotonicity of iso_exec at a fixed (K, p): per-rollout success ~ q^D, so
    # iso_exec pass@1 should DECAY with depth. Report the iso_exec pass@1 per depth at
    # the primary (max K, first p) slice.
    K_ref = max(config["K_grid"]) if config["K_grid"] else None
    p_ref = config["p_grid"][0] if config["p_grid"] else None
    iso_by_depth = {}
    savi_by_depth = {}
    for c in per_cell:
        if c["K"] == K_ref and c["p"] == p_ref:
            iso_by_depth[c["depth"]] = c["pass_at_1"]["iso_exec"]
            savi_by_depth[c["depth"]] = c["pass_at_1"]["savi_freq"]
    depths_sorted = sorted(iso_by_depth)
    iso_monotone_decay = all(
        iso_by_depth[depths_sorted[i]] >= iso_by_depth[depths_sorted[i + 1]] - 1e-9
        for i in range(len(depths_sorted) - 1)) if len(depths_sorted) >= 2 else None

    controls = {
        "C3_ceiling_all_one": ceiling_all_one,
        "C3_note": "lattice solvable by construction; ceiling must be 1.0 everywhere",
        "depth_monotonicity": {
            "K_ref": K_ref, "p_ref": p_ref,
            "iso_exec_pass_by_depth": {str(d): iso_by_depth[d]
                                       for d in depths_sorted},
            "savi_freq_pass_by_depth": {str(d): savi_by_depth.get(d)
                                        for d in depths_sorted},
            "iso_exec_decays_with_depth": iso_monotone_decay,
            "expect": "iso_exec decays with depth (~q^D); trellis advantage grows",
        },
    }
    return controls


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _emit_summary(result, elapsed):
    print("\n==== PLAN2 SWEEP SUMMARY ====", flush=True)
    cfg = result["config"]
    print(f"domain={cfg['domain']} depths={cfg['depths']} K={cfg['K_grid']} "
          f"p={cfg['p_grid']} seeds={cfg['seeds']}")
    for c in result["per_cell"]:
        p1 = c["pass_at_1"]
        parts = " ".join(f"{a}={p1[a]:.3f}" for a in ARMS)
        d1 = c["D1_exec"]
        d1s = (f"D1_exec={d1['delta']:+.3f}[{d1['lo']:+.3f},{d1['hi']:+.3f}]"
               if d1.get("status") == "ok" else "D1_exec=NA")
        print(f"  depth={c['depth']} K={c['K']} p={c['p']} | {parts} | {d1s} "
              f"| iso>=savi={c['iso_exec_geq_savi']}")
    if result.get("axis_artifact_cell"):
        a = result["axis_artifact_cell"]
        print(f"  axis_artifact depth={a['depth']} K={a['K']}: "
              f"iso_CAND={a['iso_candidates_pass']:.3f} "
              f"iso_EXEC={a['iso_exec_pass']:.3f} "
              f"savi_freq={a['savi_freq_pass']:.3f}")
    print(f"==== elapsed {elapsed:.1f}s ====", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_csv_floats(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_csv_ints(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _build_argparser():
    p = argparse.ArgumentParser(description="PLAN2 depth x K x p exec-parity sweep.")
    p.add_argument("--domain", default="lattice", choices=["lattice"])
    p.add_argument("--depths", default="4,8,16,24")
    p.add_argument("--K-grid", dest="K_grid", default="8,16,32")
    p.add_argument("--p-grid", dest="p_grid", default="0.7")
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    p.add_argument("--inst-per-depth", dest="inst_per_depth", type=int, default=24)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--depth", type=int, default=None,
                   help="run ONE depth this invocation (shares outdir/cache)")
    p.add_argument("--outdir", required=True)
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    result = run(
        domain_name=args.domain,
        depths=_parse_csv_ints(args.depths),
        K_grid=_parse_csv_ints(args.K_grid),
        p_grid=_parse_csv_floats(args.p_grid),
        seeds=_parse_csv_ints(args.seeds),
        N=args.N,
        inst_per_depth=args.inst_per_depth,
        outdir=args.outdir,
        tau=args.tau,
        only_depth=args.depth,
    )
    print(f"\nwrote {os.path.join(args.outdir, 'results2.json')}")
    return result


if __name__ == "__main__":
    main()
