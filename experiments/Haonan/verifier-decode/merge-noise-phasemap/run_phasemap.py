"""Merge × verifier-noise phase map runner (SPEC §6).

For each (g, depth, seed) cell: build the EXACT MergeLatticeDomain(g) + the fair generator
(emission), run the requested arms over all instances, and match the selection baseline A0
per-arm on exec. Decode arms A3/A4 run on a NoisyMergeLatticeDomain (same structure, noisy
verifier) while the generator stays on the exact domain (SPEC §2.4). Aggregates per
(g, depth) over seeds×instances: pass@1, mean exec, D1(arm)=paired_bootstrap(arm, matched
A0), and the realized merge ratio (the R mediator).

Usage (see SPEC §6):
  python run_phasemap.py --gs 0 --depths 8,16,24 --arms A1 --n 24 --seeds 0-7 \
      --out outputs/phasemap/v3_g0.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VD = os.path.dirname(_HERE)
# gen_fair is vendored locally under _deps/ (originally ../mechanism-recombination/gen_fair.py).
_MR = os.path.join(_HERE, "_deps")
for _p in (_HERE, _VD, _MR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import domain_merge as dm  # noqa: E402
import arms_phasemap as ap  # noqa: E402
import gen_fair as gf  # noqa: E402
from decode_core import decode as dc  # noqa: E402
from decode_core._deps.metrics.passk import pass_at_1, paired_bootstrap  # noqa: E402

BOOTSTRAP_N = 10000
NOISE_SEED = 0  # the imperfect verifier is ONE fixed function across sampling seeds


# ---------------------------------------------------------------------------
# Arm expansion: turn the --arms / --eps / --modes / --lams request into a list
# of (label, kind, params) the cell runner executes per instance.
# ---------------------------------------------------------------------------

def _expand_arms(arms, eps_list, modes, lams, rho):
    """Return ordered [(label, kind, params)]. A0 (selection) is always added implicitly."""
    out = []
    for a in arms:
        if a == "A0":
            continue  # implicit, matched per-arm
        if a == "A1":
            out.append(("A1", "savi_freq", {}))
        elif a == "A1x":
            out.append(("A1x", "savi_verifier_exact", {}))
        elif a == "A2":
            out.append(("A2", "savi_support", {}))
        elif a == "A5":
            out.append(("A5", "beam_no_merge_freq", {}))
        elif a == "A3":
            for mode in modes:
                for eps in eps_list:
                    lbl = f"A3_{mode}_eps{eps:g}"
                    out.append((lbl, "savi_hardmask",
                                {"eps": eps, "mode": mode, "rho": rho}))
        elif a == "A4":
            for mode in modes:
                for eps in eps_list:
                    for lam in lams:
                        lbl = f"A4_{mode}_eps{eps:g}_lam{lam:g}"
                        out.append((lbl, "savi_softval",
                                    {"eps": eps, "mode": mode, "rho": rho, "lam": lam}))
        else:
            raise ValueError(f"unknown arm {a!r}")
    return out


def _run_one(kind, params, g, exact, gen, idict, K, N, tau, seed, depth, profile="uniform"):
    """Run ONE decode arm over ONE instance; return a decode_core.Result."""
    if kind == "savi_freq":
        return dc.savi(exact, gen, idict, K, N, "freq", tau, seed,
                       verifier=False, max_depth=depth)
    if kind == "savi_verifier_exact":
        return dc.savi(exact, gen, idict, K, N, "freq", tau, seed,
                       verifier=True, max_depth=depth)
    if kind == "savi_support":
        return dc.savi(exact, gen, idict, K, N, "support", tau, seed,
                       verifier=False, max_depth=depth)
    if kind == "beam_no_merge_freq":
        return ap.beam_no_merge(exact, gen, idict, K, N, tau, seed,
                                verifier=False, max_depth=depth, edge_mode="freq")
    prof_depth = depth if profile == "boundary" else None
    if kind == "savi_hardmask":
        noisy = dm.NoisyMergeLatticeDomain(g, params["eps"], params["mode"],
                                           params["rho"], NOISE_SEED,
                                           profile=profile, prof_depth=prof_depth)
        return dc.savi(noisy, gen, idict, K, N, "freq", tau, seed,
                       verifier=True, max_depth=depth)
    if kind == "savi_softval":
        noisy = dm.NoisyMergeLatticeDomain(g, params["eps"], params["mode"],
                                           params["rho"], NOISE_SEED,
                                           profile=profile, prof_depth=prof_depth)
        return ap.savi_value(noisy, gen, idict, noisy.value, params["lam"], K, N, tau,
                             seed, max_depth=depth, edge_mode="freq")
    raise ValueError(f"unknown kind {kind!r}")


def _replay_ok(exact, res, idict):
    """True iff the returned path reaches a real (exact) goal; ok-only for the ceiling."""
    if not res.ok:
        return False
    if res.path is None:
        return bool(res.ok)
    s = exact.initial_state(idict)
    for mv in res.path:
        s = exact.apply(s, mv)
    return bool(exact.is_goal(s))


def _after_sum(res):
    """Total distinct nodes kept across layers (savi: distinct canon; beam: per-path)."""
    return sum(res.trellis_widths_after_merge or [])


def _sample_mult(res):
    """Within-node sample multiplicity (raw feasible / distinct successors) — NOT R."""
    b = sum(res.trellis_widths_before_merge or [])
    a = sum(res.trellis_widths_after_merge or [])
    return (b / a) if a > 0 else 1.0


# ---------------------------------------------------------------------------
# One (g, depth) cell, pooled over seeds × instances
# ---------------------------------------------------------------------------

def _run_cell(g, depth, seeds, n, p, K, N, tau, arm_specs, profile="uniform"):
    exact = dm.MergeLatticeDomain(g)
    max_depth = int(depth)

    # per-arm pooled flags + matched-selection flags + exec totals
    arm_ok = {lbl: [] for (lbl, _k, _pp) in arm_specs}
    a0_ok = {lbl: [] for (lbl, _k, _pp) in arm_specs}  # selection matched to this arm's exec
    arm_exec = {lbl: [] for (lbl, _k, _pp) in arm_specs}
    ceiling = []
    sample_mults = []          # within-node multiplicity of the savi ratio_source (ref)
    savi_after = 0             # Σ distinct-canon nodes (savi)        -- R denominator
    beam_after = 0             # Σ per-path nodes (beam_no_merge)     -- R numerator

    ratio_source = next((lbl for (lbl, k, _pp) in arm_specs
                         if k in ("savi_freq", "savi_verifier_exact")), None)

    for seed in seeds:
        insts = dm.make_lattice_instances(depth, n, seed)
        gen = gf.make_fair_generator(exact, dm.lattice_enumerate, p, max_depth)
        # selection cache within this (cell, seed): exec target -> ok per instance
        for inst in insts:
            idict = dm.lattice_inst_dict(inst)
            ceiling.append(bool(dc.oracle(exact, idict)))
            sel_cache = {}  # (inst.id, target) -> ok
            for (lbl, kind, pp) in arm_specs:
                res = _run_one(kind, pp, g, exact, gen, idict, K, N, tau, seed, max_depth,
                               profile=profile)
                ok = _replay_ok(exact, res, idict)
                ex = int(res.budget.exec)
                arm_ok[lbl].append(ok)
                arm_exec[lbl].append(ex)
                if lbl == ratio_source:
                    sample_mults.append(_sample_mult(res))
                    savi_after += _after_sum(res)
                if kind == "beam_no_merge_freq":
                    beam_after += _after_sum(res)
                # matched selection at THIS arm's exec
                target = max(1, ex)
                ck = (inst.id, target)
                if ck not in sel_cache:
                    sres = ap.best_of_k_isobudget(exact, gen, idict, target, tau, seed,
                                                  axis="exec")
                    sel_cache[ck] = (_replay_ok(exact, sres, idict),
                                     int(sres.budget.exec))
                sel_ok, sel_ex = sel_cache[ck]
                a0_ok[lbl].append(sel_ok)
                # exec parity guard: selection must spend >= the arm's exec
                if sel_ex < ex:
                    raise AssertionError(
                        f"exec parity broken: sel {sel_ex} < {lbl} {ex} "
                        f"(g={g}, depth={depth}, inst={inst.id})")

    cell = {"g": (None if g == math.inf else int(g)),
            "g_label": ("inf" if g == math.inf else str(int(g))),
            "depth": int(depth), "n_inst": int(n), "n_seeds": len(seeds),
            "n_pooled": len(ceiling),
            "ceiling": float(pass_at_1(ceiling)),
            "merge_R": ((beam_after / savi_after) if (savi_after and beam_after)
                        else None),  # cross-path redundancy realized (≈1 at no-merge)
            "sample_mult": (float(sum(sample_mults) / len(sample_mults))
                            if sample_mults else None),
            "pass1": {}, "mean_exec": {}, "sel_pass1": {}, "D1": {}}
    for (lbl, _k, _pp) in arm_specs:
        a = arm_ok[lbl]
        b = a0_ok[lbl]
        cell["pass1"][lbl] = float(pass_at_1(a))
        cell["sel_pass1"][lbl] = float(pass_at_1(b))
        cell["mean_exec"][lbl] = float(sum(arm_exec[lbl]) / max(1, len(arm_exec[lbl])))
        delta, lo, hi = paired_bootstrap(a, b, n=BOOTSTRAP_N, seed=12345)
        cell["D1"][lbl] = {"delta": float(delta), "lo": float(lo), "hi": float(hi),
                           "n": len(a)}
    return cell


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_gs(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        out.append(math.inf if tok in ("inf", "infty", "tree") else int(tok))
    return out


def _parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _parse_floats(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_seeds(s):
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return _parse_ints(s)


def main(argv=None):
    ap_ = argparse.ArgumentParser(description="merge × noise phase map")
    ap_.add_argument("--gs", default="0", help="merge granularities, e.g. 0,1,2,inf")
    ap_.add_argument("--depths", default="8,16,24")
    ap_.add_argument("--arms", default="A1",
                     help="comma list from A1,A1x,A2,A3,A4,A5 (A0 selection implicit)")
    ap_.add_argument("--eps", default="0,0.05,0.1,0.18,0.3")
    ap_.add_argument("--modes", default="fn,sym")
    ap_.add_argument("--lams", default="0.5,1,2,4,8")
    ap_.add_argument("--rho", type=float, default=1.0)
    ap_.add_argument("--profile", default="uniform", choices=["uniform", "boundary"],
                     help="noise geometry: 'uniform' (flat eps) or 'boundary' "
                          "(measured PRM FP profile concentrated near the boundary, "
                          "mean-preserving)")
    ap_.add_argument("--n", type=int, default=24)
    ap_.add_argument("--seeds", default="0-7")
    ap_.add_argument("--p", type=float, default=0.7)
    ap_.add_argument("--K", type=int, default=8)
    ap_.add_argument("--N", type=int, default=16)
    ap_.add_argument("--tau", type=float, default=1.0)
    ap_.add_argument("--out", required=True)
    args = ap_.parse_args(argv)

    gs = _parse_gs(args.gs)
    depths = _parse_ints(args.depths)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    eps_list = _parse_floats(args.eps)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    lams = _parse_floats(args.lams)
    seeds = _parse_seeds(args.seeds)
    arm_specs = _expand_arms(arms, eps_list, modes, lams, args.rho)

    config = {"gs": args.gs, "depths": depths, "arms": arms, "eps": eps_list,
              "modes": modes, "lams": lams, "rho": args.rho, "profile": args.profile,
              "n": args.n, "seeds": seeds, "p": args.p, "K": args.K, "N": args.N,
              "tau": args.tau, "bootstrap_n": BOOTSTRAP_N, "noise_seed": NOISE_SEED}

    cells = []
    for g in gs:
        for depth in depths:
            gl = "inf" if g == math.inf else int(g)
            print(f"[cell] g={gl} depth={depth} arms={len(arm_specs)} "
                  f"seeds={len(seeds)} n={args.n}", file=sys.stderr)
            cell = _run_cell(g, depth, seeds, args.n, args.p, args.K, args.N, args.tau,
                             arm_specs, profile=args.profile)
            cells.append(cell)

    out = {"plan": "merge-noise-phasemap", "config": config, "cells": cells}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out} ({len(cells)} cells)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
