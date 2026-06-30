"""Novel-path / recombination observation.

Does the soft global decode return correct paths that no single rollout produced
(stitched from per-step fragments living in different samples)? best-of-K can only
return a whole sampled chain; the merged trellis can recombine across samples.

Pure metrics (exact, no Monte Carlo for the probability):
  pp_probs   : exact per-step P_p(move|state) of the competence-p fair generator
  q_of_path  : joint probability Π P_p of a path (=> selection needs ~1/q rollouts)
  path_edges : [( (s,r), move )]  -- node identity is (s,r), T fixed per instance
  analyze    : novel (path ∉ sampled chains) / stitch (novel AND every edge covered
               by some chain -- selection had all fragments, never assembled them)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPS = os.path.join(_HERE, "_deps")     # vendored: domain_merge / arms_phasemap / value_lattice / gen_fair / domain_lattice / arms_ext
_VD = os.path.dirname(_HERE)             # verifier-decode/ (holds decode_core/, shipped separately)
for _p in (_DEPS, _VD, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import domain_merge as dm     # noqa: E402
import arms_phasemap as ap    # noqa: E402
import value_lattice as vl    # noqa: E402
import gen_fair as gf         # noqa: E402


# --------------------------------------------------------------------------- pure
def pp_probs(domain, state, p):
    """Exact P_p(move|state): legal moves weighted GOOD=p / BAD=(1-p) (uniform if all
    same class), matching gen_fair._classify_weights. Returns {move_int: prob}."""
    kept, good = [], []
    for c in dm.lattice_enumerate(state):
        m = domain.parse_move(c, state)
        if m is None:
            continue
        kept.append(int(m))
        good.append(bool(domain.solvable(domain.apply(state, m))))
    if not kept:
        return {}
    ng = sum(good)
    if ng == 0 or ng == len(kept):
        w = [1.0] * len(kept)
    else:
        w = [(p if g else (1.0 - p)) for g in good]
    tot = sum(w)
    return {m: wi / tot for m, wi in zip(kept, w)}


def q_of_path(domain, s0, path, p):
    """Joint probability Π_i P_p(m_i|s_i) of producing `path` as one rollout."""
    cur, q = s0, 1.0
    for m in path:
        pr = pp_probs(domain, cur, p).get(int(m))
        if not pr:
            return 0.0
        q *= pr
        cur = domain.apply(cur, int(m))
    return q


def solve_mass(domain, state, p, memo=None):
    """Exact P(one P_p rollout reaches the goal from `state`) = total weighted
    correct-path mass. Selection needs ~1/solve_mass rollouts to solve the instance
    at all (this is the instance-level compute wall, vs q which is one path)."""
    if memo is None:
        memo = {}
    key = (state.s, state.r)
    if key in memo:
        return memo[key]
    if state.r == 0:
        memo[key] = 1.0 if domain.is_goal(state) else 0.0
        return memo[key]
    tot = 0.0
    for m, prob in pp_probs(domain, state, p).items():
        tot += prob * solve_mass(domain, domain.apply(state, int(m)), p, memo)
    memo[key] = tot
    return tot


def path_edges(domain, s0, path):
    cur, edges = s0, []
    for m in path:
        edges.append(((cur.s, cur.r), int(m)))
        cur = domain.apply(cur, int(m))
    return edges


def chain_to_moves(domain, s0, text):
    mv = domain.parse_chain(text, s0)
    return tuple(int(x) for x in mv) if mv is not None else None


def reaches_goal(domain, s0, moves):
    cur = s0
    for m in moves:
        cur = domain.apply(cur, int(m))
    return bool(domain.is_goal(cur))


def analyze(domain, s0, decode_path, chains):
    """chains: list of move-tuples (full sampled rollouts). Returns novelty/stitch."""
    dp = tuple(int(m) for m in decode_path)
    chain_set = set(chains)
    cover = set()
    for ch in chains:
        cover.update(path_edges(domain, s0, ch))
    dedges = path_edges(domain, s0, dp)
    n_cov = sum(1 for e in dedges if e in cover)
    novel = dp not in chain_set
    all_cov = (n_cov == len(dedges))
    return {"novel": novel, "stitch": bool(novel and all_cov),
            "all_edges_covered": all_cov, "n_edges": len(dedges), "n_covered": n_cov}


# --------------------------------------------------------------------------- runner
def _train_value(depth, n, seeds=(100, 101, 102)):
    Xtr, ytr = vl.make_dataset(list(seeds), [depth], n)
    return vl.LatticeValueModel("logreg", seed=seeds[0]).fit(Xtr, ytr)


def run_cell(p, g, depth, value_kind, model, seeds, n, K, N, tau, lam, M):
    exact = dm.MergeLatticeDomain(g)
    gen = gf.make_fair_generator(exact, dm.lattice_enumerate, p, depth)
    vfn = model.value_fn() if value_kind == "learned" else (lambda s: 1.0)
    use_lam = lam if value_kind == "learned" else 0.0

    agg = {k: 0 for k in ("inst", "dec_ok", "chain_ok", "only_dec", "novel",
                          "stitch", "dec_and_sel")}
    inv_q_log = []   # log10(1/q) for decode-solved paths (selection needs ~ this for THE PATH)
    cover_fracs = []  # fraction of decode-path edges sampled by some chain
    inv_mass_only = []  # log10(1/solve_mass) over only-decode instances (instance-level wall)
    for seed in seeds:
        for inst in dm.make_lattice_instances(depth, n, seed):
            idict = dm.lattice_inst_dict(inst)
            s0 = exact.initial_state(idict)
            agg["inst"] += 1
            res = ap.savi_value(exact, gen, idict, vfn, use_lam, K, N, tau, seed,
                                max_depth=depth, edge_mode="freq")
            dec_ok = bool(res.ok and res.path is not None
                          and reaches_goal(exact, s0, res.path))
            chains = [m for m in (chain_to_moves(exact, s0, t)
                                  for t in gen(s0, M, tau, seed, "chain")) if m]
            chain_ok = any(reaches_goal(exact, s0, ch) for ch in chains)
            agg["chain_ok"] += int(chain_ok)
            if not dec_ok:
                continue
            agg["dec_ok"] += 1
            agg["only_dec"] += int(not chain_ok)
            agg["dec_and_sel"] += int(chain_ok)
            info = analyze(exact, s0, res.path, chains)
            agg["novel"] += int(info["novel"])
            agg["stitch"] += int(info["stitch"])
            cover_fracs.append(info["n_covered"] / max(1, info["n_edges"]))
            q = q_of_path(exact, s0, res.path, p)
            if q > 0:
                inv_q_log.append(math.log10(1.0 / q))
            if not chain_ok:  # only-decode: how many rollouts would selection need?
                mass = solve_mass(exact, s0, p)
                if mass > 0:
                    inv_mass_only.append(math.log10(1.0 / mass))

    d = agg["dec_ok"] or 1
    inv_q_log.sort()
    inv_mass_only.sort()
    med_inv_q = inv_q_log[len(inv_q_log) // 2] if inv_q_log else None
    med_inv_mass_only = (inv_mass_only[len(inv_mass_only) // 2]
                         if inv_mass_only else None)
    mean_cover = sum(cover_fracs) / len(cover_fracs) if cover_fracs else None
    return {
        "p": p, "g": ("inf" if g == math.inf else int(g)), "value": value_kind,
        "lam": use_lam, "M": M, **agg,
        "dec_solve_rate": agg["dec_ok"] / agg["inst"],
        "chain_solve_rate": agg["chain_ok"] / agg["inst"],
        "only_decode_rate": agg["only_dec"] / agg["inst"],
        "novel_rate_of_solved": agg["novel"] / d,
        "stitch_rate_of_solved": agg["stitch"] / d,
        "mean_edge_cover_of_solved": mean_cover,  # frac of decode-path edges selection sampled
        "median_log10_inv_q": med_inv_q,   # selection rollouts to sample DECODE'S path
        "median_log10_inv_mass_onlydec": med_inv_mass_only,  # rollouts to SOLVE only-decode insts
    }


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--p", default="0.5,0.6,0.7,0.9")
    a.add_argument("--gs", default="0,inf")
    a.add_argument("--depth", type=int, default=24)
    a.add_argument("--values", default="learned,freq")
    a.add_argument("--seeds", default="0-7")
    a.add_argument("--n", type=int, default=24)
    a.add_argument("--K", type=int, default=8)
    a.add_argument("--N", type=int, default=16)
    a.add_argument("--tau", type=float, default=1.0)
    a.add_argument("--lam", type=float, default=1.0)
    a.add_argument("--M", type=int, default=256, help="selection rollout budget")
    a.add_argument("--out", required=True)
    args = a.parse_args(argv)

    ps = [float(x) for x in args.p.split(",") if x.strip()]
    gs = [math.inf if t.strip() in ("inf", "tree") else int(t)
          for t in args.gs.split(",") if t.strip()]
    if "-" in args.seeds and "," not in args.seeds:
        lo, hi = args.seeds.split("-"); seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    values = [v.strip() for v in args.values.split(",") if v.strip()]

    model = _train_value(args.depth, args.n) if "learned" in values else None
    cells = []
    for value_kind in values:
        for p in ps:
            for g in gs:
                gl = "inf" if g == math.inf else int(g)
                print(f"[cell] value={value_kind} p={p} g={gl}", file=sys.stderr)
                cells.append(run_cell(p, g, args.depth, value_kind, model, seeds,
                                      args.n, args.K, args.N, args.tau, args.lam, args.M))

    out = {"plan": "recombination-novelpath",
           "config": {"p": ps, "gs": args.gs, "depth": args.depth, "values": values,
                      "seeds": seeds, "n": args.n, "K": args.K, "N": args.N,
                      "tau": args.tau, "lam": args.lam, "M": args.M},
           "cells": cells}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({len(cells)} cells)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
