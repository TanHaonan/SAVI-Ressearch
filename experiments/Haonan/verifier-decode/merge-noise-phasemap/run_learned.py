"""Step-2 runner: soft-λ decode with a LEARNED GRADED value (not the binary oracle).

Trains K capacity-limited value models on instances from TRAIN seeds (disjoint from EVAL
seeds), reports each model's generalization metrics, then runs the soft-λ decode arm (A4L,
``savi_value`` with the learned ``value_fn``) over the competence-`p` sweep vs an exec-matched
selection baseline (paired bootstrap). This replaces the binary 0.95/0.05 near-oracle value
with a genuinely imperfect, graded, multi-seed value — the adversary's main caveat.

Usage:
  python run_learned.py --p 0.5,0.6,0.7,0.9 --depths 24 --lams 0.1,0.5,1,2 \
    --train_seeds 100,101,102 --eval_seeds 0-7 --n 24 --capacity logreg \
    --out outputs/phasemap/learned_value.json
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
import value_lattice as vl  # noqa: E402
import gen_fair as gf  # noqa: E402
from decode_core import decode as dc  # noqa: E402
from decode_core._deps.metrics.passk import pass_at_1, paired_bootstrap  # noqa: E402

BOOTSTRAP_N = 10000


def _replay_ok(exact, res, idict):
    if not res.ok:
        return False
    if res.path is None:
        return bool(res.ok)
    s = exact.initial_state(idict)
    for mv in res.path:
        s = exact.apply(s, mv)
    return bool(exact.is_goal(s))


def _train_models(train_seeds, depths, n, capacity):
    """One model per train seed (independent noise realizations); plus disjoint-eval metrics."""
    models = {}
    metrics = {}
    # held-out eval set for metrics: a DIFFERENT seed block, disjoint from train AND from the
    # decode eval seeds (which are 0-7).
    Xev, yev = vl.make_dataset([900, 901], depths, max(8, n // 2))
    for sd in train_seeds:
        Xtr, ytr = vl.make_dataset([sd], depths, n)
        m = vl.LatticeValueModel(capacity=capacity, seed=sd).fit(Xtr, ytr)
        models[sd] = m
        metrics[sd] = vl.evaluate(m, Xev, yev)
    return models, metrics


def _run_cell(p, depth, g, eval_seeds, n, K, N, tau, models, lams):
    exact = dm.MergeLatticeDomain(g)
    max_depth = int(depth)
    arms = {}  # label -> {ok:[], a0:[], exec:[]}

    def _slot(lbl):
        return arms.setdefault(lbl, {"ok": [], "a0": [], "exec": []})

    for seed in eval_seeds:
        insts = dm.make_lattice_instances(depth, n, seed)
        gen = gf.make_fair_generator(exact, dm.lattice_enumerate, p, max_depth)
        for inst in insts:
            idict = dm.lattice_inst_dict(inst)
            sel_cache = {}

            def matched_sel(ex):
                target = max(1, int(ex))
                if target not in sel_cache:
                    sres = ap.best_of_k_isobudget(exact, gen, idict, target, tau, seed,
                                                  axis="exec")
                    sel_cache[target] = (_replay_ok(exact, sres, idict),
                                         int(sres.budget.exec))
                return sel_cache[target]

            # A1 freq, no verifier (anchor)
            r1 = dc.savi(exact, gen, idict, K, N, "freq", tau, seed,
                         verifier=False, max_depth=max_depth)
            ok1 = _replay_ok(exact, r1, idict); ex1 = int(r1.budget.exec)
            s1ok, s1ex = matched_sel(ex1)
            sl = _slot("A1_freq"); sl["ok"].append(ok1); sl["a0"].append(s1ok); sl["exec"].append(ex1)
            if s1ex < ex1:
                raise AssertionError(f"exec parity A1 p={p} d={depth} {inst.id}")

            # A4L learned soft value, per model x lambda
            for mseed, model in models.items():
                vfn = model.value_fn()
                for lam in lams:
                    res = ap.savi_value(exact, gen, idict, vfn, lam, K, N, tau, seed,
                                        max_depth=max_depth, edge_mode="freq")
                    ok = _replay_ok(exact, res, idict); ex = int(res.budget.exec)
                    sok, sex = matched_sel(ex)
                    lbl = f"A4L_m{mseed}_lam{lam:g}"
                    sl = _slot(lbl)
                    sl["ok"].append(ok); sl["a0"].append(sok); sl["exec"].append(ex)
                    if sex < ex:
                        raise AssertionError(f"exec parity {lbl} p={p} d={depth} {inst.id}")

    cell = {"p": float(p), "depth": int(depth),
            "g_label": ("inf" if g == math.inf else str(int(g))),
            "n_pooled": len(arms["A1_freq"]["ok"]), "D1": {}, "pass1": {}, "sel_pass1": {}}
    for lbl, d in arms.items():
        delta, lo, hi = paired_bootstrap(d["ok"], d["a0"], n=BOOTSTRAP_N, seed=12345)
        cell["D1"][lbl] = {"delta": float(delta), "lo": float(lo), "hi": float(hi),
                           "n": len(d["ok"])}
        cell["pass1"][lbl] = float(pass_at_1(d["ok"]))
        cell["sel_pass1"][lbl] = float(pass_at_1(d["a0"]))
    return cell


def _csv_f(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _csv_i(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _seeds(s):
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return _csv_i(s)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--p", default="0.5,0.6,0.7,0.9")
    a.add_argument("--gs", default="0", help="merge granularities, e.g. 0,inf")
    a.add_argument("--depths", default="24")
    a.add_argument("--lams", default="0.1,0.5,1,2")
    a.add_argument("--train_seeds", default="100,101,102")
    a.add_argument("--eval_seeds", default="0-7")
    a.add_argument("--n", type=int, default=24)
    a.add_argument("--K", type=int, default=8)
    a.add_argument("--N", type=int, default=16)
    a.add_argument("--tau", type=float, default=1.0)
    a.add_argument("--capacity", default="logreg")
    a.add_argument("--out", required=True)
    args = a.parse_args(argv)

    ps = _csv_f(args.p)
    gs = [math.inf if t.strip() in ("inf", "tree") else int(t)
          for t in args.gs.split(",") if t.strip()]
    depths = _csv_i(args.depths)
    lams = _csv_f(args.lams)
    train_seeds = _csv_i(args.train_seeds)
    eval_seeds = _seeds(args.eval_seeds)
    # train-depths = the eval depths (cover the same horizon)
    models, metrics = _train_models(train_seeds, depths, args.n, args.capacity)
    print("value metrics (disjoint eval):", file=sys.stderr)
    for sd, mt in metrics.items():
        print(f"  m{sd}: acc={mt['acc']:.3f} auc={mt['auc']:.3f} fn={mt['fn_rate']:.3f} "
              f"fp={mt['fp_rate']:.3f} ece={mt['ece']:.3f} midband={mt['p_frac_midband']:.2f}",
              file=sys.stderr)

    cells = []
    for p in ps:
        for d in depths:
            for g in gs:
                gl = "inf" if g == math.inf else int(g)
                print(f"[learned cell] p={p} depth={d} g={gl}", file=sys.stderr)
                cells.append(_run_cell(p, d, g, eval_seeds, args.n, args.K, args.N,
                                       args.tau, models, lams))

    out = {"plan": "merge-noise-phasemap-learned-value",
           "config": {"p": ps, "gs": args.gs, "depths": depths, "lams": lams,
                      "train_seeds": train_seeds, "eval_seeds": eval_seeds, "n": args.n,
                      "K": args.K, "N": args.N, "tau": args.tau, "capacity": args.capacity,
                      "bootstrap_n": BOOTSTRAP_N},
           "value_metrics": {str(k): v for k, v in metrics.items()},
           "cells": cells}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
