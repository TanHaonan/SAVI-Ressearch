"""STAGE A — real-LM Countdown competence-flip test (does the lattice flip reproduce?).

The lattice phase map says: a soft LEARNED imperfect value beats exec/token-matched
selection ONLY at LOW emission-competence (frequency stops tracking correctness), and is
redundant/inert at high competence. T1.4 found the learned value LOSES on Countdown at
τ=1.0 (−0.58) — the prediction is that τ=1.0 is the HIGH-competence regime, and raising τ
(flattening the policy until its frequencies no longer track feasibility) should flip the
sign on a REAL model.

This sweeps temperature τ as the competence dial, MEASURES realized competence via the
exact `reachable` oracle (so each cell is placed on the competence axis), and reports
D1 = savi_value(soft, best-λ) − best_of_many(iso-token) per (τ, k). Ground truth (pass@1,
oracle ceiling) stays exact; only the decode value is the learned imperfect V.

Usage (shard one τ per GPU):
  CUDA_VISIBLE_DEVICES=4 python run_competence_sweep.py --adapter <dec> --ks 6 --n 12 \
    --taus 1.0 --lams 0.5,1,2 --out outputs/comp_tau1.0.json
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import core_boot as cb
import domain_countdown_deep as dcd
import instruments as ins
import arms_local
import value_model as vm


def _ok(domain, di, target, r):
    return int(bool(cb._result_pass1(domain, di, target, r)))


def _mean(xs):
    xs = [x for x in xs if x == x]
    return (sum(xs) / len(xs)) if xs else float("nan")


def train_or_load_value(cache_path, log):
    p = Path(cache_path)
    if p.is_file():
        log(f"loading cached value model {p}")
        with open(p, "rb") as f:
            return pickle.load(f)
    log("training value model (disjoint seeds)...")
    Xtr, ytr, _ = vm.make_dataset(seeds=range(1000, 1030), ks=(4, 5, 6), n_per=60,
                                  max_states=8, seed=1)
    Xte, yte, _ = vm.make_dataset(seeds=range(2000, 2010), ks=(4, 5, 6), n_per=60,
                                  max_states=8, seed=2)
    m = vm.ValueModel(capacity="mlp", seed=0).fit(Xtr, ytr)
    metrics = vm.evaluate(m, Xte, yte)
    log(f"value model: {metrics}")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump({"model": m, "metrics": metrics}, f)
    return {"model": m, "metrics": metrics}


def measure_competence(exact, emit, di, md, N, tau, seed):
    """Realized competence along the model's OWN sampled trajectory at temperature τ.

    Walk from s0 for up to `md` steps; at each visited state sample N step-candidates,
    and over the PARSEABLE (legal) ones compute the fraction whose successor stays
    solvable (the Countdown analog of the lattice good-move weight `p`). Advance by the
    first parseable move (model-driven, so the walk stays on the policy's distribution).
    Returns realized competence `p`, parse_rate, and counts.
    """
    s = exact.initial_state(di)
    feas = legal = sampled = 0
    for d in range(md):
        step_seed = (int(seed) * 100003 + 7000 + d) & 0x7FFFFFFF
        cands = emit(s, N, tau, step_seed, "step")
        if not cands:
            break
        advance = None
        for c in cands:
            sampled += 1
            mv = exact.parse_move(c, s)
            if mv is None:
                continue
            legal += 1
            sp = exact.apply(s, mv)
            if exact.solvable(sp):
                feas += 1
            if advance is None:
                advance = sp
        if advance is None:
            break
        s = advance
    return {"competence": (feas / legal) if legal else float("nan"),
            "parse_rate": (legal / sampled) if sampled else float("nan"),
            "n_legal": legal}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", default="6")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--taus", default="1.0,1.5,2.0,2.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inst_seed", type=int, default=7)
    ap.add_argument("--lams", default="0.5,1,2")
    ap.add_argument("--ref_lam", type=float, default=1.0)
    ap.add_argument("--bom_max_rollouts", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--value_cache", default="outputs/value_model.pkl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ks = [int(x) for x in a.ks.split(",")]
    taus = [float(x) for x in a.taus.split(",")]
    lams = [float(x) for x in a.lams.split(",")]

    logs = []
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logs.append(line)

    vmodel = train_or_load_value(a.value_cache, log)
    V = vmodel["model"].value_fn()
    log(f"value metrics: {vmodel['metrics']}")

    exact = dcd.DeepCountdownDomain()
    log(f"loading {a.adapter}")
    model, tok = cb.load_countdown_model(a.adapter, device=a.device)
    emit = cb.make_real_sampler(model, tok, exact, a.device)
    log("loaded")

    cells = []
    for k in ks:
        iset = dcd.make_deep_instances(k, a.n, a.inst_seed + k)
        md = dcd.max_depth_for_k(k)
        for tau in taus:
            t0 = time.time()
            recs = []
            for i, inst in enumerate(iset.instances):
                di = inst.as_dict()
                comp = measure_competence(exact, emit, di, md, a.N, tau, a.seed)
                arms = {}
                sv_ex = cb.savi(exact, emit, di, K=a.K, N=a.N, edge_mode="freq", tau=tau,
                                seed=a.seed, verifier=True, max_depth=md)
                arms["savi_exact"] = {"ok": _ok(exact, di, inst.target, sv_ex),
                                      "tokens": sv_ex.budget.tokens}
                sv0 = cb.savi(exact, emit, di, K=a.K, N=a.N, edge_mode="freq", tau=tau,
                              seed=a.seed, verifier=False, max_depth=md)
                arms["savi_l0_freq"] = {"ok": _ok(exact, di, inst.target, sv0),
                                        "tokens": sv0.budget.tokens}
                ref_tokens = None
                for lam in lams:
                    svv = arms_local.savi_value(exact, emit, di, V, lam, a.K, a.N, tau,
                                                a.seed, md, edge_mode="freq")
                    arms[f"savi_value_soft_lam{lam}"] = {
                        "ok": _ok(exact, di, inst.target, svv), "tokens": svv.budget.tokens}
                    if lam == a.ref_lam:
                        ref_tokens = svv.budget.tokens
                if ref_tokens is None:
                    ref_tokens = arms[f"savi_value_soft_lam{lams[-1]}"]["tokens"]
                bom = arms_local.best_of_k_isobudget(exact, emit, di, target=ref_tokens,
                                                     tau=tau, seed=a.seed, axis="tokens",
                                                     max_rollouts=a.bom_max_rollouts)
                arms["best_of_many"] = {"ok": _ok(exact, di, inst.target, bom),
                                        "tokens": bom.budget.tokens}
                recs.append({"id": inst.id, "headroom": inst.headroom,
                             "competence": comp, "arms": arms})
                log(f"  k={k} tau={tau} {i+1}/{len(iset.instances)} "
                    f"comp={comp['competence']:.2f} parse={comp['parse_rate']:.2f} "
                    f"exact={arms['savi_exact']['ok']} soft@{a.ref_lam}="
                    f"{arms[f'savi_value_soft_lam{a.ref_lam}']['ok']} "
                    f"bom={arms['best_of_many']['ok']} ({time.time()-t0:.0f}s)")

            hr = [r for r in recs if r["headroom"]]
            armnames = list(recs[0]["arms"].keys()) if recs else []
            def flags(arm, rs): return [r["arms"][arm]["ok"] for r in rs]
            bomf = flags("best_of_many", hr)
            # best-λ soft among the swept λ (by headroom pass@1)
            soft_arms = [f"savi_value_soft_lam{lam}" for lam in lams]
            best_soft = max(soft_arms, key=lambda arm: _mean(flags(arm, hr))) if hr else soft_arms[0]
            cell = {"k": k, "depth": md, "tau": tau, "n": len(recs), "n_headroom": len(hr),
                    "realized_competence": _mean([r["competence"]["competence"] for r in recs]),
                    "realized_competence_hr": _mean([r["competence"]["competence"] for r in hr]),
                    "parse_rate": _mean([r["competence"]["parse_rate"] for r in recs]),
                    "pass1_hr": {arm: _mean(flags(arm, hr)) for arm in armnames},
                    "tokens_mean": {arm: _mean([r["arms"][arm]["tokens"] for r in recs])
                                    for arm in armnames},
                    "best_soft_arm": best_soft,
                    "D1_bestsoft_vs_bom": ins.paired_delta(flags(best_soft, hr), bomf,
                                                           n_boot=a.n_boot, seed=a.seed),
                    "D1_exact_vs_bom": ins.paired_delta(flags("savi_exact", hr), bomf,
                                                        n_boot=a.n_boot, seed=a.seed),
                    "D1_freq_vs_bom": ins.paired_delta(flags("savi_l0_freq", hr), bomf,
                                                       n_boot=a.n_boot, seed=a.seed),
                    "per_instance": recs}
            cells.append(cell)
            d1 = cell["D1_bestsoft_vs_bom"]
            log(f"k={k} tau={tau}: comp={cell['realized_competence_hr']:.2f} "
                f"bestsoft={best_soft} D1(soft-bom)={d1.get('delta')} "
                f"[{d1.get('lo')},{d1.get('hi')}] exactD1={cell['D1_exact_vs_bom'].get('delta')}")

    result = {"plan": "STAGE-A-competence-flip-realLM", "adapter": a.adapter,
              "value_metrics": vmodel["metrics"],
              "config": {"ks": ks, "n": a.n, "K": a.K, "N": a.N, "taus": taus,
                         "lams": lams, "ref_lam": a.ref_lam, "inst_seed": a.inst_seed},
              "cells": cells, "log": logs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=lambda o: None))
    log(f"wrote {a.out}")
    return result


if __name__ == "__main__":
    main()
