"""STAGE A' — real-LM Countdown with a WORKING competence dial (move-degradation).

Temperature failed to move realized competence (run_competence_sweep: comp~0.33 at every
τ). Here competence is dialed directly: each sampled STEP candidate is kept (model move)
with prob `c`, else replaced by a UNIFORM random legal move. This both lowers competence
(c: 1→0 walks toward chance) AND manufactures the step/chain gap — the model's good steps
still appear in the candidate pool (so the verifier can pick them) while complete correct
chains get rare (so selection loses its exact-terminal-check advantage). The SAME degraded
step policy feeds every arm (decode step-mode AND a step-built selection baseline), so the
comparison is fair and token-matched in step units.

Prediction (lattice): D1(soft learned − selection) is negative at high competence (c≈1,
the T1.4 regime) and FLIPS positive as c drops and the step/chain gap opens.

Usage (shard one c per GPU):
  CUDA_VISIBLE_DEVICES=4 python run_degrade_sweep.py --adapter <dec> --k 6 --n 12 \
    --tau 1.0 --cs 1.0 --lams 0.5,1,2 --out outputs/degrade_c1.0.json
"""

import argparse
import json
import random as _random
import time
from pathlib import Path

import core_boot as cb
import domain_countdown_deep as dcd
import instruments as ins
import arms_local
from run_competence_sweep import train_or_load_value, _ok, _mean


def make_degraded_step(real_emit, domain, c, deg_seed):
    """Wrap the real step-sampler: keep each candidate w.p. c, else a uniform legal move."""
    c = float(c)

    def demit(state, N, tau, seed, mode):
        if mode != "step":
            raise ValueError("degraded emit is step-mode only (build chains from steps)")
        cands = list(real_emit(state, N, tau, seed, mode))
        if c >= 1.0 or not cands:
            return cands
        ops = domain.legal_ops(state)
        if not ops:
            return cands
        legal_texts = [domain.render_op(state, op) for op in ops]
        rng = _random.Random((int(seed) * 911 + deg_seed) & 0x7FFFFFFF)
        return [(rng.choice(legal_texts) if rng.random() > c else t) for t in cands]

    return demit


def measure_competence_step(exact, demit, di, md, N, tau, seed):
    """Realized competence on the (degraded) step policy along its own walk."""
    s = exact.initial_state(di)
    feas = legal = sampled = 0
    for d in range(md):
        step_seed = (int(seed) * 100003 + 7000 + d) & 0x7FFFFFFF
        cands = demit(s, N, tau, step_seed, "step")
        if not cands:
            break
        adv = None
        for c in cands:
            sampled += 1
            mv = exact.parse_move(c, s)
            if mv is None:
                continue
            legal += 1
            sp = exact.apply(s, mv)
            if exact.solvable(sp):
                feas += 1
            if adv is None:
                adv = sp
        if adv is None:
            break
        s = adv
    return {"competence": (feas / legal) if legal else float("nan"),
            "parse_rate": (legal / sampled) if sampled else float("nan")}


def step_built_bom(domain, demit, di, target_tokens, tau, seed, md, max_rollouts):
    """Selection baseline built from the SAME degraded step policy (step-mode rollouts).

    Each rollout autoregressively samples 1 degraded step per state to the goal/terminal;
    ok iff ANY rollout reaches the goal. Token-matched (step units) to the soft arm.
    """
    s0 = domain.initial_state(di)
    budget = cb.Budget()
    win_path = None
    nr = 0
    while True:
        rs = (int(seed) * 1000003 + nr) & 0x7FFFFFFF
        cur = s0
        path = []
        for d in range(md):
            if domain.is_goal(cur):
                break
            cands = demit(cur, 1, tau, (rs * 7919 + d) & 0x7FFFFFFF, "step")
            budget.record_sample(cands)
            if not cands:
                break
            mv = domain.parse_move(cands[0], cur)
            if mv is None:
                break
            cur = domain.apply(cur, mv)
            budget.exec += 1
            path.append(mv)
        if win_path is None and domain.is_goal(cur):
            win_path = list(path)
        nr += 1
        if budget.tokens >= target_tokens or nr >= max_rollouts:
            break
    return cb.Result(ok=win_path is not None, path=win_path, budget=budget,
                     detail={"n_rollouts": nr})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--cs", default="1.0,0.6,0.3", help="competence-keep probabilities")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inst_seed", type=int, default=7)
    ap.add_argument("--deg_seed", type=int, default=20260630)
    ap.add_argument("--lams", default="0.5,1,2")
    ap.add_argument("--ref_lam", type=float, default=1.0)
    ap.add_argument("--bom_max_rollouts", type=int, default=80)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--value_cache", default="outputs/value_model.pkl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cs = [float(x) for x in a.cs.split(",")]
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
    real_emit = cb.make_real_sampler(model, tok, exact, a.device)
    log("loaded")

    iset = dcd.make_deep_instances(a.k, a.n, a.inst_seed + a.k)
    md = dcd.max_depth_for_k(a.k)

    cells = []
    for c in cs:
        demit = make_degraded_step(real_emit, exact, c, a.deg_seed)
        t0 = time.time()
        recs = []
        for i, inst in enumerate(iset.instances):
            di = inst.as_dict()
            comp = measure_competence_step(exact, demit, di, md, a.N, a.tau, a.seed)
            arms = {}
            sv_ex = cb.savi(exact, demit, di, K=a.K, N=a.N, edge_mode="freq", tau=a.tau,
                            seed=a.seed, verifier=True, max_depth=md)
            arms["savi_exact"] = {"ok": _ok(exact, di, inst.target, sv_ex),
                                  "tokens": sv_ex.budget.tokens}
            sv0 = cb.savi(exact, demit, di, K=a.K, N=a.N, edge_mode="freq", tau=a.tau,
                          seed=a.seed, verifier=False, max_depth=md)
            arms["savi_l0_freq"] = {"ok": _ok(exact, di, inst.target, sv0),
                                    "tokens": sv0.budget.tokens}
            ref_tokens = None
            for lam in lams:
                svv = arms_local.savi_value(exact, demit, di, V, lam, a.K, a.N, a.tau,
                                            a.seed, md, edge_mode="freq")
                arms[f"savi_value_soft_lam{lam}"] = {
                    "ok": _ok(exact, di, inst.target, svv), "tokens": svv.budget.tokens}
                if lam == a.ref_lam:
                    ref_tokens = svv.budget.tokens
            if ref_tokens is None:
                ref_tokens = arms[f"savi_value_soft_lam{lams[-1]}"]["tokens"]
            bom = step_built_bom(exact, demit, di, ref_tokens, a.tau, a.seed, md,
                                 a.bom_max_rollouts)
            arms["best_of_many"] = {"ok": _ok(exact, di, inst.target, bom),
                                    "tokens": bom.budget.tokens}
            recs.append({"id": inst.id, "headroom": inst.headroom,
                         "competence": comp, "arms": arms})
            log(f"  c={c} {i+1}/{len(iset.instances)} comp={comp['competence']:.2f} "
                f"parse={comp['parse_rate']:.2f} exact={arms['savi_exact']['ok']} "
                f"soft@{a.ref_lam}={arms[f'savi_value_soft_lam{a.ref_lam}']['ok']} "
                f"bom={arms['best_of_many']['ok']} ({time.time()-t0:.0f}s)")

        hr = [r for r in recs if r["headroom"]]
        armnames = list(recs[0]["arms"].keys()) if recs else []
        def flags(arm, rs): return [r["arms"][arm]["ok"] for r in rs]
        bomf = flags("best_of_many", hr)
        soft_arms = [f"savi_value_soft_lam{lam}" for lam in lams]
        best_soft = max(soft_arms, key=lambda arm: _mean(flags(arm, hr))) if hr else soft_arms[0]
        cell = {"k": a.k, "depth": md, "tau": a.tau, "c": c, "n": len(recs),
                "n_headroom": len(hr),
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
                "per_instance": recs}
        cells.append(cell)
        d1 = cell["D1_bestsoft_vs_bom"]
        log(f"c={c}: comp={cell['realized_competence_hr']:.2f} D1(soft-bom)={d1.get('delta')} "
            f"[{d1.get('lo')},{d1.get('hi')}] exactD1={cell['D1_exact_vs_bom'].get('delta')}")

    result = {"plan": "STAGE-A-prime-degradation-dial", "adapter": a.adapter,
              "value_metrics": vmodel["metrics"],
              "config": {"k": a.k, "n": a.n, "K": a.K, "N": a.N, "tau": a.tau, "cs": cs,
                         "lams": lams, "ref_lam": a.ref_lam, "deg_seed": a.deg_seed},
              "cells": cells, "log": logs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=lambda o: None))
    log(f"wrote {a.out}")
    return result


if __name__ == "__main__":
    main()
