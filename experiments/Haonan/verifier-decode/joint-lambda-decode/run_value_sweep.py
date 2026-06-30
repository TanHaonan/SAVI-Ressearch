"""PLAN4 decisive — learned value + soft λ: does ANY edge survive an imperfect verifier?

Replaces the EXACT mask with a learned, imperfect value V (calibrated, fn≈0.18, fp≈0.05;
see value_model). Runs, iso-token on the headroom stratum, the ladder:

  best_of_many   — unmasked selection (baseline)
  savi λ=0 freq  — emission-only floor (loses)
  savi_value SOFT, learned V, λ ∈ sweep — graded verifier (down-weights, never deletes)
  savi_value HARD, learned V (thresh 0.5) — the 'learned hard mask' (~T1.2 fn=0.18 point)
  savi EXACT mask — idealized ceiling (the PLAN3 headline arm)

Two questions: (1) does soft-λ learned-V beat best_of_many for some λ (edge survives an
imperfect value)? (2) is SOFT > HARD for the same learned V (soft weighting rescues the
false-negative fragility that made the hard mask lose in T1.2)?

best_of_many is matched to the soft savi_value token budget (no hard pruning → full-depth
sampling, ~constant across λ). V is trained once (cached) on disjoint seeds.

Usage:
  CUDA_VISIBLE_DEVICES=g python run_value_sweep.py --adapter <dec> --ks 6 --n 12 \
    --lams 0,0.5,1,2,4,8 --out outputs/t14_value_k6.json
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", default="6")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inst_seed", type=int, default=7)
    ap.add_argument("--lams", default="0,0.5,1,2,4,8")
    ap.add_argument("--ref_lam", type=float, default=2.0, help="lam whose tokens set the bom budget")
    ap.add_argument("--hard_thresh", type=float, default=0.5)
    ap.add_argument("--bom_max_rollouts", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--value_cache", default="outputs/value_model.pkl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ks = [int(x) for x in a.ks.split(",")]
    lams = [float(x) for x in a.lams.split(",")]

    logs = []
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logs.append(line)

    vmodel = train_or_load_value(a.value_cache, log)
    V = vmodel["model"].value_fn()

    exact = dcd.DeepCountdownDomain()
    log(f"loading {a.adapter}")
    model, tok = cb.load_countdown_model(a.adapter, device=a.device)
    emit = cb.make_real_sampler(model, tok, exact, a.device)
    log("loaded")

    out = {}
    for k in ks:
        iset = dcd.make_deep_instances(k, a.n, a.inst_seed + k)
        md = dcd.max_depth_for_k(k)
        recs = []
        t0 = time.time()
        for i, inst in enumerate(iset.instances):
            di = inst.as_dict()
            rec = {"id": inst.id, "headroom": inst.headroom, "arms": {}}
            # idealized ceiling: exact hard mask
            sv_ex = cb.savi(exact, emit, di, K=a.K, N=a.N, edge_mode="freq", tau=a.tau,
                            seed=a.seed, verifier=True, max_depth=md)
            rec["arms"]["savi_exact"] = {"ok": _ok(exact, di, inst.target, sv_ex),
                                         "tokens": sv_ex.budget.tokens}
            # emission floor
            sv0 = cb.savi(exact, emit, di, K=a.K, N=a.N, edge_mode="freq", tau=a.tau,
                          seed=a.seed, verifier=False, max_depth=md)
            rec["arms"]["savi_l0_freq"] = {"ok": _ok(exact, di, inst.target, sv0),
                                           "tokens": sv0.budget.tokens}
            # soft-lambda learned value
            ref_tokens = None
            for lam in lams:
                svv = arms_local.savi_value(exact, emit, di, V, lam, a.K, a.N, a.tau,
                                            a.seed, md, edge_mode="freq")
                rec["arms"][f"savi_value_soft_lam{lam}"] = {
                    "ok": _ok(exact, di, inst.target, svv), "tokens": svv.budget.tokens}
                if lam == a.ref_lam:
                    ref_tokens = svv.budget.tokens
            if ref_tokens is None:
                ref_tokens = rec["arms"][f"savi_value_soft_lam{lams[-1]}"]["tokens"]
            # learned HARD mask
            svh = arms_local.savi_value(exact, emit, di, V, 0.0, a.K, a.N, a.tau, a.seed,
                                        md, edge_mode="freq", hard_thresh=a.hard_thresh)
            rec["arms"]["savi_value_hard"] = {"ok": _ok(exact, di, inst.target, svh),
                                              "tokens": svh.budget.tokens}
            # iso-token baseline matched to the soft (ref_lam) budget
            bom = arms_local.best_of_k_isobudget(exact, emit, di, target=ref_tokens,
                                                 tau=a.tau, seed=a.seed, axis="tokens",
                                                 max_rollouts=a.bom_max_rollouts)
            rec["arms"]["best_of_many"] = {"ok": _ok(exact, di, inst.target, bom),
                                           "tokens": bom.budget.tokens}
            recs.append(rec)
            log(f"  k={k} {i+1}/{len(iset.instances)} exact={rec['arms']['savi_exact']['ok']} "
                f"soft@{a.ref_lam}={rec['arms'][f'savi_value_soft_lam{a.ref_lam}']['ok']} "
                f"hard={rec['arms']['savi_value_hard']['ok']} bom={rec['arms']['best_of_many']['ok']} "
                f"({time.time()-t0:.0f}s)")

        hr = [r for r in recs if r["headroom"]]
        arms = list(recs[0]["arms"].keys()) if recs else []
        def flags(arm): return [r["arms"][arm]["ok"] for r in hr]
        bomf = flags("best_of_many")
        agg = {"k": k, "depth": md, "n": len(recs), "n_headroom": len(hr),
               "pass1_hr": {arm: _mean(flags(arm)) for arm in arms},
               "tokens_mean": {arm: _mean([r["arms"][arm]["tokens"] for r in recs]) for arm in arms},
               "H1_vs_best_of_many": {arm: ins.paired_delta(flags(arm), bomf,
                                      n_boot=a.n_boot, seed=a.seed) for arm in arms},
               "per_instance": recs}
        out[k] = agg
        log(f"k={k}: " + " ".join(f"{arm.replace('savi_value_soft_','sv')}={agg['pass1_hr'][arm]:.2f}"
                                  for arm in arms))

    result = {"plan": "PLAN4-T1.4-learned-value-softlambda", "adapter": a.adapter,
              "value_metrics": vmodel["metrics"],
              "config": {"ks": ks, "n": a.n, "K": a.K, "N": a.N, "tau": a.tau,
                         "lams": lams, "ref_lam": a.ref_lam, "hard_thresh": a.hard_thresh},
              "by_k": out, "log": logs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=lambda o: None))
    log(f"wrote {a.out}")
    return result


def _mean(xs):
    xs = [x for x in xs if x == x]
    return (sum(xs) / len(xs)) if xs else float("nan")


if __name__ == "__main__":
    main()
