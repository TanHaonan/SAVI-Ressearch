"""PLAN4 T1.2 — does the λ>0 win survive a NON-exact verifier? (efficient sweep)

best_of_many never uses the mask, so it is ε-invariant: we compute savi(exact) + the
iso-token best_of_many ONCE per instance, then re-run only savi under a NoisyVerifierDomain
at each (ε, mode). H1(ε) = paired Δ(savi_noisy − best_of_many) on the headroom stratum.

best_of_many is matched to the EXACT-mask savi token budget (the intended compute);
savi_noisy's own token spend is logged so any drift is visible (fn-noise prunes more →
savi under-spends = conservative; fp-noise prunes less → over-spends). is_goal + the
oracle ceiling stay EXACT — only the decode mask is corrupted.

Usage:
  CUDA_VISIBLE_DEVICES=g python run_noise_sweep.py --adapter <dec> --ks 6 --n 16 \
    --eps_list 0,0.05,0.1,0.2,0.3 --modes sym,fp,fn --out outputs/t12_noise.json
"""

import argparse
import json
import time
from pathlib import Path

import core_boot as cb
import domain_countdown_deep as dcd
import instruments as ins
import arms_local


def _ok(domain, di, target, result):
    return int(bool(cb._result_pass1(domain, di, target, result)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", default="6")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inst_seed", type=int, default=7)
    ap.add_argument("--eps_list", default="0,0.05,0.1,0.2,0.3")
    ap.add_argument("--modes", default="sym,fp,fn")
    ap.add_argument("--bom_max_rollouts", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ks = [int(x) for x in a.ks.split(",")]
    eps_list = [float(x) for x in a.eps_list.split(",")]
    modes = a.modes.split(",")

    logs = []
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logs.append(line)

    exact = dcd.DeepCountdownDomain()
    log(f"loading {a.adapter}")
    model, tok = cb.load_countdown_model(a.adapter, device=a.device)
    emit = cb.make_real_sampler(model, tok, exact, a.device)
    log("loaded")

    # One NoisyVerifierDomain per (eps,mode), reused across instances (shares its memo).
    noisy_domains = {}
    for mode in modes:
        for eps in eps_list:
            if eps == 0.0:
                continue
            noisy_domains[(mode, eps)] = dcd.NoisyVerifierDomain(epsilon=eps, mode=mode)

    out = {}
    for k in ks:
        iset = dcd.make_deep_instances(k, a.n, a.inst_seed + k)
        md = dcd.max_depth_for_k(k)
        recs = []
        t0 = time.time()
        for i, inst in enumerate(iset.instances):
            di = inst.as_dict()
            sv0 = cb.savi(exact, emit, di, K=a.K, N=a.N, edge_mode="freq", tau=a.tau,
                          seed=a.seed, verifier=True, max_depth=md)
            T = sv0.budget.tokens
            bom = arms_local.best_of_k_isobudget(exact, emit, di, target=T, tau=a.tau,
                                                 seed=a.seed, axis="tokens",
                                                 max_rollouts=a.bom_max_rollouts)
            rec = {"id": inst.id, "headroom": inst.headroom,
                   "savi_exact_ok": _ok(exact, di, inst.target, sv0),
                   "savi_exact_tokens": T,
                   "bom_ok": _ok(exact, di, inst.target, bom),
                   "noisy": {}}
            for (mode, eps), nd in noisy_domains.items():
                svn = cb.savi(nd, emit, di, K=a.K, N=a.N, edge_mode="freq", tau=a.tau,
                              seed=a.seed, verifier=True, max_depth=md)
                rec["noisy"][f"{mode}:{eps}"] = {"ok": _ok(exact, di, inst.target, svn),
                                                 "tokens": svn.budget.tokens}
            recs.append(rec)
            log(f"  k={k} {i+1}/{len(iset.instances)} savi0={rec['savi_exact_ok']} "
                f"bom={rec['bom_ok']} ({time.time()-t0:.0f}s)")

        hr = [r for r in recs if r["headroom"]]
        bom_flags = [r["bom_ok"] for r in hr]
        agg = {"k": k, "depth": md, "n": len(recs), "n_headroom": len(hr),
               "savi_exact_pass1_hr": _mean([r["savi_exact_ok"] for r in hr]),
               "bom_pass1_hr": _mean(bom_flags),
               "H1_exact": ins.paired_delta([r["savi_exact_ok"] for r in hr], bom_flags,
                                            n_boot=a.n_boot, seed=a.seed),
               "by_noise": {}}
        for key in [f"{m}:{e}" for m in modes for e in eps_list if e != 0.0]:
            sv_flags = [r["noisy"][key]["ok"] for r in hr]
            agg["by_noise"][key] = {
                "savi_pass1_hr": _mean(sv_flags),
                "savi_tokens_mean": _mean([r["noisy"][key]["tokens"] for r in hr]),
                "H1": ins.paired_delta(sv_flags, bom_flags, n_boot=a.n_boot, seed=a.seed),
            }
        agg["per_instance"] = recs
        out[k] = agg
        log(f"k={k}: exact H1={agg['H1_exact']['delta']:+.3f} | "
            + " ".join(f"{key}={agg['by_noise'][key]['H1']['delta']:+.3f}"
                       for key in agg["by_noise"]))

    result = {"plan": "PLAN4-T1.2-noise", "adapter": a.adapter,
              "config": {"ks": ks, "n": a.n, "K": a.K, "N": a.N, "tau": a.tau,
                         "eps_list": eps_list, "modes": modes}, "by_k": out, "log": logs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=lambda o: None))
    log(f"wrote {a.out}")
    return result


def _mean(xs):
    xs = [x for x in xs if x == x]
    return (sum(xs) / len(xs)) if xs else float("nan")


if __name__ == "__main__":
    main()
