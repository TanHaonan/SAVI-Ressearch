"""STAGE A'' — verifier-quality diagnostic on the forced-open gap (why did learned V fail?).

Stage A' showed: degrading the real policy opens large headroom (exact verifier captures
+0.83) but the LEARNED value (fn=0.18) stays at the floor. Two hypotheses: (a) fn=0.18 is
simply too high to capture this headroom (a RATE problem -> Stage B needs a low-fn PRM); or
(b) the learned value is fine on rate but bad at DECODE-TIME discrimination / off the
degraded-walk distribution (a DISCRIMINATION problem -> deeper).

Test: at a fixed forced-open gap (degraded emit, c small), build a SOFT verifier from the
EXACT oracle CORRUPTED at a controlled rate ε (NoisyVerifierDomain, fn mode) and sweep ε.
Compare D1(corrupted-soft − selection) across ε to D1(learned-soft − selection) at the same
nominal fn (0.18). If the controlled ε=0.18 verifier captures the headroom but the learned
fn=0.18 does not, the learned value's failure is discrimination/OOD, not rate — and the ε at
which the controlled verifier stops capturing the headroom is the verifier-quality bar a
real PRM must clear.

Usage (shard one c per GPU):
  CUDA_VISIBLE_DEVICES=4 python run_verifier_quality.py --adapter <dec> --k 6 --n 12 \
    --c 0.3 --eps 0,0.05,0.1,0.18,0.3,0.5 --lams 0.5,1 --out outputs/vq_c0.3.json
"""

import argparse
import json
import time
from pathlib import Path

import core_boot as cb
import domain_countdown_deep as dcd
import instruments as ins
import arms_local
from run_competence_sweep import train_or_load_value, _ok, _mean
from run_degrade_sweep import make_degraded_step, step_built_bom, measure_competence_step


def corrupted_soft_value(epsilon, noise_seed, v_hi=0.95, v_lo=0.05):
    """A SOFT value from the exact oracle corrupted at rate ε (fn mode, deterministic per
    canon). Same shape as the lattice's binary value, but the corruption is on the EXACT
    Countdown reachability — a controlled-fn verifier (vs the learned one)."""
    noisy = dcd.NoisyVerifierDomain(epsilon=float(epsilon), mode="fn", noise_seed=noise_seed)
    return lambda state: (v_hi if noisy.solvable(state) else v_lo)


def _best_soft(domain, emit, di, target_ok, V, lams, K, N, tau, seed, md):
    """Run savi_value over λ for value_fn V; return (best_ok, best_tokens, best_lam)."""
    best = (-1, None, None)
    for lam in lams:
        r = arms_local.savi_value(domain, emit, di, V, lam, K, N, tau, seed, md,
                                  edge_mode="freq")
        ok = _ok(domain, di, target_ok, r)
        if ok > best[0]:
            best = (ok, r.budget.tokens, lam)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=0.3, help="competence-keep prob (the gap)")
    ap.add_argument("--eps", default="0,0.05,0.1,0.18,0.3,0.5")
    ap.add_argument("--lams", default="0.5,1")
    ap.add_argument("--ref_lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inst_seed", type=int, default=7)
    ap.add_argument("--deg_seed", type=int, default=20260630)
    ap.add_argument("--noise_seed", type=int, default=12345)
    ap.add_argument("--bom_max_rollouts", type=int, default=80)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--value_cache", default="outputs/value_model.pkl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    eps_list = [float(x) for x in a.eps.split(",")]
    lams = [float(x) for x in a.lams.split(",")]

    logs = []
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logs.append(line)

    vmodel = train_or_load_value(a.value_cache, log)
    Vlearned = vmodel["model"].value_fn()
    log(f"value metrics: {vmodel['metrics']}")
    # controlled-ε corrupted-exact soft verifiers (one per ε)
    Vcorr = {e: corrupted_soft_value(e, a.noise_seed) for e in eps_list}

    exact = dcd.DeepCountdownDomain()
    log(f"loading {a.adapter}")
    model, tok = cb.load_countdown_model(a.adapter, device=a.device)
    real_emit = cb.make_real_sampler(model, tok, exact, a.device)
    demit = make_degraded_step(real_emit, exact, a.c, a.deg_seed)
    log(f"loaded; gap c={a.c}")

    iset = dcd.make_deep_instances(a.k, a.n, a.inst_seed + a.k)
    md = dcd.max_depth_for_k(a.k)

    recs = []
    t0 = time.time()
    for i, inst in enumerate(iset.instances):
        di = inst.as_dict(); tgt = inst.target
        comp = measure_competence_step(exact, demit, di, md, a.N, a.tau, a.seed)
        arms = {}
        # exact hard-mask ceiling
        sx = cb.savi(exact, demit, di, K=a.K, N=a.N, edge_mode="freq", tau=a.tau,
                     seed=a.seed, verifier=True, max_depth=md)
        arms["savi_exact"] = {"ok": _ok(exact, di, tgt, sx), "tokens": sx.budget.tokens}
        # learned soft (best λ)
        lok, ltok, llam = _best_soft(exact, demit, di, tgt, Vlearned, lams, a.K, a.N,
                                     a.tau, a.seed, md)
        arms["soft_learned"] = {"ok": lok, "tokens": ltok, "lam": llam}
        # controlled-ε corrupted-exact soft (best λ) per ε
        ref_tokens = None
        for e in eps_list:
            cok, ctok, clam = _best_soft(exact, demit, di, tgt, Vcorr[e], lams, a.K, a.N,
                                         a.tau, a.seed, md)
            arms[f"soft_corr_eps{e}"] = {"ok": cok, "tokens": ctok, "lam": clam}
            if e == 0.0:
                ref_tokens = ctok
        if ref_tokens is None:
            ref_tokens = ltok
        # step-built selection, token-matched to the ε=0 soft budget
        bom = step_built_bom(exact, demit, di, ref_tokens, a.tau, a.seed, md,
                             a.bom_max_rollouts)
        arms["best_of_many"] = {"ok": _ok(exact, di, tgt, bom), "tokens": bom.budget.tokens}
        recs.append({"id": inst.id, "headroom": inst.headroom, "competence": comp,
                     "arms": arms})
        log(f"  {i+1}/{len(iset.instances)} comp={comp['competence']:.2f} "
            f"exact={arms['savi_exact']['ok']} learned={lok} "
            f"corr@.18={arms['soft_corr_eps0.18']['ok']} bom={arms['best_of_many']['ok']} "
            f"({time.time()-t0:.0f}s)")

    hr = [r for r in recs if r["headroom"]]
    def flags(arm): return [r["arms"][arm]["ok"] for r in hr]
    bomf = flags("best_of_many")
    armnames = list(recs[0]["arms"].keys()) if recs else []
    D1 = {arm: ins.paired_delta(flags(arm), bomf, n_boot=a.n_boot, seed=a.seed)
          for arm in armnames if arm != "best_of_many"}
    out = {"plan": "STAGE-A2-verifier-quality", "adapter": a.adapter,
           "value_metrics": vmodel["metrics"],
           "config": {"k": a.k, "n": a.n, "K": a.K, "N": a.N, "tau": a.tau, "c": a.c,
                      "eps": eps_list, "lams": lams, "deg_seed": a.deg_seed,
                      "noise_seed": a.noise_seed},
           "n_headroom": len(hr),
           "realized_competence_hr": _mean([r["competence"]["competence"] for r in hr]),
           "pass1_hr": {arm: _mean(flags(arm)) for arm in armnames},
           "D1_vs_bom": D1, "per_instance": recs, "log": logs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=lambda o: None))
    log(f"wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
