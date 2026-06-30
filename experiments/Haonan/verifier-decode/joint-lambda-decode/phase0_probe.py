"""Phase 0 generalization probe (GPU): does the trained model keep live branching at k=6?

PLAN3 §3b / §13.3. The decoupled/coupled Countdown adapters were SFT'd on k=4; k=6 is
out-of-distribution. Before the Tier-B decisive run, this probe loads a real adapter and,
on SOLVABLE k in {4,5,6} states, measures:

  * parse rate      — fraction of emitted step texts that parse to a LEGAL move,
  * root K_eff      — distinct canonical successors among N step samples at the START
                      state (cheap; one generate per instance), and
  * in-trellis K_eff — distinct canon(s') per node over the nodes the headline arm
                      actually expands (``instruments.measure_keff``, freq + verifier).

The DISAMBIGUATION (PLAN3 §13.3): measure K_eff at k=4 (in-distribution) AND k=6 (OOD).

  K_eff healthy at k=6                  -> proceed straight to Tier B at depth.
  K_eff healthy at k=4 but collapses k=6 -> OOD (fixable): short LoRA SFT on a k in
                                           {4,5,6} mixture, then Tier B.
  K_eff ~ 1 even at k=4                  -> the genuine marginal-vs-path kill criterion
                                           (H3 fails for the right reason): redirect to
                                           the emission objective, not the decoder.

Usage (one GPU per adapter; run decoupled and coupled in parallel):
    CUDA_VISIBLE_DEVICES=4 python phase0_probe.py --adapter <decoupled_dir> \
        --profile decoupled --ks 4,5,6 --n 10 --N 16 --tau 1.0 --device cuda \
        --out outputs/phase0_decoupled.json
"""

import argparse
import json
import time
from pathlib import Path

import core_boot as cb
import domain_countdown_deep as dcd
import instruments as ins


def root_keff_and_parse(domain, emit, di, N, tau, seed):
    """Root-state distinct-canon K_eff + parse/legality rate from one step sample."""
    s0 = domain.initial_state(di)
    cands = emit(s0, N, tau, seed, "step")
    canons = set()
    n_parsed = 0
    for c in cands:
        mv = domain.parse_move(c, s0)
        if mv is None:
            continue
        n_parsed += 1
        canons.add(domain.canon(domain.apply(s0, mv)))
    return {
        "root_keff": len(canons),
        "parse_rate": (n_parsed / len(cands)) if cands else float("nan"),
        "n_emitted": len(cands),
        "sample_texts": cands[:8],   # a peek for the log
    }


def probe_k(domain, emit, k, n, N, tau, seed, inst_seed, target_range, K, log):
    """Probe one k: generate n solvable instances, measure root + in-trellis K_eff."""
    iset = dcd.make_deep_instances(k, n, inst_seed + k, target_range=target_range)
    max_depth = dcd.max_depth_for_k(k)
    recs = []
    t0 = time.time()
    for i, inst in enumerate(iset.instances):
        di = inst.as_dict()
        root = root_keff_and_parse(domain, emit, di, N, tau, seed)
        keff = ins.measure_keff(domain, emit, di, K, N, tau, seed,
                                edge_mode="freq", verifier=True, max_depth=max_depth)
        recs.append({"id": inst.id, "root": root, "intrellis": keff})
        log(f"    k={k} {i+1}/{len(iset.instances)} root_keff={root['root_keff']} "
            f"parse={root['parse_rate']:.3f} intrellis_keff={keff['keff_parsed_mean']:.2f} "
            f"({time.time()-t0:.0f}s)")

    def mean(xs):
        xs = [x for x in xs if x == x]
        return (sum(xs) / len(xs)) if xs else float("nan")

    return {
        "k": k, "depth": max_depth, "n": len(recs),
        "root_keff_mean": mean([r["root"]["root_keff"] for r in recs]),
        "root_parse_rate": mean([r["root"]["parse_rate"] for r in recs]),
        "intrellis_keff_parsed_mean": mean([r["intrellis"]["keff_parsed_mean"] for r in recs]),
        "intrellis_keff_feasible_mean": mean([r["intrellis"]["keff_feasible_mean"] for r in recs]),
        "intrellis_parse_rate": mean([r["intrellis"]["parse_rate"] for r in recs]),
        "per_instance": recs,
    }


def decide(by_k, keff_healthy=1.5, keff_collapse=1.2):
    """Apply the §13.3 decision rule from the per-k root K_eff (in-distribution k=4 vs
    OOD k=6). Uses root K_eff as the headline branching signal (cheapest, cleanest)."""
    k4 = by_k.get(4, {}).get("root_keff_mean")
    k6 = by_k.get(6, {}).get("root_keff_mean")
    if k4 is None:
        return {"decision": "indeterminate", "reason": "no k=4 measurement"}
    if k4 <= keff_collapse:
        return {"decision": "KILL_marginal_vs_path",
                "reason": f"K_eff~1 even at k=4 ({k4:.2f}) -> redirect to emission objective",
                "k4": k4, "k6": k6}
    if k6 is None:
        return {"decision": "proceed_tierB_k4_only",
                "reason": f"k=4 healthy ({k4:.2f}); no k=6 probe run", "k4": k4}
    if k6 >= keff_healthy:
        return {"decision": "PROCEED_tierB_at_depth",
                "reason": f"K_eff healthy at k=6 ({k6:.2f}) -> Tier B at depth",
                "k4": k4, "k6": k6}
    return {"decision": "OOD_run_kmix_SFT_then_tierB",
            "reason": f"k=4 healthy ({k4:.2f}) but k=6 collapses ({k6:.2f}) -> short "
                      "LoRA SFT on k in {4,5,6} mixture, then Tier B",
            "k4": k4, "k6": k6}


def build_argparser():
    ap = argparse.ArgumentParser(description="PLAN3 Phase 0 generalization probe")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--profile", choices=["decoupled", "coupled"], required=True)
    ap.add_argument("--ks", default="4,5,6")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inst_seed", type=int, default=7)
    ap.add_argument("--target_lo", type=int, default=10)
    ap.add_argument("--target_hi", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    ks = [int(x) for x in args.ks.split(",")]
    logs = []
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logs.append(line)

    log(f"Phase0 probe profile={args.profile} adapter={args.adapter} ks={ks} "
        f"n={args.n} N={args.N} device={args.device}")
    domain = dcd.DeepCountdownDomain()
    log("loading model...")
    model, tok = cb.load_countdown_model(args.adapter, device=args.device)
    emit = cb.make_real_sampler(model, tok, domain, args.device)
    log("model loaded.")

    by_k = {}
    for k in ks:
        log(f"=== probing k={k} (depth {dcd.max_depth_for_k(k)}) ===")
        by_k[k] = probe_k(domain, emit, k, args.n, args.N, args.tau, args.seed,
                          args.inst_seed, (args.target_lo, args.target_hi), args.K, log)
        log(f"  k={k}: root_keff={by_k[k]['root_keff_mean']:.2f} "
            f"parse={by_k[k]['root_parse_rate']:.3f} "
            f"intrellis_keff={by_k[k]['intrellis_keff_parsed_mean']:.2f}")

    decision = decide(by_k)
    log(f"DECISION ({args.profile}): {decision['decision']} — {decision['reason']}")

    result = {"plan": "PLAN3-phase0-probe", "profile": args.profile,
              "adapter": args.adapter,
              "config": {"ks": ks, "n": args.n, "N": args.N, "K": args.K,
                         "tau": args.tau, "seed": args.seed,
                         "target_range": [args.target_lo, args.target_hi]},
              "by_k": by_k, "decision": decision, "log": logs}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2, default=lambda o: None))
    log(f"wrote {outp}")
    return result


if __name__ == "__main__":
    main()
