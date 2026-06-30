"""PLAN3 joint harness: arm matrix + iso-token + instruments + gates, both tiers.

One entry point for both tiers (the only thing that changes is the ``sample`` backend):

  Tier A (CPU mock, prereg freeze):
    python run_joint.py --tier A --ks 4,5,6 --n 30 --K 8 --N 16 --tau 1.0 \
        --seed 0 --abstain 0.0 --out outputs/tierA.json

  Tier B (real model, decisive; GPU):
    CUDA_VISIBLE_DEVICES=g python run_joint.py --tier B \
        --adapter_decoupled <dir> --adapter_coupled <dir> --device cuda \
        --ks 4,5,6 --n 30 --K 8 --N 16 --tau 1.0 --seed 0 --out outputs/tierB.json

For EACH backend profile (decoupled = headline, coupled = H3 contrast/necessity control)
it runs the §3 arm matrix per instance, on the SAME deep instance sets and SAME domain
object (so the persistent reachability memo and the token ledger are shared), then
computes the instruments (in-trellis K_eff, depth curve D1_token, freq-edge calibration
ECE, Phi-merge ratio + ablation), the validity gates (V1-V3, Phi-works), and the
hypothesis deltas (H1-H5) with paired bootstrap + Holm across {H1,H4}. Results -> JSON.
"""

import argparse
import json
import time
from pathlib import Path

import core_boot as cb
import domain_countdown_deep as dcd
import instruments as ins
import arms_local


# ---------------------------------------------------------------------------
# Per-instance arm matrix on one backend
# ---------------------------------------------------------------------------

def _pass1(domain, di, target, result):
    return int(bool(cb._result_pass1(domain, di, target, result)))


def self_consistency(domain, backend, di, target, N, tau, seed):
    """SC floor (= lambda=0 yardstick): N chains, majority canonical terminal == goal.

    ok iff the modal canonical terminal state is a goal. Token budget recorded. NOT on
    the iso-token headline axis; a control floor.
    """
    s0 = domain.initial_state(di)
    cands = backend(s0, N, tau, seed, "chain")
    budget = cb.Budget()
    budget.record_sample(cands)
    terminals = {}
    for text in cands:
        moves = domain.parse_chain(text, s0)
        if moves is None:
            continue
        cur = s0
        for mv in moves:
            cur = domain.apply(cur, mv)
            budget.exec += 1
        key = domain.canon(cur)
        terminals[key] = terminals.get(key, 0) + 1
    ok = 0
    if terminals:
        modal_key = max(terminals, key=lambda k: terminals[k])
        # modal_key is (sorted values, target); goal iff single value == target.
        vals, tgt = modal_key
        ok = int(len(vals) == 1 and vals[0] == tgt)
    return {"ok": ok, "tokens": budget.tokens, "exec": budget.exec}


def run_arm_matrix(domain, backend, di, target, K, N, tau, seed, max_depth,
                   bom_max_rollouts=1000000, with_masked_bom=False):
    """Run the full §3 arm matrix for ONE instance on ONE backend. Returns a dict of
    per-arm {ok, tokens, exec, ...} plus the savi(lambda>0,freq) Result (for instruments).
    """
    out = {}

    # Floors.
    g = cb.greedy(domain, backend, di, seed)
    out["greedy"] = {"ok": _pass1(domain, di, target, g),
                     "tokens": g.budget.tokens, "exec": g.budget.exec}

    out["self_consistency"] = self_consistency(domain, backend, di, target, N, tau, seed)

    # savi lambda=0 (support + freq) — the emission-only floors (replicate the loss).
    sv_l0_sup = cb.savi(domain, backend, di, K=K, N=N, edge_mode="support", tau=tau,
                        seed=seed, verifier=False, max_depth=max_depth)
    out["savi_l0_support"] = {"ok": _pass1(domain, di, target, sv_l0_sup),
                              "tokens": sv_l0_sup.budget.tokens, "exec": sv_l0_sup.budget.exec}

    sv_l0_freq = cb.savi(domain, backend, di, K=K, N=N, edge_mode="freq", tau=tau,
                         seed=seed, verifier=False, max_depth=max_depth)
    out["savi_l0_freq"] = {"ok": _pass1(domain, di, target, sv_l0_freq),
                           "tokens": sv_l0_freq.budget.tokens, "exec": sv_l0_freq.budget.exec}

    # THE HEADLINE ARM: savi lambda>0 freq (freq edge + exact verifier mask).
    sv_l1_freq = cb.savi(domain, backend, di, K=K, N=N, edge_mode="freq", tau=tau,
                         seed=seed, verifier=True, max_depth=max_depth)
    out["savi_l1_freq"] = {"ok": _pass1(domain, di, target, sv_l1_freq),
                           "tokens": sv_l1_freq.budget.tokens, "exec": sv_l1_freq.budget.exec,
                           "merge_ratio": ins.merge_ratio_from_result(sv_l1_freq)}

    # HEADLINE BASELINE: best_of_many at the SAVI(lambda>0) TOKEN budget (iso-token).
    bom = arms_local.best_of_k_isobudget(domain, backend, di,
                                         target=sv_l1_freq.budget.tokens, tau=tau,
                                         seed=seed, axis="tokens",
                                         max_rollouts=bom_max_rollouts)
    out["best_of_many"] = {"ok": _pass1(domain, di, target, bom),
                           "tokens": bom.budget.tokens, "exec": bom.budget.exec,
                           "n_rollouts": bom.detail.get("n_rollouts"),
                           "token_target": sv_l1_freq.budget.tokens,
                           "token_ratio": (bom.budget.tokens / sv_l1_freq.budget.tokens
                                           if sv_l1_freq.budget.tokens else float("nan"))}

    # Phi-ablation (H4 / E6c): beam_no_merge at lambda>0 freq, same K/N budget.
    bnm = arms_local.beam_no_merge(domain, backend, di, K=K, N=N, tau=tau, seed=seed,
                                   verifier=True, max_depth=max_depth, edge_mode="freq")
    out["beam_no_merge_l1_freq"] = {"ok": _pass1(domain, di, target, bnm),
                                    "tokens": bnm.budget.tokens, "exec": bnm.budget.exec}

    # PLAN4 T1.1: masked best-of-many (verifier-in-loop selection, no beam/merge), matched
    # to savi(lambda>0)'s token budget. Isolates the per-step mask from the global decode.
    if with_masked_bom:
        mbom = arms_local.masked_best_of_many(domain, backend, di,
                                              target=sv_l1_freq.budget.tokens, N=N, tau=tau,
                                              seed=seed, max_depth=max_depth, axis="tokens",
                                              max_rollouts=bom_max_rollouts)
        out["masked_best_of_many"] = {"ok": _pass1(domain, di, target, mbom),
                                      "tokens": mbom.budget.tokens, "exec": mbom.budget.exec,
                                      "n_rollouts": mbom.detail.get("n_rollouts")}

    # Oracle ceiling (= absolute solvability; 1 on the solvable set). Use the EXACT oracle
    # (uncorrupted) even when a noisy-verifier domain corrupts the decode mask.
    exact_solv = getattr(domain, "exact_solvable", domain.solvable)
    out["oracle"] = {"ok": int(bool(exact_solv(domain.initial_state(di)))),
                     "tokens": 0, "exec": 1}

    return out, sv_l1_freq


# ---------------------------------------------------------------------------
# A backend profile over the whole depth curve
# ---------------------------------------------------------------------------

ARMS = ["greedy", "self_consistency", "savi_l0_support", "savi_l0_freq",
        "savi_l1_freq", "best_of_many", "beam_no_merge_l1_freq", "oracle"]


def run_profile(domain, backend, curve, cfg, log):
    """Run the arm matrix + instruments for one backend over every k in the curve.

    ``curve`` is {k: DeepInstanceSet}. Returns a per-k dict of per-instance records,
    aggregates, and instrument outputs. The domain (with its persistent reach memo) is
    shared across k so the memo warms across the whole curve.
    """
    K, N, tau, seed = cfg["K"], cfg["N"], cfg["tau"], cfg["seed"]
    keff_n = cfg.get("keff_n", 0) or len(next(iter(curve.values())).instances)
    bom_cap = cfg.get("bom_max_rollouts", 0) or 1000000
    prof = backend.profile if hasattr(backend, "profile") else "?"
    out = {}
    for k in sorted(curve):
        iset = curve[k]
        max_depth = dcd.max_depth_for_k(k)
        per_inst = []
        t0 = time.time()
        for i, inst in enumerate(iset.instances):
            di = inst.as_dict()
            arms, sv_l1 = run_arm_matrix(domain, backend, di, inst.target,
                                         K, N, tau, seed, max_depth,
                                         bom_max_rollouts=bom_cap,
                                         with_masked_bom=cfg.get("with_masked_bom", False))
            # Instruments (K_eff + freq-edge calibration) from a SINGLE combined walk,
            # only on the first keff_n instances (population estimates, not paired).
            if i < keff_n:
                keff = ins.measure_keff(domain, backend, di, K, N, tau, seed,
                                        edge_mode="freq", verifier=True,
                                        max_depth=max_depth, with_calibration=True)
                calib = keff.pop("calib_records", [])
            else:
                keff, calib = None, []
            per_inst.append({
                "id": inst.id, "headroom": inst.headroom, "depth": inst.depth,
                "arms": arms, "keff": keff, "calib_records": calib,
            })
            log(f"    [{prof}] k={k} {i+1}/{len(iset.instances)} "
                f"savi_l1={arms['savi_l1_freq']['ok']} bom={arms['best_of_many']['ok']} "
                f"(savi_tok={arms['savi_l1_freq']['tokens']} "
                f"bom_tok={arms['best_of_many']['tokens']} "
                f"r={arms['best_of_many']['n_rollouts']}) "
                f"{time.time()-t0:.0f}s memo={domain.memo_stats()['size']}")
        out[k] = _aggregate_k(k, iset, per_inst, cfg)
        log(f"  k={k}: {_one_line_summary(out[k])}")
    return out


def _aggregate_k(k, iset, per_inst, cfg):
    """Aggregate per-instance records for one k into pass@1, deltas, gates, instruments."""
    headroom = [r for r in per_inst if r["headroom"]]
    arms_present = list(per_inst[0]["arms"].keys()) if per_inst else ARMS

    def flags(records, arm):
        return [r["arms"][arm]["ok"] for r in records]

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")

    pass1_all = {arm: mean(flags(per_inst, arm)) for arm in arms_present}
    pass1_hr = {arm: mean(flags(headroom, arm)) for arm in arms_present}
    tokens_mean = {arm: mean([r["arms"][arm]["tokens"] for r in per_inst]) for arm in arms_present}

    # H1: savi(lambda>0,freq) - best_of_many at iso-token, on headroom stratum.
    h1 = ins.paired_delta(flags(headroom, "savi_l1_freq"),
                          flags(headroom, "best_of_many"),
                          n_boot=cfg["n_boot"], seed=cfg["seed"])
    # H4 / E6c: savi - beam_no_merge at lambda>0 freq, on headroom stratum.
    h4 = ins.paired_delta(flags(headroom, "savi_l1_freq"),
                          flags(headroom, "beam_no_merge_l1_freq"),
                          n_boot=cfg["n_boot"], seed=cfg["seed"])

    # PLAN4 T1.1 decomposition deltas (present only when masked_best_of_many was run).
    decomp = {}
    if "masked_best_of_many" in arms_present:
        decomp["masked_minus_best_of_many"] = ins.paired_delta(
            flags(headroom, "masked_best_of_many"), flags(headroom, "best_of_many"),
            n_boot=cfg["n_boot"], seed=cfg["seed"])
        decomp["savi_minus_masked"] = ins.paired_delta(
            flags(headroom, "savi_l1_freq"), flags(headroom, "masked_best_of_many"),
            n_boot=cfg["n_boot"], seed=cfg["seed"])

    # K_eff (H3) aggregated over the instrumented subset (keff is None elsewhere).
    instr = [r for r in per_inst if r["keff"] is not None and r["keff"]["n_nodes"] > 0]
    keff_parsed = mean([r["keff"]["keff_parsed_mean"] for r in instr])
    keff_feasible = mean([r["keff"]["keff_feasible_mean"] for r in instr])
    parse_rate = mean([r["keff"]["parse_rate"] for r in instr])

    # Calibration (H5): pool all (p, label) records for this k.
    calib_records = [rec for r in per_inst for rec in r["calib_records"]]
    ece = ins.reliability_ece(calib_records, n_bins=10)

    # Phi-merge ratio (mean over instances with a defined ratio).
    mratios = [r["arms"]["savi_l1_freq"]["merge_ratio"] for r in per_inst
               if r["arms"]["savi_l1_freq"]["merge_ratio"] == r["arms"]["savi_l1_freq"]["merge_ratio"]]
    merge_ratio = mean(mratios)

    return {
        "k": k, "depth": iset.depth,
        "n_instances": len(per_inst), "n_headroom": len(headroom),
        "headroom_ids": list(iset.headroom_ids),
        "pass1_all": pass1_all, "pass1_headroom": pass1_hr, "tokens_mean": tokens_mean,
        "H1_savi_vs_best_of_many_isotoken": h1,
        "H4_savi_vs_beam_no_merge": h4,
        "T1_decomposition": decomp,
        "keff_parsed_mean": keff_parsed, "keff_feasible_mean": keff_feasible,
        "parse_rate": parse_rate,
        "H5_freq_edge_ece": ece,
        "merge_ratio": merge_ratio,
        "n_instrumented": len(instr),
        # per-instance kept compact (drop bulky per-node + calib records from the dump).
        "per_instance": [{"id": r["id"], "headroom": r["headroom"], "arms": r["arms"],
                          "keff": (None if r["keff"] is None
                                   else {kk: vv for kk, vv in r["keff"].items()
                                         if kk != "per_node_parsed"})}
                         for r in per_inst],
    }


def _one_line_summary(agg):
    p = agg["pass1_headroom"]
    h1 = agg["H1_savi_vs_best_of_many_isotoken"]
    return (f"savi_l1={p['savi_l1_freq']:.3f} best_of_many={p['best_of_many']:.3f} "
            f"H1={h1['delta']:+.3f}[{h1['lo']:+.3f},{h1['hi']:+.3f}] "
            f"Keff={agg['keff_parsed_mean']:.2f} merge={agg['merge_ratio']:.1f} "
            f"ECE={agg['H5_freq_edge_ece']['ece']:.3f}")


# ---------------------------------------------------------------------------
# Cross-profile synthesis: gates + depth curve + decoupled-necessity (H2,H3)
# ---------------------------------------------------------------------------

def synthesize(profiles, cfg):
    """Combine decoupled (+ coupled) profile results into gates + H2/H3 verdicts."""
    dec = profiles["decoupled"]
    cou = profiles.get("coupled")
    ks = sorted(dec)

    # H2 depth curve: D1_token(k) = H1 delta at each k (decoupled).
    depth_curve = {k: dec[k]["H1_savi_vs_best_of_many_isotoken"] for k in ks}
    deltas = [depth_curve[k]["delta"] for k in ks]
    increasing = all(deltas[i] <= deltas[i + 1] + 1e-9 for i in range(len(deltas) - 1))
    deepest = ks[-1]
    shallow = ks[0]
    h2 = {
        "deltas_by_k": {k: depth_curve[k]["delta"] for k in ks},
        "ci_by_k": {k: [depth_curve[k]["lo"], depth_curve[k]["hi"]] for k in ks},
        "strictly_increasing": increasing,
        "shallow_k": shallow, "shallow_delta_le_0": depth_curve[shallow]["delta"] <= 0,
        "deepest_k": deepest,
        "deepest_delta_gt_0_ci_excl_0": (depth_curve[deepest]["lo"] > 0),
    }

    # H3: K_eff(decoupled) > K_eff(coupled) and > 1; H1 holds decoupled, not coupled.
    h3 = {"keff_decoupled_by_k": {k: dec[k]["keff_parsed_mean"] for k in ks}}
    if cou is not None:
        h3["keff_coupled_by_k"] = {k: cou[k]["keff_parsed_mean"] for k in ks}
        h3["decoupled_gt_coupled"] = all(
            dec[k]["keff_parsed_mean"] > cou[k]["keff_parsed_mean"] for k in ks)
        h3["decoupled_gt_1"] = all(dec[k]["keff_parsed_mean"] > 1.0 for k in ks)
        h3["h1_holds_decoupled_deepest"] = dec[deepest]["H1_savi_vs_best_of_many_isotoken"]["lo"] > 0
        h3["h1_fails_coupled_deepest"] = not (cou[deepest]["H1_savi_vs_best_of_many_isotoken"]["lo"] > 0)

    # Validity gates.
    gates = {}
    gates["V1_headroom_nonempty"] = all(dec[k]["n_headroom"] > 0 for k in ks)
    if cou is not None:
        gates["V2_keff_decoupled_gt_coupled_and_gt1"] = (
            all(dec[k]["keff_parsed_mean"] > cou[k]["keff_parsed_mean"] for k in ks)
            and all(dec[k]["keff_parsed_mean"] > 1.0 for k in ks))
    gates["V3_oracle_gt_best_of_many_isotoken"] = all(
        dec[k]["pass1_all"]["oracle"] > dec[k]["pass1_all"]["best_of_many"] for k in ks)
    gates["Phi_works_merge_ratio_gt1"] = all(dec[k]["merge_ratio"] > 1.0 for k in ks)

    # H1 at the deepest tier (the prereg headline) + H4 pooled across k.
    h1_deepest = dec[deepest]["H1_savi_vs_best_of_many_isotoken"]
    h4_by_k = {k: dec[k]["H4_savi_vs_beam_no_merge"] for k in ks}

    # Holm across {H1 (deepest), H4 (deepest)} using a CI-excludes-0 pseudo decision.
    holm = {
        "H1_deepest_reject": h1_deepest["lo"] > 0,
        "H4_deepest_reject": h4_by_k[deepest]["lo"] > 0,
    }

    return {
        "ks": ks,
        "H1_deepest": h1_deepest,
        "H2_depth_curve": h2,
        "H3_keff": h3,
        "H4_by_k": h4_by_k,
        "H5_ece_by_k": {k: dec[k]["H5_freq_edge_ece"]["ece"] for k in ks},
        "gates": gates,
        "holm_family": holm,
    }


# ---------------------------------------------------------------------------
# Backends per tier
# ---------------------------------------------------------------------------

def build_backends(args, log):
    """Return {profile: sample_closure} for the chosen tier."""
    if args.tier == "A":
        import mock_backend
        log(f"Tier A: CPU mock backends (decoupled_kind={args.decoupled_kind}, "
            f"p={args.competence_p}, abstain={args.abstain}).")
        return mock_backend.make_tier_a_backends(
            abstain=args.abstain, decoupled_kind=args.decoupled_kind,
            competence_p=args.competence_p)
    # Tier B: real model per adapter (GPU). Imports torch lazily inside load.
    backends = {}
    domain_for_sampler = dcd.DeepCountdownDomain()  # only render/parse used by sampler
    if args.adapter_decoupled:
        log(f"Tier B: loading decoupled adapter {args.adapter_decoupled} on {args.device}")
        m, t = cb.load_countdown_model(args.adapter_decoupled, device=args.device)
        backends["decoupled"] = _tag(cb.make_real_sampler(m, t, domain_for_sampler, args.device),
                                     "decoupled")
    if args.adapter_coupled:
        log(f"Tier B: loading coupled adapter {args.adapter_coupled} on {args.device}")
        m, t = cb.load_countdown_model(args.adapter_coupled, device=args.device)
        backends["coupled"] = _tag(cb.make_real_sampler(m, t, domain_for_sampler, args.device),
                                   "coupled")
    if not backends:
        raise SystemExit("Tier B needs --adapter_decoupled and/or --adapter_coupled")
    return backends


def _tag(closure, profile):
    closure.profile = profile
    closure.tier = "B-real"
    return closure


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description="PLAN3 joint lambda>0 decode harness")
    ap.add_argument("--tier", choices=["A", "B"], required=True)
    ap.add_argument("--ks", default="4,5,6", help="comma list of k (numbers) -> depth k-1")
    ap.add_argument("--n", type=int, default=30, help="instances per k (solvable)")
    ap.add_argument("--K", type=int, default=8, help="beam width")
    ap.add_argument("--N", type=int, default=16, help="step samples per node")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--only_profile", choices=["decoupled", "coupled"], default=None,
                    help="run a single backend profile (for GPU sharding); merge later")
    ap.add_argument("--bom_max_rollouts", type=int, default=0,
                    help="cap best_of_many rollouts (0 = unbounded); token_ratio logged")
    ap.add_argument("--keff_n", type=int, default=0,
                    help="instrument only the first keff_n instances per k (0 = all)")
    # PLAN4 extras
    ap.add_argument("--with_masked_bom", action="store_true",
                    help="T1.1: add masked_best_of_many (verifier-in-loop selection)")
    ap.add_argument("--noise_eps", type=float, default=0.0,
                    help="T1.2: corrupt the decode mask at rate eps (0 = exact)")
    ap.add_argument("--noise_mode", choices=["sym", "fp", "fn"], default="sym",
                    help="T1.2: symmetric | false-positive-only | false-negative-only")
    ap.add_argument("--noise_seed", type=int, default=12345)
    ap.add_argument("--inst_seed", type=int, default=7, help="instance-gen base seed")
    ap.add_argument("--target_lo", type=int, default=10)
    ap.add_argument("--target_hi", type=int, default=100)
    # Tier A
    ap.add_argument("--abstain", type=float, default=0.0)
    ap.add_argument("--decoupled_kind", choices=["competence", "oraclechain"],
                    default="competence",
                    help="competence (faithful P_p, no oracle inject) | oraclechain (M1 mock)")
    ap.add_argument("--competence_p", type=float, default=0.7,
                    help="per-step good-move mass for the competence mock")
    # Tier B
    ap.add_argument("--adapter_decoupled", default=None)
    ap.add_argument("--adapter_coupled", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    ks = [int(x) for x in args.ks.split(",")]
    cfg = {"K": args.K, "N": args.N, "tau": args.tau, "seed": args.seed,
           "n_boot": args.n_boot, "keff_n": args.keff_n,
           "bom_max_rollouts": args.bom_max_rollouts,
           "with_masked_bom": args.with_masked_bom}

    def make_domain():
        if args.noise_eps > 0.0:
            return dcd.NoisyVerifierDomain(epsilon=args.noise_eps, mode=args.noise_mode,
                                           noise_seed=args.noise_seed)
        return dcd.DeepCountdownDomain()

    logs = []
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logs.append(line)

    log(f"PLAN3 run_joint tier={args.tier} ks={ks} n={args.n} K={args.K} N={args.N} "
        f"tau={args.tau} seed={args.seed}")

    backends = build_backends(args, log)
    if args.only_profile:
        backends = {p: b for p, b in backends.items() if p == args.only_profile}
        if not backends:
            raise SystemExit(f"--only_profile {args.only_profile} but that backend was "
                             "not built (missing adapter?)")
        log(f"sharding: running only profile={args.only_profile}")
    profiles = {}
    for profile, backend in backends.items():
        log(f"=== profile: {profile} ===")
        # Fresh domain per profile so the reach memo is shared within a profile's curve
        # (and instruments see a consistent memo). Instance sets are identical across
        # profiles (same seeds) so pairing is valid. Under --noise_eps this is a
        # NoisyVerifierDomain (corrupted decode mask; exact is_goal + ceiling).
        domain = make_domain()
        curve = dcd.make_depth_curve(ks, args.n, args.inst_seed,
                                     target_range=(args.target_lo, args.target_hi))
        profiles[profile] = run_profile(domain, backend, curve, cfg, log)
        log(f"  {profile} memo final: {domain.memo_stats()}")

    synthesis = synthesize(profiles, cfg) if "decoupled" in profiles else {}

    result = {
        "plan": "PLAN3-joint-lambda-decode",
        "tier": args.tier,
        "config": {"ks": ks, "n": args.n, "K": args.K, "N": args.N, "tau": args.tau,
                   "seed": args.seed, "n_boot": args.n_boot, "inst_seed": args.inst_seed,
                   "target_range": [args.target_lo, args.target_hi],
                   "abstain": args.abstain if args.tier == "A" else None,
                   "adapter_decoupled": args.adapter_decoupled,
                   "adapter_coupled": args.adapter_coupled,
                   "with_masked_bom": args.with_masked_bom,
                   "noise_eps": args.noise_eps, "noise_mode": args.noise_mode,
                   "competence_p": args.competence_p if args.tier == "A" else None},
        "profiles": profiles,
        "synthesis": synthesis,
        "log": logs,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2, default=_json_default))
    log(f"wrote {outp}")
    if synthesis:
        log("SYNTHESIS gates: " + json.dumps(synthesis["gates"]))
        log("H1_deepest: " + json.dumps(synthesis["H1_deepest"]))
        log("H2 depth curve: " + json.dumps(synthesis["H2_depth_curve"]["deltas_by_k"]))
    return result


def _json_default(o):
    try:
        import math
        if isinstance(o, float) and math.isnan(o):
            return None
    except Exception:
        pass
    return str(o)


if __name__ == "__main__":
    main()
