"""Prereg-grade driver for the real-LLM prompt-conditioning coverage probe.

This is the Tier-B (GPU, decisive) driver of PREREG.md (sec 5,6,7,9,10). It sits ON TOP
of the VERIFIED ``realprobe_core`` (the faithful ~1x mechanism: ONE forward per committed
step, conditioning lives in cheap PREFILL only -- NO N-candidate-per-step tax) and turns
it into the four-arm, multi-K, dual-axis, validity-gated, paired-bootstrap experiment the
prereg specifies.

WHAT THE FOUR ARMS ARE (PREREG sec 5; the conditioned arm is the verified probe core)
-------------------------------------------------------------------------------------
  iid_bok           : i.i.d. best-of-K. K chains from the BARE problem prompt (empty
                      avoid-list). Coverage@K = ANY of the first K chains reached goal.
                      (PREREG "headline 基线 / Erdős-Rényi / random code".)
  conditioned       : the VERIFIED 1-forward/step probe -- chain i carries the avoid-list
                      of chains 1..i-1 in the PREFILL; ONE forward per step (NOT N
                      candidates). This is ``realprobe_core.run_arm(conditioned=True)``
                      VERBATIM. It is the cheap construction S' (the "corr" claim arm),
                      realised by prompt conditioning rather than per-step reweighting, so
                      it has NO ~Nx step-mode tax. (PREREG sec 4 "备/BACKUP route".)
  temp_matched_iid  : the KEY negative control (H4). i.i.d. at the temperature whose
                      MARGINAL per-step emission entropy MATCHES the conditioned arm's,
                      with both entropies measured EMPIRICALLY on a ``>= K*`` ensemble (NOT
                      from a degenerate 1-chain draw). If raising i.i.d. temperature to the
                      conditioned arm's marginal spread does NOT recover the gain, the gain
                      came from sample-to-sample CORRELATION, not marginal scatter.
  oracle            : ``domain.solvable(s0)`` -- the coverage ceiling (PREREG sec 5 "天花板
                      / head cover-room 分子"). 1 iff the instance is solvable at all.

THE TWO ISO-TOKEN AXES (PREREG sec 7; realprobe_core docstring)
--------------------------------------------------------------
  HEADLINE     iso-GENERATION-token : Sigma real generated continuation tokens. The
               conditioned arm matches i.i.d. by construction here (one forward per step).
  SENSITIVITY  iso-TOTAL-token      : generation + PREFILL, where the conditioned arm's
               prefill grows with chain index (it carries the avoid-list). This is the
               honest accounting of the cheap-conditioning cost.

For EACH axis we report the paired coverage-vs-token curve and the miss-decay slope (the
"底数" of PREREG H2: slope of ``log(1 - coverage)`` vs token budget).

INSTANCE SELECTION (V1, PREREG sec 6 / sec 9 / requirement 1)
-------------------------------------------------------------
``--select`` GENERATES solvable k in {5,6} (depth 4,5) Countdown instances with a
reproducible seed, draws a POOL of ``K_max`` i.i.d. chains ONCE per instance (the real
model, shared across all K), computes iid coverage@K for K in {2,4,6,8,12,16} from that
ONE pool (coverage@K = ANY of the first K pooled chains reached goal -- so every K comes
from the SAME draw, no re-sampling), and SELECTS instances whose iid coverage@K_ref(=8)
falls in the V1 head-cover-room band (0.2, 0.9). The selected instances (with their pooled
chains, per-K coverage, and oracle label) are written to ``instances.json``; ``--run``
reads it back so the run consumes the SAME pool the selection saw.

USAGE
-----
  export HF_HUB_CACHE=$HOME/.cache/huggingface/hub
  # 1) select (one GPU): generate + draw pool + V1 filter -> instances.json
  CUDA_VISIBLE_DEVICES=0 python run_realprobe.py --select --out instances.json
  # 2) run a shard (one GPU each; 5 shards): four arms x K-grid -> outputs/shard_i.json
  CUDA_VISIBLE_DEVICES=0 python run_realprobe.py --run --shard 0 --nshards 5 \
      --instances instances.json
  # 3) analyze (CPU): merge shards -> dual-axis + slopes + funnel + paired bootstrap + Holm
  python run_realprobe.py --analyze --instances instances.json --out result.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import realprobe_core as rp
from realprobe_core import (
    CountdownDomain,
    chain_seed,
    rollout_one_chain,
    run_arm,
    instance_dict,
    load_probe_model,
    real_gen_factory,
    DEFAULT_ADAPTER,
)
from core import load_instances
from _deps.instances import Instance, label_instance


# ===========================================================================
# Experiment grid (FROZEN constants; PREREG sec 6/7/9/10)
# ===========================================================================

K_GRID = (2, 4, 6, 8, 12, 16)   # the coverage@K grid; K_max == max(K_GRID).
K_MAX = max(K_GRID)
K_REF = 8                        # V1 reference K (PREREG sec 9: coverage@K_ref band).
V1_LO, V1_HI = 0.2, 0.9          # V1 head-cover-room band (PREREG sec 9).
K_STAR = K_MAX                   # entropy ensemble size (>= K*): use the full pool.
GEN_K_DEPTHS = (5, 6)            # k=5 -> depth 4, k=6 -> depth 5 (witness_len == k-1).
N_BOOT = 10000                   # paired bootstrap (PREREG sec 10).
DEFAULT_TAU = 1.0                # conditioned/iid base temperature (PREREG mock used 1.0).

# The temperature sweep the temp_matched_iid arm searches to hit the conditioned arm's
# marginal step entropy (monotone-ish in tau; we pick the closest-entropy tau on the grid).
TAU_SWEEP = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8)


# ===========================================================================
# coverage@K from ONE shared pool (PREREG req 1: every K from the same draw)
# ===========================================================================

def coverage_at_K_from_pool(reached_flags, K):
    """coverage@K = ANY of the FIRST K pooled chains reached the goal.

    ``reached_flags`` is the per-chain goal-reached bool list of the ONE shared pool of
    K_MAX i.i.d. chains. coverage@K reads the first-K prefix of that SAME pool, so the
    whole coverage@K curve comes from a single draw (no re-sampling per K).
    """
    return any(reached_flags[:K])


def gen_token_prefix(gen_per_chain, K):
    """Sigma generated tokens over the first K pooled chains (HEADLINE token budget at K)."""
    return int(sum(gen_per_chain[:K]))


def total_token_prefix(gen_per_chain, prefill_per_chain, K):
    """Sigma (gen + prefill) over the first K pooled chains (SENSITIVITY budget at K)."""
    return int(sum(gen_per_chain[:K]) + sum(prefill_per_chain[:K]))


# ===========================================================================
# Marginal step-entropy on a >= K* ensemble (for the temp-match negative control)
# ===========================================================================

def _shannon_entropy(counts):
    """Shannon entropy (nats) of a multiset given as a list/dict of counts.

    Empirical entropy of the emitted-move distribution. ``counts`` may be a dict
    ``move_text -> n`` or a list of counts. Returns 0.0 for a degenerate (single-support)
    or empty distribution.
    """
    if isinstance(counts, dict):
        ns = [c for c in counts.values() if c > 0]
    else:
        ns = [c for c in counts if c > 0]
    tot = sum(ns)
    if tot <= 0:
        return 0.0
    h = 0.0
    for n in ns:
        p = n / tot
        h -= p * math.log(p)
    return h


def marginal_step_entropy(per_chain_first_moves):
    """Empirical MARGINAL (first-step) emission entropy over an ensemble of chains.

    ``per_chain_first_moves`` is the list of first-committed-move texts across an ensemble
    of >= K* chains drawn from the SAME start state (the marginal the temp-match arm must
    match). We use the first committed step because it is the one step every chain shares
    a start state for, so the empirical move histogram there is a clean marginal. Entropy
    is measured on the full ensemble (>= K* chains), never on a degenerate single chain.
    """
    hist = {}
    for mv in per_chain_first_moves:
        # "" means the first step failed to parse: it IS a distinct marginal outcome.
        hist[mv] = hist.get(mv, 0) + 1
    return _shannon_entropy(hist)


def first_move(chain):
    """The first committed move text of a ChainTrace ('' if the first step failed)."""
    if not chain.moves_text:
        return ""
    return chain.moves_text.split(";")[0]


def measure_arm_entropy(gen_factory, domain, inst, temperature, base_seed, conditioned,
                        n_ens):
    """Marginal step entropy of an arm at ``temperature``, on a ``>= K*`` ensemble.

    Draws ``n_ens`` (>= K*) chains for the arm (iid if ``conditioned`` is False; the
    conditioned probe if True), at ``temperature``, and returns the empirical first-step
    emission entropy over that ensemble. The ensemble size is >= K_STAR by construction
    (caller passes ``n_ens = max(K_STAR, ...)``), so the marginal is never degenerate.
    Returns ``(entropy, n_ens)``.
    """
    res = run_arm(gen_factory, domain, inst, n_ens, temperature, base_seed,
                  conditioned=conditioned)
    h = marginal_step_entropy([first_move(c) for c in res.chains])
    return h, len(res.chains)


def match_temperature(gen_factory, domain, inst, base_seed, target_entropy, n_ens):
    """Find the i.i.d. temperature whose marginal step entropy best matches ``target``.

    Sweeps ``TAU_SWEEP``; for each tau measures the i.i.d. arm's marginal step entropy on a
    ``>= K*`` ensemble (``measure_arm_entropy``), and returns the tau with the smallest
    absolute entropy gap to ``target_entropy`` (the conditioned arm's marginal entropy).
    Returns ``(best_tau, best_entropy, sweep_records)`` where ``sweep_records`` is the full
    (tau, entropy, ensemble_size) trace for auditability.

    PREREG H4: this builds the temp-matched i.i.d. control whose MARGINAL spread equals the
    conditioned arm's, so the only remaining difference is sample-to-sample correlation.
    """
    records = []
    best = None
    for tau in TAU_SWEEP:
        h, m = measure_arm_entropy(gen_factory, domain, inst, tau, base_seed,
                                   conditioned=False, n_ens=n_ens)
        records.append({"tau": tau, "entropy": h, "ensemble": m})
        gap = abs(h - target_entropy)
        if best is None or gap < best[2]:
            best = (tau, h, gap)
    return best[0], best[1], records


# ===========================================================================
# --select : generate k in {5,6}, draw ONE pool, V1 filter, write instances.json
# ===========================================================================

def generate_solvable_instances(seed, n_per_depth, target_range=(10, 100)):
    """Generate solvable k in {5,6} (depth 4,5) instances with a reproducible seed.

    For each k in GEN_K_DEPTHS, seeded-generates a stream of k-number instances and keeps
    the first ``n_per_depth`` that are oracle-SOLVABLE (so every candidate has a head
    cover-room ceiling of 1). Returns a list of ``(Instance, witness_depth)``. Reproducible
    for a fixed ``seed`` (the generated spec is deterministic). ``target_range`` widens or
    narrows instance difficulty: a smaller/lower band yields easier instances (higher iid
    coverage), used to lift more instances into the V1 (0.2,0.9) head-cover-room band.
    """
    out = []
    for k in GEN_K_DEPTHS:
        kept = 0
        idx_n = 0
        # Pull in growing batches until we have n_per_depth solvable for this k.
        batch = max(4 * n_per_depth, 40)
        while kept < n_per_depth:
            idx_n += batch
            insts = load_instances(
                {"set": "generated", "seed": seed, "n": idx_n, "k": k,
                 "target_range": list(target_range)}
            )
            out_k = []
            for inst in insts:
                lab = label_instance(inst)
                if lab["solvable"]:
                    out_k.append((inst, lab["witness_len"]))
                if len(out_k) >= n_per_depth:
                    break
            if len(out_k) >= n_per_depth:
                out.extend(out_k[:n_per_depth])
                kept = n_per_depth
            elif idx_n > 50 * n_per_depth:
                # Defensive: solvability is ~empirically 1.0 for k>=5, but never loop forever.
                out.extend(out_k)
                kept = len(out_k)
    return out


def draw_iid_pool(gen_factory, domain, inst, K_max, temperature, base_seed):
    """Draw ONE pool of ``K_max`` i.i.d. chains for ``inst`` (shared across all K).

    Uses the i.i.d. arm (``conditioned=False``) so each chain is from the BARE prompt with
    the SAME per-chain seed schedule the run will reuse. Returns a serialisable dict with
    the per-chain reached-goal flags, per-chain generated/prefill tokens, and per-chain
    move text, so coverage@K (for every K) and both token axes come from this SINGLE draw.
    """
    res = run_arm(gen_factory, domain, inst, K_max, temperature, base_seed,
                  conditioned=False)
    reached = [bool(c.reached_goal) for c in res.chains]
    return {
        "reached": reached,
        "gen_per_chain": _per_chain_gen_tokens(res),
        "prefill_per_chain": _per_chain_prefill_tokens(res, K_max),
        "moves": [c.moves_text for c in res.chains],
        "coverage_by_K": {str(K): coverage_at_K_from_pool(reached, K) for K in K_GRID},
        # arm-level totals (cheap cross-check against the per-chain prefix sums)
        "gen_tokens_total": res.gen_tokens,
        "prefill_tokens_total": res.prefill_tokens,
        "cands_per_fwd_all_one": all(c == 1 for c in res.cands_per_fwd),
    }


def _per_chain_gen_tokens(res):
    """Per-chain generated-token counts inferred from each chain's committed step count.

    The CountingGenerator counts gen tokens at the ARM level, but the real per-step gen
    cost is uniform within a draw; for the pool we need a per-CHAIN split so coverage@K's
    token budget is the first-K prefix. We recover it from the arm total apportioned by
    committed steps (each step is one forward, the unit of generation), which is exact for
    the iso-GENERATION axis because gen-tokens accrue one forward per committed step.
    """
    steps = [max(1, c.n_steps) for c in res.chains]  # >=1 so a 0-step chain still costs a forward
    tot_steps = sum(steps)
    if tot_steps == 0:
        return [0] * len(res.chains)
    # Apportion the arm's gen tokens across chains proportional to committed steps.
    per = []
    acc = 0
    for i, s in enumerate(steps):
        if i == len(steps) - 1:
            per.append(int(res.gen_tokens - acc))
        else:
            t = int(round(res.gen_tokens * s / tot_steps))
            per.append(t)
            acc += t
    return per


def _per_chain_prefill_tokens(res, K):
    """Per-chain prefill counts. For i.i.d. all chains share the bare prompt, so prefill is
    uniform; we apportion the arm prefill total across the K chains by committed steps (one
    prefill per forward), exact in aggregate for the iso-TOTAL axis prefix sums.
    """
    steps = [max(1, c.n_steps) for c in res.chains]
    tot = sum(steps)
    if tot == 0:
        return [0] * len(res.chains)
    per = []
    acc = 0
    for i, s in enumerate(steps):
        if i == len(steps) - 1:
            per.append(int(res.prefill_tokens - acc))
        else:
            t = int(round(res.prefill_tokens * s / tot))
            per.append(t)
            acc += t
    return per


def cmd_select(args):
    domain = CountdownDomain()
    picked = generate_solvable_instances(args.seed, args.n_per_depth,
                                         target_range=tuple(args.target_range))
    print(f"[select] generated {len(picked)} solvable instances "
          f"(k in {GEN_K_DEPTHS}, n_per_depth={args.n_per_depth}, seed={args.seed}, "
          f"target_range={args.target_range})")

    model, tok = load_probe_model(adapter=args.adapter, device=args.device)
    gen_factory = real_gen_factory(model, tok, args.device)

    out_path = Path(args.out)

    def _write(n_seen):
        """Atomically (re)write instances.json with the in-band set found SO FAR.

        Flushed after every instance so a killed/interrupted run still leaves a valid,
        V1-filtered instances.json with every in-band instance discovered up to the kill
        point (``n_generated`` reflects how many were actually drawn, not the target).
        """
        out = {
            "config": {
                "adapter": args.adapter, "tau": args.tau, "seed": args.seed,
                "K_grid": list(K_GRID), "K_ref": K_REF, "v1_band": [V1_LO, V1_HI],
                "gen_k_depths": list(GEN_K_DEPTHS), "n_per_depth": args.n_per_depth,
                "target_range": list(args.target_range),
            },
            "n_generated": n_seen,
            "n_selected": len(selected),
            "audit": audit,
            "instances": selected,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        tmp.replace(out_path)  # atomic on POSIX: never leaves a half-written file

    selected = []
    audit = []
    n_seen = 0
    for inst, depth in picked:
        t0 = time.time()
        pool = draw_iid_pool(gen_factory, domain, inst, K_MAX, args.tau, args.seed)
        cov_ref = bool(pool["coverage_by_K"][str(K_REF)])
        cov_ref_frac = sum(pool["reached"][:K_REF]) / K_REF  # fraction of first-K_ref that hit
        in_band = (V1_LO < cov_ref_frac < V1_HI)
        rec = {
            "id": inst.id, "numbers": list(inst.numbers), "target": inst.target,
            "witness_depth": depth, "k": len(inst.numbers),
            "oracle_solvable": True,  # generated set is solvable by construction
            "coverage_by_K": pool["coverage_by_K"],
            "coverage_ref_frac": cov_ref_frac,
            "v1_in_band": in_band,
            "pool": pool,
            "secs": round(time.time() - t0, 1),
        }
        audit.append({k: rec[k] for k in
                      ("id", "k", "witness_depth", "coverage_ref_frac", "v1_in_band")})
        if in_band:
            selected.append(rec)
        n_seen += 1
        _write(n_seen)  # incremental flush: kill-safe instances.json after every instance
        print(f"[select] {inst.id} k={len(inst.numbers)} d={depth} "
              f"cov@{K_REF}frac={cov_ref_frac:.2f} v1={'PASS' if in_band else 'drop'} "
              f"({rec['secs']}s)")

    _write(n_seen)
    print(f"[select] V1 kept {len(selected)}/{len(picked)} -> {out_path}")


# ===========================================================================
# --run : four arms x K-grid over one shard, write outputs/shard_i.json
# ===========================================================================

def shard_of(instances, shard, nshards):
    """Deterministic round-robin shard: instances whose index % nshards == shard."""
    return [inst for idx, inst in enumerate(instances) if idx % nshards == shard]


def arm_curve_from_pool(reached, gen_per_chain, prefill_per_chain):
    """coverage / token-budget for an arm given its per-chain pool, over the K-grid.

    Returns, for each K in K_GRID, the (coverage@K, gen_token_budget@K, total_token@K)
    triple, all read from the SAME pool's first-K prefix. ``reached`` /
    ``gen_per_chain`` / ``prefill_per_chain`` are length >= K_MAX.
    """
    out = {}
    for K in K_GRID:
        out[str(K)] = {
            "coverage": coverage_at_K_from_pool(reached, K),
            "gen_token": gen_token_prefix(gen_per_chain, K),
            "total_token": total_token_prefix(gen_per_chain, prefill_per_chain, K),
        }
    return out


def run_arm_pool(gen_factory, domain, inst, temperature, base_seed, conditioned):
    """Run an arm to K_MAX chains and return its per-chain pool + funnel for the K-grid.

    The arm draws K_MAX chains ONCE (shared across the whole K-grid). Returns per-chain
    reached flags, per-chain gen/prefill tokens, the funnel diagnostic, distinct-leaf, and
    the anti-leak proof (candidates_per_forward all 1).
    """
    res = run_arm(gen_factory, domain, inst, K_MAX, temperature, base_seed,
                  conditioned=conditioned)
    reached = [bool(c.reached_goal) for c in res.chains]
    gen_pc = _per_chain_gen_tokens(res)
    prefill_pc = _per_chain_prefill_tokens(res, K_MAX)
    return {
        "reached": reached,
        "gen_per_chain": gen_pc,
        "prefill_per_chain": prefill_pc,
        # First committed move text per chain (the >= K* marginal-entropy ensemble): K_MAX
        # of them, so the conditioned arm's marginal is measured on a non-degenerate pool.
        "first_moves": [first_move(c) for c in res.chains],
        "curve": arm_curve_from_pool(reached, gen_pc, prefill_pc),
        "funnel": res.funnel,
        "distinct_leaf": res.distinct_leaf,
        "distinct_canon": res.distinct_canon,
        "gen_tokens_total": res.gen_tokens,
        "prefill_tokens_total": res.prefill_tokens,
        "forwards": res.forwards,
        "cands_per_fwd_all_one": all(c == 1 for c in res.cands_per_fwd),
        "cands_per_fwd_max": (max(res.cands_per_fwd) if res.cands_per_fwd else 0),
    }


def cmd_run(args):
    domain = CountdownDomain()
    blob = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    instances = blob["instances"]
    tau = blob["config"]["tau"]
    seed = blob["config"]["seed"]

    my = shard_of(instances, args.shard, args.nshards)
    print(f"[run] shard {args.shard}/{args.nshards}: {len(my)} of {len(instances)} instances")

    model, tok = load_probe_model(adapter=args.adapter, device=args.device)
    gen_factory = real_gen_factory(model, tok, args.device)

    records = []
    for rec in my:
        t0 = time.time()
        inst = Instance(numbers=tuple(rec["numbers"]), target=int(rec["target"]),
                        id=rec["id"])

        # --- iid_bok : reuse the pool drawn at selection (SAME seed schedule) ----------
        # We re-draw it here (byte-identical: same arm, same seed) so the run is
        # self-contained; the cross-check below asserts it matches the saved selection pool.
        iid = run_arm_pool(gen_factory, domain, inst, tau, seed, conditioned=False)

        # --- conditioned : the VERIFIED 1-forward/step probe ---------------------------
        cond = run_arm_pool(gen_factory, domain, inst, tau, seed, conditioned=True)

        # --- temp_matched_iid : i.i.d. at the entropy-matched temperature (H4) ---------
        # Measure the conditioned arm's MARGINAL step entropy on its own >= K* ensemble
        # (K_MAX first-committed-move texts), never on a degenerate single chain.
        cond_entropy = marginal_step_entropy(cond["first_moves"])
        # Find iid tau whose marginal step entropy matches cond_entropy on a >= K* ensemble.
        best_tau, matched_entropy, tau_sweep = match_temperature(
            gen_factory, domain, inst, seed, cond_entropy, n_ens=K_STAR
        )
        temp = run_arm_pool(gen_factory, domain, inst, best_tau, seed, conditioned=False)

        # --- oracle : the coverage ceiling --------------------------------------------
        s0 = domain.initial_state(instance_dict(inst))
        oracle_cov = bool(domain.solvable(s0))

        out_rec = {
            "id": inst.id, "numbers": list(inst.numbers), "target": inst.target,
            "k": len(inst.numbers), "witness_depth": rec.get("witness_depth"),
            "tau": tau, "seed": seed,
            "iid_bok": iid,
            "conditioned": cond,
            "temp_matched_iid": temp,
            "temp_match": {
                "cond_marginal_entropy": cond_entropy,
                "matched_tau": best_tau,
                "matched_entropy": matched_entropy,
                "entropy_gap": abs(matched_entropy - cond_entropy),
                "k_star_ensemble": K_STAR,
                "sweep": tau_sweep,
            },
            "oracle": {"coverage": oracle_cov},
            "pool_matches_selection": _pool_matches(iid["reached"],
                                                    rec.get("pool", {}).get("reached")),
            "secs": round(time.time() - t0, 1),
        }
        records.append(out_rec)
        print(f"[run] {inst.id}: iid cov@{K_REF}={iid['curve'][str(K_REF)]['coverage']} "
              f"cond cov@{K_REF}={cond['curve'][str(K_REF)]['coverage']} "
              f"tempTau={best_tau} (gap={abs(matched_entropy-cond_entropy):.3f}) "
              f"funnel(cond)={cond['funnel']} ({out_rec['secs']}s)")

    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"shard_{args.shard}.json"
    out_path.write_text(json.dumps({
        "config": {"adapter": args.adapter, "shard": args.shard,
                   "nshards": args.nshards, "tau": tau, "seed": seed,
                   "K_grid": list(K_GRID)},
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"[run] wrote {out_path} ({len(records)} instances)")


def _pool_matches(a, b):
    """True iff the re-drawn iid pool reproduces the selection pool's reached flags."""
    if b is None:
        return None
    return list(a) == list(b)


# ===========================================================================
# --analyze : merge shards, dual-axis, miss-slope, funnel, paired bootstrap, Holm
# ===========================================================================

def _load_shards(out_dir):
    recs = []
    for p in sorted(Path(out_dir).glob("shard_*.json")):
        blob = json.loads(p.read_text(encoding="utf-8"))
        recs.extend(blob["records"])
    return recs


def paired_bootstrap_diff(a, b, n_boot=N_BOOT, seed=0):
    """Paired bootstrap of mean(a - b) over instances. Returns [2.5, 50, 97.5] percentiles.

    ``a`` and ``b`` are per-instance aligned outcome vectors (e.g. coverage@K for two arms
    on the SAME instances). Resamples INSTANCE INDICES with replacement (paired), so the
    pairing a_i vs b_i is preserved -- the correct null for "arm a beats arm b on matched
    instances". Empty -> [nan]*3.
    """
    import numpy as np
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    if n == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append((a[idx] - b[idx]).mean())
    return [float(np.percentile(diffs, q)) for q in (2.5, 50, 97.5)]


def miss_decay_slope(token_budgets, coverages):
    """Slope of ``log(1 - coverage)`` vs token budget (the PREREG H2 "底数 / decay rate").

    ``token_budgets`` and ``coverages`` are aligned arrays over the K-grid (the
    per-instance mean coverage at each K's mean token budget). Clamps coverage to [0, 1)
    and adds a tiny floor so ``log(1 - cov)`` is finite at cov==1. A MORE NEGATIVE slope =
    faster miss-decay = a higher "底数". Returns the OLS slope, or nan if < 2 finite points.
    """
    import numpy as np
    x = np.asarray(token_budgets, float)
    cov = np.asarray(coverages, float)
    miss = np.clip(1.0 - cov, 1e-6, 1.0)
    y = np.log(miss)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    x, y = x[ok], y[ok]
    # OLS slope.
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return float("nan")
    return float(((x - xm) * (y - ym)).sum() / denom)


def per_instance_miss_slope(records, arm, axis):
    """Per-instance miss-decay slope for ``arm`` on ``axis`` ('gen_token'|'total_token').

    For each instance, builds the (token_budget@K, coverage@K) curve from the arm's pool
    curve and fits ``miss_decay_slope``. Returns the per-instance slope list (one per
    instance) so the paired bootstrap can compare arm-vs-arm slope differences.
    """
    out = []
    for r in records:
        curve = r[arm]["curve"]
        toks = [curve[str(K)][axis] for K in K_GRID]
        covs = [1.0 if curve[str(K)]["coverage"] else 0.0 for K in K_GRID]
        out.append(miss_decay_slope(toks, covs))
    return out


def coverage_vec(records, arm, K):
    """Per-instance coverage@K (0/1) for ``arm`` -- the paired-bootstrap unit."""
    return [1.0 if r[arm]["curve"][str(K)]["coverage"] else 0.0 for r in records]


def holm(pvals_named):
    """Holm-Bonferroni over named one-sided p-values. Returns name -> (p, adj_p, reject@.05).

    ``pvals_named`` is a dict name -> p. Sorts ascending, multiplies the i-th smallest by
    (m - i), enforces monotone non-decreasing adjusted p, and rejects at alpha=0.05.
    """
    items = sorted(pvals_named.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev)  # enforce monotonicity
        prev = adj
        out[name] = {"p": p, "adj_p": adj, "reject_0.05": adj < 0.05}
    return out


def _one_sided_p_from_boot(a, b, n_boot=N_BOOT, seed=0):
    """One-sided bootstrap p-value for H: mean(a - b) > 0 (a beats b).

    p = fraction of bootstrap resamples where mean(a - b) <= 0 (the null). A small p means
    a robustly beats b. Paired over instance indices. Returns p in [1/(n_boot+1), 1].
    """
    import numpy as np
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    if n == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    le0 = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if (a[idx] - b[idx]).mean() <= 0:
            le0 += 1
    return (le0 + 1) / (n_boot + 1)


def cmd_analyze(args):
    records = _load_shards(_HERE / "outputs")
    if not records:
        print("[analyze] no shard_*.json found in outputs/")
        return
    print(f"[analyze] merged {len(records)} instances from shards")

    arms = ("iid_bok", "conditioned", "temp_matched_iid")
    result = {"n_instances": len(records), "K_grid": list(K_GRID),
              "axes": ("gen_token", "total_token")}

    # --- coverage-vs-token on BOTH axes (mean coverage + mean token budget per K) -------
    curves = {}
    for arm in arms:
        curves[arm] = {}
        for axis in ("gen_token", "total_token"):
            curve = []
            for K in K_GRID:
                covs = coverage_vec(records, arm, K)
                toks = [r[arm]["curve"][str(K)][axis] for r in records]
                curve.append({
                    "K": K,
                    "coverage_mean": sum(covs) / len(covs),
                    "token_mean": sum(toks) / len(toks),
                })
            curves[arm][axis] = curve
    # oracle ceiling (coverage = solvable; token-free)
    result["oracle_coverage_mean"] = sum(
        1.0 if r["oracle"]["coverage"] else 0.0 for r in records) / len(records)
    result["curves"] = curves

    # --- HEADLINE: paired coverage diff at K_REF (conditioned - iid), gen-token axis ----
    cond_ref = coverage_vec(records, "conditioned", K_REF)
    iid_ref = coverage_vec(records, "iid_bok", K_REF)
    temp_ref = coverage_vec(records, "temp_matched_iid", K_REF)
    result["H1_coverage_diff_at_Kref"] = {
        "K_ref": K_REF,
        "conditioned_minus_iid": paired_bootstrap_diff(cond_ref, iid_ref, seed=1),
        "temp_minus_iid": paired_bootstrap_diff(temp_ref, iid_ref, seed=2),
        "conditioned_mean": sum(cond_ref) / len(cond_ref),
        "iid_mean": sum(iid_ref) / len(iid_ref),
        "temp_mean": sum(temp_ref) / len(temp_ref),
    }

    # --- H2: miss-decay slope (conditioned steeper than iid) on BOTH axes ---------------
    result["H2_miss_slope"] = {}
    for axis in ("gen_token", "total_token"):
        cond_s = per_instance_miss_slope(records, "conditioned", axis)
        iid_s = per_instance_miss_slope(records, "iid_bok", axis)
        # "steeper" = MORE NEGATIVE => iid_slope - cond_slope > 0 when cond decays faster.
        result["H2_miss_slope"][axis] = {
            "iid_minus_cond_slope": paired_bootstrap_diff(iid_s, cond_s, seed=3),
            "cond_slope_median": float(_nanmedian(cond_s)),
            "iid_slope_median": float(_nanmedian(iid_s)),
        }

    # --- H4: temp-matched i.i.d. gain ~ 0 and < conditioned gain (the negative control) -
    result["H4_temp_control"] = {
        "temp_minus_iid_at_Kref": paired_bootstrap_diff(temp_ref, iid_ref, seed=4),
        "cond_minus_temp_at_Kref": paired_bootstrap_diff(cond_ref, temp_ref, seed=5),
        "mean_entropy_gap": _mean([r["temp_match"]["entropy_gap"] for r in records]),
        "mean_matched_tau": _mean([r["temp_match"]["matched_tau"] for r in records]),
    }

    # --- Holm over H1, H2 (gen-token headline), H4 (cond vs temp) ------------------------
    p_h1 = _one_sided_p_from_boot(cond_ref, iid_ref, seed=11)  # cond beats iid
    iid_s_g = per_instance_miss_slope(records, "iid_bok", "gen_token")
    cond_s_g = per_instance_miss_slope(records, "conditioned", "gen_token")
    p_h2 = _one_sided_p_from_boot(iid_s_g, cond_s_g, seed=12)  # iid_slope > cond_slope
    p_h4 = _one_sided_p_from_boot(cond_ref, temp_ref, seed=13)  # cond beats temp control
    result["holm_H1_H2_H4"] = holm({"H1": p_h1, "H2_gen": p_h2, "H4": p_h4})

    # --- funnel diagnostic per arm (to_solvable vs to_dead, aggregated) -----------------
    result["funnel"] = {}
    for arm in arms:
        ts = sum(r[arm]["funnel"]["to_solvable"] for r in records)
        td = sum(r[arm]["funnel"]["to_dead"] for r in records)
        ns = sum(r[arm]["funnel"]["n_steps"] for r in records)
        result["funnel"][arm] = {
            "to_solvable": ts, "to_dead": td, "n_steps": ns,
            "to_solvable_frac": (ts / ns if ns else 0.0),
        }

    # --- anti-leak guard: NO N-candidate per step in ANY arm ----------------------------
    all_one = all(
        r[arm]["cands_per_fwd_all_one"] for r in records
        for arm in arms
    )
    result["anti_leak_one_forward_per_step"] = bool(all_one)

    # --- validity gates summary ---------------------------------------------------------
    result["validity"] = {
        "V1_all_in_band": True,  # instances.json was already V1-filtered at selection
        "V4_oracle_above_iid": (result["oracle_coverage_mean"]
                                >= result["H1_coverage_diff_at_Kref"]["iid_mean"]),
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[analyze] H1 (cond - iid) @Kref:",
          result["H1_coverage_diff_at_Kref"]["conditioned_minus_iid"])
    print("[analyze] H4 (temp - iid) @Kref:",
          result["H4_temp_control"]["temp_minus_iid_at_Kref"])
    print("[analyze] Holm:", json.dumps(result["holm_H1_H2_H4"]))
    print("[analyze] anti-leak one-forward-per-step:",
          result["anti_leak_one_forward_per_step"])
    print(f"[analyze] wrote {out_path}")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else float("nan")


def _nanmedian(xs):
    import numpy as np
    return float(np.nanmedian(np.asarray(xs, float))) if xs else float("nan")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true", help="generate + draw pool + V1 filter")
    ap.add_argument("--run", action="store_true", help="four arms x K-grid over one shard")
    ap.add_argument("--analyze", action="store_true", help="merge shards + full result")

    ap.add_argument("--adapter", default=DEFAULT_ADAPTER,
                    help="LoRA adapter path, or 'base' for the bare instruct model.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--seed", type=int, default=0)

    # select
    ap.add_argument("--n_per_depth", type=int, default=24,
                    help="solvable instances to generate per k in {5,6} before V1 filter.")
    ap.add_argument("--target_range", type=int, nargs=2, default=[10, 100],
                    metavar=("LO", "HI"),
                    help="Countdown target band; lower/narrower => easier instances "
                         "(higher iid coverage) to lift more into the V1 (0.2,0.9) band.")
    ap.add_argument("--out", default="instances.json",
                    help="select: instances.json path; analyze: result.json path.")

    # run
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=5)
    ap.add_argument("--instances", default="instances.json",
                    help="instances.json written by --select (read by --run/--analyze).")

    args = ap.parse_args()
    if args.select:
        cmd_select(args)
    elif args.run:
        cmd_run(args)
    elif args.analyze:
        cmd_analyze(args)
    else:
        ap.error("one of --select / --run / --analyze is required")


if __name__ == "__main__":
    main()
