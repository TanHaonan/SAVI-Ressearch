"""TINY GPU smoke for the real-LLM prompt-conditioning coverage probe.

Run (one GPU):
    export HF_HUB_CACHE=$HOME/.cache/huggingface/hub
    CUDA_VISIBLE_DEVICES=3 python smoke_realprobe.py

Confirms on ~3 solvable depth-3/4 instances, small K:
  * the model + decoupled LoRA load and generate parseable chains (parse rate);
  * the conditioned arm ACTUALLY changes the chains vs the i.i.d. arm (and chain 1 is
    identical, since chain 1's avoid-list is empty in both arms);
  * token accounting is sane: gen tokens ~= equal across arms (the ~1x mechanism: ONE
    forward per step, NO N-candidate tax), prefill GROWS with chain index in the
    conditioned arm; candidates_per_forward is ALWAYS 1 (anti-leak);
  * the narrow-funnel diagnostic (to_solvable vs to_dead steps) and any coverage signal.

It writes outputs/smoke_result.json and prints a compact summary.
"""

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import realprobe_core as rp
from realprobe_core import (
    CountdownDomain,
    run_arm,
    instance_dict,
    load_probe_model,
    real_gen_factory,
    DEFAULT_ADAPTER,
)
from core import Instance, load_instances
from core.metrics import boot_ci  # may be unused; imported defensively
from _deps.instances import label_instance


def pick_instances(n_want=3, depths=(3, 4)):
    """Pick ``n_want`` builtin instances that are solvable with witness depth in ``depths``.

    Depth = len(solve_one witness) = number of combines to the goal. The builtin set is
    all 4-number target-24 instances, so witnesses are depth 3 (4 numbers -> 3 combines)
    almost always; we keep the depth filter explicit so the smoke documents what it ran.
    """
    insts = load_instances({"set": "builtin"})
    out = []
    for inst in insts:
        lab = label_instance(inst)
        if lab["solvable"] and lab["witness_len"] in depths:
            out.append((inst, lab["witness_len"]))
        if len(out) >= n_want:
            break
    return out


def parse_rate(chains):
    """Fraction of committed steps over the rollout cap (a proxy for staying on-task).

    Here: total committed moves / total forwards. A forward that parse-fails commits no
    move, so committed/forwards == the per-step legality rate of the emission.
    """
    committed = sum(c.n_steps for c in chains)
    return committed


def summarize_arm(name, res):
    return {
        "arm": name,
        "coverage": bool(res.coverage),
        "n_goal": res.n_goal,
        "K": len(res.chains),
        "gen_tokens": res.gen_tokens,
        "prefill_tokens": res.prefill_tokens,
        "total_tokens": res.gen_tokens + res.prefill_tokens,
        "forwards": res.forwards,
        "committed_steps": sum(c.n_steps for c in res.chains),
        "cands_per_forward_max": (max(res.cands_per_fwd) if res.cands_per_fwd else 0),
        "cands_per_forward_all_one": all(c == 1 for c in res.cands_per_fwd),
        "distinct_leaf": res.distinct_leaf,
        "distinct_canon": res.distinct_canon,
        "funnel": res.funnel,
        "split_tokens": res.split_tokens,
        "chains_preview": [c.moves_text for c in res.chains[:4]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER,
                    help="LoRA adapter path, or 'base' for the bare instruct model.")
    ap.add_argument("--n_inst", type=int, default=3)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    domain = CountdownDomain()
    picked = pick_instances(n_want=args.n_inst)
    print(f"[smoke] instances ({len(picked)}): "
          + ", ".join(f"{i.id}{i.numbers}->{i.target} d={d}" for i, d in picked))

    print(f"[smoke] loading model adapter={args.adapter!r} device={args.device} ...")
    model, tok = load_probe_model(adapter=args.adapter, device=args.device)
    gen_factory = real_gen_factory(model, tok, args.device)
    print(f"[smoke] model loaded in {time.time()-t0:.1f}s")

    records = []
    for inst, depth in picked:
        ti = time.time()
        iid = run_arm(gen_factory, domain, inst, args.K, args.tau, args.seed,
                      conditioned=False)
        cond = run_arm(gen_factory, domain, inst, args.K, args.tau, args.seed,
                       conditioned=True)
        # Did conditioning change the chains? Chain 1 must match (empty avoid in both);
        # at least one later chain should differ if conditioning is biting.
        chain1_same = (iid.chains[0].moves_text == cond.chains[0].moves_text)
        later_changed = sum(
            1 for k in range(1, args.K)
            if iid.chains[k].moves_text != cond.chains[k].moves_text
        )
        rec = {
            "id": inst.id,
            "numbers": list(inst.numbers),
            "target": inst.target,
            "witness_depth": depth,
            "iid": summarize_arm("iid_bok", iid),
            "conditioned": summarize_arm("conditioned", cond),
            "chain1_identical": bool(chain1_same),
            "later_chains_changed": later_changed,
            "secs": round(time.time() - ti, 1),
        }
        records.append(rec)
        print(f"[smoke] {inst.id}: "
              f"iid cov={iid.coverage}(n={iid.n_goal}) genT={iid.gen_tokens} "
              f"preT={iid.prefill_tokens} distinctLeaf={iid.distinct_leaf} "
              f"funnel={iid.funnel} || "
              f"cond cov={cond.coverage}(n={cond.n_goal}) genT={cond.gen_tokens} "
              f"preT={cond.prefill_tokens} distinctLeaf={cond.distinct_leaf} "
              f"funnel={cond.funnel} || "
              f"chain1same={chain1_same} laterChanged={later_changed}/{args.K-1} "
              f"({rec['secs']}s)")

    # Aggregate sanity flags.
    all_one_fwd = all(
        r["iid"]["cands_per_forward_all_one"] and r["conditioned"]["cands_per_forward_all_one"]
        for r in records
    )
    any_conditioning_bit = any(r["later_chains_changed"] > 0 for r in records)
    prefill_grows = all(
        r["conditioned"]["prefill_tokens"] >= r["iid"]["prefill_tokens"]
        for r in records
    )
    iid_cov = sum(r["iid"]["coverage"] for r in records)
    cond_cov = sum(r["conditioned"]["coverage"] for r in records)
    total_committed = sum(
        r["iid"]["committed_steps"] + r["conditioned"]["committed_steps"] for r in records
    )
    total_forwards = sum(
        r["iid"]["forwards"] + r["conditioned"]["forwards"] for r in records
    )
    parse_rate_overall = (total_committed / total_forwards) if total_forwards else 0.0

    summary = {
        "config": {
            "adapter": args.adapter, "n_inst": len(picked), "K": args.K,
            "tau": args.tau, "seed": args.seed,
        },
        "sanity": {
            "one_forward_per_step": all_one_fwd,
            "conditioning_changes_chains": any_conditioning_bit,
            "prefill_grows_in_conditioned": prefill_grows,
            "parse_rate_committed_over_forwards": round(parse_rate_overall, 3),
        },
        "coverage": {
            "iid_solved_instances": iid_cov,
            "conditioned_solved_instances": cond_cov,
            "n_instances": len(records),
        },
        "records": records,
        "wall_secs": round(time.time() - t0, 1),
    }

    out_path = _HERE / "outputs" / "smoke_result.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[smoke] SANITY:", json.dumps(summary["sanity"]))
    print("[smoke] COVERAGE:", json.dumps(summary["coverage"]))
    print(f"[smoke] wrote {out_path} ({summary['wall_secs']}s total)")


if __name__ == "__main__":
    main()
