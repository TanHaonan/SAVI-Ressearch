"""Merge sharded Tier-B run_joint outputs into one results.json + synthesis.

Tier B is sharded across GPUs by (profile, k-group) for parallelism; each shard writes a
run_joint JSON with a single profile and a subset of k. This merges them: it unions the
per-k aggregates per profile, re-runs ``run_joint.synthesize`` over the full
{decoupled, coupled} × {4,5,6} grid, and writes the combined results.

Usage:
    python merge_shards.py --shards outputs/tierB_*.json --out outputs/tierB.json
"""

import argparse
import glob
import json
from pathlib import Path

import run_joint


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True,
                    help="shard JSON paths (globs allowed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    paths = []
    for pat in a.shards:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        raise SystemExit("no shard files matched")

    profiles = {}
    base_cfg = None
    tier = None
    logs = []
    for p in paths:
        r = json.load(open(p))
        tier = r.get("tier", tier)
        base_cfg = base_cfg or r.get("config")
        for prof, by_k in r.get("profiles", {}).items():
            profiles.setdefault(prof, {})
            for k, agg in by_k.items():
                profiles[prof][int(k)] = agg     # normalize k to int
        logs.append(f"merged {p}: profiles={list(r.get('profiles',{}))} "
                    f"ks={[list(v) for v in r.get('profiles',{}).values()]}")

    cfg = {"K": base_cfg.get("K"), "N": base_cfg.get("N"), "tau": base_cfg.get("tau"),
           "seed": base_cfg.get("seed"), "n_boot": base_cfg.get("n_boot", 10000)}
    synthesis = run_joint.synthesize(profiles, cfg) if "decoupled" in profiles else {}

    result = {"plan": "PLAN3-joint-lambda-decode", "tier": tier,
              "config": base_cfg, "merged_from": paths,
              "profiles": profiles, "synthesis": synthesis, "merge_log": logs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=run_joint._json_default))
    print(f"wrote {a.out}")
    if synthesis:
        print("gates:", json.dumps(synthesis["gates"]))
        print("H1_deepest:", json.dumps(synthesis["H1_deepest"]))
        print("H2:", json.dumps(synthesis["H2_depth_curve"]["deltas_by_k"]))
        print("H3 keff:", json.dumps({"dec": synthesis["H3_keff"].get("keff_decoupled_by_k"),
                                      "cou": synthesis["H3_keff"].get("keff_coupled_by_k")}))
    return result


if __name__ == "__main__":
    main()
