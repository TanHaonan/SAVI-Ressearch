"""Analysis: turn saved per-step rewards into the boundary-step calibration
verdict. GPU-free. Reads results/<name>.json ({sol_id: [r0,...]}).

- θ calibrated by max per-step F1 on gsm8k; uniformity evaluated on the held-out
  harder configs (math/olympiadbench/omnimath) pooled, plus per-config.
- ProcessBench first-error F1 reported as a sanity anchor vs published numbers.
- θ-sensitivity: verdict re-checked at nearby thresholds.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import data as D
import metrics as M

HELD_OUT = ["math", "olympiadbench", "omnimath"]


def processbench_f1(records, rw, theta):
    """Standard ProcessBench metric: predict first step with reward<θ (else 'no
    error'); error-acc = predicted index == true first-error index."""
    acc = defaultdict(lambda: {"ct": 0, "ch": 0, "et": 0, "eh": 0})
    for r in records:
        v = rw.get(r["id"])
        if v is None:
            continue
        pred = next((i for i, x in enumerate(v) if x == x and x < theta), -1)
        d = acc[r["config"]]
        if r["label"] == -1:
            d["ct"] += 1; d["ch"] += int(pred == -1)
        else:
            d["et"] += 1; d["eh"] += int(pred == r["label"])
    out = {}
    for c, d in acc.items():
        ca = d["ch"] / d["ct"] if d["ct"] else 0.0
        ea = d["eh"] / d["et"] if d["et"] else 0.0
        out[c] = {"correct_acc": round(ca, 3), "error_acc": round(ea, 3),
                  "f1": round(2 * ca * ea / (ca + ea), 3) if ca + ea else 0.0}
    return out


def _fmt(rep):
    bd = {k: (v["n"], round(v["mean"], 3)) for k, v in rep["by_distance"].items()}
    bp = {k: (v["n"], round(v["mean"], 3)) for k, v in rep["by_position"].items()}
    return (f"verdict={rep['verdict']} flags={rep['flags']} "
            f"| FP={rep['fp_rate'][1]:.3f}[{rep['fp_rate'][0]:.3f},{rep['fp_rate'][2]:.3f}] "
            f"FN={rep['fn_rate'][1]:.3f}[{rep['fn_rate'][0]:.3f},{rep['fn_rate'][2]:.3f}] "
            f"FN/FP={rep['polarity_ratio']:.2f}\n  by_dist(FP|good)={bd}\n  by_pos(err)={bp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards", required=True)
    ap.add_argument("--out", default=None, help="results.json to write")
    a = ap.parse_args()

    rw = json.load(open(a.rewards))
    if isinstance(rw, dict) and "rewards" in rw:   # generative-judge wrapped format
        rw = rw["rewards"]
    recs = D.load_processbench(D.CONFIGS)
    recs = [r for r in recs if r["id"] in rw]

    gsm = [r for r in recs if r["config"] == "gsm8k"]
    obs_gsm = M.build_step_table(gsm, rw)
    theta, f1 = M.choose_threshold(obs_gsm)
    print(f"θ (max per-step F1 on gsm8k) = {theta:.3f}  (gsm8k step-F1={f1:.3f})\n")

    pbf1 = processbench_f1(recs, rw, theta)
    print("ProcessBench first-error F1 (sanity vs published ~64 avg for 7B PRM):")
    for c in D.CONFIGS:
        if c in pbf1:
            print(f"  {c:14s} {pbf1[c]}")
    print()

    held = [r for r in recs if r["config"] in HELD_OUT]
    obs_held = M.build_step_table(held, rw)
    rep_held = M.uniformity_report(obs_held, theta)
    print(f"== HELD-OUT (math+olympiad+omni), n_good={rep_held['n_good']} "
          f"n_bad={rep_held['n_bad']} ==\n{_fmt(rep_held)}\n")

    per_cfg = {}
    for c in D.CONFIGS:
        o = M.build_step_table([r for r in recs if r["config"] == c], rw)
        rep = M.uniformity_report(o, theta)
        per_cfg[c] = rep
        print(f"-- {c} (n_good={rep['n_good']} n_bad={rep['n_bad']}) --\n  {_fmt(rep)}")
    print()

    print("θ-sensitivity (held-out verdict at nearby thresholds):")
    sens = {}
    for t in [round(theta - 0.15, 3), round(theta - 0.05, 3), theta,
              round(min(theta + 0.05, 0.99), 3), round(min(theta + 0.15, 0.99), 3)]:
        rep = M.uniformity_report(obs_held, t)
        sens[t] = {"verdict": rep["verdict"], "flags": rep["flags"]}
        print(f"  θ={t:.3f}: {rep['verdict']} {rep['flags']}")

    if a.out:
        def pack(rep):
            return {"verdict": rep["verdict"], "flags": rep["flags"],
                    "fp_rate": rep["fp_rate"], "fn_rate": rep["fn_rate"],
                    "polarity_ratio": rep["polarity_ratio"],
                    "n_good": rep["n_good"], "n_bad": rep["n_bad"],
                    "by_distance": rep["by_distance"], "by_position": rep["by_position"]}
        json.dump({"theta": theta, "gsm8k_step_f1": f1, "processbench_f1": pbf1,
                   "held_out": pack(rep_held), "per_config": {c: pack(r) for c, r in per_cfg.items()},
                   "theta_sensitivity": sens}, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
