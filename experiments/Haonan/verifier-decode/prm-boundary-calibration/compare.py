"""Compare PRM vs LM-judge calibration geometry on the common subset.

Answers the disambiguation: does the better-/differently-calibrated judge show
the SAME false-positive spike at the boundary (d=-1) as the PRM (=> the last
correct step is genuinely borderline, a data property any value inherits) or a
FLATTER profile (=> the spike is PRM-specific)?
"""
from __future__ import annotations

import argparse
import json

import data as D
import metrics as M

HELD_OUT = ["math", "olympiadbench", "omnimath"]
DIST_ORDER = ["-1", "-2", "-3", "-4", "<=-5"]


def load_rewards(path):
    r = json.load(open(path))
    if isinstance(r, dict) and "rewards" in r:
        r = r["rewards"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prm", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    prm = load_rewards(a.prm)
    jud = load_rewards(a.judge)
    ids = set(prm) & set(jud)
    recs = D.load_processbench(D.CONFIGS)
    recs = [r for r in recs if r["id"] in ids]
    print(f"common subset: {len(recs)} solutions "
          f"({sum(1 for r in recs if r['label']>=0)} erroneous)\n")

    gsm = [r for r in recs if r["config"] == "gsm8k"]
    prm_theta, _ = M.choose_threshold(M.build_step_table(gsm, prm))

    held = [r for r in recs if r["config"] in HELD_OUT]
    results = {}
    for name, rw, th in [("PRM (Qwen2.5-Math-PRM-7B)", prm, prm_theta),
                         ("JUDGE (Qwen2.5-7B-Instruct, generative)", jud, 0.5)]:
        rep = M.uniformity_report(M.build_step_table(held, rw), th)
        results[name] = {"theta": th, "rep": rep}
        bd = rep["by_distance"]
        row = "  ".join(f"{k}:{bd[k]['mean']:.3f}(n{bd[k]['n']})" for k in DIST_ORDER if k in bd)
        print(f"== {name}  (θ={th:.2f}) ==")
        print(f"  FP={rep['fp_rate'][1]:.3f}  FN={rep['fn_rate'][1]:.3f}  "
              f"FN/FP={rep['polarity_ratio']:.2f}  verdict={rep['verdict']} {rep['flags']}")
        print(f"  FP-by-distance: {row}")
        d = rep["by_distance"]
        if "-1" in d and "<=-5" in d and d["<=-5"]["mean"] > 0:
            print(f"  boundary gradient (d=-1 / d<=-5) = {d['-1']['mean']/d['<=-5']['mean']:.2f}x\n")
        else:
            print()

    if a.out:
        def pack(x):
            r = x["rep"]
            return {"theta": x["theta"], "fp": r["fp_rate"], "fn": r["fn_rate"],
                    "polarity_ratio": r["polarity_ratio"], "verdict": r["verdict"],
                    "flags": r["flags"], "by_distance": r["by_distance"]}
        json.dump({"n_solutions": len(recs),
                   "models": {k: pack(v) for k, v in results.items()}},
                  open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
