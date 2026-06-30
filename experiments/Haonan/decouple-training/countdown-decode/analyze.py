"""Merge the 4 split eval runs (decoupled/coupled x A/B) and report the capstone numbers.

For each backend (decoupled, coupled) the two halves' per_instance records are unioned, then:
  - per-arm pass@1 over the union (the arm ladder: greedy / self_consistency / savi / best_of_k / oracle);
  - the PAIRED savi-vs-self_consistency comparison (the core claim — same instances): McNemar
    counts b=savi&!sc, c=!savi&sc, net=b-c, and a paired bootstrap CI on (savi_i - sc_i);
  - savi(decoupled) vs savi(coupled) paired on the shared instance ids (does decoupled training help);
  - emission diagnostics (coverage, parse_rate) and savi trellis widths (from the arms block).

Usage: python analyze.py   (reads outputs/eval/<tag>/results.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
import core  # noqa: E402
from core import metrics  # noqa: E402

ARMS = ["greedy", "self_consistency", "savi", "best_of_k", "oracle"]


def _load(tag):
    p = _ROOT / "outputs" / "eval" / tag / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def _merge(tag_a, tag_b):
    ra, rb = _load(tag_a), _load(tag_b)
    if ra is None or rb is None:
        return None, None
    pi = {x["id"]: x for x in ra["per_instance"]}
    for x in rb["per_instance"]:
        pi[x["id"]] = x
    # arms-block diagnostics (coverage/parse/widths) averaged across the two halves, n-weighted
    diag = {}
    for key in ["correct_terminal_coverage", "parse_rate"]:
        va = ra["arms"]["greedy"]["diagnostics"].get(key)
        vb = rb["arms"]["greedy"]["diagnostics"].get(key)
        na, nb = len(ra["per_instance"]), len(rb["per_instance"])
        diag[key] = (va * na + vb * nb) / (na + nb) if (va is not None and vb is not None) else None
    for key in ["trellis_width_before_merge", "trellis_width_after_merge"]:
        va = ra["arms"]["savi"]["diagnostics"].get(key)
        vb = rb["arms"]["savi"]["diagnostics"].get(key)
        na, nb = len(ra["per_instance"]), len(rb["per_instance"])
        diag[key] = (va * na + vb * nb) / (na + nb) if (va is not None and vb is not None) else None
    return list(pi.values()), diag


def _paired_diff_ci(pairs, seed=0):
    """Bootstrap CI on mean(a_i - b_i) for a list of (a,b) bool pairs."""
    d = np.array([1.0 * a - 1.0 * b for a, b in pairs])
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(2000)]
    return float(np.mean(d)), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def report(tag, recs, diag):
    n = len(recs)
    print(f"\n===== {tag}  (n={n}) =====")
    for a in ARMS:
        m = np.mean([r[a] for r in recs])
        ci = metrics.boot_ci([1.0 * r[a] for r in recs], seed=0)
        print(f"  {a:16s} pass@1 = {m:.3f}  ci=[{ci[0]:.3f},{ci[2]:.3f}]")
    # paired savi vs self_consistency (the core claim)
    b = sum(1 for r in recs if r["savi"] and not r["self_consistency"])
    c = sum(1 for r in recs if not r["savi"] and r["self_consistency"])
    md, ci = _paired_diff_ci([(r["savi"], r["self_consistency"]) for r in recs])
    print(f"  PAIRED savi - SC: mean_diff={md:+.3f} ci=[{ci[0]:+.3f},{ci[1]:+.3f}]  "
          f"(McNemar b=savi&!sc={b}, c=!savi&sc={c}, net={b-c})")
    print(f"  diagnostics: coverage={diag.get('correct_terminal_coverage')}, "
          f"parse_rate={diag.get('parse_rate')}, "
          f"trellis width {diag.get('trellis_width_before_merge')}->{diag.get('trellis_width_after_merge')}")
    return {r["id"]: r for r in recs}


def main():
    dec, dec_diag = _merge("decoupled_A", "decoupled_B")
    cup, cup_diag = _merge("coupled_A", "coupled_B")
    out = {}
    if dec is not None:
        out["decoupled"] = report("decoupled", dec, dec_diag)
    if cup is not None:
        out["coupled"] = report("coupled", cup, cup_diag)
    if dec is not None and cup is not None:
        # paired savi(decoupled) vs savi(coupled) on shared ids
        di, ci_ = out["decoupled"], out["coupled"]
        shared = [i for i in di if i in ci_]
        md, ci = _paired_diff_ci([(di[i]["savi"], ci_[i]["savi"]) for i in shared])
        print(f"\n===== decoupled vs coupled (savi, paired n={len(shared)}) =====")
        print(f"  savi(decoupled) - savi(coupled): mean_diff={md:+.3f} ci=[{ci[0]:+.3f},{ci[1]:+.3f}]")
        # also greedy/SC marginals side by side
        for a in ARMS:
            dm = np.mean([di[i][a] for i in shared]); cm = np.mean([ci_[i][a] for i in shared])
            print(f"  {a:16s} decoupled={dm:.3f}  coupled={cm:.3f}")
    # persist a compact summary
    summ = {}
    for name, recs in (("decoupled", dec), ("coupled", cup)):
        if recs is None:
            continue
        summ[name] = {a: float(np.mean([r[a] for r in recs])) for a in ARMS}
        summ[name]["n"] = len(recs)
    (_ROOT / "outputs" / "summary.json").write_text(json.dumps(summ, indent=2))
    print("\nwrote outputs/summary.json")


if __name__ == "__main__":
    main()
