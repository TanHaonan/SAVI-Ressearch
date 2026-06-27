"""Extract the structured-commit A/B (decoupled vs decoupled_genreg) from the state-emission diversity_<group>.json.

Reports, by number of survivors j (pooled over k, temp), for each group:
  calibration_tv, coverage, readout_tv, gap (=calibration_tv - readout_tv), abstain_rate.
Pooling weights each cell by its finite-n so the per-j number reflects item counts, not cell counts.

Usage: python extract_ab.py <dir-with-diversity_*.json> [decoupled decoupled_genreg]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CELL = re.compile(r"k(?P<k>\d+)_j(?P<j>\d+)_T(?P<t>[\d.]+)")
KEYS = ["calibration_tv", "coverage", "readout_tv", "gap", "abstain_rate"]


def by_j(diversity_json, regime="b1"):
    """{j: {metric: weighted-mean}} pooling cells of that j over k and temp by finite-n."""
    blob = json.loads(Path(diversity_json).read_text())
    cells = blob["regimes"].get(regime, {})
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))   # j -> metric -> [sum(mean*n), sum(n)]
    for cell, agg in cells.items():
        m = CELL.match(cell)
        if not m:
            continue
        j = int(m.group("j"))
        for key in KEYS:
            v = agg.get(key)
            if v is None:
                continue
            n = v.get("n", 0)
            mean = v.get("mean")
            if n and mean is not None and mean == mean:        # finite, n>0
                acc[j][key][0] += mean * n
                acc[j][key][1] += n
    out = {}
    for j, md in acc.items():
        out[j] = {key: (s / n if n else float("nan")) for key, (s, n) in md.items()}
        out[j]["_n"] = max((md[key][1] for key in KEYS), default=0)
    return out


def main():
    d = Path(sys.argv[1])
    groups = sys.argv[2:] or ["decoupled", "decoupled_genreg"]
    tables = {g: by_j(d / f"diversity_{g}.json") for g in groups}
    js = sorted(set().union(*[set(t) for t in tables.values()]))

    print(f"\n=== structured-commit A/B by j  (pooled over k, temp; dir={d}) ===")
    hdr = f"{'j':>2} | " + " | ".join(f"{g:>16}" for g in groups)
    print(hdr); print("-" * len(hdr))
    for metric in ["calibration_tv", "coverage", "readout_tv", "gap", "abstain_rate"]:
        print(f"[{metric}]")
        for j in js:
            cells = " | ".join(f"{tables[g].get(j, {}).get(metric, float('nan')):>16.3f}" for g in groups)
            print(f"{j:>2} | {cells}")
    # explicit gap delta (genreg - decoupled) if both present
    if set(groups) >= {"decoupled", "decoupled_genreg"}:
        print("\n[gap delta = decoupled_genreg - decoupled]  (negative = gap shrank)")
        for j in js:
            gd = tables["decoupled_genreg"].get(j, {}).get("gap", float("nan"))
            dc = tables["decoupled"].get(j, {}).get("gap", float("nan"))
            ad = tables["decoupled_genreg"].get(j, {}).get("abstain_rate", float("nan"))
            ac = tables["decoupled"].get(j, {}).get("abstain_rate", float("nan"))
            print(f"  j={j}: dgap={gd - dc:+.3f}   dabstain={ad - ac:+.3f}")


if __name__ == "__main__":
    main()
