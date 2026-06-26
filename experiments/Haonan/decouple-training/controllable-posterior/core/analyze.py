"""Aggregate outputs/run_*.json into the pre-registered tables + decision checks."""
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent / "outputs"

def load():
    # main grid only: (mode,regime) -> seed -> cells; skip cross-talk (--only_j) runs so the table stays clean
    runs = defaultdict(dict)
    for f in glob.glob(str(OUT / "run_*.json")):
        d = json.loads(Path(f).read_text())
        if d["cfg"].get("only_j"):
            continue
        runs[(d["mode"], d["regime"])][d["cfg"]["seed"]] = d["cells"]
    return runs

def load_records():
    # every run as a flat record (includes --only_j runs) for the gap/objective/cross-talk decisions
    recs = []
    for f in glob.glob(str(OUT / "run_*.json")):
        d = json.loads(Path(f).read_text())
        recs.append(dict(mode=d["mode"], regime=d["regime"], seed=d["cfg"]["seed"],
                         only_j=d["cfg"].get("only_j", ""), cells=d["cells"]))
    return recs

def _cell_mean(recs, key, field):
    """mean over the given runs of cells[key][field], skipping runs lacking that cell."""
    vals = [r["cells"][key][field] for r in recs if key in r["cells"] and r["cells"][key].get(field) is not None]
    return float(np.mean(vals)) if vals else None

def decision_inference_gap(recs):
    # self.tv - oracle.tv at matched (k,j), decoupled regime; positive = inference costs
    o = [r for r in recs if r["mode"] == "oracle" and r["regime"] == "decoupled" and not r["only_j"]]
    s = [r for r in recs if r["mode"] == "self" and r["regime"] == "decoupled" and not r["only_j"]]
    if not o or not s:
        print("\n=== inference gap (self - oracle TV, decoupled) ===\n  needs both oracle+self decoupled runs (absent)")
        return
    keys = sorted(set().union(*[r["cells"] for r in o], *[r["cells"] for r in s]),
                  key=lambda x: tuple(map(int, x.split(","))))
    byj = defaultdict(list)
    for key in keys:
        ov, sv = _cell_mean(o, key, "tv_mean"), _cell_mean(s, key, "tv_mean")
        if ov is not None and sv is not None:
            byj[int(key.split(",")[1])].append(sv - ov)
    print("\n=== inference gap (self - oracle TV, decoupled); positive = inference costs ===")
    for j in sorted(byj):
        print(f"  j={j}: mean(self-oracle TV)={np.mean(byj[j]):+.3f}")

def decision_objective(recs):
    # decoupled vs coupled within-survivor non-uniformity on ambiguous (j>=2); shuffle TV (should be high/chance)
    print("\n=== objective: ambiguous (j>=2) within-survivor non-uniformity; shuffle TV ===")
    for mode in sorted(set(r["mode"] for r in recs)):
        def amb_nonunif(regime):
            rs = [r for r in recs if r["mode"] == mode and r["regime"] == regime and not r["only_j"]]
            if not rs:
                return None
            keys = sorted(set().union(*[r["cells"] for r in rs]))
            vals = [_cell_mean(rs, key, "nonunif_mean") for key in keys if int(key.split(",")[1]) >= 2]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None
        def mean_tv(regime):
            rs = [r for r in recs if r["mode"] == mode and r["regime"] == regime and not r["only_j"]]
            if not rs:
                return None
            keys = sorted(set().union(*[r["cells"] for r in rs]))
            vals = [_cell_mean(rs, key, "tv_mean") for key in keys]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None
        dec, cou, shf = amb_nonunif("decoupled"), amb_nonunif("coupled"), mean_tv("shuffle")
        parts = []
        if dec is not None: parts.append(f"decoupled nonunif={dec:.3f}")
        if cou is not None: parts.append(f"coupled nonunif={cou:.3f}")
        if shf is not None: parts.append(f"shuffle meanTV={shf:.3f}")
        if dec is not None and cou is not None:
            parts.append("(decoupled < coupled expected)" if dec < cou else "(coupled NOT worse)")
        print(f"  {mode:6s} | " + ("  ".join(parts) if parts else "no decoupled/coupled/shuffle runs"))

def decision_crosstalk(recs):
    # k=4 mixed vs separate (--only_j): ambiguous tv_mean / nonunif_mean delta
    onlyj = [r for r in recs if r["only_j"]]
    if not onlyj:
        print("\n=== cross-talk (k=4 mixed vs separate) ===\n  cross-talk runs not present (run --only_j cells)")
        return
    mixed = [r for r in recs if r["mode"] == "oracle" and r["regime"] == "decoupled" and not r["only_j"]]
    sep_amb = [r for r in onlyj if set(r["only_j"].replace(",", "")) - {"1"}]   # an only_j run covering j>=2
    if not mixed or not sep_amb:
        print("\n=== cross-talk (k=4 mixed vs separate) ===\n  needs both mixed oracle+decoupled and an ambiguous --only_j run (absent)")
        return
    amb_keys = [k for k in mixed[0]["cells"] if k.startswith("4,") and int(k.split(",")[1]) >= 2]
    print("\n=== cross-talk (k=4): mixed - separate (ambiguous); positive = mixed worse ===")
    for key in sorted(amb_keys, key=lambda x: tuple(map(int, x.split(",")))):
        mt, st = _cell_mean(mixed, key, "tv_mean"), _cell_mean(sep_amb, key, "tv_mean")
        mn, sn = _cell_mean(mixed, key, "nonunif_mean"), _cell_mean(sep_amb, key, "nonunif_mean")
        if mt is not None and st is not None:
            extra = f" nonunif d={mn - sn:+.3f}" if (mn is not None and sn is not None) else ""
            print(f"  {key}: TV mixed-sep={mt - st:+.3f}{extra}")

def main():
    runs = load()
    print("=== mean TV (held-out) per (k,j); lower=closer to true posterior ===")
    for (mode, regime), seeds in sorted(runs.items()):
        cells = list(seeds.values())
        keys = sorted(cells[0], key=lambda s: tuple(map(int, s.split(","))))
        row = "  ".join(f"{key}:{np.mean([c[key]['tv_mean'] for c in cells]):.2f}" for key in keys)
        print(f"{mode:6s} {regime:9s} | {row}")
    # capacity check: oracle decoupled TV by j
    od = runs.get(("oracle", "decoupled"))
    if od:
        cells = list(od.values()); cap = {}
        for key in cells[0]:
            k, j = map(int, key.split(","))
            cap.setdefault(j, []).append(np.mean([c[key]["tv_mean"] for c in cells]))
        print("\n=== capacity (oracle+decoupled): mean TV by j ===")
        for j in sorted(cap):
            print(f"  j={j}: TV={np.mean(cap[j]):.3f}  -> {'OK(<=.10)' if np.mean(cap[j])<=0.10 else 'HIGH'}")
        print("  prediction: TV<=.10 for ALL j -> capacity present (k>=3 not a structural limit)")

    recs = load_records()
    decision_inference_gap(recs)
    decision_objective(recs)
    decision_crosstalk(recs)

if __name__ == "__main__":
    main()
