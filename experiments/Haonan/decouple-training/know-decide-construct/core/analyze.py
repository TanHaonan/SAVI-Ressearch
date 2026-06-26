"""Analysis: frontier + matched-sharpness paired-bootstrap test + in-the-forward summary.
Reads outputs/run_*.json and outputs/internalize_*.json. Pure CPU. Writes outputs/frontier.json.
"""
import json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def boot_paired(diff, n_boot=2000, seed=0):
    """CI on the mean of a paired difference array."""
    v = np.asarray(diff, float)
    if len(v) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [float(np.percentile(means, q)) for q in (2.5, 50, 97.5)]


def boot_mean(vals, n_boot=2000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [float(np.percentile(means, q)) for q in (2.5, 50, 97.5)]


def main():
    runs = {}
    for f in glob.glob(str(OUT / "run_*.json")):
        d = json.loads(Path(f).read_text())
        runs[d["tag"]] = d

    # group
    dec = defaultdict(dict)   # lam -> seed -> run
    coup, shuf = {}, {}
    for tag, d in runs.items():
        cfg = d["cfg"]; s = cfg["seed"]
        if d["regime"] == "decoupled":
            dec[int(cfg["lam_shape"])][s] = d
        elif d["regime"] == "coupled":
            coup[s] = d
        else:
            shuf[s] = d

    # frontier (mean over seeds)
    frontier = []
    for lam in sorted(dec):
        ds = list(dec[lam].values())
        frontier.append(dict(lam=lam, n_seed=len(ds),
                             det=float(np.mean([x["final"]["det_acc"] for x in ds])),
                             amb=float(np.mean([x["final"]["amb_abs_dev"] for x in ds])),
                             ent=float(np.mean([x["final"]["amb_entropy"] for x in ds])),
                             ppl_ratio=float(np.mean([x["final"]["ppl_ratio"] for x in ds]))))
    coup_det = float(np.mean([coup[s]["final"]["det_acc"] for s in coup])) if coup else float("nan")
    coup_amb = float(np.mean([coup[s]["final"]["amb_abs_dev"] for s in coup])) if coup else float("nan")
    shuf_det = [shuf[s]["final"]["det_acc"] for s in shuf] if shuf else []

    print("=== frontier (decoupled, mean over seeds) ===")
    for r in frontier:
        print(f"  lam={r['lam']:3d}  det={r['det']:.3f}  amb|p-.5|={r['amb']:.3f}  ent={r['ent']:.3f}  ppl x{r['ppl_ratio']:.2f}")
    print(f"  coupled: det={coup_det:.3f} amb={coup_amb:.3f}   shuffle det={np.mean(shuf_det) if shuf_det else float('nan'):.3f}")

    # matched-sharpness: smallest lam with mean det >= coupled det
    cand = [r for r in frontier if r["det"] >= coup_det]
    matched = min(cand, key=lambda r: r["lam"]) if cand else (max(frontier, key=lambda r: r["det"]) if frontier else None)
    ms = dict(coupled_det=coup_det, coupled_amb=coup_amb)
    if matched:
        lam = matched["lam"]
        ms["matched_lam"] = lam
        ms["matched_decoupled_det"] = matched["det"]
        ms["matched_decoupled_amb"] = matched["amb"]
        ms["reached_coupled_sharpness"] = bool(matched["det"] >= coup_det)
        per_seed = []
        pooled = []
        for s in sorted(set(dec[lam]) & set(coup)):
            dd = dec[lam][s]; cc = coup[s]
            # pair by amb id
            cid = {i: v for i, v in zip(cc["final"]["amb_ids"], cc["final"]["amb_abs_dev_items"])}
            did = {i: v for i, v in zip(dd["final"]["amb_ids"], dd["final"]["amb_abs_dev_items"])}
            ids = [i for i in cid if i in did]
            diff = [cid[i] - did[i] for i in ids]   # +ve = decoupled better calibrated
            ci = boot_paired(diff, seed=s)
            per_seed.append(dict(seed=s, n=len(ids), mean_diff=float(np.mean(diff)), ci=ci, win=bool(ci[0] > 0)))
            pooled += diff
        ms["per_seed"] = per_seed
        ms["pooled_ci"] = boot_paired(pooled)
        ms["n_seed_win"] = sum(p["win"] for p in per_seed)
        ms["WIN"] = bool(ms["n_seed_win"] >= 2)
    print("\n=== matched-sharpness (decoupled vs coupled, +diff = decoupled better calibrated) ===")
    if matched:
        print(f"  matched lam={ms.get('matched_lam')} (det {ms.get('matched_decoupled_det'):.3f} vs coupled {coup_det:.3f}, "
              f"reached={ms.get('reached_coupled_sharpness')})")
        for p in ms.get("per_seed", []):
            print(f"   seed{p['seed']}: mean_diff={p['mean_diff']:+.3f} CI={[round(x,3) for x in p['ci']]} win={p['win']}")
        print(f"  pooled CI={[round(x,3) for x in ms['pooled_ci']]}  seeds_win={ms['n_seed_win']}/  WIN={ms['WIN']}")

    # internalize
    intern = []
    for f in glob.glob(str(OUT / "internalize_*.json")):
        d = json.loads(Path(f).read_text())
        diff = np.array(d["readout_corr"]) - np.array(d["native_corr"])  # paired per item
        ci = boot_paired(diff)
        intern.append(dict(tag=d["tag"], native=d["native_det"], readout=d["readout_det"],
                           delta=d["readout_minus_native"], ci=ci,
                           internalized=bool(ci[2] <= 0.03), offloaded=bool(ci[0] >= 0.10)))
    print("\n=== internalize (readout on adapted forward minus native; <=+.03=internalized, >=+.10=offload) ===")
    for d in sorted(intern, key=lambda x: x["tag"]):
        print(f"  {d['tag']:22s} native={d['native']:.3f} readout={d['readout']:.3f} "
              f"delta={d['delta']:+.3f} CI={[round(x,3) for x in d['ci']]} "
              f"{'INTERNALIZED' if d['internalized'] else ('OFFLOAD' if d['offloaded'] else 'partial')}")

    summary = dict(frontier=frontier, coupled=dict(det=coup_det, amb=coup_amb),
                   shuffle_det=shuf_det, shuffle_det_ci=boot_mean(shuf_det) if shuf_det else None,
                   matched_sharpness=ms, internalize=intern)
    (OUT / "frontier.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'frontier.json'}")


if __name__ == "__main__":
    main()
