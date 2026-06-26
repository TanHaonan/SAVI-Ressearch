"""Aggregate outputs/run_*.json into four summary tables and print a decision for each. Reads only the
on-disk JSONs written by run.py; loads no model.

Plain-language summary of what each table answers (the headline finding goes first when you read the output):
  Table 1 (readout)  Does keeping the correct option alive in the per-slot readout matter?  -> compare
      oracle-edge + marginal joint exact-match across the three emission sources {base, coupled, decoupled},
      and report the paired CI of (decoupled - coupled). If decoupled clearly beats coupled, the calibrated
      readout is necessary; if they tie, calibration buys nothing on this carrier (a conditions-bound null).
  Table 2 (edges)    Does global decoding beat per-slot greedy when the edges come from a cheap, separate
      model?  -> with the decoupled emission, marginal joint-EM by edge source {greedy baseline, self, cheap,
      oracle, shuffle}, each with a CI of the gain over the greedy baseline.
  Table 3 (path)     Does single-best-path (viterbi) add accuracy over the per-position marginal?  -> viterbi
      - marginal joint-EM, pooled across cells, with a CI. Expected small (<= ~0.05); reported, not headline.
  Table 4 (decorrelation)  Are the cheap-model edges actually independent of the self-model edges?  -> edge
      phi of the cheap builder vs the self builder. Low phi (< 0.5) = independent.

Bootstrap CIs:
  - per-source/per-cell means use the item-bootstrap percentile CI already stored by run.py (boot_ci).
  - paired differences (decoupled-coupled, source-vs-greedy, viterbi-marginal) are recomputed here from the
    per-item joint-EM vectors (`em_vec`) via metrics._bootstrap_ci so the difference is properly paired.
"""
import argparse, importlib.util as ilu, json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _bp(name, p):
    s = ilu.spec_from_file_location(name, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


metrics = _bp("dep_metrics", HERE / "_deps" / "metrics.py")


def _load_runs(outdir):
    """Return {emission: [run_dict, ...]} keyed by emission, one entry per seed file."""
    runs = defaultdict(list)
    for p in sorted(outdir.glob("run_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"  [warn] could not parse {p.name}: {e!r}")
            continue
        runs[d.get("emission", p.stem)].append(d)
    return runs


def _emvec(run, src, dec):
    """Per-item joint-EM vector for (src, dec) in a run, or None if absent."""
    cells = run.get("cells", {})
    if src not in cells or dec not in cells[src]:
        return None
    return np.asarray(cells[src][dec].get("em_vec", []), float)


def _pool_emvec(runs_for_emission, src, dec):
    """Concatenate the per-item joint-EM vectors across seeds for one emission/source/decoder."""
    parts = []
    for run in runs_for_emission:
        v = _emvec(run, src, dec)
        if v is not None and v.size:
            parts.append(v)
    return np.concatenate(parts) if parts else np.array([], float)


def _mean(v):
    return float(v.mean()) if v.size else float("nan")


def _paired_ci(a, b):
    """Bootstrap CI of mean(a - b) when a,b are paired (same length); else a per-element-truncated pair."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(a.size, b.size)
    if n == 0:
        return [float("nan"), float("nan")]
    return metrics._bootstrap_ci(a[:n] - b[:n])


def table_readout(runs):
    print("\n" + "=" * 78)
    print("Table 1 (readout) - Node subspace necessity (oracle edges, marginal decoder, joint exact-match)")
    print("     Does keeping the correct option alive in the readout matter?")
    print("=" * 78)
    em = {}
    for emission in ("base", "coupled", "decoupled"):
        v = _pool_emvec(runs.get(emission, []), "oracle", "marginal")
        em[emission] = v
        print(f"  emission={emission:10s} oracle+marginal joint-EM = {_mean(v):.3f}  (n={v.size})")
    dec_v, cou_v = em.get("decoupled", np.array([])), em.get("coupled", np.array([]))
    if dec_v.size and cou_v.size:
        diff = _mean(dec_v[:min(dec_v.size, cou_v.size)]) - _mean(cou_v[:min(dec_v.size, cou_v.size)])
        ci = _paired_ci(dec_v, cou_v)
        print(f"  paired (decoupled - coupled) = {diff:+.3f}  CI95 = [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        gr = _pool_emvec(runs.get("decoupled", []), "oracle", "greedy")
        dec_greedy = _mean(dec_v) - _mean(gr) if gr.size else float("nan")
        # decision
        if ci[0] > 0:
            print("  DECISION (readout): decoupled > coupled (CI excludes 0) -> calibrated readout is NECESSARY "
                  "for oracle edges to recover gold.")
        elif gr.size and abs(_mean(dec_v) - _mean(gr)) < 1e-9 and abs(_mean(cou_v) - _mean(gr)) < 1e-9:
            print("  DECISION (readout): FORK B: even decoupled+oracle ~ greedy: the correct option is "
                  "not readable from the forward pass (node side is the blocker).")
        else:
            print("  DECISION (readout): FORK A: coupled+oracle ~ decoupled+oracle: the subspace was alive "
                  "anyway on this carrier; calibration buys nothing here (conditions-bound null).")
    else:
        print("  DECISION (readout): insufficient runs (need both coupled and decoupled emission files).")


def table_edges(runs):
    print("\n" + "=" * 78)
    print("Table 2 (edges) - Edge make-or-break (decoupled emission, marginal decoder, joint exact-match)")
    print("     Does global decoding beat per-slot greedy with cheap decorrelated edges?")
    print("=" * 78)
    dec_runs = runs.get("decoupled", [])
    if not dec_runs:
        print("  DECISION (edges): no decoupled emission runs found.")
        return
    # greedy baseline: the greedy decoder ignores edges, so any source's greedy column is the baseline.
    base_src = None
    for s in ("oracle", "self", "cheap", "shuffle"):
        if _pool_emvec(dec_runs, s, "greedy").size:
            base_src = s; break
    greedy_v = _pool_emvec(dec_runs, base_src, "greedy") if base_src else np.array([])
    print(f"  greedy baseline (edge-free)   joint-EM = {_mean(greedy_v):.3f}  (n={greedy_v.size})")
    rows = {}
    for src in ("self", "cheap", "oracle", "shuffle"):
        v = _pool_emvec(dec_runs, src, "marginal")
        rows[src] = v
        if not v.size:
            print(f"  edges={src:8s} marginal joint-EM = (absent)")
            continue
        gain = _mean(v) - _mean(greedy_v) if greedy_v.size else float("nan")
        ci = _paired_ci(v, greedy_v) if greedy_v.size else [float("nan")] * 2
        print(f"  edges={src:8s} marginal joint-EM = {_mean(v):.3f}  gain vs greedy = {gain:+.3f}  "
              f"CI95 = [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    # decision logic
    def beats(src):
        v = rows.get(src, np.array([]))
        if not v.size or not greedy_v.size:
            return False
        return _paired_ci(v, greedy_v)[0] > 0
    oracle_beats = beats("oracle")
    cheap_beats = beats("cheap")
    if oracle_beats and cheap_beats:
        print("  DECISION (edges): cheap edges beat greedy (CI excludes 0) -> a small decorrelated model supplies "
              "usable constraint edges; the gain is globality + decorrelation, not scale.")
    elif oracle_beats and not cheap_beats:
        print("  DECISION (edges): FORK A: oracle beats greedy but cheap ~ greedy: constraint extraction "
              "(parsing the NL constraint) is the bottleneck, not the decoder.")
    elif not oracle_beats:
        print("  DECISION (edges): FORK B: even oracle ~ greedy: no genuine cross-slot coupling to exploit "
              "on this carrier (fix the carrier: raise density / strengthen the conflict).")


def table_path(runs):
    print("\n" + "=" * 78)
    print("Table 3 (path) - Viterbi vs marginal (viterbi - marginal joint-EM, pooled over cells)")
    print("     Does single-best-path add accuracy over the per-position marginal? (expected small)")
    print("=" * 78)
    any_row = False
    kill_cells = []          # cells where viterbi >> marginal: diff > 0.05 AND CI lower bound > 0
    max_lo = -np.inf         # largest CI lower bound seen (for the "all CIs include 0" summary)
    for emission in ("base", "coupled", "decoupled"):
        for src in ("oracle", "self", "cheap", "shuffle"):
            vit = _pool_emvec(runs.get(emission, []), src, "viterbi")
            mar = _pool_emvec(runs.get(emission, []), src, "marginal")
            if not (vit.size and mar.size):
                continue
            any_row = True
            n = min(vit.size, mar.size)
            diff = float((vit[:n] - mar[:n]).mean())
            ci = metrics._bootstrap_ci(vit[:n] - mar[:n])     # [lo, hi]
            max_lo = max(max_lo, ci[0])
            if diff > 0.05 and ci[0] > 0.0:                   # CI excludes 0 from below AND beats the 0.05 bar
                kill_cells.append((emission, src, diff, ci))
            print(f"  emission={emission:10s} edges={src:8s} "
                  f"viterbi-marginal = {diff:+.3f}  CI95 = [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    if not any_row:
        print("  DECISION (path): no runs with both viterbi and marginal columns.")
        return
    if not kill_cells:
        print(f"  DECISION (path): viterbi-marginal <= ~0.05 with CI including 0 in every cell -> the gain is "
              f"globality, not path; use the marginal version as primary (expected outcome, NOT a failure).")
    else:
        worst = max(kill_cells, key=lambda r: r[2])
        print(f"  DECISION (path): SURPRISE (path is special): viterbi >> marginal in {len(kill_cells)} cell(s) "
              f"(diff > 0.05, CI excludes 0); worst at emission={worst[0]}/edges={worst[1]} "
              f"diff={worst[2]:+.3f} -> genuine fan structure exists (a surprise vs prior).")


def table_decorr(runs):
    print("\n" + "=" * 78)
    print("Table 4 (decorrelation) - Decorrelation sanity (edge phi: cheap builder vs self builder)")
    print("     Are the cheap-model edges actually independent of the Qwen3-4B edges?")
    print("=" * 78)
    # phi is stored per run (computed in evaluate_with against the self builder).
    cheap_phis, self_phis = [], []
    for emission, runlist in runs.items():
        for run in runlist:
            phi = run.get("phi", {}) or run.get("cells", {}).get("_phi", {})
            if "cheap" in phi:
                cheap_phis.append(phi["cheap"])
            if "self" in phi:
                self_phis.append(phi["self"])
    if cheap_phis:
        cm = float(np.mean(cheap_phis))
        print(f"  cheap-builder phi vs self = {cm:.3f}  (mean over {len(cheap_phis)} runs)")
        if cm < 0.5:
            print("  DECISION (decorrelation): cheap phi < 0.5 -> the cheap model's edge errors are decorrelated from "
                  "Qwen3-4B's; any P2 gain can be read as coming from independence.")
        else:
            print("  DECISION (decorrelation): FAIL: cheap phi >= 0.5: the 'decorrelated' model is not actually "
                  "independent here; its net gain (if any) is not from independence.")
    else:
        print("  DECISION (decorrelation): no cheap-builder phi recorded (run with --edges including cheap).")
    if self_phis:
        sm = float(np.mean(self_phis))
        print(f"  self-builder phi vs self  = {sm:.3f}  (sanity: should be ~1.0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE / "outputs"))
    a = ap.parse_args()
    outdir = Path(a.outdir)
    runs = _load_runs(outdir)
    if not runs:
        raise SystemExit(f"no run_*.json found in {outdir}; run run.py first")
    print(f"loaded runs: " + ", ".join(f"{em}x{len(rl)}" for em, rl in sorted(runs.items())))
    table_readout(runs)
    table_edges(runs)
    table_path(runs)
    table_decorr(runs)


if __name__ == "__main__":
    main()
