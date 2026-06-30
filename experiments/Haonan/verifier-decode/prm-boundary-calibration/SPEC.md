# SPEC — PRM boundary-step calibration check

## Purpose
Gate the math harness. The `merge-noise-phasemap` "competence flip" finding
(verifier-decode beats selection at low generator competence) was established
under **uniform** label noise: `domain_merge.py:223` corrupts labels at a fixed
rate ε keyed by `sha256(state)`, i.e. errors are spread evenly by construction.
Real PRMs do not behave this way — their errors are directional and
position-dependent (positive/negative reward bias, late-detection shift). If the
chosen PRM's errors are **concentrated** near the on-path/off-path boundary, then
a decode win is reducible to "exploiting a structured blind spot" — the
`near-exact mask is the only lever` critique that closed the `joint-lambda`
framing. This check measures the error *geometry* before we commit the harness.

Question: **is the PRM's error rate on near-boundary steps roughly uniform, or
concentrated (by polarity, by distance-to-boundary, or by absolute position)?**

## Candidates
- **Primary:** `Qwen/Qwen2.5-Math-PRM-7B` (SOTA open 7B PRM; positive/lenient
  bias expected). Scoring: `AutoModel`, `<extra_0>` step separator, per-step
  positive-class probability (`make_step_rewards`).
- **Control:** `Qwen/Qwen2.5-7B-Instruct` as LM-as-step-judge (size-matched;
  expected better-calibrated). Per-step good/bad probability via a judging prompt.
- Free contrast (cached): `peiyi9979/math-shepherd-mistral-7b-prm`.

## Substrate
ProcessBench (local JSON: gsm8k/math/olympiadbench/omnimath), human-annotated
first-error step. Per solution with first-error index `k`:
- `k == -1`: all steps good (`truth=1`, no boundary).
- `k >= 0`: steps `[0..k-1]` good (`truth=1`, `dist=i-k<0`); step `k` bad
  (`truth=0`, `dist=0`); steps `> k` **excluded** (true quality unknown).

"Near-boundary, on-path vs off-path close" = `dist=-1` (last correct) vs `dist=0`
(first error) — the hardest discrimination.

## Per-step signal
PRM reward `r_i ∈ (0,1)`. Threshold θ → predict bad if `r_i < θ`.
- **FP** = good step scored bad (`r_i < θ`, `truth=1`).
- **FN** = bad step scored good (`r_i ≥ θ`, `truth=0`).
θ chosen by max per-step F1 on a calibration split (gsm8k); uniformity evaluated
on the held-out configs. Report θ-sensitivity.

## Uniformity axes (the deliverable)
1. **Polarity:** overall FP rate vs FN rate. Asymmetry > `ratio_thresh` (default
   2×) ⇒ directional concentration (reward bias). Qwen-PRM expected FN ≫ FP.
2. **Distance-to-boundary:** FP rate among good steps binned by
   `d ∈ {-1,-2,-3,-4,≤-5}`. Any bin > 2× the pooled good-FP rate ⇒ concentration
   near the boundary.
3. **Absolute position:** error rate by `early/mid/late` thirds of the solution.
   `late/early` ratio > 2× ⇒ positional concentration (late-detection shift).
4. *(optional, deferred)* PRMBench 9-subcategory breakdown.

All rates with percentile bootstrap CI (`boot_ci`, n_boot=2000), mirroring
`cb.metrics.boot_ci`'s `[lo, mean, hi]` convention.

## Pass / fail
- **UNIFORM** ⇔ none of axes 1–3 flags concentration. The phasemap's uniform-ε
  basis is defensible; competence-flip finding can transfer as-is.
- **CONCENTRATED** ⇔ ≥1 axis flags. Name the dominant axis. Then the harness must
  (a) recalibrate θ and/or (b) replace the uniform `sha256` flip in
  `domain_merge.py:223` with the **measured** error profile and re-run the
  phasemap to test whether the finding survives realistic error geometry.

## Code-level feedback (regardless of verdict)
`domain_merge.py` has only `sym` and `fn` (negative-bias) noise modes. Qwen-PRM's
failure direction is **positive** (bad scored good): add an `fp` mode before the
harness can faithfully simulate it.

## Layout
- `data.py` — ProcessBench loader (local JSON, no network).
- `metrics.py` — pure: step table, FP/FN, by-distance / by-position profiles,
  threshold selection, uniformity verdict. (GPU-free, unit-tested.)
- `test_metrics.py` — unit tests for the above.
- `score_prm.py` / `score_judge.py` — per-step scorers (GPU).
- `run.py` — orchestrate: score → table → report → VERDICT.md.

## Validation
Small-slice (~20 solutions) end-to-end before the full multi-config run.
