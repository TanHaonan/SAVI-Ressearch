# VERDICT — PRM boundary-step calibration check

**Result in one line:** real value functions concentrate their errors at the
on/off-path boundary (~3.5× FP gradient), robustly across two independent value
families — so the phasemap's uniform-ε basis fails and the math harness must
simulate boundary-clustered noise before the competence-flip finding can transfer.

**Candidate:** `Qwen/Qwen2.5-Math-PRM-7B` (primary). **Control:**
`Qwen/Qwen2.5-7B-Instruct` generative critic.
**Substrate:** ProcessBench, all 4 configs, 3400 solutions → 16,548 labeled steps
(14,327 good + 2,221 first-error). θ = 0.850 (max per-step F1 on gsm8k).

## Reconstruction validated
The published checkpoint's remote modeling code NaNs under transformers 5.7
(legacy `DynamicCache` + attention-mask API). We rebuilt it from native parts —
maintained `Qwen2Model` backbone + the PRM's exact head
`Sequential(Linear(h,h), ReLU, Linear(h,2))` (keys `score.0/score.2`). Faithful:
gsm8k ProcessBench first-error F1 = **0.822**, matching Qwen's published **82.4**.
(Hard-config F1 runs lower — math .717, olympiad .567, omni .550 — mainly because
θ is tuned on gsm8k; this is the sanity anchor, not the deliverable, and the
verdict is checked across θ below.)

## Headline: errors are CONCENTRATED at the boundary
Held-out (math+olympiad+omni), n_good=12,966 n_bad=2,014, θ=0.85:

- **Polarity is roughly balanced** at the honest operating point: FP=0.111
  [0.106,0.116], FN=0.123 [0.109,0.138], **FN/FP=1.11**. The strong *positive*
  bias reported in the literature (Socratic-PRMBench) appears only at low θ
  (FN/FP=3.06 at θ=0.70); at max-F1 θ the PRM is polarity-balanced.

- **False positives concentrate on the steps adjacent to the first error**
  (FP rate among *good* steps, by signed distance to the boundary):

  | dist to boundary | -1 (last correct) | -2 | -3 | -4 | ≤-5 |
  |---|---|---|---|---|---|
  | FP rate | **0.234** | 0.156 | 0.121 | 0.102 | 0.066 |

  Monotone: the last correct step is flagged bad **~3.5× more often** than steps
  far from the boundary, and **2.11× the pooled FP rate**. This is exactly the
  "on-path vs off-path close" regime — the PRM is least reliable precisely at the
  hardest discrimination.

- **No strong positional (late-detection) effect** at this θ: error rate by
  early/mid/late thirds = 0.112 / 0.123 / 0.099 (flat).

**Robust to threshold:** the held-out verdict is CONCENTRATED at every θ in
[0.70, 0.99]; only the flagged axis shifts (polarity at extreme θ, distance in
the middle). Per-config: gsm8k / math / olympiadbench CONCENTRATED; omnimath
reads UNIFORM *only* because its high baseline FP (0.139) compresses the ratio —
the d=-1 gradient (0.231 vs 0.069 at ≤-5) is still present.

## Implication for the math harness (the gate)
**FAIL the uniformity gate.** The `merge-noise-phasemap` competence-flip finding
was established under uniform ε (`domain_merge.py:223`, sha256-keyed flips). The
real PRM violates that assumption: its errors cluster at the on/off-path boundary
— the exact region where decode and end-selection diverge. Therefore the finding
**cannot be claimed to transfer as-is**. The harness must:
1. Replace the uniform flip in `domain_merge.py:223` with the **measured
   distance-dependent FP profile** above (and add an `fp`/positive-bias mode —
   currently only `sym`/`fn` exist), then re-run the phasemap to test whether the
   decode-vs-select advantage survives boundary-concentrated noise.
2. Use the calibrated θ, not a default.

## Control comparison — the caveat is RESOLVED
Control: `Qwen/Qwen2.5-7B-Instruct` as a **generative critic** (reasons, then
emits a first-error verdict; ProcessBench protocol). Compared to the PRM on the
common 1,600-solution subset (1,407 erroneous), good steps = correct prefixes:

| value (held-out) | FP d=-1 | d=-2 | d=-3 | d=-4 | d≤-5 | **gradient d=-1/d≤-5** | FN/FP |
|---|---|---|---|---|---|---|---|
| PRM (θ=0.85) | 0.231 | 0.140 | 0.114 | 0.103 | 0.065 | **3.58×** | 0.84 |
| 7B critic (θ=0.50) | 0.078 | 0.035 | 0.033 | 0.021 | 0.023 | **3.43×** | 16.68 |

**Two independent value families — a trained PRM and a generative LLM critic
(different architecture, training, and mechanism) — concentrate false positives
at the boundary by the same ~3.5× factor, both monotone in distance.** The d=-1
spike is therefore **data-driven, not PRM-specific**: the step immediately before
the first error is intrinsically the hardest to classify, and *any* imperfect
value inherits the concentration. This makes the gate failure **robust and
mechanism-independent** — it cannot be engineered away by swapping PRMs.

What *does* differ by family: **polarity** (PRM balanced FN/FP≈0.84; the 7B
critic is strongly positive-biased, FN/FP≈16.7, FN=0.73 — it misses ~¾ of
errors) and **position** (the critic is far worse on late steps, late/early≈3.0;
the PRM is flat). So polarity/position are knobs to vary; boundary concentration
is a fixed feature of the regime.

Caveats: (i) report the **gradient**, not the binary UNIFORM/CONCENTRATED flag —
the flag is baseline-sensitive (on this all-erroneous subset the PRM's pooled FP
rises to 0.143, pushing the d=-1 ratio just under the 2× flag even though the
gradient is 3.58×). (ii) The 400/config subsample drew only erroneous solutions
for math/olympiad/omni (those files are ordered erroneous-first), so the critic's
*standalone* ProcessBench F1 is degenerate (no all-correct solutions to score);
this does not affect the FP-by-distance comparison, which uses the shared good
steps. A stratified re-sample would tighten the critic's standalone numbers.

## Bottom line for the math harness
The competence-flip finding rests on **uniform** ε; real value functions —
robustly, across families — put their errors **at the on/off-path boundary**
(~3.5× gradient), exactly where decode and end-selection diverge.

**Transfer test run (`../merge-noise-phasemap/VERDICT_boundary_transfer.md`).**
Wired the measured profile into `domain_merge.py` as a mean-preserving
`profile="boundary"` (same total ε, redistributed to the boundary; the `fp` mode
already existed) and re-ran the load-bearing cell. Result: the competence-flip
**survives in its soft form but the hard mask dies**. At p=0.5/0.6/0.7 (d24,
ε=0.18, fn), A4 soft (λ=0.1) holds CI>0 (+0.245→+0.214, +0.432→+0.333,
+0.464→+0.464) while A3 hard mask collapses from clearly positive to tie-or-loss
(+0.354→−0.047, +0.297→−0.109, +0.219→−0.177). Since ε is mean-matched, this is a
pure geometry effect: the soft down-weight-not-delete decode is robust to
boundary-clustered errors; hard pruning is not. The gate "failure" is thus a
**positive** result for the soft verifier-decode and a negative one for masking.

## Artifacts
- `results/prm_all.json`, `results/prm_verdict.json` — PRM rewards + numbers.
- `results/judge_all.json`, `results/judge_verdict.json` — critic preds + numbers.
- `results/compare.json` — PRM-vs-critic comparison.
- `score_prm.py` (native-backbone scorer), `score_judge.py` (generative critic),
  `metrics.py` (+`test_metrics.py`), `data.py`, `run.py`, `compare.py`.
