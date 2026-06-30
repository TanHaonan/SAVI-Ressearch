# RESULTS — Recombination mechanism check (fair, same-source decode)

> **⚠ POST-HOC CORRECTION (read first — added after the workflow, by direct probe).**
> The headline verdict below ("no band; selection beats the trellis everywhere") is computed
> under **candidate parity**, and that is **the wrong iso-compute axis**: candidate parity
> prices *one full multi-step rollout the same as one single decoder step*, so it hands the
> selection baseline `~depth ×` more complete attempts. The Verify "fairness" agent flagged
> the asymmetry but then argued (incorrectly) that a moves/tokens axis "would give the trellis
> *even less* budget, so the sign would not reverse." **That reasoning is backwards.** A direct
> moves/exec-parity probe (Countdown, p=0.7, N=6, K∈{4,8}, depths 3–5) shows:
>
> - The candidate→exec axis switch **collapses selection's pass@1 in every cell** (e.g. depth-3
>   K=4: 1.00→0.50; depth-4 K=8: 0.75→0.25). The workflow's −0.6…−0.9 D1 was **largely a
>   candidate-parity artifact**.
> - On the honest exec axis the trellis is **competitive, not crushed**: `savi_freq −
>   iso_EXEC` ≈ ±0.1–0.4 (noisy, 8 samples/cell), with a **clear trellis win at K=8, depth 4
>   (+0.375)** and a loss at depth 5 (−0.375).
> - `savi_freq ≫ savi_support` in every cell — **the graded (frequency) edge carries the
>   signal; the support-only headline arm (all-zero edges, no ranking) is the weakest** and is
>   not where to look. Wider beam helps the trellis (K=8 > K=4 throughout).
>
> **Corrected bottom line:** the SAVI premise is **not refuted**. The original study stacked
> three choices that each disfavor the trellis — candidate parity + support-only headline arm +
> depth-3 substrate. The decisive design lesson for the real-LM run: **iso-compute must be
> tokens/moves, never candidate count**, and the headline arm must be `freq` (or
> verifier-masked), not `support`. A CI-grade exec/token-parity sweep is the proper next run.
> See `results.json → axis_correction_addendum` for the numbers. The original analysis below is
> retained verbatim for the record.
>
> ---

**This is a semi-synthetic mechanism check, NOT the real-LM result.** Every arm consumes a
synthetic per-step generator `P_p(move | state)` whose move quality is defined by the exact
backward oracle (`domain.solvable`). The competence parameter `p` is a stand-in for "a model
that proposes goal-reachable moves with probability `p`," not a measurement of any real model.
No conclusion here is a statement about how a real LM behaves end-to-end; the decisive number
is still gated on the emission line.

## Headline verdict

**No competence band exists.** The study's success condition was a band of `p` where the
trellis beats compute-matched selection (`D1 = savi_support − best_of_k_isobudget > 0`, CI
excluding 0) **and** that lift is attributable to the Φ-merge (`D_merge > 0`). The first
condition fails everywhere: **D1 is strongly negative at every competence below 1.0, with
bootstrap CIs excluding 0, and is exactly 0 only at `p = 1.0`** (where both arms saturate).
Compute-matched selection beats the trellis across the entire `p`-grid. The predicted
inverted-U does not appear on this budget axis.

The result is therefore the **opposite** of the hypothesis under the candidate-parity budget
definition, and we report it as such without softening.

### D1 — global decode vs. compute-matched selection (headroom stratum, n=80, paired bootstrap)

| p | savi_support pass@1 | best_of_k_iso pass@1 | D1 = savi − iso | 95% CI |
|----|----|----|----|----|
| 0.3 | 0.0125 | 0.275 | **−0.2625** | [−0.3625, −0.1625] |
| 0.5 | 0.0625 | 0.788 | **−0.7250** | [−0.825, −0.625] |
| 0.6 | 0.150 | 0.913 | **−0.7625** | [−0.8625, −0.650] |
| 0.7 | 0.1125 | 0.988 | **−0.8750** | [−0.9375, −0.800] |
| 0.8 | 0.225 | 1.000 | **−0.7750** | [−0.8625, −0.6875] |
| 0.9 | 0.3875 | 1.000 | **−0.6125** | [−0.7125, −0.500] |
| 1.0 | 1.000 | 1.000 | +0.000 | [0, 0] |

Every CI for `p < 1.0` lies entirely below zero. There is no "peak" with D1 > 0 to report.

## The Φ-merge is real and load-bearing, but the lift it produces is tiny

The merge is active and correctly ablated (Verify confirmed `beam_no_merge` is a clean
one-variable ablation of the canonical-state collapse). It produces a **small but positive**
`D_merge = savi_support − beam_no_merge` at several competences:

| p | D_merge | 95% CI | merge_ratio (before/after) | dp_only_wins (Σ5 seeds) | phi_attributed_wins (Σ5 seeds) |
|----|----|----|----|----|----|
| 0.3 | +0.0125 | [0.000, 0.0375] | 3.17 | 1 | 1 |
| 0.5 | +0.0125 | [0.000, 0.0375] | 3.16 | 0 | 1 |
| 0.6 | +0.0500 | [0.0125, 0.100] | 3.19 | 2 | 4 |
| 0.7 | +0.0125 | [0.000, 0.0375] | 3.17 | 0 | 1 |
| 0.8 | +0.0500 | [0.0125, 0.100] | 3.24 | 4 | 4 |
| **0.9** | **+0.0750** | **[0.025, 0.1375]** | 3.41 | 0 | 6 |
| 1.0 | +0.000 | [0, 0] | 15.29 | 0 | 0 |

`D_merge` excludes 0 at `p ∈ {0.6, 0.8, 0.9}` (and is marginal, `lo = 0`, at 0.3/0.5/0.7).
The peak is at **`p = 0.9`: D_merge = +0.075 [0.025, 0.1375]**, with 6 Φ-attributed wins
across 5 seeds. The merge demonstrably fires wherever `D_merge > 0` (min merge_ratio ≈ 3.1 in
those cells; never a no-op). So the Φ-merge does recover a handful of instances the per-path
beam misses — but at a magnitude (≤ 0.075 pass@1, single-digit instance counts summed over
five seeds) far too small to overturn the −0.6 to −0.9 deficit that the trellis carries
against the iso-budget selection baseline. **The band that the PLAN required (D1 > 0 AND
D_merge > 0) does not exist:** the only place D1 ≥ 0 is `p = 1.0`, where D_merge = 0.

### Secondary: D2 (graded vs. support edges) is strongly positive

`D2 = savi_freq − savi_support` is positive and rising across the band: `p0.5 +0.1875`,
`p0.6 +0.325`, `p0.7 +0.625 [0.5125, 0.7375]`, `p0.8 +0.725 [0.625, 0.825]`, `p0.9 +0.6125`.
Within the trellis family, frequency-weighted edges substantially beat support edges — the
graded posterior carries real signal. But even `savi_freq` (e.g. 0.95 at p0.8, 1.00 at p0.9)
only reaches parity with, not above, `best_of_k_isobudget` at the top of the range, and the
headline arm is `savi_support` by design.

## Controls — all hold

- **C1 (high competence, p=1.0):** `best_of_k = savi_support = 1.0`, `dp_only_wins = 0`. When
  the generator is perfect, the DP is redundant, as predicted.
- **C2 (low competence, p=0.3):** all arms near zero (`savi_support = 0.0125`,
  `best_of_k = 0.0`); `dp_only_wins` averages 0.2/cell. Almost nothing to recombine, as
  predicted. (The one dp-only and one phi win at p0.3 are within the predicted "≈0" noise.)
- **C3 (unsolvable):** 4 unsolvable instances; **no arm ever solves one** (no false-positive
  wins). Ceiling = 0 on those instances.
- **C4 (verifier on vs. off):** `savi_verifier_on ≥ savi_support` at every `p` (e.g. p0.6:
  0.8875 vs. 0.150; p0.8: 1.0 vs. 0.225). The oracle mask closes most of the gap, quantifying
  how much of the deficit is the synthetic model load (verifier-off) vs. the oracle. The
  headline stays verifier-off.

The controls behaving exactly as designed means the negative D1 is a genuine property of this
budget axis, not a broken harness.

## Algebra confirmation

A single confirmatory point on the real CAS state graph (`p=0.6`, K=6, N=8, depth=8, seed 1,
8 headroom instances, all solvable). The cell is a **null**: `savi_support = 0.0`,
`best_of_k_isobudget = 0.0`, `beam_no_merge = 0.0` → `D1 = +0.000 [0,0]`,
`D_merge = +0.000 [0,0]`, `dp_only_wins = 0`, `phi_attributed_wins = 0`. **No band confirmed
here** (single point; the tiny config solves nothing). The one positive thing it shows is that
the **Φ-merge is genuinely active on a real CAS Φ**: `merge_ratio = 8.0` (72 distinct
expansions collapsed to 9 canonical states), so the mechanism is not a Countdown-specific
artifact even though it produces no lift at this config. Fairness and determinism controls pass
(iso candidates 136 == savi candidates 136; independent re-run identical).

## Caveats raised by the Verify agents (folded in)

All three adversarial Verify passes returned `ok: true`, and all flagged the same load-bearing
interpretive point rather than a defect:

1. **Fairness audit — PASS, with a caveat on the budget axis.** No arm receives oracle/solver
   injection (the frozen `_sample_chain` oracle path was instrumented and reached **0 times**
   by any arm; all 3100 sampler calls on the probe were `mode="step"`). The iso-budget baseline
   is candidate-matched **exactly per instance** across all 980 (p, seed, instance) triples
   (candidate diff identically 0; on tokens/exec the iso arm gets *more*, so matching is
   generous to the baseline). D1 reproduces by hand from the per-instance flags. The caveat:
   **the candidate-parity axis is not semantically symmetric** — an iso "candidate" is one full
   root-to-leaf chain rollout, whereas a savi "candidate" is one single step, so at equal
   candidate counts the selection baseline explores many more *complete* paths. This is exactly
   the PLAN-specified budget axis, met precisely, so the comparison is fair by the study's own
   definition; but it is the structural reason D1 is negative. Under a tokens- or moves-parity
   axis the trellis would get *even less* relative budget, so the sign would not reverse — which
   strengthens (does not weaken) the negative finding.

2. **Φ-merge ablation validity — PASS.** `beam_no_merge` mirrors `savi` statement-for-statement
   and differs in exactly one variable (canonical-state keying vs. per-path nodes). It equals
   `savi` wherever no merge fires (K=1 on all 28 instances; verified grid) and diverges with
   strictly larger trellis widths wherever the merge fires (p0.9, K=8: all 28/28). `phi_attributed_wins`
   is the tight definition (validated solves `savi_support` makes that `beam_no_merge` does not,
   both through the same replay gate). `beam_no_merge` is never compute-starved relative to
   `savi`.

3. **Determinism — PASS.** A fresh-process, cache-free re-run of Countdown cell (p=0.7, seed=3)
   reproduced all 8 arms' pass@1, `dp_only_wins`, `phi_attributed_wins`, and `merge_ratio` to
   full float precision, plus the full per-instance maps and budgets bit-identically.

No corrections to the *numbers* were required; the only recommendations were reporting-honesty
items (state D1 < 0 plainly; document the budget-axis asymmetry), which this document follows.

## What this does NOT settle

- **The real-LM end-to-end number.** This study uses an oracle-defined synthetic generator to
  *parameterize* competence. It cannot say a real model lands in any particular regime, nor
  what its end-to-end pass@1 under SAVI would be. That number remains gated on the emission
  line delivering a real per-step generator.
- **Whether a different budget axis would change the sign.** We tested only candidate parity
  (the PLAN axis). The Verify audit argues tokens/moves parity would make the trellis deficit
  larger, not smaller, so the negative D1 is likely robust — but that has not been run.
- **Whether the Φ-merge ever produces a *decision-relevant* lift.** It produces a real but
  small `D_merge` (≤ 0.075); we have not found a regime where it is large.

### How this de-risks / predicts the decisive run

The honest read is that this **does not de-risk in the direction hoped**, and that is itself
informative for the decisive run:

- It establishes that under fair, equal-candidate compute, **the trellis mechanism alone does
  not beat plain compute-matched selection** in either domain on these instances. For SAVI to
  win on a real LM, the win must come from somewhere this mechanism check does not supply —
  most plausibly from the **graded posterior** (D2 is large and positive: the frequency signal
  is real) combined with a budget axis where a single LM step is genuinely cheaper than a full
  rollout (the candidate-vs-token asymmetry the Verify agent flagged), or from a verifier in
  the loop (C4 shows the oracle mask recovers most of the gap).
- It confirms the **Φ-merge is mechanically correct and active on both a toy graph and a real
  CAS Φ**, so the merge machinery is not the thing to debug. The merge is not where the
  headroom is.
- It predicts the decisive run should be designed around (a) frequency/graded edges, not bare
  support, and (b) a compute axis that prices an LM step against a full chain rollout honestly,
  rather than counting candidates — because candidate parity structurally favors selection.

In short: the mechanism is real, fair, and deterministic; the recombination-lift claim is
**not supported** on this synthetic substrate under the candidate-parity budget. The decisive
real-LM run is still required and is the only thing that can settle the end-to-end question.
