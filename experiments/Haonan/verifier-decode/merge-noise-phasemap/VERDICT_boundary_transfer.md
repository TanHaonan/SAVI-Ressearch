# VERDICT — boundary-concentrated noise transfer test

**Question.** The competence-flip (`VERDICT.md`) was established under **uniform** ε
(`domain_merge.py` flips a flat fraction of solvable nodes). The PRM
boundary-step calibration check (`../prm-boundary-calibration/`) found real value
functions concentrate their solvable→unsolvable (false-positive) errors at the
on/off-path boundary (~3.5× gradient, data-driven across two value families).
Does the finding survive when the noise has that geometry instead of being flat?

**Method.** Added a **mean-preserving** boundary profile to
`NoisyMergeLatticeDomain` (`profile="boundary"`): the same total ε, redistributed
across solvable states by distance-to-boundary using the measured shape
`_FP_BY_DIST` (per-bin weights normalized so the solvable-state-weighted mean = ε,
so this tests error *geometry*, not *amount*). Lattice distance = integer slack to
the nearest infeasibility edge (margin 0 = at the edge = the last-correct node).
Re-ran the load-bearing cell (g=0, depth 24, ε=0.18, fn mode, K=8, N=16, τ=1,
seeds 0–7, n=24; D1 = decode − exec-matched selection, 95% paired bootstrap).
Baseline reproduces the frozen numbers exactly (uniform A4 λ=0.1 = +0.245, A3 =
+0.354).

## Result — the soft decode survives; the hard mask dies

| p (d24) | A3 hard mask: uniform → boundary | A4 soft λ=0.1: uniform → boundary |
|---|---|---|
| 0.5 | +0.354 [+.28,+.43] → **−0.047 [−.14,+.05]** | +0.245 [+.18,+.31] → **+0.214 [+.15,+.28]** |
| 0.6 | +0.297 [+.22,+.37] → **−0.109 [−.20,−.02]** | +0.432 [+.36,+.51] → **+0.333 [+.27,+.40]** |
| 0.7 | +0.219 [+.14,+.30] → **−0.177 [−.27,−.08]** | +0.464 [+.40,+.54] → **+0.464 [+.40,+.54]** |

- **Hard mask (A3) collapses.** Boundary concentration turns a clear win
  (+0.22…+0.35) into a tie-or-loss (−0.05…−0.18, CI ≤ 0 at every p). The hard
  mask prunes exactly the critical last-good nodes when the false-prunes cluster
  there. Same total ε — only its location changed.
- **Soft value (A4, honest λ=0.1) survives, CI>0 at every p.** Attenuated where it
  is below ceiling (p=0.5: +0.245→+0.214; p=0.6: +0.432→+0.333) and unchanged once
  saturated (p=0.7: +0.464, pass@1 already 1.0). Down-weighting (not deleting)
  tolerates boundary-clustered errors that deletion cannot.

## Interpretation
The competence-flip **transfers to the realistic boundary-concentrated error
geometry — but only in its soft form.** This sharpens the mechanism: the
load-bearing lever is the *soft, down-weight-not-delete* verifier, precisely
because it is robust to errors at the on/off-path boundary. It also qualifies the
`joint-lambda` conclusion that "the only lever is a near-exact per-step mask" — the
hard mask is not merely uninteresting, it is **non-robust to the error geometry
real value functions actually have** (mean-matched boundary concentration alone
destroys it). The earlier "gate failure" reads, after this test, as a *positive*
result for the soft decode and a *negative* one for hard masking.

## On-policy confirmation — the LEARNED value (run_learned.py, logreg)

The imposed test above redistributes binary noise to the boundary by hand. The
genuinely learned graded value does it *organically*: logreg cannot represent the
two-sided interval `r ≤ T−s ≤ 3r`, so it errs exactly at the edges.

**Its error geometry is boundary-concentrated and STEEPER than what I imposed.**
FP rate (solvable scored unsolvable) by distance-to-boundary bin, 3 models:

| model | bin1 (edge) | bin2 | bin3 | bin4 | bin5 (far) | gradient |
|---|---|---|---|---|---|---|
| m100 | 0.599 | 0.545 | 0.473 | 0.426 | 0.103 | 5.84× |
| m101 | 0.638 | 0.576 | 0.506 | 0.433 | 0.117 | 5.48× |
| m102 | 0.609 | 0.532 | 0.475 | 0.420 | 0.100 | 6.12× |

So the PRM's 3.5× (and my imposed 3.5×) is *conservative*; a real learned value is
~5.5–6×. (overall FP ≈ 0.25; acc≈0.87, AUC≈0.93.)

**Its soft decode beats selection at low competence (non-saturated, CI>0)** —
D1 = decode − exec-matched selection, mean over 3 models, d24:

| p | A1 freq (no verifier) | A4L soft λ=0.1 | A4L soft λ=1 (non-sat.) | A4L soft λ=2 |
|---|---|---|---|---|
| 0.5 | −0.125 | −0.078 (pass1 .34) | **+0.276 [+.21,+.34]** (pass1 .69) | +0.420 (pass1 .84) |
| 0.6 | −0.047 | +0.033 (tie) | **+0.443 [+.37,+.52]** (pass1 .91) | +0.490 (pass1 .96) |
| 0.7 | +0.391 | +0.425 [+.36,+.50] (pass1 .96) | +0.460 | +0.464 |

Honest reading: the graded value must be *used* (λ≈1) — at λ=0.1 it barely moves
the soft edge and tracks freq-only A1, which loses at low p. Unlike the binary
value's best-λ (which floor-pins pass@1 to 1.0, an artifact), here the win at λ=1
is **non-saturated** (pass@1 ≈ 0.69–0.91 ≪ ceiling 1.0) — genuine graded
down-weighting, not masking. There is no viable hard-mask version of a graded
value, and §1 showed hard masking dies under this geometry anyway.

**Conclusion (loop closed):** a genuinely learned value has boundary-concentrated
errors (~5.5–6×, organic, steeper than measured on the PRM), and its *soft*
decode still beats exec-matched selection at low competence (non-saturated, CI>0).
This is the on-policy confirmation of the imposed-profile result: soft
verifier-decode is robust to the boundary error geometry real value functions
exhibit; hard masking is not.

## Caveats / scope
- Imposed test: controlled binary 0.95/0.05 verifier, mean-matched boundary
  redistribution on the fn direction, single load-bearing cell (g=0, d24, ε=0.18,
  fn), p∈{0.5,0.6,0.7}; mean-match guaranteed by construction + `tests/`.
- Learned test: logreg graded value, 3 train seeds (disjoint eval), d24, same p;
  win is non-saturated. Both at g=0; merge axis was already falsified upstream.

## Artifacts
- `domain_merge.py` — `profile="boundary"`, `boundary_bin`, `boundary_weights`
  (mean-preserving), `_FP_BY_DIST` (measured shape).
- `run_phasemap.py --profile boundary`; outputs in `outputs/boundary/`.
- `tests/test_phasemap.py` — mean-preservation + boundary-gradient tests.
