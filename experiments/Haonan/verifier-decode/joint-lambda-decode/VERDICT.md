# VERDICT — PLAN3 joint λ>0 decode (Tier B, real model)

**Date:** 2026-06-29. **Run:** `results.json` (merged from `outputs/tierB_*.json`),
Qwen3-4B + LoRA (decoupled & coupled adapters), deeper Countdown k ∈ {4,5,6} →
depth {3,4,5}, K=8, N=8, τ=1.0, seed=0, n=16/k, headroom stratum, paired bootstrap
n=10000. Config amended pre-result for compute (see `PREREG.md` §1). Harness
adversarially reviewed: 0 confirmed headline-biasing bugs.

## One-sentence answer

**Yes — verifier-weighted (λ>0) global trellis decode beats greedy and iso-token
best-of-many at equal tokens, on a real model in a real verifiable domain, and the
margin is significant at depth** (decoupled k=6: savi λ>0 = 1.00 vs best-of-many = 0.69,
Δ = **+0.31, 95% CI [+0.13, +0.56]**). The verifier flips the emission line's depth-3
λ=0 loss (−0.375) to a depth-5 λ>0 win. **But the win is attributable to the verifier
and to depth — not to decoupled training (the coupled model wins equally) and not to the
Φ-merge (inert at this beam width).**

## Pass@1 on the headroom stratum (iso-token)

DECOUPLED adapter:

| k | depth | greedy | SC | savi λ0-freq | **savi λ>0-freq** | best-of-many | Φ-ablation | oracle | **H1 Δ [CI]** | K_eff | merge | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 3 | 0.21 | 0.21 | 0.36 | **0.50** | 0.36 | 0.50 | 1.00 | +0.143 [−0.143,+0.429] | 4.30 | 5.3 | 0.084 |
| 5 | 4 | 0.00 | 0.00 | 0.40 | **0.80** | 0.40 | 0.80 | 1.00 | **+0.400 [+0.133,+0.667]** | 4.29 | 3.3 | 0.100 |
| 6 | 5 | 0.00 | 0.19 | 0.19 | **1.00** | 0.69 | 1.00 | 1.00 | **+0.312 [+0.125,+0.562]** | 4.86 | 2.4 | 0.201 |

COUPLED adapter (one-hot trained):

| k | depth | savi λ0-freq | **savi λ>0-freq** | best-of-many | **H1 Δ [CI]** | K_eff |
|---|---|---|---|---|---|---|
| 4 | 3 | 0.36 | 0.43 | 0.57 | −0.143 [−0.429,+0.143] | 4.12 |
| 5 | 4 | 0.20 | 0.60 | 0.40 | +0.200 [+0.000,+0.400] | 4.41 |
| 6 | 5 | 0.38 | 0.94 | 0.62 | **+0.312 [+0.125,+0.562]** | 4.09 |

## Hypotheses

- **H1 (λ>0 beats iso-token selection) — CONFIRMED at depth.** Decoupled k=6 Δ = +0.312,
  CI excludes 0; coupled k=6 identical (+0.312). The λ=0 floor (`savi λ0-freq`) sits at
  the greedy/SC level (0.19–0.40); the verifier mask lifts it to 0.80–1.00 — the slice
  **only the verifier can claim**. This is the precise flip the two lines predicted: the
  emission line's λ=0 was −0.375 at depth 3; λ>0 is positive at every depth and
  significant from depth 4 on.
- **H2 (margin rises with depth) — SUPPORTED in trend, not strictly monotone.** Δ =
  +0.143 → +0.400 → +0.312 over k = 4,5,6: positive throughout, significant at depth 4–5,
  rising from the shallow tier. It is NOT strictly monotone (k=5 ≥ k=6) and the prereg
  sub-clause "≤ 0 at k=4" FAILS — because the verifier already lifts the shallow tier
  above 0 (the λ=0 negative does not survive turning the verifier on, even at depth 3).
  The deepest-tier sign + significance (the primary H2 signal) holds.
- **H3 (decoupled emission necessary) — SPLIT; the necessity half is REFUTED.** The kill
  criterion does **not** fire: in-trellis K_eff(decoupled) = 4.3–4.9 ≫ 1.5, so multi-peak
  branching survives into per-step decode (marginal-vs-path risk did not realize). **But**
  K_eff(decoupled) ≈ K_eff(coupled) (4.1–4.4), and H1 holds **equally for coupled**
  (k=6 +0.312). At τ=1.0 sampling the one-hot-trained model branches just as much, so the
  decoupled-vs-coupled training distinction does not drive the win. Validity gate **V2
  fails for this reason** (not for lack of decoupled branching).
- **H4 / E6c (Φ-merge contributes) — NOT at K=8.** `savi − beam_no_merge` = 0.000 at every
  depth and profile. Merge compresses the trellis 2.4–5.3× but does not change pass@1 at
  K=8: the beam is wide enough to retain the goal-bearing path with or without merge. The
  proper E6c test is at narrow K (a K=2 ablation is running; result appended below).
- **H5 / v (freq-edge calibration) — reported.** ECE of the emission frequency p=count/N
  against the exact `reachable` oracle rises with depth: 0.084 → 0.100 → 0.201. The graded
  edge is well-calibrated shallow, less so at depth (descriptive; no pass/fail).

## Validity gates

V1 headroom non-empty ✓ · V2 K_eff(dec) > K_eff(cou) ✗ (they are equal — the H3 finding)
· V3 oracle > best-of-many at iso-token ✓ · Φ-works merge ratio > 1 ✓. Parse rate
0.98–1.00 (model on-task even OOD at k=6).

## What this settles

On a real LM in a real verifiable domain, at equal tokens: **global verifier-weighted
decode beats both greedy and iso-token selection, with the margin significant at depth.**
The win is **correctly attributed to the verifier-weighted graded edge (λ>0)** — the load-
bearing lever by a wide margin (λ=0 floor ≈ greedy; λ>0 ≈ ceiling) — operating over a
global trellis that **depth** makes increasingly valuable. This closes the seam where both
lines stopped: λ=0/support loses, the verifier (λ>0) is load-bearing, the win needs depth,
iso-compute is tokens.

## What this does NOT settle / corrects

- **Decoupled training is not the active ingredient here.** The coupled (one-hot) model
  branches as much at τ=1.0 and wins equally; the decoupled-emission advantage seen in
  data-level diversity does not translate into a decode-time K_eff gap or an H1 gap. The
  memo's "decoupled emission makes candidates comparable" clause is **not supported** on
  this model — re-attribute the win to verifier + depth, not to decoupled training.
- **Φ-merge is inert at K=8** (does not change pass@1). Its lever, if any, lives at narrow
  beam (K=2 test pending below).
- Generality beyond Countdown (one domain); free-prose emission; whether decoupled would
  help at larger scale / smaller K. **Compute caveats:** N was reduced 16→8 for iso-token
  feasibility (a fair budget for both arms, but it lowers absolute pass@1 where N=8
  emission misses a solvable move at shallow depth — visible as savi trellis collapse on a
  few k=4 instances); the token ledger is the whitespace-token proxy `len(text.split())`
  (consistent across arms and with the emission line's prior iso-compute; the ';'-join in
  chains makes best-of-many's count slightly conservative *for* savi, i.e. it can draw
  marginally more real compute than the budget implies — the comparison is not inflated in
  savi's favor).

## Memo scorecard rows addressed

- **End-to-end (the headline):** ✓ confirmed on a real model at depth, correctly attributed.
- **v (backward-value calibration):** ECE reported per layer (0.08→0.20).
- **ii-Countdown (Φ false-merge):** merge ratio > 1, parse > 0.98, trellis non-degenerate
  on the chosen depths (no false-merge pathology observed).
- **E6c (DP-backtracking helps):** merge does NOT change pass@1 at K=8 OR K=2 (see below);
  the honest E6c answer is that Φ-merge is pass@1-inert in this decoder's goal-collection
  regime, though it is a real trellis compression.

## PLAN4 — what the headline actually is (Tier-1 dissection; the deflation)

PLAN4 (`PLAN4.md`) attacked the three things PLAN3 left the win resting on. The Tier-1
result is decisive and **deflationary**: the active ingredient is the per-step EXACT
feasibility mask alone — not the global trellis decode, not decoupled emission, not
Φ-merge — and the advantage is fragile to an imperfect mask.

### T1.1 Decomposition (real decoupled, with `masked_best_of_many`; `outputs/t11_*.json`)
Ladder, iso-token, headroom: `best_of_many` (unmasked) → `masked_best_of_many` (per-step
mask, NO beam/merge/DP) → `savi K=1` (masked single path) → `savi K=8` (full decode).

| k | best_of_many | masked_bom | savi K=8 | **savi − masked_bom** | masked_bom − bom |
|---|---|---|---|---|---|
| 4 | 0.400 | 0.600 | 0.600 | **+0.000 [0, 0]** | +0.200 |
| 5 | 0.455 | 0.727 | 0.818 | +0.091 [0, +0.27] (n.s.) | +0.273 |
| 6 | 0.750 | 1.000 | 1.000 | **+0.000 [0, 0]** | +0.250 |

**The global trellis decode adds nothing beyond the per-step mask** (savi − masked_bom =
0 at k=4 and k=6, +0.09 n.s. at k=5). The entire win over `best_of_many` is the mask
itself (masked_bom − bom = +0.20…+0.25). `savi K=1` (single masked path) already beats
unmasked selection at every depth (e.g. k=5 0.455 vs bom 0.091), confirming the mask, not
the beam, is the lever.

### T1.2 Imperfect verifier (real decoupled, k=6; `outputs/t12_noise_k6.json`, n=12)
savi pass@1 (and H1 = savi_noisy − best_of_many) as the decode mask is corrupted at rate ε
(exact ε=0: savi 1.00, H1 +0.25):

| ε | false-negative (prune correct) | false-positive (allow dead-ends) | symmetric |
|---|---|---|---|
| 0.05 | 0.833, +0.08 | 1.000, +0.25 | 0.833, +0.08 |
| 0.10 | 0.833, +0.08 | 1.000, +0.25 | 0.833, +0.08 |
| 0.20 | 0.583, −0.17 | 0.750, 0.00 | 0.500, −0.25 |
| 0.30 | 0.500, −0.25 | 0.750, 0.00 | 0.417, −0.33 |

**The win requires a near-exact oracle.** A 5% false-negative rate already drops savi from
1.00 to 0.83 and removes significance; by ε=0.2 false-negatives savi LOSES to selection.
False-positives are tolerable (ceiling to ε=0.1) — letting dead-ends in only wastes tokens,
whereas a false-negative can prune the only correct path. (n=12: CIs wide; the monotone
degradation is the robust read.)

### T1.3 Degraded generator (CPU, exact mask; `outputs/t13_degraded_p*.json`)
savi λ>0 / best_of_many pass@1 with the exact mask but a competence-`p` generator:

| p (good-move mass) | k=4 | k=5 | k=6 |
|---|---|---|---|
| 0.3 (very weak) | 0.14 / 0.00 | 0.13 / 0.00 | 0.56 / 0.00 |
| **0.5 (uninformed)** | 0.50 / 0.21 | 0.73 / 0.00 | **1.00 / 0.19** |
| 0.9 (strong) | 1.00 / 0.93 | 1.00 / 1.00 | 1.00 / 1.00 |

**An uninformed (uniform) generator + the exact mask already reaches the ceiling at depth**
(savi 1.00 at k=6), matching the real model. The generator is load-bearing only in a band:
too weak (p=0.3) and finite-N coverage starves the trellis; strong enough (p=0.9) and
selection catches up. Emission probability matters through coverage, not through "making
candidates comparable."

### PLAN4 net verdict (the honest reframing)
The PLAN3 headline survives as a fact but its mechanism is now localized: **"applying an
(essentially exact) per-step feasibility check during generation beats checking only at the
end, and the gap grows with depth because unmasked rollouts compound dead-ends."** The
trellis/beam/Φ-merge ("global decode") contributes nothing measurable; decoupled emission
is not the active ingredient; and the advantage collapses once the verifier is imperfect,
especially on false-negatives. So this is **not** evidence that "verifier-weighted global
decode" is a good decoding algorithm — it is evidence that a near-exact per-step oracle,
used greedily, helps. Generalizing it (a second domain, T3.2) is therefore low-value until
the imperfect-verifier regime (a learned value, soft λ) is shown to retain any edge.

### T1.4 LEARNED value + soft λ — the kill test (real decoupled; `outputs/t14_value_*.json`)
The exact mask is the strong assumption; the only experiment that could revive the
"decoder" framing is replacing it with a learned, imperfect value used as a soft
λ-weighted edge (`edge = log P_freq + λ·log V`, no hard prune). V is a genuine learned
solvability classifier (disjoint-seed train; **acc 0.91, AUC 0.97, ECE 0.008, false-neg
0.18, false-pos 0.05** — a good model, sitting where T1.2's hard mask LOST).

k=6 (depth 5, n=12 headroom), iso-token vs best_of_many = 0.833:

| arm | pass@1 | H1 vs best_of_many |
|---|---|---|
| savi **exact mask** (idealized ceiling) | 1.000 | +0.167 [0, +0.42] |
| savi λ=0 freq (emission floor) | 0.167 | −0.667 |
| **savi_value soft, learned V, λ=1…8** | **0.250** | **−0.583 [−0.83, −0.33]** |
| savi_value **hard**, learned V (thr 0.5) | 0.000 | −0.833 |

Full depth curve (soft@λ=2 / hard / best_of_many): k=4 0.40 / 0.00 / 0.80 (soft FLAT at
the floor for every λ — the K=8 beam exceeds the shallow layer width so the value cannot
prune); k=5 0.55 / 0.00 / 0.55 (soft ties bom); k=6 0.25 / 0.00 / 0.83. The soft-λ learned
value never beats best_of_many at any depth; the hard learned mask is 0.00 everywhere; only
the exact mask wins at every k (0.60 / 0.82 / 1.00).

**The kill test fires.** With a good learned value, **soft-λ verifier-weighted decode loses
to plain iso-token best-of-many by −0.58 at every λ** — the λ knob barely lifts it off the
emission floor (0.17→0.25). The hard learned mask is catastrophic (0.00): an 18% false-
negative rate compounds over 5 steps and prunes the only correct path. **Soft does beat
hard** (0.25 > 0.00 — graded weighting is less brittle than masking, as hypothesized), but
the gap to best_of_many is far too large to call a rescue. **Only the exact oracle wins.**

**Final PLAN4 verdict.** By the pre-stated kill criterion ("if no edge survives a learned
mask, the decoder framing is dead"), the verifier-weighted **decode** line is dead as a
general method: the entire PLAN3 win is contingent on a *near-exact* per-step feasibility
oracle. A realistic learned approximation of that oracle — even a strong, well-calibrated
one — does not retain any edge over end-only selection at depth, whether applied softly or
as a hard mask. What survives is narrow and assumption-bound: *given* a near-exact per-step
reachability oracle, using it per-step (greedily) beats checking only at the end, and the
gap grows with depth. That is a statement about having an exact oracle, not about decoding.

### T3.1 strict-collection Φ-merge (mechanism footnote; CPU mock, K=2, k=6)
H4=0 was an artifact of collecting goals pre-prune. With STRICT collection (goals only from
the beam-kept frontier), Φ-merge does bind weakly: savi(merge) 0.833 vs strict
beam_no_merge 0.792 (+0.04) at K=2. So merge's value exists but only in a strict-collection,
narrow-beam regime — a footnote, not a headline lever.

---

## K=2 narrow-beam Φ-merge ablation (E6c) — decoupled, `outputs/tierB_K2_dec_*.json`

| k | savi λ>0 | beam_no_merge | **H4 Δ** | merge ratio | best-of-many | H1 Δ [CI] |
|---|---|---|---|---|---|---|
| 4 | 0.500 | 0.500 | **0.000** | 4.11 | 0.357 | +0.143 [−0.143,+0.429] |
| 5 | 0.667 | 0.667 | **0.000** | 2.18 | — | — |
| 6 | 0.562 | 0.562 | **0.000** | 1.61 | 0.250 | **+0.312 [+0.125,+0.562]** |

**E6c conclusion.** Narrowing the beam to K=2 does NOT make Φ-merge bind: H4 = 0 at every
depth, even though real merging still occurs (ratio 1.6–4.1×). The reason is structural —
`savi` collects goal nodes over the **full merged layer before beam pruning**, so a goal
that merge would have "made room for" is already counted whether or not the beam keeps it.
Φ-merge therefore compresses the trellis but is **pass@1-inert** for reach-the-goal
success at every tested beam width; its value would show only on a metric sensitive to
beam composition or path score, not on this pass@1. The headline is robust to beam width:
at K=2, savi λ>0 still beats iso-token best-of-many at k=6 by the same +0.312 [+0.125,
+0.562] (both arms' absolute pass@1 drop with the narrower beam — savi 1.00→0.56,
best-of-many 0.69→0.25 — but the iso-token margin holds).
