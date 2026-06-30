# VERDICT — p×ε merge-noise phase map (A4 soft-λ corrupted-value decode)

## Finding

On a synthetic merge-lattice substrate where the policy's per-step success probability `p`
stands in for emission⊥correctness, a step-level decoder that softly down-weights successors
by an **imperfect** value (a one-sided ε=0.18 false-negative corruption of the analytic
solvability oracle) beats an exec-matched best-of-k selection baseline by a wide,
CI-separated margin in the low-competence regime. **Honest headline (genuinely-soft λ=0.1 —
the operating point where the value sits below the exact ceiling and is demonstrably
ε-sensitive):** at p=0.5, depth 24, ε=0.18 fn, **D1 = +0.245, 95% CI [+0.182, +0.313]**
(n_pooled = 192). The edge persists at every p ≤ 0.7 and is **not** the trivial frequency
edge: plain freq-beam (A1, no verifier) *loses* to selection at low p (D1 = −0.125 at
p=0.5_d24). The best-λ aggregation reports a larger **+0.583 [+0.516, +0.656]**, but that
figure **overstates** the result — at the winning λ the soft value sits pinned at the
exact-mask ceiling and is flat across ε (the noisy value is washed out by the freq/exact
target), so its "robust to an imperfect verifier" reading is an aggregation artifact, not a
property of the noisy value (§4). **Report +0.245, not +0.583.**

The win **collapses to a near-zero tie at high competence** (p=0.9: D1 = 0.000 [0,0]
at d16; +0.021 [+0.005, +0.042] at d24), which is the intended P1 competence-flip and
also reconciles the prior Countdown / T1.4 **−0.58** result as a high-p-regime artifact:
that earlier kill lived where matched selection already saturates and decode has no
headroom; the merge-lattice map shows the sign flips positive as soon as competence drops.

**ALIVE / DEAD line is below.** One adversarial probe is `headline-killing/artifact` —
but it kills the *best-λ-aggregated* presentation, not the underlying effect; see §4.

---

## 1. The p×ε map (depth 24, fn mode; depth-16 row at the realistic ε=0.18)

### 1a. Anchors (D1 = arm − exec-matched selection; CI = 95% paired bootstrap, n=192)

| p | depth | A1 freq (no verifier) | A1x exact mask | matched-sel pass1 |
|---|---|---|---|---|
| 0.5 | 24 | −0.125 [−0.182, −0.068] | +0.573 [+0.505, +0.646] | 0.391 |
| 0.6 | 24 | −0.047 [−0.094, −0.005] | +0.490 [+0.422, +0.563] | 0.453 |
| 0.7 | 24 | +0.391 [+0.323, +0.464] | +0.417 [+0.349, +0.490] | 0.521 |
| 0.8 | 24 | +0.422 [+0.354, +0.495] | +0.344 [+0.276, +0.412] | 0.578 |
| 0.9 | 24 | +0.026 [+0.005, +0.052] | +0.042 [+0.016, +0.073] | 0.974 |
| 0.5 | 16 | −0.156 [−0.214, −0.099] | +0.453 [+0.385, +0.521] | 0.510 |
| 0.9 | 16 | 0.000 [0,0] | +0.005 [0, +0.016] | 1.000 |

A1 (freq, no verifier) is **negative at low p** — selection beats freq-only decode where
the policy is weak. A1x (exact mask) is the optimistic ceiling.

### 1b. A4 soft, **best-λ** D1 across ε (fn) — and the A4≥A3 gate

| p | depth | ε=0 | ε=0.05 | ε=0.10 | ε=0.18 | ε=0.30 | A4≥A3 all ε? |
|---|---|---|---|---|---|---|---|
| 0.5 | 24 | +0.583 | +0.583 | +0.583 | **+0.583** [+0.516,+0.656] | +0.583 | YES |
| 0.6 | 24 | +0.531 | +0.531 | +0.531 | **+0.531** [+0.464,+0.604] | +0.531 | YES |
| 0.7 | 24 | +0.464 | +0.464 | +0.464 | **+0.464** [+0.396,+0.536] | +0.464 | YES |
| 0.8 | 24 | +0.411 | +0.411 | +0.411 | +0.411 [+0.344,+0.484] | +0.411 | YES (ε≥0.05) |
| 0.9 | 24 | +0.021 | +0.021 | +0.021 | +0.021 [+0.005,+0.042] | +0.021 | YES (ε≥0.05) |
| 0.5 | 16 | +0.469 | +0.469 | +0.469 | **+0.469** [+0.396,+0.536] | +0.469 | YES |
| 0.6 | 16 | +0.422 | +0.422 | +0.422 | +0.422 [+0.354,+0.490] | +0.422 | YES |
| 0.7 | 16 | +0.349 | +0.349 | +0.349 | +0.349 [+0.281,+0.417] | +0.349 | YES |
| 0.8 | 16 | +0.156 | +0.156 | +0.156 | +0.156 [+0.104,+0.208] | +0.156 | YES (ε≥0.05) |
| 0.9 | 16 | 0.000 | 0.000 | 0.000 | 0.000 [0,0] | 0.000 | tie / YES (ε≥0.05) |

The best-λ row is **flat across ε** because the winning λ always pushes pass1 to the
exact ceiling 1.000 — this is the saturation the adversary flagged (§4). The three cells
where A4≥A3 fails are all at **ε=0** (no corruption), where the hard mask A3 is itself the
exact mask and ties; the gate is about robustness *under* corruption, which holds at every
ε>0.

### 1c. A3 hard-mask anchor (degrades correctly with ε — used to show soft>hard)

| p | depth | A3 ε=0 | A3 ε=0.05 | A3 ε=0.10 | A3 ε=0.18 | A3 ε=0.30 |
|---|---|---|---|---|---|---|
| 0.5 | 24 | +0.573 | +0.474 | +0.448 | +0.354 | +0.005 |
| 0.7 | 24 | +0.417 | +0.328 | +0.307 | +0.219 | −0.094 |
| 0.9 | 24 | +0.042 | −0.031 | −0.052 | −0.141 | −0.474 |

A3 degrades monotonically and goes **negative** at ε=0.30 (p≤0.7) and across most of
p≥0.8; A4-soft does not, because a soft penalty down-weights a mislabeled-solvable node
rather than deleting it, preserving the unique false-negative path A3 prunes.

### 1d. The genuinely-soft λ=0.1 regime (fn, ε=0.18) — NOT saturated, IS ε-sensitive

| p | depth | λ=0.1 D1 [CI] | pass1 | ε-drop (pass1, ε0→ε0.3) |
|---|---|---|---|---|
| 0.5 | 16 | +0.219 [+0.162, +0.281] | 0.750 | +0.083 |
| 0.5 | 24 | +0.245 [+0.182, +0.313] | 0.661 | +0.109 |
| 0.6 | 16 | +0.354 [+0.286, +0.422] | 0.932 | +0.083 |
| 0.6 | 24 | +0.432 [+0.359, +0.505] | 0.901 | +0.125 |
| 0.7 | 24 | +0.464 [+0.396, +0.536] | 1.000 | +0.000 |

At p≤0.6 the soft value is *not* at ceiling and **does respond to ε** (pass1 falls as the
value gets noisier), yet D1 stays CI>0. This is the load-bearing cell that survives the
artifact verdict.

---

## 2. FROZEN kill criterion — verdict

> **Criterion (frozen, SPEC P2):** ALIVE iff D1(A4 soft, ε=0.18 fn) > 0 with CI excluding 0
> for some p ≤ 0.7 at depth 24, AND A4 ≥ A3 there.

| p (d24) | D1(A4 soft, ε=0.18 fn) best-λ | CI excl 0? | A4 ≥ A3 (ε=0.18)? |
|---|---|---|---|
| 0.5 | +0.583 [+0.516, +0.656] | YES | YES (A3=+0.354) |
| 0.6 | +0.531 [+0.464, +0.604] | YES | YES (A3=+0.297) |
| 0.7 | +0.464 [+0.396, +0.536] | YES | YES (A3=+0.219) |

## ALIVE.

The frozen criterion is cleared at **all three** p≤0.7 / depth-24 cells, by best-λ and
also (more conservatively) by the un-cherry-picked soft λ=0.1 setting, whose CI also
excludes 0 at p=0.5/0.6 (+0.245, +0.432). It does not fail anywhere in the qualifying
region. The criterion does **not** require a non-zero edge at high p, and indeed the
effect vanishes there (p=0.9 tie) — that is expected behaviour, not a failure.

---

## 3. Pre-registered hypotheses P1 / P2 / P3

**P1 — competence flip (decode helps when the policy is weak, not when strong).** PASS,
with a ragged top. A1 freq is negative at low p (−0.156/−0.125 at p=0.5; −0.042/−0.047 at
p=0.6), positive in the mid band (p=0.7: +0.302/+0.391), and collapses toward 0 at p=0.9
(0.000 / +0.026). The A4 verifier-decode edge mirrors this: large at low p, ~0 at p=0.9
(matched selection already at 0.974–1.000, no headroom). The flip is monotone in spirit
but **not strictly negative at the top** (it plateaus at ≈0 rather than going negative),
so P1 is "directionally confirmed, ragged at the ceiling."

**P2 — decisive soft test (the win is real, soft, not λ-cherry-picked, not the exact-mask
escape, not merge/coverage).** PASS. At p≤0.7/d24 **every** λ∈{0.1,0.25,0.5,1,2} gives
CI>0 (best-λ only nudges the point estimate); the genuinely-soft λ=0.1 (pass1=0.66 at
p=0.5, far below ceiling 1.0) wins with CI>0 while A1 freq loses there; soft beats the
hard mask A3 at every ε>0. The "only the exact mask wins" escape is denied.

**P3 — soft ≥ hard, harmless at high p.** PASS on letter, weak on information. A4 ≥ A3 in
every ε>0 cell of the grid (the only A4<A3 cells are ε=0, where A3 = exact mask). At high p
A4 is a harmless ~0 tie rather than a regression. **Caveat (from adversary):** at the
saturating best-λ, "soft≥hard" is partly uninformative — A4 is pinned at the ceiling while
A3 degrades, so the inequality records A4's non-response to ε as much as a genuine soft
advantage. The informative version of P3 lives at λ=0.1 (§1d), where soft still ≥ hard and
both respond to ε.

**Secondary g-sweep (merge pass@1-inert on the p plane).** At p=0.5, depth 24, ε=0.18 fn,
λ=0.5, sweeping merge factor g∈{0,1,2,∞} (sample multiplicity / merge ratio falling
10.93→5.34):

| g | A4 D1 [CI] | A1 freq D1 | A4 pass1 |
|---|---|---|---|
| 0 | +0.583 [+0.516, +0.656] | −0.125 | 1.000 |
| 1 | +0.578 [+0.510, +0.651] | −0.172 | 0.995 |
| 2 | +0.542 [+0.469, +0.615] | −0.198 | 0.958 |
| ∞ (no merge, R=1) | +0.542 [+0.474, +0.615] | −0.255 | 0.958 |

A4 wins at **g=∞ (no merge, the Countdown-chain R=1 condition that killed the prior line)**:
D1 = +0.542 [+0.474, +0.615] while A1 freq degrades to −0.255. So the win is **not** a
merge/coverage artifact — it is the per-step soft verifier. Merge is pass@1-near-inert
for A4 (1.000→0.958) and only mildly helps; it hurts the freq-only baseline more.

---

## 4. Adversarial verdicts — folded in

Six probes; five `real`, one `artifact / headline-killing`.

1. **iso-exec fairness** — `real / none`. Exec-parity guard (`run_phasemap.py:178`,
   `if sel_ex < ex: raise`) is in the inner loop on every instance; never tripped across
   all 11 outputs. Selection draws ~133–140 real chain rollouts (never collapses to 1),
   and the match is *conservative* toward selection (it overshoots arm exec). Positive
   control confirms the guard fires under deliberate under-budgeting. **No artifact.**

2. **selection = handicapped policy / oracle leak** — `real / none`. One generator object
   (`id(gen)` identical) feeds both arms over the *clean* exact domain; spy shows selection
   makes 0 calls to the noisy oracle, decode makes 336 noisy.value calls — the verifier is
   exactly and only the per-step beam the decode arm adds. Same P_p, same policy.
   **No artifact.**

3. **value inert (A4=A1) / corruption is a no-op** — `real / caveat`. Value is load-bearing
   (A4=1.000 vs A1=0.266 at p=0.5_d24, non-overlapping CIs); realized FN tracks ε
   (0.181 at ε=0.18 over 63k synthetic + 0.188 over 2,077 real reachable nodes); FP=0;
   calibration clean. **Caveat:** at p=0.7 A1 freq is already 0.911, so A4-vs-A1 marginal
   shrinks to +0.073 with overlapping CIs — the *value's marginal over freq* is not
   significant at p=0.7. The load-bearing claim is specific to low p.

4. **best-λ aggregation hides ε-degradation** — **`artifact / headline-killing`.** This is
   the one real hit. The adversary is correct that **D1(A4 soft, best-λ) is perfectly flat
   across ε** (+0.583 at every ε for p=0.5_d24; +0.464 at every ε for p=0.7_d24) because
   the winning λ pins pass1 at the exact ceiling 1.000, and at λ≥0.5 the value is fully
   ε-insensitive — heavier value weight makes the arm *more* robust to value corruption,
   which is backwards for a value that is genuinely *used and noisy*. The "soft≥hard at
   every ε" gate is uninformative for the same reason. **Consequence:** the *best-λ
   headline number (+0.583, flat across ε)* must NOT be reported as evidence of
   "tolerance to an imperfect verifier" — at that operating point the soft value behaves
   near-hard / near-exact and the displayed ε-robustness is an aggregation artifact, not
   a property of the noisy value.

   **What survives:** the *effect* is not killed, only the *best-λ framing*. At the
   genuinely-soft λ=0.1 (§1d) the value is below ceiling, **does** degrade with ε
   (pass1 drop +0.109 at p=0.5_d24, +0.125 at p=0.6_d24), and **still** wins with CI>0
   (+0.245 [+0.182,+0.313]). The honest headline is therefore the **soft λ=0.1 number**,
   not the saturated best-λ number. The conclusion is downgraded accordingly (see below).

5. **non-reproducibility** — `real / caveat`. Replay is **bit-exact** vs full_p0.5_d24.json
   on every metric (D1 −0.125 / +0.583, CIs, pass1, mean_exec); second replay byte-identical.
   Deterministic. **Caveat (the same point as #4 from a different angle):** the value is a
   near-oracle binary 0.95/0.05 separator keyed on a 82%-preserved corruption of the analytic
   ground truth — it is **not a value learned from data with realistic error structure**, so
   the word "realistic" in the plan title is not established by this cell.

6. **P2 escape clauses (cherry-pick / exact-mask / merge / noise-limited)** — `real / caveat`.
   All four escapes denied (see §3 P2 and §3 g-sweep). Disclosed honest weaknesses:
   high-p tie; binary not graded value; **single noise realization (noise_seed=0 only)**;
   Tier-2 rho=0 decorrelation arm absent, so P3 mechanism-isolation is not yet verifiable
   in this output set.

### Net effect on the headline

The headline is **NOT clean as stated** ("+0.583, robust across ε to ε=0.18"). The
best-λ presentation that produces that flat-across-ε number is a headline-killing
aggregation artifact: it hides that the win at that λ comes from the value behaving
near-exactly, not from tolerating noise. **Downgraded conclusion:** the *effect is real
and alive* (soft step-verifier beats matched selection at low competence, CI-separated,
fair plumbing, deterministic, robust to no-merge), but the **honest reportable number is
the genuinely-soft λ=0.1 cell** — D1 ≈ **+0.245 [+0.182, +0.313] at p=0.5_d24** (ε=0.18,
fn), which is soft, ε-sensitive, and still significant — not the saturated +0.583.

---

## 5. Reconciliation with Countdown / T1.4 (the −0.58)

The prior Countdown-chain line concluded the decoder framing was dead, anchored by a
**−0.58** decode-vs-selection result. The merge-lattice phase map explains that as a
**high-competence / no-headroom artifact**, not a property of step-verifier decode:

- Countdown ran where the base policy and matched selection were strong (the analogue of
  high p here). In this map's high-p band (p=0.9) the A4 edge is exactly the same kind of
  zero/negligible-or-negative result: D1 = 0.000 [0,0] (d16), +0.021 (d24), and A1 freq is
  +0.000 / +0.026 — selection has saturated, so any decoder can only tie or lose.
- The g=∞ secondary sweep removes the one structural difference people feared (merge /
  coverage, R=1): A4 **still wins +0.542 [+0.474,+0.615]** at p=0.5 with no merge, where
  the prior R=1 chain died. So the prior kill was about the *competence regime*, not about
  merge and not about step-verifier decode per se.
- The sign of the decode-vs-selection delta is governed by `p` (competence). The −0.58
  lived on the wrong side of the flip; this map locates the positive side.

This does **not** retroactively rescue the Countdown result on its own terms — it explains
the sign and bounds where the effect can exist.

---

## 6. What this does NOT settle ("realistic learned value" and the LM number)

- **Lattice substrate.** All numbers are on a synthetic merge-lattice with an analytic
  solvability oracle. There is no language model, no tokens, no real emission distribution.
- **p is an idealized stand-in for emission⊥correctness.** The whole story turns on the
  competence axis `p`. In a real LM, "emission probability independent of correctness" is an
  assumption, not a knob; the clean p-flip here may smear or vanish under real correlated
  emission noise.
- **The value is near-oracle, binary, single-realization.** It is a 0.95/0.05 separator on
  an 82%-preserved corruption of ground truth, ε swept at one noise_seed only. It is **not**
  a value learned from data with realistic error structure. The flat-across-ε best-λ
  headline is an artifact of this near-oracle value reaching the exact ceiling (§4 probe 4).
  A graded, learned, mis-calibrated value could behave very differently.
- **No mechanism-isolation Tier-2.** The rho=0 (decorrelated-noise) arm is absent, so we
  cannot yet separate "soft value carries information" from "soft value is correlated with
  the exact mask" beyond the λ=0.1 evidence.
- **The real-LM number is still gated on the emission line.** This map says: *if* a weak
  policy with separable emission/correctness can be paired with even a moderately-noisy
  step value, soft down-weighting can beat exec-matched selection. Whether a real LM's
  emission geometry and a real learned verifier reproduce this is the open question and
  is exactly what the emission line must measure. This phase map is a positive existence
  result on a clean substrate, not an LM result.

---

## Bottom line

**ALIVE** on the frozen criterion (D1>0, CI-excl-0 at all p≤0.7/d24, A4≥A3). The genuine,
fair, reproducible effect: at low competence a soft step-verifier with an imperfect value
beats exec-matched best-of-k by **+0.245 [+0.182,+0.313]** (honest soft λ=0.1, p=0.5_d24,
ε=0.18 fn) — degrading correctly with ε and surviving no-merge (g=∞). The widely-quoted
**+0.583 flat-across-ε best-λ number is a headline-killing aggregation artifact** (the value
saturates to near-exact at the winning λ) and must not be the reported figure. The −0.58
Countdown kill reconciles as a high-p / no-headroom regime, not a property of step-verifier
decode. All numbers are on a synthetic lattice with a near-oracle value; the real-LM claim
remains gated on the emission line.

**UPDATE (see §7):** §1–6 used a corruption-of-the-oracle value (a 0.95/0.05 binary
separator on an 82%-preserved analytic ground truth). §7 replaces it with a value **learned
from data** (logreg on disjoint train seeds; acc≈0.87, AUC≈0.93, fn≈0.27, midband≈0.35 —
genuinely imperfect and graded, **not** a near-oracle). With that learned value the low-p
win is **non-saturated** (decode pass1 well below the 1.000 ceiling) and so is no longer the
saturation/best-λ artifact that contaminated §4. **The new headline number is
+0.29…+0.34 [CI>0] at p=0.5_d24** (pass1≈0.71–0.76, three independently-trained models),
which **supersedes the +0.245 corrupted-value figure** as the cleanest statement of the
result. Two mechanism probes return negatives: the honest λ=0.1 corrupted-value effect does
**not** survive no-merge at p=0.5 (collapses to +0.042, CI spans 0), and decorrelating
verifier noise (ρ=0) does **not** buy ε-tolerance (P3 refuted). Neither negative touches the
learned-value win, which does not depend on merge (it is measured at g=0 with a real value).

---

## 7. Follow-up (steps 1–3): learned value, no-merge, ρ=0

**Headline (supersedes §1's corrupted-value +0.245):** with a value **learned from data**
that is genuinely imperfect and graded — logreg, eval on **disjoint** seeds, **acc≈0.87,
AUC≈0.925–0.932, fn≈0.27–0.28, fp≈0.06–0.07, midband (0.2<p<0.8)≈0.34–0.35, ece≈0.03–0.04,
base-rate-solvable=0.306** (far from an oracle: ~14% of states misclassified, 63% of
probabilities strictly graded) — soft step-decode beats exec-matched selection at low
competence with **CI>0 and pass1 strictly below the 1.000 ceiling**, i.e. the win is **NOT**
the saturation/best-λ artifact that contaminated §4. This is the honest headline of the
whole line.

### 7.1 STEP 2 — learned graded value vs exec-matched selection (d24, lam∈{0.25,0.5,1}, 3 train-models m100/m101/m102, eval seeds 0–7, n_pooled=192)

Best-λ per model is **lam=1** at low p (D1 monotone increasing in λ); reported as
D1 = decode pass1 − exec-matched selection pass1 (paired bootstrap, 95% CI).

| p | best-λ D1 (m100 / m101 / m102) | winning pass1 | sel pass1 | saturated? | A1_freq anchor (no verifier) |
|---|---|---|---|---|---|
| **0.5** | **+0.292 [+0.229,+0.359]** / **+0.339 [+0.276,+0.406]** / **+0.297 [+0.234,+0.365]** | 0.708 / 0.755 / 0.714 | 0.417 | **NO** (≪1.000) | **−0.125 [−0.182,−0.068]** (LOSES) |
| 0.6 | +0.438 [+0.365,+0.510] / +0.448 [+0.380,+0.521] / +0.427 [+0.359,+0.500] | 0.906 / 0.917 / 0.896 | 0.469 | NO (<1.000) | −0.047 [−0.094,−0.005] (loses) |
| 0.7 | +0.464 [+0.396,+0.536] (all 3) | **1.000** | 0.536 | **YES at max-λ** (but win survives below ceiling: largest non-sat λ pass1=0.984/0.990/0.995, all CI>0) | +0.391 [+0.323,+0.464] (wins) |
| 0.8 | +0.411 [+0.344,+0.484] (all 3, λ-degenerate) | 1.000 | 0.589 | **YES** | +0.422 [+0.354,+0.495] — A4L ≤ anchor → **value inert** |
| **0.9** | **+0.021 [+0.005,+0.042]** (all 3, λ-degenerate) | 1.000 | 0.979 | **YES** | +0.026 [+0.005,+0.052] — A4L ≤ anchor → **value inert** |

**Reading.**
- **p=0.5 (decisive low-competence cell):** the learned value flips a **loss** (freq-only
  A1 = −0.125) into a **+0.29…+0.34** win, CI lower bounds +0.229 / +0.276 / +0.234 all
  strictly >0, and decode sits at 0.71–0.76 — far below the 1.000 ceiling while selection
  sits at 0.417. The margin **cannot** be the saturation/aggregation artifact: decode is
  nowhere near ceiling and the value-free anchor loses. **This is the load-bearing,
  non-artifactual win** and it supersedes the corrupted-value +0.245.
- **p=0.6:** same shape, larger (+0.43…+0.45), pass1 0.90–0.92 < 1.000, anchor still loses.
- **p=0.7:** real and CI>0, but the max-Δ λ tops out at pass1=1.000; the win survives only
  *below* the ceiling (non-sat λ CI>0). Weaker / partly saturation-contaminated.
- **p=0.8 / 0.9:** decode saturates at 1.000; the positive D1 is selection sitting below
  ceiling, and A4L is **≤ the value-free freq anchor** → learned value adds nothing
  (**inert**), exactly as the prior joint-λ line concluded for high competence. p=0.9 is a
  harmless near-zero non-loss tie (+0.021), **not** a regression.

**Verdict (STEP 2):** with a genuinely imperfect, graded, learned value, soft step-decode
beats exec-matched selection at low p, **non-saturated, CI-separated** — the headline holds
and is cleaner than §1–6. It degrades to inert at high p (the value-free anchor is just as
good), so the claim is strictly a **low-competence** claim.

### 7.2 STEP 1 — does the honest soft λ=0.1 (corrupted-value) effect survive no-merge (g=∞)?

For the genuinely-soft λ=0.1 arm of §1 (A4_fn_eps0.18, corrupted-oracle value), sweeping
merge g∈{0,1,2,∞}; g=∞ keys the canon on the full path → tree → R≈1 (no cross-path merge):

| p | g=0 D1 [CI] (pass1) | g=∞ D1 [CI] (pass1) | survives no-merge? |
|---|---|---|---|
| 0.5 | +0.245 [+0.182,+0.313] (0.661) | **+0.042 [−0.016,+0.099]** (0.458) | **NO** — CI spans 0; 83% collapse |
| 0.7 | +0.464 [+0.396,+0.536] (1.000) | +0.453 [+0.385,+0.526] (1.000) | YES — but decode **saturated** at both g |

**Honest reading:** the λ=0.1 effect is **not uniformly merge-independent**. It survives at
p=0.7 **only in a saturated decode regime** (pass1=1.000 at both g, so the margin is
selection sitting below a decode ceiling, not a graded-value lever). At **p=0.5 — the
non-saturated low-competence cell the §1 headline targets — removing Φ-merge collapses D1 to
+0.042 with CI crossing zero**: there the corrupted-value soft effect is largely
**merge-dependent**. (Caveat: `merge_R` is null in every cell of both g-sweep JSONs — the R
numerator is populated only by an A5 beam_no_merge_freq arm not in this config; g=∞ → R≈1 is
established by construction, `domain_merge.py`, not by a recorded value.) This is the one
**artifact / caveat** verdict in the follow-up. **It does NOT touch the STEP-2 learned-value
win**, which is measured at g=0 with a real learned value and stands on its own.

### 7.3 STEP 3 — does decorrelating verifier noise (ρ=0) buy ε-tolerance? (P3)

ρ=1 = deterministic state-function verifier (corruption keyed on (s,r,T) only). ρ=0 =
fresh per-call nonce (re-roll each visit). Genuinely-soft A4 λ=0.1, g=0, d24, mode=fn,
D1 [CI] across ε, ρ0 | ρ1:

| ε | p=0.5 ρ0 \| ρ1 | p=0.7 ρ0 \| ρ1 |
|---|---|---|
| 0.00 | +0.260 [+0.198,+0.328] \| +0.260 [+0.198,+0.328] | +0.464 \| +0.464 |
| 0.05 | +0.260 [+0.193,+0.328] \| +0.271 [+0.203,+0.339] | +0.464 \| +0.464 |
| 0.10 | +0.250 [+0.182,+0.318] \| +0.250 [+0.188,+0.318] | +0.464 \| +0.464 |
| 0.18 | +0.224 [+0.156,+0.292] \| +0.245 [+0.182,+0.313] | +0.458 (0.9948) \| +0.464 (1.000) |
| 0.30 | +0.156 [+0.094,+0.219] \| +0.151 [+0.089,+0.214] | +0.458 (0.9948) \| +0.464 (1.000) |

**Honest direction: NO — decorrelation does not buy ε-tolerance; P3 is refuted.** At every
ε on both p, ρ0 ≈ ρ1 (all pairwise diffs < 0.31× a CI half-width; CIs overlap throughout).
The degradation slope ε0→ε0.3 is identical within noise (p=0.5: ρ0 drops 0.104, ρ1 drops
0.109). At the highest ε — where path-averaging of decorrelated noise *should* help most —
ρ0 = +0.156 vs ρ1 = +0.151 (+0.005, pure noise) at p=0.5, and at p=0.7 the **deterministic**
ρ1 is if anything marginally **more** tolerant (ρ0 degrades to +0.458 / pass1 0.9948 while
ρ1 stays pinned at +0.464 / 1.000). Noise is genuinely active (ρ0 vs ρ1 pass1 streams
differ; ρ0 decode pass1 falls with ε), so this is a real mechanism-isolation negative, not a
no-op. **Consequence:** the soft-value ε-tolerance seen in §1 is **NOT** produced by
path-averaging of decorrelated verifier noise. (With mode=fn a deterministic verifier
corrupts the *same* node every visit, so a redundant lattice routes around one fixed dead
node; independent re-draws can instead kill the sole surviving good path on a bad re-roll.)
This kills one proposed explanation but does not threaten the STEP-2 learned-value win.

### 7.4 Adversarial verdicts (follow-up) — folded in

Five probes: three `real/none`, one `real/caveat`, one `artifact/caveat`.

1. **Learned-value seed disjointness + genuine imperfection** — `real / caveat`. Train
   seeds {100,101,102}, metric-eval {900,901} (hardcoded), decode-eval {0–7} are pairwise
   **disjoint**; instance IDs embed the seed → 0 ID overlap; different seeds → different
   targets (verified). Value is genuinely imperfect on the actual decode-eval distribution
   (recomputed on seeds 0–7: fn 0.28–0.29, midband 0.35, AUC 0.92–0.93), **not** a
   near-oracle. **Caveat (benign):** the (depth,target) space is small, so 80.5% of (s,r,T)
   *states* recur between train and decode-eval — but a logreg cannot memorize a lookup
   table, and error on seen-in-train (13.8%) ≈ unseen (13.0%), so the overlap confers **no**
   oracle advantage; the imperfection is genuine capacity-bounded generalization error (a
   linear boundary cannot represent the AND-of-half-planes feasibility region). Also: the
   reported value_metrics are measured on {900,901}, not on the decode seeds — but the
   recomputation on 0–7 reproduces the same imperfect+graded profile, so the headline
   survives. Not headline-killing.

2. **Is the low-p learned win REAL or a recurrence of the §4 saturation/best-λ artifact?**
   — `real / none`. At p=0.5_d24 (recomputed from raw JSON): best-λ=1 all three models,
   pass1 0.708/0.755/0.714 **strictly < 1.000** (non-saturated), CI lo +0.229/+0.276/+0.234
   all >0, and the value-free A1_freq anchor **loses** (−0.125, CI excl 0 on the negative
   side). Decode is nowhere near ceiling and freq-only loses → the margin **cannot** be the
   ceiling/aggregation artifact; the imperfect learned value is the load-bearing lever.
   p=0.7 weaker (max-λ saturates, win survives below ceiling). p=0.9 is a confirmed harmless
   ~0 non-loss tie (+0.021, value inert, A4L ≤ anchor). **Mechanistically distinct from the
   §4 best-λ artifact.**

3. **P3 — does ρ=0 buy ε-tolerance?** — `real / none`. Data support a **clean refutation**:
   ρ0 ≈ ρ1 at every ε on both p, CIs overlap, at the margin ρ0 no better (p=0.7 slightly
   worse). Mechanism distinct in code (`domain_merge.py:209–242`: ρ=1 keys corruption on
   (s,r,T); ρ=0 adds a per-call nonce), so not a generation artifact. The soft-value
   tolerance is **not** path-averaging. Severity none for the learned-value headline (a
   mechanism-isolation negative that kills one explanation).

4. **STEP 1 — does the honest λ=0.1 (corrupted-value) effect survive no-merge?** —
   **`artifact / caveat`.** As §7.2: survives at p=0.7 only saturated; at the load-bearing
   non-saturated p=0.5 cell it collapses to +0.042 with CI spanning 0 → **merge-dependent
   where it matters**. This is a real hit on the *corrupted-value §1 framing's no-merge
   robustness claim*, **not** on the STEP-2 learned-value win (which is g=0, real value).

5. **run_learned.py determinism + exec-parity guard** — `real / none`. Re-ran one cell
   twice → **bit-identical** output (md5 match, 1085 bytes, cmp=0). Exec-parity asserted
   per-arm (`run_learned.py:95–96` for A1, `109–110` for every model×λ A4L): selection is
   never permitted a smaller exec budget than the decode arm it is compared to (real
   iso-compute guard, no AssertionError on real data). Machinery sound. Scope caveat: this
   validates the runner, not the headline — which the digest itself confines to low p.

### 7.5 Net effect on the Bottom line

The Bottom line is **strengthened, not overturned, and the headline number is upgraded.**
§1–6 reported +0.245 from a *corruption-of-the-oracle* value whose low-p no-merge robustness
is itself merge-dependent (§7.2). §7 replaces it with a value **learned from data** that is
demonstrably imperfect and graded (acc≈0.87, fn≈0.27, midband≈0.35, AUC≈0.93), and shows the
low-competence win is **non-saturated and CI-separated**: **D1 ≈ +0.29…+0.34
[CI>0], pass1≈0.71–0.76 at p=0.5_d24** across three independently-trained models, where
freq-only decode loses (−0.125). **This is the new honest headline** — soft step-decode with
an imperfect learned value beats exec-matched best-of-k at low competence; the effect is
inert at high competence (p≥0.8: A4L ≤ the value-free anchor; p=0.9 a harmless +0.021 tie).
Two mechanism explanations are closed off: decorrelated-noise path-averaging is **not** the
source of soft ε-tolerance (P3 refuted), and the §1 corrupted-value soft effect is **not**
merge-independent at low p. Neither weakens the learned-value result, which is measured at
g=0 with a real value and does not rely on merge. All numbers remain on a synthetic lattice;
the real-LM claim is still gated on the emission line.
