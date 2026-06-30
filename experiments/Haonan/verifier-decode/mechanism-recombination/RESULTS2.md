# RESULTS2 — CI-grade exec-parity depth sweep (the corrected mechanism run)

> **✅ CORRECTED VERDICT (read first — supersedes the "claim NOT supported" verdict below).**
> The Verify agents caught a real contamination bug (the fair generator's memo/seed were keyed
> by `canon = (s,r)`, which omits the per-instance target `T`; one generator shared per cell
> leaked competence classification across instances). **The bug was fixed** (`LatticeDomain.canon`
> now `= (s,r,T)`; stale cache deleted) and the sweep **re-run clean with a fresh per-instance
> generator** (immune to any sharing). The contamination biased *against* the trellis; the
> verifier's predicted sign-flips landed exactly. Two independent clean re-runs AGREE — the fixed
> harness (8 seeds, **authoritative**, `outputs/lattice_sweep/results2.json`) and a per-instance
> generator probe (6 seeds, `clean_rerun.py`). Authoritative 8-seed numbers (p=0.7, N=16, 24
> inst/depth, paired bootstrap n=10000):
>
> | depth | iso_exec | savi_freq | D1_exec = savi_freq − iso_exec (K=8 / K=16 / K=32) |
> |---|---|---|---|
> | 4  | 1.000 | 1.000 | +0.000 / +0.000 / +0.000  (null: all saturate — no-headroom control) |
> | 8  | 0.911 | 1.000 | **+0.089** [+.05,+.13]  (all K) |
> | 16 | 0.68–0.76 | 0.96–1.00 | **+0.281** [+.21,+.35] / **+0.255** / **+0.245** |
> | 24 | 0.48–0.56 | 0.92–1.00 | **+0.437** [+.37,+.51] / **+0.458** / **+0.438** [+.37,+.51] |
>
> (`D_verif` = `savi_verifier_on − iso_exec` ≥ D1 and also rises with depth; `D_merge` =
> `savi_freq − beam_no_merge(freq)` is positive at depth ≥ 8 — the Φ-merge contributes a real but
> smaller share than the graded edge + the axis. The 6-seed probe corroborates: D8 +0.15, D16
> +0.31–0.38, D24 +0.38–0.42.)
>
> **Corrected bottom line: the claim IS supported on this substrate.** On the honest moves/exec
> axis, with a freq (or verifier-masked) headline arm, global decode **beats compute-matched
> selection at every depth ≥ 8 (CI excluding 0), and the margin rises monotonically with depth.**
> Both artifacts that produced the prior negatives are now removed: PLAN1's candidate-parity axis,
> and PLAN2's cross-`T` contamination. The K=8 cells that looked negative in the contaminated data
> are clean-positive (+0.146/+0.312/+0.389); **depth, not beam width, is the driver.** The Φ-merge
> contributes a positive but smaller share (D_merge > 0 at depth ≥ 8); the larger lever is the
> graded edge + the honest axis.
>
> **Scope (unchanged, important):** this is the **synthetic Viterbi lattice** — the algorithm's
> home turf, with heavy position×time merging — and an **oracle-defined** competence generator,
> **not a real LM**. The real-Φ confirm cells stay negative/null *at the shallow depths they can
> reach* (Countdown −0.083 at effective depth ~3, where the lattice D4 cell is also null;
> algebra null, generator-limited). A real domain with a **cheap solvability oracle and deep
> horizon** is the missing piece for real-domain confirmation; `countdown.reachable` is
> exponential and caps depth ~5. The real-LM end-to-end number remains gated on the emission line.
> See `results.json → corrected_clean_rerun`. The original (contaminated-data) analysis is retained
> verbatim below for the record.
>
> ---

**Substrate: SEMI-SYNTHETIC** (integer-sum Viterbi lattice + a competence-parameterized
fair generator `P_p`; not a real language model). The real-LM end-to-end number remains
gated on the emission line and is **not** settled here.

## Headline verdict

**The central PLAN2 claim is NOT supported.** The claim under test was: *on the honest
moves/exec axis, with a freq (and verifier-masked) headline arm and adequate beam width,
global decode beats compute-matched selection, and the margin GROWS with horizon depth.*

It fails on two independent grounds, the first of which is disqualifying:

1. **The sweep numbers are contaminated (material bug).** The fair generator memoizes
   move-classification (good/bad weights) and seeds its RNG by `canon(state) = (s, r)`,
   which **omits the per-instance target `T`**. A single generator is built per
   `(depth, K, p, seed)` cell (`run_mechanism2._run_cell` line 173) and shared across all
   24 instances of that cell, which have **different `T` but heavily overlapping `(s, r)`
   canons** (every instance shares `(0, D)` and intermediate nodes). Whichever instance
   first populates the memo/seed for a given `(s, r)` fixes the weights and the RNG draw
   for every later instance at that `(s, r)` **under the wrong target**. The good/bad
   classification depends on `T` (via `solvable`), so the contamination is not benign: it
   biases `savi_freq` strongly downward. Every `pass@1` and every `D1_exec`/`D_verif`/
   `D_merge` in `outputs/lattice_sweep/results2.json` was produced by this contaminated
   path (reproduced bit-for-bit by the verifier under the buggy code). **The published
   sweep must be re-run with a per-instance generator (or `canon` extended to `(s, r, T)`)
   before any scientific conclusion is drawn; the current `cache.json` must be deleted, not
   trusted.**

2. **Even taken at face value, the depth-monotonicity sub-claim is false.** In the
   (contaminated) data, `D1_exec` is **non-monotone in depth in every (K, p) slice**: at
   adequate beam it rises to a peak at D16 and then *decays* at D24; at the narrow beam
   K=8 it is **negative and worsening** for D ≥ 8. The variable that gates the sign is
   **beam width K, not depth**. "Margin grows with horizon depth" is therefore not what
   the data show even before the bug is corrected.

**`claim_supported = no.`**

What the corrected re-run is *expected* to show (per the verifier's spot re-run with a
clean per-instance generator) is a *directional* positive D1_exec that strengthens at
adequate K — the sign flips positive on exactly the cells now reported negative. But that
is a forward-looking expectation from a handful of re-run cells, **not** a CI-grade result
in hand. Until the full sweep is re-run clean, the honest status is: contaminated +
non-monotone, claim unsupported.

---

## Verification summary (all three verdicts folded in)

| Verify line | Result | What it establishes |
|---|---|---|
| Exec-parity / honest-axis accounting | **PASS** | The comparison is methodologically honest: exec-parity holds across all 4608 paired instances, exec is counted for **every** rollout (not stopped at first win), the headline numerator is `savi_freq`/`savi_verifier_on` (not `support`), and no arm receives oracle injection. |
| Oracle exactness + substrate wiring | **FAIL (ok=false)** | The `solvable` oracle is provably exact (brute-force cross-check, 0 disagreements; ceiling = 1.0 on all 768 instances). **But** the generator leaks classification + RNG across instances because the canon/seed key omits `T`. This contaminates every headline number; the verifier's clean re-run **flips signs** on the negative cells. |
| Determinism + depth claim | **FAIL (ok=false)** | Determinism reproduces bit-identically (cache-free). `iso_exec` decays with depth (the structural control holds). **But** "D1_exec rises with depth" is not CI-supported: the trend is non-monotone, and beam width K — not depth — gates the sign. |

Two of three verdicts returned `ok=false`. The first failure (contamination) is the
controlling one.

---

## The contaminated sweep numbers (reported for completeness, NOT to be trusted)

These are the values in `outputs/lattice_sweep/results2.json`. They are presented so the
reader can see the magnitude of what the bug moved, **not** as evidence for or against the
mechanism. All come from the contaminated shared-generator path.

### D1_exec = `savi_freq − best_of_k_isobudget(exec)`, paired bootstrap n=10000, n=192/cell

p = 0.7:

| depth \ K | K=8 | K=16 | K=32 |
|---|---|---|---|
| D4 | +0.193 [+0.141, +0.250] | +0.193 [+0.141, +0.250] | +0.193 [+0.141, +0.250] |
| D8 | **−0.062** [−0.109, −0.016] | +0.234 [+0.177, +0.297] | +0.234 [+0.177, +0.297] |
| D16 | **−0.161** [−0.234, −0.089] | +0.089 [+0.031, +0.146] | +0.464 [+0.391, +0.531] |
| D24 | **−0.125** [−0.203, −0.047] | +0.036 [−0.031, +0.104] (∋0) | +0.276 [+0.214, +0.339] |

p = 0.6:

| depth \ K | K=8 | K=16 | K=32 |
|---|---|---|---|
| D4 | +0.188 [+0.135, +0.245] | +0.188 [+0.135, +0.245] | +0.188 [+0.135, +0.245] |
| D8 | −0.036 [−0.083, +0.010] (∋0) | +0.292 [+0.229, +0.359] | +0.292 [+0.229, +0.359] |
| D16 | **−0.104** [−0.161, −0.047] | +0.073 [+0.026, +0.120] | +0.448 [+0.380, +0.516] |
| D24 | **−0.141** [−0.193, −0.089] | −0.042 [−0.083, +0.000] (∋0) | +0.276 [+0.214, +0.339] |

Reading of the contaminated table (with both caveats attached): the sign is gated by **K**,
not depth. At K=32 every cell is CI-positive; at K=16 it is positive up to D16 then reverts
to null at D24; at K=8 it is **negative** for D ≥ 8. Within a fixed K the depth trend
peaks (D16) then *falls* — `savi_freq` itself collapses at D24 (1.0 → 0.745 at K=32),
which pulls the margin back down. There is **no monotone rise with depth** in any slice.
Bold = CI strictly below 0 (selection beats the trellis).

### D_verif = `savi_verifier_on − best_of_k_isobudget(exec)` (oracle mask), p = 0.7

| depth | K=8 | K=16 | K=32 |
|---|---|---|---|
| D4 | +0.193 | +0.193 | +0.193 |
| D8 | +0.281 [+0.219, +0.349] | +0.234 | +0.234 |
| D16 | +0.479 [+0.406, +0.552] | +0.484 [+0.417, +0.552] | +0.464 |
| D24 | +0.562 [+0.490, +0.635] | +0.568 [+0.495, +0.635] | +0.531 [+0.458, +0.604] |

With a perfect verifier mask the margin is positive at every cell and *does* rise
monotonically with depth (`verifier_on` hits the 1.0 ceiling for D ≤ 16). This is the
cleanest depth-monotone signal in the file — but it (a) uses an exact oracle mask, not a
graded statistical edge, and (b) is still computed on the contaminated `iso_exec`
baseline, so the *level* is not trustworthy even though the *shape* is clean.

### D_merge = `savi_freq − beam_no_merge(freq)` (the Φ-ablation), p = 0.7

| depth | K=8 | K=16 | K=32 |
|---|---|---|---|
| D4 | +0.250 [+0.193, +0.313] | +0.135 [+0.089, +0.188] | +0.000 [0, 0] |
| D8 | +0.177 [+0.125, +0.234] | +0.464 [+0.391, +0.531] | +0.406 [+0.339, +0.474] |
| D16 | +0.104 [+0.063, +0.151] | +0.312 [+0.250, +0.380] | +0.698 [+0.630, +0.760] |
| D24 | +0.125 [+0.078, +0.172] | +0.255 [+0.198, +0.318] | +0.495 [+0.427, +0.568] |

Position×time merging contributes positively at every cell and the contribution grows with
depth at adequate beam (D4/K32 = 0 because a 32-wide beam already saturates the shallow
lattice and there is nothing left to merge). This is the cleanest *directional* evidence
that the Φ-merge is a real mechanism — subject to the same contamination caveat on levels.

### Axis-artifact cell (D16, K=8, p=0.6) — confirms the PLAN1 bug directly

`iso_candidates_pass = 0.677` ≫ `iso_exec_pass = 0.484` > `savi_freq_pass = 0.380` (n=192).
Candidate parity (the PLAN1 accounting) hands selection ≈ depth× more *complete* attempts
than the honest exec parity, inflating the selection baseline by ≈ +0.19 and masking/
flipping the trellis comparison. This is the artifact PLAN2 was built to remove.

### Control — `iso_exec` decays with depth (K=32, p=0.7)

`0.807 (D4) → 0.766 (D8) → 0.536 (D16) → 0.469 (D24)` — monotone decay, in all six (K, p)
slices. This is the structural reason (per-rollout success ≈ q^D) the trellis's *relative*
advantage would be expected to grow with horizon; the control holds. C3 ceiling = 1.0 on
all instances (solvable by construction).

---

## Real-Φ confirmatory cells (`outputs/confirm/results.json`)

These run the corrected exec-parity comparison on the real M1 Countdown adapter and the M2
AlgebraDomain — to show the comparison is not lattice-specific. They are single mid-depth
points, not a depth sweep, and are **not** affected by the lattice-canon bug (different
domains, different canon).

- **Countdown** (builtin_small, 16 headroom inst × seeds 1–3 = 48 flags, p=0.7, K=16, N=16,
  max_depth=6): `savi_freq = 0.875`, `iso_exec = 0.958`, `iso_candidates = 1.000`.
  - `D1_exec = −0.083 [−0.167, −0.021]` — at depth 6 the freq trellis does **not** beat
    exec-matched selection (CI below 0). Consistent with PLAN2's own expectation that the
    trellis advantage appears only at larger depth, not at this shallow real cell.
  - Axis artifact reproduced on a real domain: `iso_candidates − iso_exec = +0.042
    [0.000, 0.104]` — candidate parity lifts selection from 0.958 to 1.000.
  - Exec-parity verified per-instance (`iso_exec` total exec 21291 ≥ `savi_freq` exec
    21248).
- **Algebra** (AlgebraDomain mock, builtin_small headroom, p=0.7, K=8, N=8, max_depth=8,
  seed 1, n=5 of an intended 8 — shrunk to fit the runtime wall, reported honestly):
  `savi_freq = 0.0`, `iso_exec = 0.0`, `D1_exec = 0.0 [0, 0]`. Both arms solve nothing
  because the **mock step-generator (not the decoder) is the binding constraint** at this
  config (matches the repo M2 note). The cell confirms the corrected comparison runs
  end-to-end on real CAS Φ with exec-parity holding, but carries **no ranking signal**.

The real-domain confirm cells are therefore *negative-or-null* on D1_exec at the depths
that fit the runtime budget. They do not provide independent positive support for the
mechanism; they corroborate the PLAN1 accounting artifact and the "advantage needs depth"
framing, nothing stronger.

---

## What is settled vs not

**Settled (and solid):** (a) the PLAN1 −0.6…−0.9 D1 was substantially a candidate-parity
accounting artifact — the axis-artifact cell and the Countdown confirm both reproduce it;
(b) the honest exec axis with a graded/verifier headline is the right design; (c) the
`solvable` oracle is exact and the lattice is fair by construction; (d) `iso_exec` decays
with depth (the structural premise) and determinism is bit-stable.

**NOT settled:** whether, on the honest axis, the freq trellis beats compute-matched
selection and whether the margin grows with depth — because the sweep is contaminated by
the cross-`T` generator leak and the depth-monotonicity sub-claim is already false in the
contaminated data. Also not settled: the real-LM end-to-end number (gated on the emission
line) and whether real reasoning has lattice-like semantic merging.

## Required corrections before re-claiming

1. **Fix the leak.** Either extend `LatticeDomain.canon` to `(s, r, T)` so the canon-keyed
   memo/seed cannot collide across instances, **or** build a fresh `make_fair_generator`
   inside the per-instance loop in `run_mechanism2._run_cell`. (Extending canon to include
   `T` does not disable intra-decode Φ-merging — within one decode `T` is constant, so
   `(s, r)` already keys the node uniquely; only cross-instance sharing changes.)
2. **Delete `cache.json` and re-run the entire sweep**; regenerate `outputs/lattice_sweep/
   results2.json` and this file. Do not trust the current cache.
3. **Add a regression test:** two instances with the same `(depth, seed, p)` but different
   `T` through one shared generator must yield per-step weights/samples identical to a
   private per-instance generator (memo/seed must not leak across `T`).
4. **Restate the headline** as K-gated and non-monotone, never "rises with depth": report
   all 24 cells with CIs (not just the favorable K=32 slice), and explicitly flag the
   CI-negative K=8 cells and the deep K=16 cells that revert to null.
