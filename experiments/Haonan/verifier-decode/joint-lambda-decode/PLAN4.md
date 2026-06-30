# PLAN4 — what the Tier-B result forces us to test next

**Status: SPEC + executing.** PLAN3 confirmed the headline on a real model
(`VERDICT.md`): verifier-weighted (λ>0) global decode beats greedy and iso-token
best-of-many at depth (decoupled k=6 Δ=+0.31 CI[+0.13,+0.56]). But it left the win
resting on assumptions the result itself makes suspicious. PLAN4 attacks exactly those.

Three things the Tier-B numbers make us doubt:
1. **The lever might be only the per-step exact mask, not the global decode.** λ=0 sits at
   the greedy floor; λ>0 jumps to ceiling; H4 (Φ-merge) = 0 at K=8 and K=2. So beam +
   merge contributed nothing measured — the per-step `reachable` mask may be everything.
2. **The verifier is an EXACT oracle**, which is a strong, usually-unavailable assumption.
   The win is currently "with a perfect per-step feasibility oracle, masked decode beats
   unmasked selection." Whether it survives a noisy/learned mask is untested.
3. **The generator may not be load-bearing** at all under an exact mask: a near-random
   step generator + exact mask + beam is a masked search that may solve it anyway.

decode_core stays FROZEN. New arms/domains live in this dir. Iso-compute axis = tokens.
Paired bootstrap n=10000 on the headroom stratum, Holm within a family.

---

## Tier 1 — decisive: does the headline mean anything? (must do)

### T1.1 Decomposition ladder — mask vs global decode (E6c, sharpened)

Add one ingredient at a time, all iso-token, real decoupled model, k∈{4,5,6}:

| arm | uses exact mask? | global structure | isolates |
|---|---|---|---|
| `best_of_many` (have) | no | whole-chain selection | baseline (unmasked) |
| **`masked_best_of_many`** (new) | yes, per step | none (independent masked rollouts) | the per-step mask, in a selection setting |
| `savi` **K=1** | yes | single masked freq-greedy path | mask + freq, no beam |
| `savi` **K=8** (headline) | yes | masked freq beam + Φ-merge | full global decode |

`masked_best_of_many`: draw independent masked rollouts (each step: sample N, keep only
successors with `solvable`, pick one weighted by emission frequency, advance to goal /
dead-end / depth); accumulate tokens to the savi(K=8) target; ok iff any rollout reaches
goal. This is "verifier-in-the-loop selection" without beam/merge/cross-path DP.

**Decision rules (k=6):**
- `masked_best_of_many − best_of_many` > 0 (CI excl 0) ⇒ the per-step mask is the lever
  (expected).
- `savi(K=8) − masked_best_of_many` ≈ 0 ⇒ global decode (beam+merge) adds **nothing**
  beyond the mask; the honest claim collapses to "per-step exact mask beats end-only
  check." `> 0` (CI excl 0) ⇒ global decode has independent value.
- `savi(K=8) − savi(K=1)` isolates beam width; with H4=0 already, a null here means the
  whole "trellis/DP" framing carries no measured weight on pass@1.

### T1.2 Imperfect verifier — does the win survive a non-exact mask? (external validity)

`NoisyVerifierDomain`: the decode mask = exact `reachable` corrupted at rate ε,
deterministic per `canon` (a state's mask label is fixed within a run). `is_goal` stays
EXACT (pass@1 ground truth uncorrupted); the oracle ceiling uses the exact oracle.

Sweep at k=6 (and k=4 for contrast), real decoupled:
- **symmetric** ε ∈ {0, 0.05, 0.10, 0.20, 0.30};
- **false-positive only** (unsolvable→"solvable", ε_fp): lets dead-ends back in → wastes
  tokens on doomed paths;
- **false-negative only** (solvable→"unsolvable", ε_fn): prunes correct paths → can make
  the goal unreachable in the trellis.

**Decision rule:** report ε* = the largest ε at which `savi(λ>0) − best_of_many` still
excludes 0. ε* small (≤0.05) ⇒ the result is fragile, depends on a near-exact oracle ⇒
generality is weak. ε* ≥ 0.2 ⇒ robust to verifier quality. The fp/fn asymmetry tells us
which error mode the decoder tolerates.

### T1.3 Degraded generator — is emission load-bearing under an exact mask? (CPU)

Run the headline arm with the EXACT mask but a deliberately weak generator (the Tier-A
competence mock at p ∈ {0.5 (uninformed), 0.7, 0.9, 1.0}), k∈{4,5,6}, and compare to the
real model + exact mask.

**Decision rule:** if `savi(λ>0)` at p=0.5 (uninformed) ≈ the real model's pass@1, the
generator is **not** load-bearing — the exact mask + beam is effectively doing a masked
search and the emission only matters through coverage at finite N. If the real model is
clearly higher, emission probability is contributing. This bounds how much of the headline
is "the oracle solved it."

---

## Tier 2 — consolidate the headline (robustness reviewers will ask for)

### T2.1 N sweep — is the win an artifact of the N=16→8 compute cut?
k=6, real decoupled, N ∈ {8, 16, 32} (best_of_many matches each savi token budget). Report
H1 at each N and the absolute savi pass@1. Expect H1 stable in sign; quantify how much N=8
suppressed absolute pass@1 (the shallow trellis-collapse caveat).

### T2.2 Larger n + depth extension to k=7
n=40 per k, ks = 4,5,6,7 (the persistent `reachable` memo makes k=7 / depth 6 feasible;
verify cost first). Tightens the wide n=16 CIs and tests whether the depth margin keeps
rising or peaks at k=5 (the non-monotone hint).

### T2.3 τ sweep — is "decoupled = coupled" only a high-temperature effect?
Both adapters, k=6, τ ∈ {0, 0.3, 0.7, 1.0}. Measure in-trellis K_eff and H1 per τ.
**Decision rule:** if at τ→0 coupled K_eff collapses to ~1 while decoupled stays >1 (and
H1 holds for decoupled but not coupled), the decoupled-training value is a low-temperature
phenomenon masked by τ=1.0 sampling — i.e. decoupled matters only when you cannot use
temperature. If decoupled≈coupled at all τ, the decoupled-emission line is not buying
anything for this decoder.

---

## Tier 3 — generality + mechanism detail

### T3.1 Strict-collection savi — where does Φ-merge actually bind?
A savi variant that collects goals ONLY from the beam-kept frontier (not the full
pre-prune layer). There merge frees beam slots for goal-bearing canons, so `savi −
beam_no_merge` should turn positive at narrow K. Run K ∈ {2,4,8}, decoupled, k∈{4,5,6}.
This is the honest E6c: it locates the regime (strict collection × narrow beam) where
Φ-merge contributes, instead of reporting a flat H4=0.

### T3.2 Second domain with a cheap exact oracle — generality beyond Countdown
One domain is not generality. Need a domain where Φ strictly shrinks (so the trellis is
non-degenerate, unlike algebra) and an exact backward oracle is cheap. Candidate: a
bounded integer **reach-the-target path** task (state = (position vector, steps left);
moves = a small action set; canon collapses order-equivalent positions; oracle = O(1)
feasibility band), or a second arithmetic task with a shrinking multiset. Re-run the
decomposition (T1.1) + noise (T1.2) there. If the masked-decode-beats-selection +
depth-scaling pattern replicates, the claim is about a class of verifiable tasks, not
Countdown. Highest effort; gated on T1 outcomes (no point if T1 shows the mask is
everything — then "generality" is just "exact oracles solve search").

---

## Order of execution (gated)

1. **T1.1 + T1.3 first** (cheap, decisive): if global decode adds nothing beyond the mask
   AND an uninformed generator already solves it, the honest verdict shifts to "exact
   per-step oracle + beam = masked search," and the writeup must be reframed before any
   generality work. 2. **T1.2** sets external validity (ε*). 3. **T2** consolidates. 4.
   **T3.1** locates Φ-merge's regime; **T3.2** only if T1 leaves a non-trivial decode
   contribution worth generalizing.

Deliverables: `run_followup.py` (or run_joint flags), `results_followup.json`,
`VERDICT.md` "PLAN4" section, `EXECUTION_LOG.md` updates.
