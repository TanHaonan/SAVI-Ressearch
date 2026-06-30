# VERDICT — novel-path / recombination observation

**One line:** the recombination mechanism is **real and unique to decode at the
PATH level** (decode returns correct paths no rollout produced, ~10⁻¹¹ joint
probability, a third demonstrably stitched from selection's own sampled
fragments, and the verifier is what makes the stitch land on a *correct* path) —
**but on this lattice it does NOT translate into a compute-independent INSTANCE
win**: the same instances are selection-solvable with only ~10³·⁴–10⁴·² rollouts,
because the additive lattice has many correct paths. The categorical win the
reframe wants needs a substrate whose correct-path *set* is globally sparse.

## Setup
Lattice, competence-p fair generator, soft-λ global decode (learned logreg value,
λ=1; freq-only λ=0 as contrast), merge g=0 vs no-merge g=∞, depth 24, seeds 0–3
(n=24), selection rollout budget M=256. `P_p` and path/solve probabilities are
computed **exactly** (no Monte Carlo). All 7 metric unit tests pass.

## Results (learned value)

| p | g | dec | sel@256 | onlyDec | novel | edgeCov | stitch | log₁₀(1/q_path) | log₁₀(1/mass)·onlyDec |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0 | 0.67 | 0.46 | 0.23 | 1.00 | 0.82 | 0.36 | **11.5** | **3.4** |
| 0.5 | ∞ | 0.43 | 0.46 | 0.15 | 1.00 | 0.89 | 0.51 | 11.5 | 3.4 |
| 0.6 | 0 | 0.94 | 0.47 | 0.47 | 1.00 | 0.70 | 0.23 | 11.0 | 4.2 |
| 0.6 | ∞ | 0.76 | 0.47 | 0.30 | 1.00 | 0.83 | 0.42 | 11.0 | 3.4 |
| 0.7 | 0 | 1.00 | 0.61 | 0.39 | 1.00 | 0.79 | 0.33 | 10.0 | 3.8 |
| 0.9 | 0 | 1.00 | 1.00 | 0.00 | 0.98 | 0.97 | 0.76 | 6.2 | – |

freq-only (no verifier), p=0.5 g=0: dec=0.20, onlyDec=**0.01**; p=0.6: onlyDec=0.03.

## What is confirmed (the reframe's literal claim)
1. **Decode returns paths no rollout produced.** novel = 1.00 at low/mid p; the
   decoded correct path has joint probability ~10⁻¹⁰–10⁻¹¹·⁵, so selection would
   need ~10¹¹ rollouts to ever sample *that path*. Selection, returning whole
   sampled chains, **cannot return it at any feasible budget** — compute-independent
   at the path level. This is exactly "a path no single rollout produced."
2. **A large fraction are genuine stitches.** 23–36% of decode's correct solutions
   (p=0.5/0.6, g=0) are paths *every edge of which* selection sampled in some
   rollout, but which selection never assembled into one chain — "had all the
   pieces, never connected them."
3. **The verifier is load-bearing.** Without it, freq-only decode at low p barely
   solves (0.20) and almost never beats selection (onlyDec 0.01): raw recombination
   stitches mostly-*wrong* paths. The learned value steers recombination onto
   correct rare paths (0.67 / onlyDec 0.23). "Stitch **verified** fragments" is
   literally the mechanism.
4. **Merge increases reach at low competence.** dec-solve 0.67 (g=0) vs 0.43 (g=∞)
   at p=0.5; onlyDec 0.23 vs 0.15; the gap vanishes by p=0.7. Cross-path merge is
   what lets decode reach more correct paths when single rollouts are incomplete.

## What is NOT supported (the reframe's strong/instance claim)
On the lattice, selection solving the *instance* needs only **~10³·⁴–10⁴·²
rollouts** (`1/mass` over the only-decode instances) — a ~1.5-order-of-magnitude
sample-efficiency gap over decode, **not a wall**. Reason: the additive lattice
has *many* correct paths, so total correct mass is moderate even when each
individual path is rare. So decode's instance-level advantage here is still
**compute-efficiency** — the iso-compute axis that keeps dying under iso-token.
"Selection can't return THIS path" (true) ≠ "selection can't solve this instance"
(false here).

## Implication — the decisive next substrate
A *compute-independent* recombination win requires a task whose **correct-solution
set is globally sparse** (few correct paths, each only reachable by stitching), so
total correct mass — not just one path's probability — is tiny. The additive
lattice is the wrong substrate for that claim (correct paths are abundant by
construction). Real multi-step math is plausibly the right one: a hard problem
often has essentially one valid derivation, so `1/mass ≈ 1/q` and the path-level
wall becomes the instance-level wall. **This is the sharp, falsifiable hypothesis
the math harness should test:** measure `1/mass` (total correct mass) on real
problems; recombination is categorical iff `1/mass` exceeds any feasible
best-of-K, not merely `1/q`.

## Artifacts
`novelpath.py` (exact P_p / q / solve_mass / novelty / stitch + runner),
`test_novelpath.py` (7 tests), `outputs/sweep.json`, `SPEC.md`.
