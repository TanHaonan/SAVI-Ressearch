# PLAN2 — CI-grade exec-parity depth sweep (the corrected mechanism run)

**Status: SPEC (to be executed).** Follows the post-hoc axis correction in `results.json →
axis_correction_addendum`. Verifier-line study; writes its own `RESULTS2.md` / `results2.json`.

## Why PLAN1 needed correcting

PLAN1's headline used **candidate parity** (one full chain rollout counted == one single decoder
step) and a **support-only** headline arm on a **depth-3** substrate — three choices that each
disfavor the trellis. A direct probe showed the −0.6…−0.9 D1 collapses ~6× and flips positive
(+0.375 at K=8, depth=4) on the honest **moves/exec** axis with **freq** edges. PLAN2 turns that
directional finding into a CI-grade result, and removes the depth cap.

## The three corrections, baked in

1. **Compute axis = moves/exec (and tokens), never candidate count.** The selection baseline
   `best_of_k_isobudget(axis="exec")` draws rollouts until cumulative move-applies reach the
   trellis's per-instance exec budget (≈ `depth×` fewer rollouts than candidate parity). Exec is
   counted for **every** rollout (fix the PLAN1-probe bug where exec stopped after the first win).
2. **Headline arm = `savi_freq` and `savi_verifier_on`, not `savi_support`.** Support edges are
   all 0 ⇒ no ranking signal ⇒ the weakest arm. `savi_support` is kept only as a floor.
3. **Deep substrate with a cheap O(1) solvability oracle.** `countdown.reachable` is exponential
   (caps depth ~5). Replace it with an integer-sum lattice (below) so depth is unbounded-cheap.

## Substrate — integer-sum Viterbi lattice (`domain_lattice.py`)

- **State** `(s, r)`: current sum `s`, remaining steps `r`. Start `(0, D)`.
- **Moves** `add v` for `v ∈ M = {1, 2, 3}`. `apply((s,r), v) = (s+v, r-1)`.
- **Goal** `is_goal((s,r)) = (s == T and r == 0)`.
- **canon** `(s, r)` — the classic position×time lattice; different move-orders reaching the same
  `(s,r)` **merge** (this is where Φ has teeth, by construction).
- **solvable** `(s,r)`: reachable iff `r ≤ (T − s) ≤ 3r` (every integer in `[r, 3r]` is a sum of
  `r` moves from `{1,2,3}` since `1∈M`). **O(1), exact.**
- **Instances** `make_lattice_instances(depth, n, seed)`: `T` drawn in `[D, 3D]` (so every
  instance is solvable; ceiling = 1), `s0=0`, `r0=D`. The minimal-step solution length is exactly
  `D`, so "depth" is honest.
- **enumerate_moves(state)** returns `["add 1","add 2","add 3"]` (finite known set; no mock
  sampler needed). `gen_fair.make_fair_generator` consumes it unchanged; competence `p` upweights
  moves whose successor stays `solvable` exactly as before.

Why this substrate is fair, not rigged: it is the algorithm's home turf (a literal Viterbi
lattice), so if the trellis can ever beat compute-matched selection it must do so here; if it
cannot even here, that is strong evidence against the mechanism. Real reasoning's open question
then becomes "does it have lattice-like semantic merging," which is what Φ is for.

## Arms

`greedy`, `best_of_k_isobudget(axis="exec")` (headline baseline), `savi_support` (floor),
`savi_freq` (headline), `savi_verifier_on` (headline, oracle mask), `beam_no_merge(edge="freq")`
(Φ-ablation against the freq headline), `ceiling` (=1 by construction, sanity).

## Grid

- depth `D ∈ {4, 8, 16, 24}` (the key axis).
- `K ∈ {8, 16, 32}` (beam width / and best-of budget scales via exec parity).
- `p ∈ {0.7}` primary, `{0.6}` secondary if cheap.
- `N = 16`, seeds `1..8`, instances `24` per depth. Paired bootstrap n=10000.

## Headline metrics (per depth × K, paired bootstrap CI on the per-instance flags)

- **D1_exec = `savi_freq − best_of_k_isobudget(exec)`** — the corrected headline. Expect it to
  **rise with depth** and exceed 0 (CI excluding 0) for `D ≥` some threshold.
- **D_verif = `savi_verifier_on − best_of_k_isobudget(exec)`** — with the oracle mask.
- **D_merge = `savi_freq − beam_no_merge(freq)`** — Φ-merge contribution at the freq headline.
- Report `best_of_k_isobudget` under BOTH axes (candidate vs exec) at one (D,K) cell to exhibit
  the artifact directly.
- Exec/token budgets per arm (assert exec-parity holds: iso exec ≥ savi exec, candidates differ).

## Controls

- **C1** `p=1.0`: every arm → 1.0 (lattice always solvable; perfect generator), D1→0.
- **C3** ceiling = 1.0 on all instances (solvable by construction); any arm solving a
  non-goal-at-r=0 state would be a bug.
- **Depth monotonicity** of `best_of_k_isobudget(exec)`: should DECAY with depth (per-rollout
  success `~ q^D`), the structural reason the trellis's relative advantage grows.
- **Axis cross-check**: `best_of_k_isobudget(candidates) ≫ best_of_k_isobudget(exec)` at depth.

## Confirmatory points (real Φ)

One algebra cell and one Countdown cell at a mid (D,K,p) using `axis="exec"` headline (freq), to
show the corrected comparison on a real CAS Φ / the original toy, not only the lattice.

## Deliverables

`domain_lattice.py`, extended `arms_ext.best_of_k_isobudget(..., axis=...)`, `run_mechanism2.py`
(+ tests), `RESULTS2.md`, `results2.json`, `EXECUTION_LOG.md` (Phase 6). decode_core stays FROZEN.

## What this settles / does not

Settles: whether, on the honest moves axis with a graded/verifier-masked headline and adequate
beam, global decode beats compute-matched selection, and how that scales with horizon depth — on
a controlled substrate. Does NOT settle the real-LM end-to-end number (gated on the emission
line) or whether real reasoning exhibits lattice-like merging.
