# PREREG — countdown-decode (capstone: decoupled emission → executor-Φ → λ=0 trellis → pass@1)

> Same design as `../algebra-decode/PREREG.md`; this is the **pivot domain** (algebra's Φ was
> structurally unsuited — see `../algebra-decode/FINDING_phi_mismatch.md`). The yardstick and
> kill criteria below were fixed before the countdown run.

## Hypothesis

A model trained with decoupled emission (calibrated multi-peak over moves) feeds an
executor-Φ-merged λ=0 Viterbi (`decode_core.savi`, `verifier=False`, `edge_mode="freq"`)
candidates that the trellis aggregates into a higher pass@1 than greedy — and, if the
emission's de-peaking + the Φ-merge recover answers that surface voting misses, higher than
self-consistency.

## Design

- Domain: Countdown (value multiset + target; one move combines two values; canon = sorted
  values + target, strictly shrinking). Executor-Φ. Held-out solvable instances, seed-disjoint
  from training.
- Arms (pass@1 = executor `is_goal`, terminal single value == target): greedy / self-consistency
  / savi(K=8, N=16, freq, λ=0, max_depth=10) / best-of-K / oracle.
- Decision rule (the CORRECTED yardstick, not "+0.18 over greedy"): **savi must beat
  self-consistency, paired on the same instances.** Secondary: decoupled-savi > coupled-savi.
- Iso-compute: report the token Budget per arm; compare savi against a **token-matched**
  selection baseline (best-of-K scaled to savi's token budget), because candidate-parity is the
  wrong axis (a global-decode lesson from the verifier line).

## Kill criteria (fixed before data)

1. savi does not beat greedy → the λ=0 global decode is inert on this domain.
2. savi beats greedy but not self-consistency → the DP adds nothing beyond sampling+voting;
   Φ-merge recovered no extra mass. (This is the load-bearing null.)
3. decoupled-savi does not beat coupled-savi → data-level decoupling did not earn its keep.
4. **savi does not beat a token-matched selection baseline → the candidate-parity win is an
   iso-compute artifact; λ=0 emission-only decode is not competitive at equal compute.**
   (Report this honestly rather than reporting only the candidate-parity number.)
