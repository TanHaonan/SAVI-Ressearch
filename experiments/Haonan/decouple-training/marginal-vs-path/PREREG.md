# PREREG — single-best-path vs per-position-marginal decoding on a real LLM

*Written before running. Records the hypothesis, setup, and the predictions and kill conditions
that were fixed in advance.*

## Hypothesis

Two global decoders are often conflated: single-best-path decoding (Viterbi / max-product, which
returns the one most probable joint assignment) and per-position marginal decoding (sum-product,
which returns the most probable value at each position separately). Past demonstrations that a
global decoder beats greedy were all run on tasks with a unique solution, and on a unique solution
these two decoders are provably equal — so path decoding has never actually been shown to add
anything beyond what marginals already give.

We hypothesize that on a real LLM reading out per-step beliefs, path decoding will **not**
independently beat marginal decoding on accuracy, because a real LLM's per-step output is sharp
(its probability concentrates on the top option). Sharp per-step output means a near-unimodal
posterior, and the two decoders converge there. Path decoding should separate from marginal
decoding only when the posterior is genuinely multimodal and the competing probability mass is
shaped so that many individually-weak feasible solutions route through the same value at one
position; their summed marginal can then outvote the single best path. Real LLM output is not
expected to fall into that regime.

## Setup

- **Carrier task.** A chain-structured CRF/HMM with multiple variables, hard adjacent-exclusion
  constraints (neighbors must differ), and a deliberately multimodal posterior, so that the two
  decoders *can* diverge in principle.
- **Belief source.** Per-position unary potentials read out from a real LLM (Qwen2.5-3B), plus an
  exact-emission control on the same instances (exact unary potentials instead of the LLM's).
- **Controls.** (i) exact emission on the same instances, to confirm the task itself can separate
  the two decoders; (ii) an adversarial worst-case construction (a "fan" of weak feasible solutions
  converging on one value at one position), to confirm the mechanism by which path decoding can win.
- **Metrics.** Whole-answer exact-match accuracy gap (single-best-path minus per-position-marginal);
  posterior modality (mass on the single best path); and the infeasibility rate of the marginal
  decoder's output (how often per-position argmax violates a hard constraint).

## Pre-registered prediction

1. **Core prediction.** On real-LLM emissions, the per-step output is sharp (top-1 probability
   around 0.9), the posterior is near-unimodal, and single-best-path and per-position-marginal
   decoding give essentially the same answer: whole-answer accuracy gap near zero, with a confidence
   interval that touches zero. Path decoding does not add accuracy on top of marginals here.
2. **Exact-emission control.** With exact unary potentials on the same instances, the posterior is
   multimodal and the two decoders separate (a positive accuracy gap, lower mass on the single best
   path).
3. **Feasibility.** On random multimodal instances, the only clean advantage of path decoding is
   staying globally feasible (marginal decoding produces constraint-violating outputs at a
   noticeable rate, path decoding never does) — this is a legality advantage, not an accuracy one.
4. **Adversarial control.** A deliberately constructed fan of weak feasible solutions makes
   single-best-path decoding strictly beat per-position-marginal decoding on accuracy, confirming
   the mechanism exists.

## KILL condition

If, on real-LLM emissions, the posterior turns out genuinely multimodal and the competing mass
lands in the fan-shaped regime so that single-best-path decoding **substantially and reliably
beats** per-position-marginal decoding on whole-answer accuracy (a clearly positive gap whose
confidence interval excludes zero), then the conclusion is overturned: path decoding would carry
independent accuracy value on real LLM output, not merely the legality advantage. Under the
hypothesis, real LLM output is too sharp to land there, so this is not expected.

## Boundary

This is a synthetic chain task with real-LLM read-out as the unary source; an exact oracle is
available for ground truth. It establishes the mechanism and its magnitude. The tested tasks carry
design caveats (local lexical cues, a brittle whole-answer exact-match metric). The directional
claim (sharp output → the two decoders do not separate) is what is being tested; pinning it down
fully would need a cleaner multimodal natural-language task.
