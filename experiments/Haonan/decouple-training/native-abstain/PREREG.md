# PREREG — does the trained calibration shape enter free generation, or stay at the probe logit?

*Written before running. Records the hypothesis, setup, and the pass/kill conditions fixed in
advance.*

## Hypothesis

A companion experiment trained a small adapter so that the model's yes/no answer-position logit is
discriminative on questions with a unique correct answer and calibrated (near 50/50) on genuinely
ambiguous questions. That measurement was taken at a single answer-position token — a probe. The
behavioral claim is stronger: when the model freely writes an answer, does it commit correctly on
determinate inputs and hedge or abstain on genuinely-ambiguous ones, instead of confidently picking
a side?

We separate two things that the training fused. **Discrimination** — using a correct answer the
model already had access to — is expected to transfer to free generation: greedy decoding will write
the answer the logit favors. **Calibration** — the flat 50/50 shape on ambiguous inputs — is
expected **not** to transfer, because a logit near 0.5 still lets greedy decoding pick one token, so
the model can still write a single committed answer even though the probe reads as uncertain.

## Setup

- **Carrier.** A frozen Qwen3-4B with a small LoRA adapter, trained with an objective that decouples
  calibration from commitment. Four models are compared: base (no adapter), a one-hot
  cross-entropy variant, and the decoupled objective at two strengths.
- **Task.** Free-text answers (no "answer yes/no only" constraint, greedy decoding) to determinate
  questions (context uniquely fixes the answer) and to genuinely-ambiguous questions (both readings
  valid).
- **Judge.** A separate, cached instruction-tuned model classifies each free-text answer's stance as
  commit-yes, commit-no, hedge, or abstain, read in a single forward pass. Validated by dumping
  samples for human inspection and reporting stance counts.
- **Metrics.** Determinate selective accuracy (accuracy when the model commits) and confident-wrong
  rate; ambiguous confident-commit rate (the behavior to avoid) versus hedge/abstain rate (the
  calibrated behavior). Generic-text perplexity as a fluency control. Bootstrap confidence intervals
  per model.

## Pre-registered prediction

- **Discrimination transfers.** On determinate questions, the trained models commit correctly far
  more than base: selective accuracy rises sharply and confident-wrong rate drops, in both the
  decoupled and the one-hot variants, since both train discrimination into the answer-position logit.
- **Calibration does not transfer.** On ambiguous questions, the model whose probe logit is flattest
  still shows a confident-commit rate in free generation close to base — greedy decoding writes a
  committed answer even when the probe reads near 50/50. Its confident-commit rate is not lower than
  the one-hot variant.
- **Fluency preserved.** Generic-text perplexity stays near base across all models.

## Decision rules

- **PASS (shape entered behavior).** The decoupled model has a *lower* ambiguous confident-commit
  rate than base and the one-hot variant — the confidence interval of the difference excludes zero —
  and a higher hedge/abstain rate, while keeping determinate selective accuracy and fluency near
  base. This would mean the calibration shape transfers from the probe logit into generation
  behavior.
- **KILL (probe-only).** The decoupled model's free-generation confident-commit rate is
  indistinguishable from base and the one-hot variant (the difference's confidence interval includes
  zero). The calibration shape would then live only in the answer-position logit, not in generation
  behavior, and the claim "internalized into the native forward" must be narrowed to "into the yes/no
  logit, not into free generation."
- **Secondary kill.** If fluency collapses in free generation (perplexity far above base), the
  fluency anchor used during training was too weak.

## Boundary

The hedge/abstain judgment depends on the judge model and its threshold; absolute hedge rates are
sensitive to it, so the robust signal is the *relative* comparison across models. The calibration
shape was trained only on the answer-position yes/no logit; whether training the shape on generated
tokens (rather than the single answer-position logit) would move this boundary is left open.
