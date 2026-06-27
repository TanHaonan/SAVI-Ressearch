# PREREG — a generation-level calibration term ("commit-slot" calibration)

## Plain-language summary

Earlier work found a split. When we read the model's answer *as a probability
over the option letters at the answer position* (the "readout"), the trained
`decoupled` adapter is almost perfectly calibrated to the known posterior: its
distance-to-truth (total-variation, TV) is about **0.02**. But when we instead let the
same adapter **freely generate** a short reasoning + a `COMMIT: <thing>` line and
*count which thing it actually commits to*, the committed distribution is far from the
known posterior: calibration-TV **0.31–0.52**, and about **1/3 of samples abstain**
(no usable commit). So the model "knows" the right distribution at the readout but does
not *emit* it when generating.

This experiment tests the cheapest possible fix: add **one training term** that
directly calibrates the token the model emits in the COMMIT slot, and check whether
that shrinks the readout→generation gap without breaking anything.

## Hypothesis

Training the model so that, after a fixed reasoning preamble, the next-token
distribution **over the option nouns at the `COMMIT:` slot** matches the known
posterior, will move the *generated* committed-state distribution toward that posterior
— shrinking the gap `gap = calibration_tv − readout_tv` and lowering the abstain rate —
while leaving the answer-position readout and general fluency essentially unchanged.

## Design (clean A/B, one added term)

- **Control (`decoupled`)**: the existing oracle/seed-0 adapter. Training loss =
  `loss_decoupled` (KL of the known posterior to the softmax of the answer-position
  option logits) + a fluency KL anchor to the base model on generic text.
- **Treatment (`decoupled_genreg`)**: *same data, same schedule, same control loss and
  fluency anchor*, plus **one** additional term:
  `LAM_COMMIT · loss_commit`, where `loss_commit` is the **same KL form** applied at the
  COMMIT slot — i.e. KL(known posterior ‖ softmax of the per-noun logits read at the
  token right after a fixed reasoning template + " COMMIT:"). Default `LAM_COMMIT=1.0`.

Everything else (mode=oracle, seed=0, 6 epochs, lr 1e-4, LoRA rank 16, batch 8,
fluency `lam_kl=0.5`, target = uniform-over-survivors) is held fixed. The only
independent variable is the added commit-slot calibration term.

- **IV (independent variable):** presence of the commit-slot calibration term
  (`LAM_COMMIT · loss_commit`).
- **Control:** all other training (data, control loss, fluency anchor, schedule, seed).
- **DV (dependent variables), measured on the held-out test split, structured-commit regime (`b1`):**
  - primary: the structured-commit gap `calibration_tv − readout_tv`, by number of survivors `j`;
  - primary: the structured-commit `abstain_rate`, by `j`.
- **Guardrails (must hold for a "real" win):**
  - `readout_tv` stays ≈0.02 (the readout must not be sacrificed to move the gap);
  - generic-text perplexity / fluency does not collapse (checked via the same base-vs-adapter
    KL anchor used in training, on a held-out generic-text slice).

## Decision rule

- **Support:** structured-commit gap (and/or abstain_rate) shrinks materially for `decoupled_genreg`
  vs `decoupled`, *while* `readout_tv` stays ≈0.02 and fluency does not collapse.
- **KILL → escalate to a heavier sequence-level / RL lever:**
  - the gap does **not** move; **OR**
  - the gap moves **only by hurting** the readout (`readout_tv` rises off ≈0.02)
    or fluency (perplexity collapses).

## Main risk (stated up front)

We calibrate the commit token **given a single FIXED reasoning template**
(default `" Let me weigh the remaining clues."`), but the eval **samples** the reasoning
freely (temperature 1.0). The commit-slot fix may therefore fail to *generalize* from
the fixed preamble to sampled preambles. If the training-time commit-slot TV drops but
the eval structured-commit gap does not, that mismatch (fixed-vs-sampled reasoning) is the most likely
cause, and it is itself the signal to escalate to a sequence-level lever that sees the
model's own sampled reasoning.

## What is logged

- Training: per epoch — total loss, the two loss components (decoupled, commit), the
  answer-position mean-TV and the **commit-slot mean-TV** on a held-out slice
  (`gen-calibration/outputs/train_genreg.log`). Expectation: commit-slot mean-TV drops
  across epochs; answer-position mean-TV stays low (control guardrail at train time).
- Eval: the standard state-emission A/B (`decoupled` vs `decoupled_genreg`), both regimes
  (`b1`, `b2`), giving `calibration_tv`, `coverage`, `readout_tv`, `gap`, `abstain_rate` per cell.
- Fluency: held-out generic-text base-vs-adapter KL for both adapters.
