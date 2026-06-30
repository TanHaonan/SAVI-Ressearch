# PREREG — execution-selection (placeholder until relocation)

The frozen pre-registration for this probe currently lives with the running experiment
and will be copied here verbatim once the 7B run finishes. It registers:

- **Question.** Among code candidates that are indistinguishable on the visible (base)
  tests, does any out-of-base signal predict which also pass the hidden tests?
- **Arms.** greedy (A0); best-of-K selection (binary pass/fail); value (the generator's
  own likelihood); value_indep (an independent model's likelihood); consensus (cluster
  candidates by behavior on model-self-generated tests, pick the largest cluster); oracle
  (pick a hidden-test passer if one exists — the ceiling).
- **Stratum.** Problems where the base-passers disagree on the hidden tests (the only
  place selection among them can matter).
- **Validity gates (must pass or the cell is instrument-invalid, not a null).** the
  disagreement stratum is non-empty; the generated tests actually split base-passers;
  the oracle beats best-of-K (there is selectable headroom).
- **Decision gates.** value − random ≥ 3 points; consensus − random ≥ 3 points; both on
  the disagreement stratum, paired bootstrap, Holm-corrected.

This file will be replaced by the exact frozen text (no threshold changes) at relocation.
