# verifier-decode — the verifier + global-decode half of SAVI

This line is the complement of `Haonan/decouple-training/`. That line works on the
**emission** side: training an LLM's output to be calibrated and multimodal so a
global decoder has comparable candidates. This line works on the **decode + verifier**
side: given candidates, what does an execution/verifier signal let you recover, and
through which channel.

The two lines meet at one interface — "N candidate continuations from a state" — and
one shared metric set (`../_shared/`).

## Result so far (verifier side)

In a verifiable domain (code, where execution is a ground-truth checker), the execution
signal recovers correctness only through **selection over independent samples**:

- Execution-filtered best-of-K selection lifts code pass@1 by +10 points (1B) and +18
  points (7B) over greedy.
- Directing the model to repair its own output (localize the wrong line, regenerate it)
  does not help: detection of the wrong line works, but the model regenerates the same
  mistake. Detection and correction are separate.
- Richer per-candidate signals do not beat plain binary pass/fail selection: the model's
  own likelihood does not separate candidates that pass the visible tests, and consensus
  over self-generated tests rarely splits them (the distinguishing inputs are adversarial
  edge cases the model does not generate).
- On HumanEval+ the selectable headroom is small: once a candidate passes the visible
  tests it usually passes the hidden ones (an oracle that always picks a hidden-test
  passer barely beats best-of-K). This says the verifier half has little to show on
  EvalPlus, and points the first joint experiment toward a constraint-heavy domain.

For SAVI: the verifier's role is to **select / constrain over the beam**, not to direct
regeneration. The payoff is bounded by candidate diversity (the emission side) and by
whether the domain has real selectable headroom.

## Layout

```
execution-selection/   the code probe: A0 / best-of-K / value / consensus / oracle
  PREREG.md            hypotheses + gates (frozen before the decisive run)
  VERDICT.md           result + conditions
  core/                minimal reproducible harness (self-contained: evalplus + transformers)
  tests/               CPU unit tests
  results.json         the distilled headline numbers the report cites
```

## Provenance

The probe code was developed in a separate program repo and relocated here. Large raw
run logs are reproducible and are not committed (see `execution-selection/core/README.md`).
The backward-error-detection experiments that established "detection ≠ correction" live in
a separate package and are referenced from `execution-selection/VERDICT.md`, not duplicated
here.
