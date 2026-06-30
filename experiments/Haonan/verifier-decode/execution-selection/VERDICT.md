# VERDICT — execution-selection (interim; final numbers pending the 7B run)

**Status: interim.** The 1B cell and two smokes are in; the 7B cell is finishing. The
headline numbers below are filled in at relocation from the final result files.

## Finding

In a verifiable code domain, an execution signal recovers correctness only as a
**selector over independent samples**, not as a corrector of a single sample, and richer
per-candidate signals do not beat plain binary selection.

- **Selection works.** Execution-filtered best-of-K over independently sampled programs:
  +10.2 points pass@1 at 1B, +18.4 at 7B, over greedy.
- **Self-correction does not.** A backward localizer identifies the wrong line (dev-set
  top-1 ≈ 0.7–0.8) but regenerating that line gives no pass@1 gain over regenerating the
  whole function (difference within the bootstrap interval at both scales). The model
  reproduces the same mistake regardless of where it is told to look. Detection and
  correction are separate. (Established separately; see the backward-error-detection
  package referenced below.)
- **Richer selection does not beat binary.** On the stratum where base-passers disagree on
  the hidden tests: the generator's own likelihood does not order them above chance, and
  consensus over self-generated tests rarely splits them (the distinguishing inputs are
  adversarial edge cases the model does not generate).
- **Headroom is small on EvalPlus.** An oracle that always selects a hidden-test passer
  barely beats best-of-K (1B: oracle ≈ best-of-K; 7B: oracle − best-of-K ≈ 5 points). Once
  a candidate passes the visible tests it usually passes the hidden ones.

## Conditions

The result holds for whole-program generation on HumanEval+ at StarCoderBase 1B and 7B.
The small oracle−best-of-K gap is a property of EvalPlus (visible tests already imply
hidden tests for most problems). It does not transfer to domains where the constraint
genuinely decides the answer; the first joint experiment with the emission line should use
such a domain.

## For SAVI

Use the verifier to **select / constrain over the beam**, not to direct regeneration. The
ceiling on what selection can add is set by candidate diversity (the emission line) and by
the domain's selectable headroom.

## Pointers

- Pre-registration: `../PREREG.md` (full frozen text copied at relocation).
- Backward-error-detection (the "detection ≠ correction" evidence): packaged separately at
  a separate `write_time_ownership/` package (not duplicated here).
- Result files: `results.json` (distilled); raw run logs are reproducible and not committed.
