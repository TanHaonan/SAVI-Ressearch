# SPEC — novel-path / recombination observation

## Reframed success criterion
Stop scoring verifier-decode on "cheaper than best-of-K" (iso-token, where it
keeps dying). Score it on **doing what selection cannot do by definition**:
returning a correct path that **no single rollout produced** — assembled
("stitched") from per-step fragments that live in different samples. best-of-K can
only return a whole chain it sampled; global decode over a merged trellis can
recombine verified prefixes/suffixes across samples. This is compute-independent.

## Substrate
The integer-sum lattice (`mechanism-recombination/domain_lattice.py` via
`merge-noise-phasemap/domain_merge.py`), the competence-`p` fair generator
(`gen_fair.make_fair_generator`), and the soft-λ global decode
(`arms_phasemap.savi_value`). Value: the **learned** logreg value
(`value_lattice`, the on-policy realistic verifier); freq-only (λ=0) as contrast.

## Exact per-step probability (the compute-independence lever)
At r>0 the legal moves are {1,2,3}; the generator weights GOOD (`solvable(apply)`)
= p, BAD = (1−p), normalized (uniform if all same class). So `P_p(move|state)` is
recomputable exactly, and the joint probability of any path
`q(path) = Π_i P_p(m_i | s_i)`. Selection sampling M chains hits a specific path
with prob ≈ M·q, so it needs ≈ 1/q rollouts. If 1/q ≫ any feasible M, selection
**categorically cannot** return that path — no matter the budget.

## Node identity & edges
Node id = `(s, r)` (T fixed per instance). A path's edges = `[((s_i,r_i), m_i)]`.
A sampled chain "covers" an edge if it contains that exact (node, move) transition
(possibly reached via a different prefix — that is the recombination).

## Classifications (per decode-solved instance)
Sample a rollout set of M chains from the same generator (selection's view).
- **novel**: decode's correct path ∉ {sampled chains} (as whole move-sequences).
- **stitch** (strict recombination): decode's correct path is novel AND **every one
  of its edges is covered by some sampled chain** — i.e., selection sampled all the
  fragments but never connected them into one path. This is the literal
  "stitch verified fragments selection had but couldn't assemble."
- **only-decode@M**: decode solves AND no sampled chain solves (selection fails at
  budget M).
- **q / (1/q)**: joint probability of decode's path and the implied rollouts needed.

## Predictions
1. novel/stitch rates and only-decode@M are **substantial at low competence**
   (p=0.5/0.6) and shrink as p→0.9 (single rollouts already complete).
2. With **no merge (g=∞)** the decode cannot stitch across lineages → novel/stitch
   rate collapses toward 0 and the only-decode advantage drops. With merge (g=0)
   it is positive. This ties novelty causally to recombination.
3. q for decode's paths (esp. only-decode) is small enough that 1/q ≫ M (selection
   can't reach them at feasible budget) — the compute-independent statement.

## Sweep
p ∈ {0.5, 0.6, 0.7, 0.9}; g ∈ {0 (merge), ∞ (no-merge)}; depth 24; value =
learned (λ=1, primary) + freq-only (λ=0, contrast); eval seeds 0–7, n=24; chain
budget M (matched-ish + large). D1 / pass-rates with bootstrap CI where relevant.

## Layout
`novelpath.py` (pure metrics: `pp_probs`, `q_of_path`, edge/novelty/stitch + runner),
`test_novelpath.py` (unit tests), `outputs/`, `VERDICT.md`.
