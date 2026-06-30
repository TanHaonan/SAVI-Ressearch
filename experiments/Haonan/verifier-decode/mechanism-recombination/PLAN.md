# PLAN — Recombination mechanism check (fair, same-source decode)

**Status: SPEC (to be executed).** This is a verifier-line study. It does NOT edit the
master scorecard or ledgers; it writes its own `RESULTS.md` / `results.json` /
`EXECUTION_LOG.md` here.

## Why this study exists

The M1/M2 mock comparisons are rigged against the trellis and we found exactly where:

- `best_of_many` consumes **chain** mode, and the decoupled chain mock injects the exact
  solver for `n_solver = max(1, N // 8)` (Countdown, `core/sampler.py:_sample_chain`) /
  `max(1, N // 4)` (algebra, `core/algebra_sampler.py:_sample_chain`) of the draws. So on
  any solvable instance `best_of_many` is *handed a correct chain* → pass@1 = 1.0.
- `savi` consumes **step** mode, which is undirected (spread over distinct next-states, no
  goal direction). So the trellis gets a strictly weaker generator.

The 1.0-vs-0.125 gap is therefore an artifact of *which generator each arm consumes*, not
of the decoder. Both VERDICTs already say "the mock can't show the science." This study
removes the asymmetry and asks the real question:

> **When best-of-K and the trellis consume the SAME per-step generator, at EQUAL compute,
> does global decode recover correct paths that selection misses — and is it the Φ-merge
> (semantic-state merging) that does it?**

This is a **semi-synthetic mechanism check**, not the real-LM result. It cannot and does
not claim a real model behaves this way. What it CAN do, before the emission line's real
generator arrives: (a) prove the decoder mechanism has headroom to recover at all (de-risk
the decisive run); (b) map *where* in generator-quality space global decode helps (predicts
the regime where SAVI helps a real LM); (c) attribute any lift to the Φ-merge specifically.

## Core design — the fair, competence-parameterized generator

A single per-step policy `P_p(move | state)`, consumed identically by every arm. No arm
ever gets oracle injection.

- Enumerate the legal moves at a state as TEXTS (reuse the existing per-domain
  `mock_decoupled` step sampler at high temperature + dedup-by-outcome — the same
  enumeration the Markov/false-merge audits already use).
- Classify each legal move by its successor: **good** if `domain.solvable(s')` is True
  (goal still reachable), else **bad** (dead end). `domain.solvable` is the exact backward
  oracle each domain already exposes (`countdown.reachable` / algebra CAS solvable).
- `P_p`: good moves get weight `p`, bad moves get weight `(1 - p)`, normalized within the
  enumerated legal set. (All-good or all-bad ⇒ uniform.)

Properties (the inverted-U we are testing):
- `p → 1`: every step stays on the solvable manifold; since state strictly progresses
  (Countdown values shrink each op; algebra under a depth cap), a single rollout reaches
  the goal ⇒ `best_of_k ≈ 1.0` ⇒ **DP redundant** (high-competence control).
- `p → low`: good moves rare; even a width-K beam cannot assemble an all-good path within
  depth ⇒ everyone ≈ 0 ⇒ **nothing to recombine** (low-competence control).
- `p` mid-band: single rollouts often step off the manifold and the whole chain fails, but
  the **beam keeps surviving good states and the Φ-merge collapses duplicate canonical
  states**, spending beam width on *distinct* states ⇒ recall > best-of-K. **This is the
  predicted lift.**

**Methodological note (load-bearing honesty).** The generator uses the backward oracle to
*define competence p*; this is a synthetic stand-in for "a model that proposes good moves
with probability p," not a claim about a real model. It is fair as a decoder comparison
because (1) **all arms consume the identical generator** — the oracle is in the emission
equally, never in one arm's decoder; (2) the headline decoder runs `verifier=False` — it
never calls `solvable`; (3) the result is reported as competence→lift, clearly labeled
semi-synthetic. The real-LM number still waits on the emission line.

## Arms

All consume `P_p` via the same `sample(state, N, tau, seed, mode)` closure.

| arm | what | role |
|---|---|---|
| `greedy` | 1 rollout, τ=0 (argmax-weight move each step) | weak baseline |
| `best_of_k` | K independent rollouts of `P_p`; ok if any reaches goal | selection baseline |
| `best_of_k_isobudget` | draw rollouts until cumulative candidate count ≥ the trellis's candidate budget for that instance; ok if any solved | **compute-matched selection (headline baseline)** |
| `savi_support` | trellis + Φ-merge, support edges, `verifier=False` | **the claim** |
| `savi_freq` | trellis + Φ-merge, freq edges, `verifier=False` | does graded prob add anything (D2) |
| `beam_no_merge` | width-K beam over `P_p`, **no Φ-merge** (nodes keyed per-path), same K/N/depth/verifier=False | **isolates Φ (ablation)** |
| `ceiling` | `domain.solvable(s0)` | headroom numerator / sanity |
| `savi_verifier_on` | trellis with `verifier=True` (oracle per-step mask) | quantifies model-vs-oracle gap (control C4) |

## Metrics (per (domain, p, seed), aggregated with paired bootstrap over instances)

Computed on the **headroom stratum** (the denominator) and on the full set.

- pass@1 per arm.
- **D1 (global > selection, equal compute):** `savi_support − best_of_k_isobudget`,
  paired bootstrap CI (n=10000). Headline.
- **D_merge (Φ is load-bearing):** `savi_support − beam_no_merge`, same K/N/budget, CI.
- **D2 (graded vs support):** `savi_freq − savi_support`, CI.
- **DP-only wins:** count of instances `savi_support` solves but `best_of_k_isobudget`
  does not (the irreducible recombination lift), as a fraction of headroom.
- **Φ-attributed wins:** instances `savi_support` solves but `beam_no_merge` does not.
- **merge ratio:** Σ `trellis_widths_before_merge` / Σ `after_merge` (how much the Φ-merge
  actually collapsed) — expect it to track where D_merge is positive.
- **iso-compute table:** total candidates / tokens / exec per arm; assert
  `best_of_k_isobudget` candidate budget ≥ `savi_support` candidate budget (matched).

## Controls / falsification (predictions that must hold or the story is wrong)

- **C1 high competence (p=1.0):** `best_of_k ≈ savi ≈ 1.0`, DP-only wins ≈ 0. (DP redundant
  when the generator is already good.)
- **C2 low competence (p≈0.3):** all arms low, DP-only wins ≈ 0. (Nothing to recombine.)
- **C3 unsolvable instances:** ceiling = 0 and every arm = 0 (no false-positive wins).
- **C4 verifier on vs off:** `savi_verifier_on ≥ savi_support`; the gap is how much the
  oracle mask does vs the model-load-bearing decode. Headline stays `verifier=False`.

The scientific claim succeeds iff there is a competence band where **D1 > 0 with CI
excluding 0 AND D_merge > 0**, flanked by C1/C2 nulls (an inverted-U, not a monotone
artifact).

## Substrate

- **Primary: Countdown** (~0.02 s/savi call). Full sweep. `builtin_small` (28 = 16 headroom
  + 8 control + 4 unsolvable). Grid `p ∈ {0.3,0.5,0.6,0.7,0.8,0.9,1.0}`, seeds 1..5,
  K=8, N=16, max_depth=6.
- **Confirmatory: algebra** (~14 s/savi call → tiny). `builtin_small` 8-headroom stratum,
  1–2 competence points in the band (e.g. p=0.6[,0.7]), 1 seed, K=6, N=8, max_depth=8,
  arms {`best_of_k_isobudget`, `savi_support`, `beam_no_merge`, `ceiling`}. Shows the
  mechanism is not Countdown-shaped on a real (CAS) Φ.

## Deliverable code (this directory; decode_core stays FROZEN — Countdown parity preserved)

- `gen_fair.py` — domain-agnostic fair generator.
  - `make_fair_generator(domain, enumerate_moves, p, depth_cap) -> sample(state,N,tau,seed,mode)`.
    `mode="step"` → N i.i.d. draws from `P_p`; `mode="chain"` → N autoregressive rollouts
    of `P_p` (no oracle injection); τ=0 → deterministic argmax-weight.
  - `enumerate_legal_texts(domain, step_sampler, state) -> [text]` (dedup by successor canon).
  - Deterministic, process-stable seeding (sha256 of seed, canon(state), p, mode, N, temp),
    mirroring the existing samplers.
- `arms_ext.py` — `best_of_k_isobudget(...)` and `beam_no_merge(...)`, reusing
  `decode_core` `Budget/Node/Result/_stable_topk/_backtrack`. `beam_no_merge` runs the same
  layer-by-layer loop as `savi` but keys nodes per-path (no canonical collapse); a test
  asserts it equals `savi` on an instance whose canons never coincide.
- `run_mechanism.py` — sweep harness.
  - CLI: `--domain {countdown,algebra} --set builtin_small --p-grid 0.3,0.5,... --seeds 1,2,..
    --K 8 --N 16 --max-depth 6 --arms <csv> --outdir <dir>`.
  - Writes `<outdir>/results.json` with schema: `{config, per_cell:[{domain,p,seed,
    pass_at_1{arm}, per_instance{arm:{id:bool}}, budgets{arm:{candidates,tokens,exec}},
    dp_only_wins, phi_attributed_wins, merge_ratio}], gates:{D1,D_merge,D2 per p with
    delta,lo,hi}, controls:{C1,C2,C3,C4}}`. Deterministic + resumable (cache by cell key).
- `tests/` — TDD per module: generator determinism + in-domain + the p→1/p→0 limits;
  arms_ext equivalence test; harness smoke (tiny grid) finite + deterministic.

## Execution (ultracode workflow)

1. **Build** — implement the three modules + tests (TDD), run pytest, run a tiny smoke
   (Countdown, p=0.6, seed 1, 4 instances) and print the numbers.
2. **Run** — (parallel) Countdown full sweep; algebra confirm.
3. **Verify** — (parallel, adversarial) equal-compute fairness audit; Φ-merge ablation
   validity; reproduce one cell from scratch (determinism).
4. **Synthesize** — write `RESULTS.md` (labeled semi-synthetic) + `results.json` +
   `EXECUTION_LOG.md`; state the competence band (if any), D1/D_merge/D2 with CIs, the
   controls, and the algebra confirmation; list what still requires the real generator.

## What this does NOT settle

The real-LM end-to-end number (still gated on the emission line). This study is about the
**decoder mechanism**: whether, and where in generator-quality space, global decode +
Φ-merge can recover headroom that compute-matched selection cannot. A positive, banded,
merge-attributed result de-risks and *predicts* the decisive run; it does not replace it.
