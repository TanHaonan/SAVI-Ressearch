# VERDICT — 1/mass math harness on real-model Countdown

**One line:** the **recombination/stitching** mechanism is **NEGATIVE on real AR
everywhere** (stitch = 0.00 at k=4 and k=6 — decode never reuses selection's
sampled fragments). A *separate*, **depth-emergent, oracle-dependent** "decode
solves what best-of-64 can't" effect does appear at k=6 (exact-oracle only_decode
= 1.00 on n=3 sparse), but it is **verifier-guided search reaching chain-uncovered
paths, NOT recombination** — and a realistic learned verifier captures only ~1/3
of it. So the reframe's specific claim (stitch verified fragments) dies on real
Countdown; what survives is the previously-identified near-exact-mask lever.

### k=4 builtin (28): flat null
Across 28: only_decode = 0.00, stitch = 0.00 for every arm. Sparse subset (n=4):
exact / learned / freq all 0/4 — **knowledge** failures (model never samples the
needed fragments at this shallow depth).

### k=6 generated (24): depth-emergent only-decode, but not stitch
All-24: exact solve 1.00 (only_decode 0.12), learned 0.50 (0.04), freq 0.25.
**Sparse (n=3): exact only_decode = 1.00 (3/3), learned 0.33, freq 0.** Every
solved path is novel (∉ chains) but **stitch = 0** with edge-coverage ≈ 0–1/5 —
the chains never sampled decode's transitions, so decode reaches them by
oracle-masked per-step exploration, not by stitching selection's fragments. The
effect is **oracle-dependent** (learned captures 1/3) and rests on n=3.

## Setup
`CountdownDomain`, real Qwen3-4B+LoRA (adapter_decoupled_s0), builtin set (28;
4 unsolvable filtered → 24 solvable). K_big=64 selection rollouts, decode K=8
N=8 τ=1.0. Sparse subset = oracle-solvable ∧ best-of-64 finds 0 correct.
Metrics: node identity = `canon(state)`, edge = canonical transition; 4 metric
unit tests pass; 2-instance GPU smoke passed.

## Results
| subset | arm | solve | only_decode | novel(of solved) | **stitch(of solved)** |
|---|---|---|---|---|---|
| ALL (28) | exact | 0.50 | **0.00** | 0.21 | **0.00** |
| ALL (28) | learned | 0.43 | 0.00 | 0.25 | 0.00 |
| ALL (28) | freq | 0.36 | 0.00 | 0.10 | 0.00 |
| **SPARSE (4)** | exact | **0.00** | 0.00 | – | – |
| SPARSE (4) | learned | 0.00 | – | – | – |
| SPARSE (4) | freq | 0.00 | – | – | – |

Sparse instances bs-01/04/06/08: all 0/64 chains, all arms fail including exact.
(`novel>0` but `stitch=0`: a few solved paths aren't in the chain set, but their
edges aren't all sampled by selection either — that is decode's per-step sampler
finding a different move, not recombination of selection's fragments.)

## Why the lattice win did not transfer
A categorical recombination win needs **both**: (i) sparse solutions (so selection
can't sample a whole correct chain) AND (ii) a generator whose per-step sampling
still covers the correct *fragments* densely (so there is something to stitch).
The two regimes we've run satisfy exactly one each, for opposite reasons:

- **Synthetic lattice**: generator uniform at low p → fragments densely, diversely
  sampled (ii ✓) → stitch 23–36%. But additive lattice has many correct paths
  (i ✗) → not categorical at the instance level.
- **Real Countdown**: solutions are genuinely sparse (i ✓), but the AR model's
  per-step coverage is peaked/correlated → the correct fragments at deep nodes are
  not sampled (ii ✗) → stitch = 0, and hard instances are knowledge-bound (exact
  decode 0/4).

Real autoregressive generators structurally violate (ii): once a rollout drifts it
stays drifted, so marginal fragment coverage at depth collapses together with
joint coverage. This matches the **correlated-coverage** negative (no free AR
analog of decorrelated coverage; coverage ceiling = base model + tokens). The
lattice's recombination win was an artifact of a synthetic decorrelated generator.

## Bottom line
On real AR Countdown: (1) **recombination/stitching is dead** — stitch = 0 at both
depths; the AR generator's correlated, peaked per-step sampling does not lay down
decorrelated fragments to recombine (matches the correlated-coverage negative).
(2) A **depth-emergent only-decode effect** exists (k=6 exact 3/3 sparse) but it is
**oracle-dependent verifier-guided search**, not recombination, and a learned
value keeps only ~1/3 — i.e. the same "near-exact mask is the only lever" from
joint-lambda, now shown to also produce chain-uncovered paths. The binding
constraints remain the **generator's fragment coverage** and the **verifier's
exactness**, not the decode-vs-select choice. Recombination as a categorical lever
needs high-marginal / low-joint fragment coverage, which standard AR sampling does
not provide.

## Caveats / scope
- Sparse subset is n=4 on builtin; a harder **generated k=6** run is in progress
  (`outputs/gen_k6_K64.json`) to firm up the knowledge-bound finding. The robust
  signal is stitch=0 / only_decode=0 across all 28 (and 24 solvable).
- Single model/adapter (decoupled_s0), raw (non-degraded) emitter. A
  decorrelation intervention on the emitter (higher τ / diverse decoding / a
  trained anti-collapse sampler) is the only thing that could revive (ii); whether
  any feasible AR intervention yields stitchable fragments is the open question
  (cf. the belief-decode / anti-collapse thread).

## Artifacts
`mass_harness.py`, `test_mass_harness.py` (4 tests), `SPEC.md`,
`outputs/builtin_K64.json` (+ `gen_k6_K64.json` pending).
