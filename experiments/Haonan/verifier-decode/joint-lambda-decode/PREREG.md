# PREREG — PLAN3 joint λ>0 decode (FROZEN on Tier A, before the Tier-B decisive run)

**Status: FROZEN 2026-06-29.** All thresholds, the arm matrix, the iso-token protocol,
the instance generation, and the seeds below are fixed BEFORE any real-model (Tier B)
number is read. Tier A (CPU mock) has been run to validate the harness + instruments +
gates and to freeze this document; it is explicitly **not** the science.

## 1. Frozen design (cannot change for Tier B)

- **Domain.** Deeper Countdown, k ∈ {4,5,6} numbers → derivation depth {3,4,5}. State =
  multiset of `Fraction` values + target; Φ canon = `(sorted values, target)` (strictly
  shrinks each step). Exact backward oracle `reachable` is the verifier mask (λ→∞) and
  the H5 ground truth, memoized run-wide by canon (`DeepCountdownDomain.solvable`).
- **Instances.** `make_depth_curve(ks, n, inst_seed=7, target_range=[10,100])`: per k,
  seeded generation (seed = `inst_seed + k`), filtered to oracle-SOLVABLE (oracle = 1),
  headroom = solvable ∧ greedy-myopic-proxy fails. Deltas are read on the **headroom
  stratum**. Tier B: `n = 40` per k (or as compute allows; ≥ 30).
- **Arm matrix (§3).** greedy · self_consistency · savi(λ=0,support) · savi(λ=0,freq) ·
  **savi(λ>0,freq)** [headline] · best_of_many(iso-**token**) · beam_no_merge(λ>0,freq)
  [Φ-ablation] · oracle. λ>0 ≡ `savi(edge="freq", verifier=True)`; the verifier is the
  per-step `reachable` legality+still-solvable mask + the exact leaf check — never a
  move-proposing solver.
- **Iso-compute axis = TOKENS** (`Budget.tokens` = Σ `len(text.split())`), never
  candidates. `best_of_many = best_of_k_isobudget(axis="tokens",
  target = savi(λ>0,freq).budget.tokens)` per instance: draw chain rollouts until
  cumulative tokens ≥ the savi arm's tokens. Paired comparison at equal tokens.
- **Decoder config.** K = 8, N = 16, τ = 1.0, seed = 0, `max_depth = k − 1`.
  **Amended 2026-06-29, pre-Tier-B (no real result seen): N = 16 → 8, n = 40 → 16 per k,
  `best_of_many` rollouts capped (token_ratio logged), instruments on the first 8
  instances/k.** Reason: the iso-token baseline must spend savi's full token budget via
  sequential ~5-token chains (~3.4 s each at k=6); at N=16 that is ~135 rollouts ≈ 8
  min/instance — infeasible at scale. N=8 halves savi's token budget (K=8 keeps beam
  coverage) so the decisive run fits ~1.5 h sharded across GPUs 4–7. This is a compute
  scaling of the SAME frozen arms/axis, fixed before any Tier-B number was read.
- **Statistics.** Paired bootstrap n_boot = 10000 on the headroom stratum; Holm across
  the {H1, H4} family. A hypothesis "passes" iff its paired-delta CI excludes 0 in the
  predicted direction.
- **decode_core is FROZEN.** The trellis decoder + arms are the vendored copy
  (byte-identical to `verifier-decode/decode_core`, verified 2026-06-29). All new code
  lives in `joint-lambda-decode/`. The two extra arms (`beam_no_merge`,
  `best_of_k_isobudget`) are verbatim ports in `arms_local.py`.

## 2. Frozen hypotheses + thresholds (the prereg bar for Tier B)

These describe the **real-model** expectation; they were drafted in PLAN3 §9 and are
frozen here unchanged. (Where Tier A cannot stand in for the real model, see §4.)

- **δ(H1).** `savi(λ>0,freq) − best_of_many` ≥ **+3 pts** at iso-token, deepest tier
  (k=6), paired-bootstrap CI excluding 0.
- **H2.** `D1_token(depth)` (= the H1 delta at each k) is increasing over k ∈ {4,5,6};
  ≤ 0 at k=4 (replicating the emission line's depth-3 λ=0 loss) and > 0 (CI excl 0) at
  k=6. *Primary signal = the deepest-tier sign + the monotone trend.*
- **H3.** in-trellis `K_eff(decoupled) ≥ 1.5` and `> K_eff(coupled)`; H1 holds for
  decoupled and **not** for coupled. If `K_eff(decoupled) ≈ 1` in-trellis (the
  marginal-vs-path kill criterion), PLAN3 fails for a stated, measured reason.
- **H4 / E6c.** `savi(λ>0,freq) − beam_no_merge(λ>0,freq)` > 0, CI excluding 0 (expected
  positive but the smaller lever).
- **H5 / v.** ECE of the freq edge `p = count/N` vs `reachable` reported per layer
  (descriptive; no pass/fail).
- **Validity gates** (instrument-valid iff all hold): V1 headroom non-empty; V2
  `K_eff(dec) > K_eff(cou)` and `> 1`; V3 oracle pass@1 > best_of_many pass@1 at
  iso-token; Φ-works merge ratio > 1.

## 3. Tier-A validation (machinery + instruments + gates — run 2026-06-29)

Config: tier A, ks 4,5,6, n 40, K 8, N 16, τ 1.0, seed 0, competence mock p = 0.7,
n_boot 10000 (`outputs/tierA.json`). Decoupled = faithful competence-`p` generator (no
oracle injection; best_of_many is a real, beatable baseline); coupled = one-hot mock.

| k | depth | savi λ0-freq | **savi λ>0-freq** | best_of_many | oracle | H1 [CI] | K_eff dec/cou | merge | ECE | parse |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 3 | 0.595 | **0.946** | 0.730 | 1.000 | +0.216 [+0.054,+0.378] | 6.76 / 1.0 | 5.42 | 0.079 | 1.000 |
| 5 | 4 | 0.595 | **1.000** | 0.676 | 1.000 | +0.324 [+0.189,+0.486] | 7.94 / 1.0 | 4.41 | 0.122 | 1.000 |
| 6 | 5 | 0.615 | **1.000** | 0.769 | 1.000 | +0.231 [+0.103,+0.359] | 8.95 / 1.0 | 2.58 | 0.230 | 1.000 |

**All four validity gates pass.** The harness detects a **CI-significant λ>0 win** over
iso-token best_of_many at every depth on a controlled generator, the verifier lifts
pass@1 from ~0.60 (λ=0) to ~1.0 (λ>0) — the slice only the verifier claims — and every
instrument (K_eff, depth curve, merge ratio, ECE, parse rate) produces sane numbers.
The plumbing the Tier-B run rests on is validated. 10/10 deliverable tests pass
(`tests/test_plan3.py`: reachable-memo correctness, K_eff, iso-token accounting,
determinism).

## 4. Where Tier A CANNOT pre-judge Tier B (honest caveats)

The mock is a stand-in for the harness, not the model. Three places the real run may
differ, flagged now so Tier-B interpretation stays honest:

1. **H2 shallow sign.** On the competence mock the verifier already helps at k=4
   (H1 = +0.216 > 0), so the mock does **not** reproduce the predicted "≤ 0 at depth 3".
   That shallow-negative is a property of the *real* generator (whose λ=0 lost −0.375 at
   depth 3); only Tier B can test it. We hold the H2 threshold as written.
2. **H4 = 0 on the mock.** Φ-merge compresses the trellis 2.6–5.4× but does **not**
   change pass@1 at K=8 (beam_no_merge collects goals over the full expanded layer, so a
   wide beam keeps the goal path either way). H4's lever likely only manifests at
   **narrow K**; Tier B additionally runs the H4 ablation at K=2 to give merge a chance
   to bind. We keep the H4 threshold but treat K=8 H4≈0 as expected, not a null.
3. **Coupled is not collapsed on the real model.** The Phase-0 probe shows the
   one-hot-trained ("coupled") adapter still branches at τ=1.0 sampling (root K_eff 6–7),
   unlike the coupled *mock* (K_eff≡1). So H3's "H1 fails for coupled" may be weaker than
   on the mock; Tier B measures the real decoupled-vs-coupled contrast rather than
   assuming the mock's collapse.

## 5. Phase-0 gate before Tier B (§13.3)

Tier B proceeds only per the Phase-0 probe decision rule on in-trellis K_eff:
`PROCEED_tierB_at_depth` (k=6 K_eff healthy) → run as specified; `OOD_…` (k=6 collapse)
→ short k-mixture LoRA SFT first; `KILL_marginal_vs_path` (K_eff≈1 even at k=4) → the
stated H3 negative, redirect to the emission objective. (Probe smoke 2026-06-29:
decoupled k=4 K_eff 3.40, k=6 6.26, parse 1.0 — pointing to PROCEED; full probe pending.)
