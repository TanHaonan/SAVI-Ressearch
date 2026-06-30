# MECHANISM — canonical-state repulsion sampler S′ (correlated-coverage)

**Scope.** This is the detailed, build-to spec for the constructive method `S′` of `PREREG.md`
(§4, §7, §8). It pins down the exact per-step reweighting, the β=0 ⇒ i.i.d. equivalence (V2),
the iso-token accounting (V3, the binding fairness constraint), the inverted-U mechanism in β
(§0/§3 H3) and its tie to the sphere-graph dimension knob k≈ℓ², plus complexity / edge cases /
the joint-diversity instrument (§8.3). `decode_core` stays **FROZEN**: S′ is a new sampler that
consumes the *same* `domain` (the `Domain` Protocol: `canon`, `is_goal`, `apply`, `parse_move`,
`parse_chain`, `solvable`) and the *same* per-state emission primitive
`sample(state, N, temperature, seed, backend, mode)` already in `core/_deps/sampler.py`.

Throughout, "depth t" indexes trellis layers (one `apply` per step), "chain i" is the i-th of K
sequentially drawn chains, and `S` / `S′` are **canonical** states, i.e. images under
`domain.canon(·)` (the Φ-merge key). `n(·)` always counts in canonical space — that is the whole
point of S′ (we repel from canonical *identities*, not from surface text or raw states).

---

## 1. The per-step reweighting `P̃(S′|S)`

### 1.1 What the base emission gives us

`decode_core` never reads logits; it reads candidate **texts**. So the "emission distribution"
at a state `S` is reconstructed empirically from a step-mode draw:

```
cands = sample(state, N, temperature, seed, backend, mode="step")   # N single-op texts
```

For each candidate text `c` we parse-and-execute exactly as `savi` does:

```
move = domain.parse_move(c, state)         # None ⇒ malformed/illegal: dropped
sp   = domain.apply(state, move)           # successor (raw) state
sp_key = domain.canon(sp)                  # canonical successor S′
```

Tallying the surviving candidates by `sp_key` gives the **empirical emission over canonical
successors**:

```
P_emission(S′ | S) = count(S′) / Σ_{S''} count(S'')          (1)
```

where `count(S′)` = number of the N candidates whose parsed move lands in canonical state `S′`.
(Normalization is over the **feasible, parsed** candidates only — see §5 for parse-failure /
dead-state handling.) This is precisely the support the `savi` trellis already builds per layer;
S′ reuses it verbatim so β=0 can be made identical (§2).

### 1.2 The repulsion reweight

Maintain a per-depth visit counter over canonical states:

> **`n(S′, t)`** = the number of *previously drawn* chains (chains `1 .. i−1`, i.e. **strictly
> before** the current chain `i`) whose realized path passed through canonical state `S′` at
> depth `t`.

When drawing **chain i**, at each step the transition from the current canonical state `S`
(reached at depth `t−1`, so the successor sits at depth `t`) is reweighted multiplicatively by a
Boltzmann repulsion factor and renormalized:

> **`P̃_i(S′ | S, t) ∝ P_emission(S′ | S) · exp(−β · n(S′, t))`**  (2)
>
> i.e. `P̃_i(S′ | S, t) = P_emission(S′|S)·e^{−β·n(S′,t)} / Z_i(S,t)`,
> with `Z_i(S,t) = Σ_{S''} P_emission(S''|S)·e^{−β·n(S'',t)}`.

`β ≥ 0` is the single correlation-strength knob (the analogue of the sphere-graph dimension knob;
§4). The next canonical state is sampled `S′ ∼ P̃_i(·|S,t)`, we set the current state to a *raw*
representative of `S′` (we keep one concrete `sp` per `sp_key`, exactly as `savi`'s `sp_cache`
does, so `apply` can continue), and proceed to `t+1`. We stop the chain at `is_goal`, at a dead /
terminal state (no parsed feasible successor), or at `max_depth`.

**Sequential draw discipline (binding).** Chains are drawn **one at a time, in order**, and the
counter is updated **after** each chain completes (or incrementally as each step is committed):

```
n(·,·) ← 0                                  # empty before chain 1
for i in 1..K:
    path_i = draw_one_chain(S0, β, n, seed_i)     # uses n built from chains < i
    for t, S' in enumerate(path_i, start=1):
        n(S', t) += 1                       # chain i now repels chains i+1 .. K
```

So chain 1 is drawn from the *unmodified* emission (its `n(·,·)` is all zero ⇒ factor `e^0=1`);
each later chain is pushed off the canonical states its predecessors occupied, **per depth**.
The repulsion is **layer-local**: a state visited at depth 3 repels future chains at depth 3, not
at depth 5 — this matches "avoid the canonical states the earlier chains walked through" (PREREG
§4) and keeps the bookkeeping `O(distinct states × depth)`.

**Design commitments here:**
- Repulsion lives in **canonical** space (`domain.canon`), never surface text. Two chains that
  reach the same `canon` by different operations are correctly treated as *redundant* and repelled.
- The factor is **multiplicative in probability / additive in log-space** (`log P̃ = log P_em −
  β·n + const`), so it composes cleanly with a future `freq`/verifier log-edge if S′ is later
  fused with `savi` (out of scope here; PREREG §14).
- `n` counts **chains**, not candidates and not raw token visits — the instrument that matters for
  coverage is *how many of my K committed samples already sit here*.
- Counter scope is **per problem instance** (reset between instances). Within an instance it is
  shared across the K chains; that sharing is the entire correlation mechanism.

---

## 2. β=0 reduces EXACTLY to i.i.d. best-of-K (V2 well-posedness)

The equivalence test must be **well-posed**: at β=0, S′ and `iid_bok` must draw from the *same*
distribution, so any measured gap at β>0 is attributable to correlation alone, not to a sampler
re-implementation artifact.

### 2.1 Distributional proof sketch

At β=0, `exp(−β·n) = exp(0) = 1` for every `S′`, every depth, every chain. Substituting into (2):

```
P̃_i(S′ | S, t) = P_emission(S′|S)·1 / Σ_{S''} P_emission(S''|S)·1 = P_emission(S′|S).   (3)
```

The reweight is the identity map and `Z_i ≡ 1`. Hence:

1. **No cross-chain coupling.** `n(·,·)` enters only through `e^{−β·n}=1`, so chain i's law does
   not depend on chains `1..i−1`. The K chains are mutually **independent**.
2. **Identical per-chain law.** Each chain is drawn step-by-step from `P_emission(·|S)` =
   exactly the empirical step-emission the existing `sample(..., mode="step")` / parse / `canon`
   pipeline defines. A whole chain's probability is `∏_t P_emission(S_t | S_{t−1})`.

So at β=0, S′ = "K i.i.d. chains, each generated by stepwise sampling from the per-state
emission, scored by `is_goal` at the leaf" — which is the **definition** of the coverage of
best-of-K under stepwise generation. The equivalence is therefore at the level of the *chain
distribution and the coverage statistic*, and the test is well-posed.

### 2.2 The seeding discipline for byte-identical identity (preferred)

Distributional equivalence is necessary but we want the **strongest** V2 the harness allows: a
byte-identical assertion (`PREREG` §8.5, §9 V2: "corr(β=0) 的链分布逐位 == iid_bok"). The
obstacle is that `iid_bok` calls `sample(s0, K, tau, seed, "chain")` **once** in chain-mode, while
S′ is intrinsically **step-mode** (it must reweight per step). These two emission paths are *not*
guaranteed byte-identical because the existing sampler derives its RNG seed from a SHA-256 digest
that **includes `mode` and `N`** (`_digest_seed(seed, state, backend, mode, N, temperature)` in
`core/_deps/sampler.py`). `mode="chain", N=K` and `mode="step", N` digest differently, so even at
β=0 the two would consume *different* random streams. Therefore byte-identity against the existing
chain-mode `best_of_k` is **infeasible without changing decode_core** (which is FROZEN).

We commit to the following, in order of strength:

- **(A) Byte-identity against a step-mode i.i.d. reference (the well-posed target).** Define the
  headline `iid_bok` arm itself as **stepwise** i.i.d.: each of K chains generated by stepwise
  sampling from `P_emission` with the *same* per-step `sample(..., mode="step", seed=seed_i)`
  calls and the *same* per-chain seed schedule `seed_i` that S′ uses. Then S′ at β=0 and this
  `iid_bok` issue **identical `sample` calls with identical arguments** at every step (because the
  only difference, the reweight, is the identity at β=0), draw from the identical SHA-256-seeded
  streams, and select the next state with the identical categorical-draw routine. The test asserts
  **path-list equality byte-for-byte** across K chains for many `(instance, seed)` pairs. This is
  the V2 we build and freeze: i.i.d. best-of-K is *defined* as the β=0 limit of S′, so the
  reduction is exact by construction.
- **(B) Per-chain seed derivation.** `seed_i` must be a deterministic, process-stable function of
  `(base_seed, instance_id, i)` — e.g. `seed_i = int.from_bytes(sha256(f"{base_seed}:{inst}:{i}"),...)`
  — so that (i) chain i in S′ and chain i in `iid_bok` share a seed, (ii) seeds are independent
  across i (no accidental chain-to-chain coupling at β=0), and (iii) reruns reproduce. The
  categorical draw `S′ ∼ P̃` must use a single `random.Random(seed_i)` advanced deterministically
  per step (e.g. one `rng.random()` per step fed to an inverse-CDF over the sorted-by-`canon`
  support), so the *only* thing β changes is the weight vector, never the random stream consumed.
- **(C) Statistical fallback (only if (A) cannot be made exact for a backend).** If some backend's
  emission cannot be expressed step-wise identically, fall back to **statistical equivalence**:
  over `S ≥ 2000` seeds, the paired difference `coverage_corr(β=0) − coverage_iid` has a 95%
  bootstrap CI (n=10000 resamples) containing 0, **and** a two-sample test on the per-instance
  coverage vectors fails to reject (p > 0.20), **and** the joint-diversity distributions (§5) are
  indistinguishable (KS p > 0.20). We prefer (A); (C) is the documented escape hatch, not the plan.

**Commitment:** build (A)+(B); ship a `tests/test_beta0_equivalence.py` that asserts byte-identical
K-chain path lists between S′(β=0) and the stepwise `iid_bok` over a grid of instances and seeds.
The chain-mode `best_of_k` in `decode_core` remains available as a *sanity* cross-check (coverage
should be statistically equal), but it is **not** the byte-identity target — that would require
touching the frozen seeding, which we will not do.

---

## 3. Iso-token accounting (V3 — the binding fairness constraint)

This line has previously lost on exactly this point (the capstone memory: λ=0 trellis beats SC on
candidate-parity but **loses** to token-matched best-of-K). So the comparison is **at matched
generated-token budget**, period. `tokens` is the headline axis (PREREG §7).

### 3.1 What a token is, and who pays

`decode_core.Budget` defines `tokens = Σ over every returned candidate text of len(text.split())`
(see `Budget.record_sample`). We adopt this **identically** so cross-arm accounting matches the
frozen core. Three counters per arm: `tokens` (headline), `forwards`/`sample_calls`, and `exec`
(`apply`+`solvable` during *evaluation only*; the verifier is **not** in S′'s method, PREREG §5).

- **`iid_bok` (chain-mode reference)** pays, per instance, `Σ_chains len(chain_text.split())`. A
  chain text is the `;`-joined op sequence; one depth-`d` chain costs `≈ 3d + (d−1)` tokens
  (`3` space-delimited tokens per op, plus the `;` separators absorbed by `split()` — concretely
  `len("3 + 7;8 * 9".split()) = 6`, i.e. `3·#ops`). So a K-chain best-of-K over depth `d` costs
  `≈ K · 3d` tokens.
- **`corr(β)` (step-mode)** pays per **step**: at every node of every chain it draws `N`
  single-op candidates to reconstruct `P_emission`, each `≈ 3` tokens. One chain of depth `d`
  therefore costs `≈ Σ_{t=1..d} 3N = 3Nd` tokens; K chains cost `≈ K · 3Nd`.

**The overhead is the factor `N`.** Step-mode S′ is `~N×` more expensive per realized chain than
chain-mode best-of-K, because each *committed* step is backed by `N` *sampled* candidates that are
mostly discarded (only one canonical successor is kept). This is the "逐步互斥需要 step-mode 采样
…比 chain-mode 的 best-of-K 贵" tax flagged in PREREG §7. It is **real generated text** and it
**counts**.

### 3.2 Two honest ways to match the budget

Per PREREG §7, both routes charge the tax explicitly; we build the **hard** one as primary:

- **(MAIN, mode="step", hard test).** Fix a token budget grid `B`. For each `B`: run `corr(β)` to
  produce as many repelled chains as the budget allows (so K_corr ≈ `B / (3Nd)`), and run
  `iid_bok` to produce as many i.i.d. chains as the **same** `B` allows (K_iid ≈ `B / (3d)`, i.e.
  `~N×` **more** chains). Compare coverage **at equal B**. This is the binding test: S′ must beat
  i.i.d. *even though i.i.d. gets ~N× more chains for the same tokens*. The reweight must buy back
  the step-mode tax through better-than-i.i.d. coverage per chain. If it cannot, H1 fails honestly.
- **(BACKUP, prompt-level conditioning).** Draw cheaply in chain-mode but append to the prompt a
  short "已走过 canonical 前缀,请走不同解法" conditioning string. Generated-continuation tokens
  are unchanged vs `iid_bok`; only cheap prompt tokens grow. We still **count prompt tokens** in a
  separate ledger and report both, but the headline `tokens` (generated continuation) is matched by
  construction. This route is cheaper but depends on the model obeying; it is the weaker evidence.

### 3.3 Tiers and what each settles (PREREG §6, §4 last bullet)

- **Tier A (CPU mock).** Use an **explicit ensemble**: exact `P_emission` is known (synthetic
  generator with knobs `p` = good-step prob and a mode-collapse knob), so we can do *exact* i.i.d.
  vs *exact* repulsion **mechanism-cost-free** — Tier A proves the *principle* (can favorable
  correlation push the coverage base数 up at all) and **freezes the prereg thresholds**, separate
  from the step-mode token tax.
- **Tier B (GPU, decisive).** Real `sample` (frozen Qwen3-4B + `adapter_decoupled_s0`), deeper
  Countdown head-cover-room instances, **same harness, same token ledger**. Tier B answers whether
  a *real, cheap* mechanism cashes the gain **at iso-token** (MAIN route).

**Commitments:** `tokens` per `Budget` semantics is the only fairness axis for the headline; per-B
paired comparison on identical instances; `corr`'s full step-mode overhead is inside its ledger;
an `exec-parity guard` asserts the evaluation `is_goal`/`apply` path is identical across arms
(V4); Tier A freezes thresholds before any Tier B run.

---

## 4. Why β yields an inverted-U, and the link to k≈ℓ²

### 4.1 The two competing bad structures

Coverage = `P(at least one of K chains hits the goal) = 1 − P(all K miss)`. Maximizing coverage
at fixed K (or fixed tokens) means minimizing the joint-miss probability `P(all miss)`. Two
failure modes bracket the optimum, exactly mirroring the red-clique / blue-clique pair in the
sphere-graph union bound (PREREG §1, §2 table last row):

- **β small ⇒ collapse (redundancy / "red clique").** With weak repulsion the K chains are
  ≈ i.i.d. and pile onto the same high-probability canonical states (LLM samples are notoriously
  redundant). Many of the K samples are *duplicates in canonical space*, so the **effective**
  number of independent goal-attempts is far below K. `P(all miss) → (1−c)^{K_eff}` with
  `K_eff ≪ K`: coverage is stuck at the Erdős–Rényi / i.i.d. ceiling. This is the loss the whole
  experiment is trying to break.
- **β large ⇒ diffusion ("blue clique").** With strong repulsion the reweight `e^{−β·n}`
  overwhelms `P_emission`: a state visited even once is suppressed by `e^{−β}`, so later chains
  are **pushed off the high-mass region into low-`P_emission` (dead / near-dead) canonical
  states** from which the narrow goal is unreachable. Here chains are maximally *distinct* but
  individually *bad*: each chain's own success prob collapses, so `P(all miss) → 1` again — for the
  opposite reason.

Because one bad structure dominates as β→0 and the other as β→∞, and both raise `P(all miss)`,
coverage as a function of β is **non-monotone with an interior maximum β\*>0** — the inverted-U of
H3. The optimum is the β that spreads chains across *distinct* canonical states **while keeping
each chain on the high-`P_emission` manifold**: maximal favorable negative correlation without
spilling into the dead region. This is exactly the union-bound sweet spot where the *sum* of the
two competing bad-event rates is minimized (PREREG §1: "ε = 两个底率之和 < 1 的严格余量").

### 4.2 Why a *single* multiplicative knob produces the U (sketch)

Let `q(β)` = per-chain success prob (decreasing in β: repulsion erodes each chain's own mass) and
`ρ(β)` = cross-chain redundancy / positive correlation among the K chains (also decreasing in β:
repulsion is anti-correlation). Under a simple negatively-correlated-Bernoulli surrogate,
`P(all miss) ≈ (1−q(β))^{K_eff(β)}` with `K_eff` increasing in (1−ρ). At β=0: `q` is maximal but
`K_eff` minimal (collapse). As β↑: `K_eff` rises (good) faster than `q` falls (the early gain),
so `P(all miss)` drops — coverage rises. Past β\*: `q` falls off a cliff as chains hit the dead
region (`e^{−β}` dominates a finite `P_emission` support), and the `K_eff` gain saturates (you
cannot exceed the number of *viable* distinct states), so `P(all miss)` rises again. The product
of a rising-then-saturating factor with a slowly-then-sharply-falling factor is single-peaked ⇒
inverted-U. `β\*` is where `d/dβ [K_eff·log(1−q)] = 0`.

### 4.3 The sphere-graph correspondence (k≈ℓ²)

In Ma–Shen–Xie the continuous knob is the **sphere dimension k**: `k→∞` ⇒ edges become
independent (no gain, = Erdős–Rényi = our β→0 collapse-to-i.i.d.); `k` too small ⇒ over-correlated,
the *competing* (blue) clique blows up (= our β→∞ diffusion into the dead region); the sweet spot
is **k≈ℓ²**. Our β is the **monotone-inverse image** of k:

| sphere graph | S′ |
|---|---|
| k → ∞ (independent edges, no gain) | β → 0 (independent chains, collapse to i.i.d.) |
| k too small (over-correlated, blue clique explodes) | β too large (over-repelled, chains diffuse to dead region) |
| sweet spot k ≈ ℓ² | sweet spot β\* (interior max of coverage) |

So **large k ↔ small β** and **small k ↔ large β**: both are *one* continuous correlation knob
interpolating between "independent" and "over-correlated," and both have an interior optimum where
the favorable correlation is maxed before the competing bad structure detonates. The experiment's
H3 is the LLM-decoding image of "the gain is an interior dimension/correlation sweet spot, not a
monotone benefit of more correlation." (We do **not** claim β\* equals any function of ℓ²
numerically; the correspondence is structural — same shape, same two-competing-bad-structures
cause — which is all the analogy licenses.)

---

## 5. Complexity, edge cases, joint-diversity instrument

### 5.1 Complexity

- **Per chain:** `d` steps; each step one `sample(..., N, ..., "step")` (cost `O(N)` candidates,
  `~3N` tokens), `O(N)` parse+apply, plus an `O(|support|)` reweight+categorical draw where
  `|support| ≤ N`. Per chain: `O(d·N)` time / tokens.
- **K chains:** `O(K·d·N)` time and tokens. The counter `n(S′,t)` is a dict keyed by
  `(canon, depth)`: `O(distinct canonical states × d)` space, `O(1)` amortized update per step.
- **β sweep:** `|β-grid|` independent passes; β\* located by the peak (no gradient needed). Tier A
  is exact (closed-form `P_emission`), so its `p×β` heat-map is cheap.

### 5.2 Edge cases (commit to explicit handling)

- **Parse failures.** Candidates with `parse_move == None` are **dropped before** building
  `P_emission` (eq. 1 normalizes over surviving candidates only) — identical to `savi`'s
  `if move is None: continue`. The dropped candidates' tokens **still count** in `Budget.tokens`
  (you generated them). If *all* N parse-fail at a step, the chain **dies** at that step (treated
  as a non-goal terminal; see dead states).
- **Terminal / dead states.** A state with no parsed feasible successor (Countdown: ≤1 value left,
  or `legal_ops` empty) ends the chain. `is_goal` is checked at the realized leaf only (verifier
  excluded from S′ by design, PREREG §5). A dead non-goal chain is a *miss* that consumed tokens —
  this is exactly the diffusion penalty at large β and must be charged.
- **Empty support after reweight.** `Z_i(S,t) = Σ P_emission·e^{−β·n}` is strictly positive
  whenever the support is non-empty (every factor > 0), so renormalization never divides by zero
  for β < ∞. At extreme β + heavily-visited support the weights underflow; compute the draw in
  **log-space** (`log P_em − β·n`, subtract the max, exponentiate) for numerical safety.
- **Ties.** When two canonical successors have equal reweighted mass, break ties by a deterministic
  key = `canon` (the same total-order discipline `decode_core._stable_order` uses), so the draw is
  reproducible. The categorical draw iterates the support **sorted by `canon`** before applying the
  inverse-CDF, so β=0 byte-identity (§2) holds.
- **β = 0 short-circuit.** Skip the reweight entirely (return `P_emission`) so the β=0 path is
  literally the i.i.d. code path — this is what makes the V2 byte-identity assertion robust.
- **Counter reset.** `n(·,·)` is reset to empty per instance; never leaks across instances.

### 5.3 Joint-diversity instrument (distinguishes correlation from marginal spread)

The headline danger (PREREG §2 last ¶, §8.3, H4) is mistaking **marginal temperature spread** for
**joint anti-correlation**. The instrument:

> **`distinct_canon@K`** = number of **distinct canonical states** realized across the K chains of
> one ensemble — measured both per-depth (`distinct_canon(t)` = `|{S : n(S,t) > 0}|`) and at the
> leaf (distinct canonical terminal states). Report the mean over instances and seeds.

This is a **joint** (ensemble-level) statistic: it counts *non-redundancy of the K committed
samples together*, which is exactly what coverage depends on, and exactly what mere marginal
entropy does **not** control (K i.i.d. high-temperature draws can be marginally broad yet jointly
pile onto the same few canonical states). Used three ways:

1. **β-axis (H3 corroboration):** `distinct_canon@K` should rise with β (more repulsion ⇒ more
   distinct), while *coverage* peaks then falls — confirming the U is "distinctness up, but viability
   down past β\*," not a monotone artifact.
2. **τ-match calibration for H4 (the key negative control).** Set the `temp_matched_iid` arm's
   temperature so its **marginal** per-step entropy equals `corr(β\*)`'s marginal entropy
   (measured from the realized step distributions). Then test: at matched marginal entropy and
   matched tokens, does `temp_matched_iid` reach `corr`'s `distinct_canon@K` and coverage? H4
   predicts **no** — the temperature arm spreads marginally but the K draws stay jointly redundant
   (≈ i.i.d.), so it under-covers. This is the operational separation of "相关" from "边际散开"
   and the direct re-adjudication of the old τ=1 null (PREREG §2: decoupled≈coupled because
   temperature gives marginal, not joint, spread).
3. **Sanity at β=0:** `distinct_canon@K` for `corr(0)` must match `iid_bok` (part of V2 / §2(C)).

**Commitment:** report a 3-arm table (`iid_bok`, `corr(β\*)`, `temp_matched_iid`) of
`(coverage@token-B, distinct_canon@K per-depth and leaf, marginal step-entropy)`; the experiment's
claim is corroborated **only if** `corr` is high on *both* coverage and joint distinctness while
`temp_matched_iid` matches on marginal entropy yet trails on joint distinctness and coverage.

---

## Build checklist (cross-ref deliverables, PREREG §13)

- `corr_sampler.py`: eq. (2) reweight in log-space; per-`(canon,depth)` chain counter; sequential
  K-chain draw with per-chain `seed_i = f(base_seed, inst, i)`; β=0 short-circuit; deterministic
  `canon`-sorted inverse-CDF draw.
- `tests/test_beta0_equivalence.py`: byte-identical K-chain path lists, `corr(β=0)` vs stepwise
  `iid_bok`, over an instance×seed grid (V2 route A+B); chain-mode `best_of_k` as statistical
  sanity only.
- `tests/test_iso_token.py`: per-B token ledger matched across arms within tolerance; `corr`'s
  step-mode `N×` overhead present in its `Budget.tokens` (V3).
- `instruments.py`: coverage-vs-token + miss-slope fit (H2); β inverted-U sweep (H3);
  `distinct_canon@K` joint-diversity (per-depth + leaf); τ-match by marginal entropy (H4); paired
  bootstrap n=10000 with Holm across H1/H2/H4.
- `mock_backend.py`: exact `P_emission` with knobs `p` (good-step) × mode-collapse for Tier A
  exact i.i.d./repulsion and the `p×β` heat-map (H5).
