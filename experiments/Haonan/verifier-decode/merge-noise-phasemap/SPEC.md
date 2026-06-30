# SPEC / PREREG — Merge × Verifier-Noise phase map

**Status: DRAFT (freeze before reading any biased-value cell).** One-substrate, CPU-only
experiment that decouples the two variables prior work confounded and decides whether the
"verifier-weighted global decode" line is alive for *realistic* (learned, imperfect)
verifiers, or dead for good.

Branch: `line/verifier-decode`. Lives under `verifier-decode/merge-noise-phasemap/`.
Reuses the FROZEN `decode_core.savi`, the mechanism-recombination lattice substrate
(`../mechanism-recombination/`), and the joint-lambda value/noise instruments
(`../joint-lambda-decode/`).

---

## PIVOT (2026-06-29, post-Phase-0) — supersedes the merge-axis design below

Phase 0 (V2/V3 controls + a competence probe, all on this substrate, n≤192) **falsified the
merge axis** and found the real one. Operative design is now the **`p × ε` plane**; the
merge sections below are retained for the record.

### What Phase 0 established (the reframe)
1. **Merge `g`/R is pass@1-inert for `savi`.** `D1(savi_freq)` is ~identical at `g=0`
   (merge, realized R≈2.1) and `g=∞` (no merge, R=1.0): +0.30/+0.39 vs +0.29/+0.40 at
   depth 16/24. `savi` collects goals over the full pre-prune layer, so freeing beam slots
   doesn't change pass@1 (the `H4=0` mechanism from joint-lambda). The original primary
   knob is dead; demote `g` to a logged secondary.
2. **The lever is the graded freq edge, not coverage.** `A2` (support edge = pure beam
   coverage, no graded signal) LOSES badly at every `g` and `p` (−0.36…−0.88). The per-step
   beam wins only when it has a graded signal to rank with.
3. **The decisive axis is emission competence `p`** (how well sampling frequency tracks
   correctness; `p=0.5` ⇒ frequent⊥good = the "greedy-associative"/hard regime, `p=0.9` ⇒
   frequent≈good = easy/Countdown-like). The kill test **flips** on it. Depth-24,
   best-λ soft (preliminary n=64):

   | regime | A1 freq (no verifier) | A3 hard mask fn=.18 | **A4 soft-λ fn=.18** | A1x exact |
   |---|---|---|---|---|
   | p=0.5 (hard) | −0.14 (loses) | +0.38 | **+0.58 [+0.45,+0.70]** | +0.55 |
   | p=0.7 | +0.44 | +0.23 | **+0.44 [+0.31,+0.56]** | +0.38 |
   | p=0.9 (easy) | +0.02 | −0.12 (hurts) | **+0.00 (harmless)** | +0.00 |

   **At low `p` the realistic learned value (fn=0.18 — the T1.4 value that died −0.58 on
   Countdown) used as a soft λ-edge wins +0.44…+0.58, matching/beating even the exact mask,
   and is harmless at high `p`.** The earlier "decoder framing dead" verdict was a
   high-competence-regime artifact (Countdown's emission concentrated on feasible moves ≈
   high `p`). fn-tolerance comes from **emission diversity** (low `p` surfaces many solution
   paths so the verifier's false-negatives are covered), not canonical merge.

### Operative design (p × ε plane)
- **Substrate / generator / decoder:** unchanged (§2.1; exact `MergeLatticeDomain` for the
  generator, `NoisyMergeLatticeDomain` for the decode verifier, two-handle wiring §2.4).
- **Primary axis 1 — competence `p` ∈ {0.5, 0.6, 0.7, 0.8, 0.9}** (the generator's
  good-move weight; `gen_fair`). This is the "frequency⊥correctness" / greedy-associative
  axis.
- **Primary axis 2 — verifier corruption `ε` ∈ {0, 0.05, 0.10, 0.18, 0.30} × mode{fn,sym}**,
  ρ=1 (deterministic state-function verifier; §2.3). `ε=0.18 fn` = the T1.4 operating point.
- **Headline arm: A4 `savi_value`** (soft λ-edge over the corrupted value), **λ ∈ {0.1,
  0.25, 0.5, 1, 2}** (extended down per Phase-0: best λ was the smallest swept). Anchors:
  A1 (freq, no verifier), A1x (exact mask), A3 (hard corrupted mask), A2 (support floor),
  A5 (no-merge ablation). Selection A0 matched per-arm on exec.
- **Secondary (logged):** `g ∈ {0,1,2,∞}` at one `(p,depth)` to record that merge stays
  pass@1-inert on the `p`-pivoted plane (report realized merge_R; expect D1 flat in `g`).
- **Depths {16, 24}; n=24; seeds 0–7; paired bootstrap n=10000.** All CPU.

### Restated hypotheses + kill criterion (FROZEN on the p × ε plane)
- **P1 — competence flip.** `D1(A1 freq)` decreases as `p` falls (frequency stops tracking
  correctness): > 0 at high `p`, ≤ 0 by `p=0.5` at depth. The freq beam is *not* a
  general decoder; it rides emission competence.
- **P2 — THE decisive test (soft realistic value).** At `ε=0.18 fn, ρ=1`, is there a
  competence band where `D1(A4 soft-λ) > 0` (CI excl 0)?
  - *Prediction (Phase-0-supported):* **Yes at low/mid `p`** (≤ ~0.7), where the verifier is
    load-bearing; **≈0/≤0 at high `p`**, where emission already carries it.
  - **KILL CRITERION (pre-committed):** the decoder framing is **alive** iff `D1(A4 soft,
    ε=0.18 fn) > 0` (CI excl 0) for some `p ≤ 0.7` at depth 24 **and** A4 ≥ A3 there (soft no
    worse than hard). If A4 ≤ 0 across all `p` at ε=0.18 (or only the *exact* mask ever
    wins), the realistic-verifier framing is **dead** — stop and write the negative.
    *Map the `p×ε` boundary either way.*
- **P3 — soft ≥ hard, and harmless when redundant.** A4 (soft) ≥ A3 (hard) at every `(p,ε)`
  (down-weight beats delete under false-negatives), and A4 does not *hurt* relative to
  selection at high `p` (where A3 does). The ε-tolerance widens with depth (more diversity).
- **Validity (Phase-0, passed):** V3 g=0 reproduces RESULTS2 (+0.089/+0.281/+0.437); V2 g=∞
  gives merge_R=1.0 with D1(A1) unchanged (merge inert); V4/V5 unit + A0-invariance tests
  pass (`tests/test_phasemap.py`, 11/11). New p-axis control: at ε=0, A3==A1x and A4(λ→ )
  tracks the exact-mask ceiling.

---

## 1. The question, and the two results it reconciles  *(original merge-axis framing — for the record)*

Two prior lines, same program, opposite verdicts — *because they sat at opposite ends of a
hidden axis*:

- **joint-lambda (Countdown, PLAN3+4):** the entire decode win is a *near-exact per-step
  mask*; global trellis adds nothing (`savi − masked_bom = 0`); a good learned value
  (acc .91/AUC .97/fn .18) used soft-λ **loses to selection by −0.58 at every λ**; hard
  learned mask = 0.00. Decoder framing declared dead. *Measured cause:* the feasible
  lattice is near-degenerate — `keff_feasible ≈ 1.96` at k=6 (≈ a single chain), so one
  false-negative prunes the only path and there is nothing to merge (`H4 = 0`).

- **mechanism-recombination (synthetic Viterbi lattice, RESULTS2):** the *frequency* edge
  (`count/N`, NOT an exact oracle) **beats** iso-exec selection, margin rising with depth
  (D8 +0.089, D16 +0.281, D24 +0.437). *But* the substrate has heavy position×time
  merging and an oracle-defined competence generator; merge (`D_merge`) is a real positive
  share.

The hidden axis is **state merging / path redundancy R**. Countdown sat at R≈1 (chain);
the lattice sat at high R. Neither line varied R while holding everything else fixed, and
neither tested a *realistic biased* value on a *high-R* substrate. This experiment does
exactly that: it sweeps R as a controlled knob and crosses it with verifier corruption.

**Decisive question.** Holding the emission (competence `p`, sample count `N`) fixed, does
increasing merge/redundancy R let an *imperfect* per-step verifier beat iso-exec selection
— and specifically, does R rescue a **deterministic** learned value (the realistic case),
or only a **frequency / sample-averaged** signal?

**Mechanistic prior (to be tested, not assumed).** Merge's lever is **beam coverage**, not
count-pooling (the frozen `savi` keeps the higher-scoring history at a merged canon; it does
*not* sum counts across parents). Collapsing duplicate paths to a shared canonical node
frees beam slots, so a width-K beam covers ~R× more *distinct* states (R = paths/node) — a
benefit that grows with R and depth and vanishes as K→∞ (the RESULTS2 `D_merge` pattern).
The freq edge (A1) is an **exact-competence emission statistic** (`count/N` under the
generator's exact good/bad classification), so it is already reliable; merge amplifies it via
coverage. The kill test is whether a **corrupted** verifier survives: for a hard mask at
ρ=1, a false-negative on a canon prunes it **uniformly across all parents** (same
`solvable(canon)`), so coverage cannot rescue it; at ρ=0, independent per-evaluation draws
give redundant parents independent chances to pass, so coverage **can**. A soft value only
*down-weights* (not deletes) a wrongly-low-V correct node, so whether the wider effective
beam keeps it or the extra distinct competitors crowd it out is genuinely uncertain — hence
measured. Prediction (to test): coverage rescues a corrupted verifier only when its error
**decorrelates** across redundant paths (ρ low); the realistic learned value (ρ=1,
deterministic in the state) is not rescued. If confirmed, the decode win reduces to
exact-competence emission + coverage — not "a verifier-weighted decoder is a better
algorithm" — and the framing is dead for realistic verifiers, not just on Countdown.

---

## 2. Frozen design

### 2.1 Substrate
Integer-sum Viterbi lattice (`../mechanism-recombination/domain_lattice.py`): state
`(s, r)` carrying target `T`, moves `add v, v∈{1,2,3}`, goal `s==T ∧ r==0`, **exact O(1)
oracle** `solvable = (r ≤ T−s ≤ 3r)`. Chosen because the oracle is cheap at *unbounded*
depth (Countdown's `reachable` caps depth ~5) — the only place R, depth, and verifier
noise can all be swept independently. Instances: `make_lattice_instances(depth, n, seed)`,
all solvable by construction (ceiling = 1.0). Generator: the competence-`p` fair generator
`make_fair_generator` (`gen_fair.py`), `p = 0.7`, `N = 16`, `τ = 1.0` — **identical to
RESULTS2's authoritative config**, consumed identically by every arm (no oracle injection
into any decoder; `solvable` enters emission only to classify move competence).

### 2.2 Knob 1 — merge granularity `g` → realized redundancy R  *(the new axis)*
Add an optional history tag to the lattice state and a granularity `g` controlling how much
of it `canon` retains (new `MergeLatticeDomain(g)`, §5):

- `apply` sets `child.tag = (state.tag + (v,))` truncated to the last `g` moves
  (`g=0` ⇒ tag always `()`; `g=∞` ⇒ full history).
- `canon = (s, r, T, tag)`.
  - **`g=0`** → `canon=(s,r,T)` → **maximum merge** (current lattice; high R).
  - **`g=∞`** (≥ D) → every path distinct → **tree, R≈1** (reproduces the Countdown chain).
  - **`g=1,2`** → intermediate merge.

`g` is provably **emission-invariant**: enumeration dedups successors by canon, but the
three moves always yield distinct sums (distinct successors) so dedup never fires at a node
regardless of `g`; and competence weights depend on `solvable(successor)` ⇒ on `(s,r,T)`
only, never on `tag`. Therefore `g` changes **only** what the decoder merges, holding the
sampled distribution fixed. (The emission *distribution* is fixed; realized interior-node
*samples* still vary with `g`, because the generator's RNG is canon-keyed — a merged node is
sampled once and reused, an unmerged one re-samples per path. That is the faithful
computational meaning of merge-vs-no-merge, not a confound; the clean invariance control is
on the competence *weights* and on root-seeded selection, see V4.) The IV is `g`; report the
**realized** `merge_ratio` (already in `Result.trellis_widths_before/after`) and `K_eff` as
the measured mediator R.

Levels: **`g ∈ {0, 1, 2, ∞}`**.

### 2.3 Knob 2 — verifier corruption (ε, mode, ρ)  *(decoder-only)*
The verifier that the *decoder* applies is a corrupted copy of `solvable`
(`NoisyMergeLatticeDomain(g, eps, mode, rho, noise_seed)`, §5), corruption rate `ε`, mode
∈ {`fn` (prune a solvable node — the killer per T1.2), `fp` (admit a dead node — tolerant
control), `sym`}, and noise correlation **ρ**:

- **ρ=1 (deterministic, the realistic learned-value case):** corruption is a stable
  function of `canon(state)` via sha256 ⇒ same node, same (wrong) label on every path. This
  is what a learned `V` does. *Tier 1.*
- **ρ=0 (iid, stochastic):** corruption re-drawn per candidate evaluation ⇒ averages under
  merge. Requires the per-candidate verifier arm (§5). *Tier 2, explanatory.*

Levels: **`ε ∈ {0, 0.05, 0.10, 0.18, 0.30}`**, **mode ∈ {fn, sym}** (+ `fp` at the
realistic point only), **ρ ∈ {1 (Tier 1), 0 (Tier 2)}**. `ε=0.18 fn` is the **T1.4
operating point** (the learned value that lost on Countdown) — the cell that decides the
line.

### 2.4 The two oracle handles  *(the wiring that must not be confused)*
`solvable` is used in **two** places; corrupting the wrong one silently confounds the map:

1. **Emission competence** (`gen_fair._classify_weights`) — sets coverage. **Stays EXACT.**
2. **Decode verifier** (`savi`/`beam_no_merge`'s `if verifier: solvable(sp)`) — **this** is
   what ε corrupts.

⇒ **generator** is built on `MergeLatticeDomain(g)` (exact); **decode arms** run with
`NoisyMergeLatticeDomain(g, ε, …)` (same `canon`/`apply`/`is_goal`, noisy `solvable` only);
**selection** runs on the exact domain (it never queries the decoder verifier). A regression
test (V5) asserts corrupting the decoder verifier does **not** move selection/emission.

### 2.5 Arms (all reuse FROZEN `decode_core.savi`; ⊕ = one new arm to add)
| id | arm | edge | decoder verifier | role |
|----|-----|------|------------------|------|
| A0 | `best_of_k_isobudget(axis="exec")` | — | none (exact final `is_goal`) | **selection baseline** |
| A1 | `savi(edge="freq", verifier=False)` | `log(cnt/N)` | none | **RESULTS2 headline = freq + coverage (V3 positive control)** |
| A1x | `savi(edge="freq", verifier=True)` exact | `log(cnt/N)` | **exact** `solvable` | exact-mask ceiling (the ε=0 anchor of A3) |
| A2 | `savi(edge="support", verifier=False)` | `0` | none | pure-coverage floor (no edge signal) |
| A3 | `savi(edge="freq", verifier=True)` on noisy domain | `log(cnt/N)` | **ε-noisy** hard mask, ρ | realistic learned *hard mask* (ε=0 → A1x) |
| A4 ⊕ | `savi_value(value_fn=noisy.value, λ)` | `log(cnt/N) + λ·log V_noisy` | **ε-noisy** soft value, ρ | realistic learned *soft value* (λ=0 → A1) — most likely to survive |
| A5 | `beam_no_merge(edge="freq", verifier=False)` | `log(cnt/N)` | none | **Φ/merge ablation** of A1 (A1 − A5 = `D_merge`, isolates R) |

A4 ports `../joint-lambda-decode/arms_local.py::savi_value` (soft λ-edge) to the lattice
(now `arms_phasemap.py`); λ ∈ {0.5, 1, 2, 4, 8} (report the best per cell, as T1.4 did). A5
already exists (`arms_ext.beam_no_merge`). The RESULTS2 headline (+0.089/+0.281/+0.437) is
**A1 = `savi(freq, verifier=False)`** — emission frequency + merge coverage, **no mask**;
the exact mask (A1x) wins by more (`D_verif`). The kill test corrupts the *verifier* (A3/A4),
never A1's emission edge.

### 2.6 Iso-compute axis, headroom, statistics
- **Axis = exec** (move-applies; the abstract lattice has no tokens). `best_of_k_isobudget`
  target = the **exec budget of the specific savi arm it is differenced against** (per-arm
  iso-exec; verifier queries cost `exec += 1`, so each savi arm carries its own budget). The
  exec-parity fix from RESULTS2 (`budget.exec` counted for every rollout, not stopped at
  first win) is inherited verbatim.
- **Headroom stratum:** solvable ∧ greedy-myopic-proxy fails (the stratum where selection
  has room — deltas read here only). `n = 24` instances/depth, `8` seeds.
- **Primary metric** `D1(arm) = pass@1_headroom(arm) − pass@1_headroom(A0)`, **paired
  bootstrap n_boot = 10000**; a result "holds" iff its CI excludes 0 in the stated
  direction. Holm across the arm family per (g, D).

### 2.7 Grid
Tier 1 (ρ=1): `g{0,1,2,∞} × D{8,16,24} × {A0, A1, A2, A5} ∪ {A3,A4 × ε{0,.05,.10,.18,.30} × mode{fn,sym}}`,
× 8 seeds × 24 inst. Tier 2 (ρ=0): the realistic point `ε{.05,.10,.18} fn × g{0,2,∞} × D{16,24}`,
A4 only, + an `N ∈ {8,16,32}` sweep at `g=0` (does pooling = just more samples?). All CPU.

---

## 3. Hypotheses, thresholds, and the kill criterion (FROZEN)

- **P1 — freq mechanism (expected positive).** `D1(A1)` (= `savi(freq, verifier=False)` −
  selection) increases with realized R at each depth and with depth at high R. Bar: at
  **g=0**, `D1(A1)@D24 ≥ +0.30` (reproduce RESULTS2 +0.437, CI excl 0); at **g=∞**,
  `D1(A1) ∈ [−0.05, +0.05]` (collapse to selection as merge → 1).

- **P2 — THE decisive kill test (biased value).** At the realistic point
  **ε=0.18 fn, ρ=1**: does any `g` give `D1(A4 soft-λ) > 0`, CI excl 0?
  - *Prediction (to test):* **No** — a deterministic per-node error is not averaged by
    merge; `D1(A4) ≤ 0` for ε ≥ 0.10 fn at **every** `g`, while `D1(A1 freq)` stays
    positive at low `g`. (A3 hard mask ≤ A4, per T1.4 soft > hard.)
  - **KILL CRITERION (pre-committed):** *if* `D1(A4) ≤ 0` (or n.s.) for ε ≥ 0.10 fn at
    every `g` up to full merge, the verifier-weighted **decode** line is **dead
    universally** — redundancy does not rescue a realistic learned value — **stop the
    domain hunt and write the negative.** *Conversely*, if ∃ `g` with `D1(A4) > 0` (CI excl
    0) at ε=0.18, the line is **alive**: record the realized **R\*** (merge_ratio at that
    `g`) as the admission threshold and proceed to screen real domains for R ≥ R\*.

- **P3 — mechanism isolation (Tier 2).** The ρ sweep shows coverage (R) rescues a corrupted
  verifier only at **low ρ** (decorrelated noise), not at ρ=1 (the realistic learned-value
  case, where every path through a canon reads the same wrong label). This pins the honest
  reframing: the decode win is *exact-competence emission + beam coverage*, not
  "verifier-weighted decode is a better algorithm." Corroborating `N`-sweep at fixed `g`: if
  `D1(A1)` rises with `N`, part of the win is sample count, partly recoverable by giving
  selection more samples.

---

## 4. Validity gates (instrument valid iff all hold)
- **V1** headroom non-empty at every D.
- **V2 — negative control / R-axis is real.** At `g=∞`: `D1(A1) ≈ 0` (CI ∋ 0) **and**
  `merge_ratio ≈ 1.0`. Proves the substrate reproduces the no-merge (Countdown) degeneracy
  and that `g` genuinely controls R.
- **V3 — positive control.** At `g=0`: `D1(A1)` (`savi(freq, verifier=False)` − selection)
  reproduces RESULTS2 (D8 +0.089 / D16 +0.281 / D24 +0.437) within CI; and `D1(A1x)` (exact
  mask) reproduces the larger `D_verif`. Fails ⇒ harness regressed; stop.
- **V4 — EMISSION INVARIANCE (critical).** (a) *unit:* `gen_fair._classify_weights` returns
  identical (texts, weights) for a fixed `(s,r,T)` under `MergeLatticeDomain(g)` for every
  `g` — the emission *distribution* is g-invariant by construction. (b) *integration:*
  `pass@1(A0)` (selection) is g-invariant within MC noise (its chains are root-seeded; the
  root tag is empty for every `g`). NOT asserted on A1/A2: their interior-node realized
  samples legitimately vary with `g` (merge ⇒ sample-once-reuse vs no-merge ⇒ resample
  per path), which is the mechanism, not a leak. A leak would show as the *weights* moving
  with `g` (caught by (a)) or A0 moving with `g` (caught by (b)).
- **V5 — oracle-handle separation.** Corrupting the decoder verifier (ε: 0 → 0.30) leaves
  `pass@1(A0)` and `pass@1(A2)` bit-identical (selection/emission never read the decoder
  verifier). Asserted as a test.

---

## 5. Implementation map (reuse / add / freeze)

**Freeze:** `decode_core/` (`savi`, `Budget`, `Node`, `Result`) — byte-identical, never
edited. PREREG frozen before any A3/A4 cell is read.

**Reuse as-is:** `gen_fair.make_fair_generator`, `domain_lattice.make_lattice_instances`,
`arms_ext.best_of_k_isobudget` (A0) and `arms_ext.beam_no_merge` (A5),
`instruments.{keff, merge_ratio, ece, paired_bootstrap}`.

**Add (all new code in `merge-noise-phasemap/`):**
1. `domain_merge.py`
   - `MergeLatticeDomain(LatticeDomain)`: `__init__(self, g)`; extend `LatticeState` with
     `tag: tuple = ()`; override `apply` (update+truncate tag to last `g`), `canon`
     (`(s,r,T,tag)`), `initial_state` (tag `()`). Exact `solvable` inherited.
   - `NoisyMergeLatticeDomain(MergeLatticeDomain)`: `__init__(self, g, eps, mode, rho,
     noise_seed)`; override **`solvable` only** — ρ=1: corrupt iff
     `sha256(canon, mode, noise_seed) < eps` per `mode`; ρ=0 handled in the arm (below).
2. `arms_ext.py` (append) — `savi_value(domain_decode, sample, inst, K, N, tau, seed,
   value_fn, lam, max_depth, edge_mode="freq")`: mirror `decode_core.savi`'s layer loop but
   set `edge = log(cnt/N) + lam·log(value_fn(sp))`; `value_fn` reads `domain_decode`'s
   (possibly noisy) `solvable` → V∈{ε,1−ε}-style probability, or a per-candidate-nonce noisy
   value for ρ=0. Account `exec += 1` per verifier query (iso-exec parity). Add a
   `savi_value == savi(freq,verifier=True)`-equivalence assertion when `lam→∞` & exact.
3. `run_phasemap.py` — grid runner over (`g`, D, arm, ε, mode, ρ, λ); **wires generator ←
   exact `MergeLatticeDomain(g)`, decoder ← `NoisyMergeLatticeDomain(g,ε,…)`**; headroom
   stratification; per-arm iso-exec target; paired bootstrap; merge_ratio/K_eff logging;
   writes `outputs/phasemap/<cell>.json`. Flags: `--gs --depths --arms --eps --modes --rho
   --lams --n --seeds --out`. Shard by (`g`, D), one cell per process.

**Phase order:** (0) **Tier-A validation** — run the *full grid on the mock at ε=0* to pass
V1–V5 and **freeze this SPEC** (machinery only, not the science). (1) Tier 1 ρ=1 — the
decisive P1/P2 map. (2) Tier 2 ρ=0 + N-sweep — P3, only if Tier 1 is ambiguous or alive.

---

## 6. Runner commands (shape)
```
# V3 positive control (must reproduce RESULTS2 before anything else)
python run_phasemap.py --gs 0 --depths 8,16,24 --arms A0,A1 --eps 0 --rho 1 \
  --n 24 --seeds 0-7 --out outputs/phasemap/v3_g0.json
# V2 negative control
python run_phasemap.py --gs inf --depths 8,16,24 --arms A0,A1,A2 --eps 0 --rho 1 \
  --n 24 --seeds 0-7 --out outputs/phasemap/v2_ginf.json
# Tier 1 decisive cell (the kill test), sharded per (g,D)
python run_phasemap.py --gs 0,1,2,inf --depths 16,24 --arms A0,A3,A4 \
  --eps 0,0.05,0.10,0.18,0.30 --modes fn,sym --rho 1 --lams 0.5,1,2,4,8 \
  --n 24 --seeds 0-7 --out outputs/phasemap/t1_<g>_<D>.json
```

## 7. Tests (`tests/test_phasemap.py`)
- **V4** emission invariance: `pass@1(A0)` over `g∈{0,1,2,∞}` equal within MC noise (fixed seeds).
- **V5** oracle separation: `A0`/`A2` identical at ε∈{0,0.3}.
- **R↔g monotone:** `merge_ratio(g=0) > merge_ratio(g=2) > merge_ratio(g=∞)≈1`.
- **ρ=1 determinism:** same (g,ε,mode,seed) ⇒ bit-identical `solvable` labels and pass@1.
- **savi_value equivalence:** `λ→∞` exact ⇒ `savi_value == savi(freq,verifier=True)`.
- **g=0 ≡ current lattice:** `MergeLatticeDomain(0).canon == LatticeDomain.canon` (modulo tag).

## 8. Compute
All CPU, O(1) oracle. Tier 1 ≈ `4 g × 3 D × (4 base + 2 arms × 5 ε × 2 mode) ≈ 288 cells`,
each `8 seeds × 24 inst` of cheap small-K decodes — shardable to well under an hour.

## 9. What Tier A cannot pre-judge / caveats
- Tier A (mock, ε=0) validates plumbing + V1–V5; it is **not** the science (P1/P2 read on
  Tier 1). - The lattice is Viterbi's home turf with maximal structural merge; a **real**
  domain's R depends on whether a *semantic canonicalization* makes distinct reasoning paths
  reconverge (math/code plausibly; free prose not). R\* from this map is a **necessary**
  admission threshold for a real task, not a sufficient guarantee. - ρ=1 models a learned
  value as a deterministic function of the canonical state; a real learned value may carry
  *some* decorrelated (feature-noise) component — Tier 2's ρ sweep brackets that. - This map
  decides the *decode-vs-selection* question on a controlled substrate; the real-LM
  end-to-end number stays gated on the emission line.
