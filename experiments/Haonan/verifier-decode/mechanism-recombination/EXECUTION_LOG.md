# Recombination mechanism check — execution log (subagent-driven, ultracode workflow)

Fair, competence-parameterized generator `P_p(move|state)` consumed identically by every arm.
Question: at equal compute, does global decode + Φ-merge recover headroom that compute-matched
selection misses, and where in p-space. **Semi-synthetic mechanism check — NOT the real-LM
result.** decode_core stays FROZEN (Countdown parity preserved); the study imports it read-only.

## Phase 1 — Build (gen_fair.py, arms_ext.py, run_mechanism.py + tests) ✅
**Result:** three modules implemented TDD; pytest green; tiny Countdown smoke ran and printed numbers.
**Evidence:**
- `gen_fair.py` — domain-agnostic `make_fair_generator(domain, enumerate_moves, p, depth_cap)` returning the `sample(state,N,tau,seed,mode)` closure decode_core calls. `P_p`: enumerate legal move TEXTS via the frozen `mock_decoupled` step sampler (dedup by successor canon), classify each by `domain.solvable(s')` (good→weight p, bad→1−p, all-good/all-bad→uniform). `mode="step"`→N iid draws; `mode="chain"`→autoregressive rollouts of P_p (NO oracle injection); τ=0→argmax-weight. Seeding = sha256(seed, canon, p, mode, N, temp), process-stable (no Python hash()).
- `arms_ext.py` — `best_of_k_isobudget` (draw rollouts until cumulative candidate count ≥ the trellis's per-instance candidate budget) and `beam_no_merge` (savi loop, nodes keyed per-path, no canonical collapse; `_stable_topk_seq` keyed (-score,canon,seq)). Reuses decode_core Budget/Node/Result/_stable_topk/_backtrack.
- Countdown adapter replicated from `decode_core/tests/_countdown_adapter.py` (reachable=solvable, solve_one, legal_ops); AlgebraDomain used directly.
- Tests: generator determinism + in-domain + p→1/p→0 limits; arms_ext equivalence (beam_no_merge==savi where canons never coincide; tests/test_arms_ext.py 5 passed); harness smoke finite + deterministic.

## Phase 2 — Run ✅
**Countdown full sweep** (primary, ~0.02s/savi call):
`--domain countdown --set builtin_small --p-grid 0.3,0.5,0.6,0.7,0.8,0.9,1.0 --seeds 1..5 --K 8 --N 16 --max-depth 6 --tau 0.7`. 8 arms. 28 instances (16 headroom + 8 control + 4 unsolvable). Output: `outputs/countdown_sweep/results.json`. iso_matched=true all p.
**Algebra confirm** (confirmatory, ~14s/savi call → tiny):
`--domain algebra --p-grid 0.6 --seeds 1 --K 6 --N 8 --max-depth 8` over the 8-headroom stratum, arms {best_of_k_isobudget, savi_support, beam_no_merge, ceiling}. Output: `outputs/algebra_confirm/results.json`. iso_matched=true.

## Phase 3 — Verify (parallel, adversarial) — all 3 PASS (ok=true) ✅
- **Equal-compute fairness audit → FAIR.** No oracle/solver injection (frozen `_sample_chain` path instrumented, reached 0× by any arm; 3100/3100 probe sampler calls were mode="step"). iso candidate-matched exactly per instance across all 980 (p,seed,instance) triples (diff identically 0; on tokens/exec iso gets ≥ savi). D1 reproduces by hand. verifier=False headline makes 0 solvable calls. CAVEAT (not a defect): candidate-parity axis is not semantically symmetric (chain candidate=full rollout vs savi candidate=one step) — the structural reason D1<0; tokens/moves parity would give the trellis *less* budget, not reverse sign.
- **Φ-merge ablation validity → PASS.** beam_no_merge mirrors savi statement-for-statement, one variable (canonical keying vs per-path). Equals savi where no merge fires (K=1 all 28; grid), diverges with strictly larger widths where it does (p0.9 K=8: 28/28). phi_attributed_wins is the tight definition (validated solves savi makes that beam_no_merge does not, same replay gate). beam_no_merge never compute-starved. merge_ratio ≥ 3.09 wherever D_merge>0.
- **Determinism → PASS.** Fresh cache-free re-run of cell p=0.7 seed=3 reproduced all 8 arms' pass@1, dp/phi wins, merge_ratio to full float precision, plus per-instance maps and budgets bit-identically. (`outputs/verify_repro/`)

No corrections to the numbers required; only reporting-honesty recommendations (state D1<0 plainly; document budget-axis asymmetry) — followed in RESULTS.md.

## Phase 4 — Synthesize ✅
Wrote RESULTS.md (labeled semi-synthetic), results.json (distilled), this log.

### VERDICT: NO COMPETENCE BAND.
- **D1 (savi_support − best_of_k_isobudget, headroom n=80, paired bootstrap)** NEGATIVE at every p<1.0, CIs exclude 0; exactly 0 only at p=1.0. p0.3 −0.263[−0.363,−0.163]; p0.5 −0.725[−0.825,−0.625]; p0.6 −0.763[−0.863,−0.650]; p0.7 −0.875[−0.938,−0.800]; p0.8 −0.775[−0.863,−0.688]; p0.9 −0.613[−0.713,−0.500]; p1.0 +0.000. Compute-matched selection beats the trellis everywhere. **Opposite of the hypothesis.**
- **D_merge (savi_support − beam_no_merge)** small but POSITIVE, CI excludes 0 at p∈{0.6 +0.050, 0.8 +0.050, 0.9 +0.075}; peak p0.9 +0.075[0.025,0.1375]. Merge fires (ratio ≈3.1–3.4) wherever positive — real but tiny; never co-occurs with D1>0. dp_only_wins ≈0 (Σ5: 1,0,2,0,0,0,0), phi_attributed_wins small (Σ5: 1,1,4,1,4,6,0).
- **D2 (savi_freq − savi_support)** strongly POSITIVE and rising: peak p0.8 +0.725[0.625,0.825]; graded edges carry real signal within the trellis family, but savi_freq only reaches parity (not above) best_of_k_isobudget at the top.
- **Controls all HOLD:** C1 p1.0 best_of_k=savi=1.0 dp=0; C2 p0.3 all~0 (savi 0.0125, dp 0.2/cell); C3 4 unsolvable, 0 solved by any arm; C4 verifier_on ≥ verifier_off all p (p0.6 0.8875 vs 0.150).
- **Algebra confirm:** NULL cell (all arms 0.0) → D1=+0.0, D_merge=+0.0; no band. Φ-merge IS active on real CAS Φ (merge_ratio 8.0, 72→9). Fairness + determinism pass.

## Phase 5 — POST-HOC AXIS CORRECTION (main session, direct probe) ⚠ supersedes the headline
The Phase-3 fairness caveat was load-bearing and its conclusion was WRONG. Candidate parity prices
one full multi-step rollout the same as one single decoder step → hands selection ~depth× more
complete attempts. The claim "tokens/moves parity would give the trellis *less* budget, not reverse
the sign" is backwards: exec/moves parity gives the SELECTION baseline ~depth× FEWER rollouts.
**Probe** (Countdown, seeded solvable instances, p=0.7, N=6, 4 inst/depth × 2 seeds; directional,
not CI-grade), iso matched on candidates vs on exec(move-applies):
- depth3 K4: iso_CAND 1.00 → iso_EXEC 0.50 (savi_freq 0.375); K8: 1.00→0.75 (freq 0.625)
- depth4 K4: 0.75→0.25 (freq 0.125); **K8: 0.75→0.25, savi_freq 0.625 ⇒ freq−isoEXEC = +0.375 (trellis WINS)**
- depth5 K4: 1.00→0.75 (freq 0.125); K8: 1.00→0.75 (freq 0.375)
**Readings:** (1) candidate→exec switch collapses selection pass@1 in every cell — the −0.6…−0.9 D1
was largely a candidate-parity artifact; (2) on the honest axis the trellis is competitive (±0.1–0.4,
noisy), with a clear win at K8/depth4; (3) savi_freq ≫ savi_support everywhere (graded edge is the
lever, support-only has zero ranking info); (4) wider beam helps (K8>K4). Caveats: 8 samples/cell;
semi-synthetic; freq>>support partly because synthetic P_p makes frequency track quality.
See `results.json → axis_correction_addendum`.

## Status: STUDY COMPLETE, headline CORRECTED. The original "selection beats the trellis everywhere"
is a candidate-parity artifact; on the honest moves/exec axis the SAVI premise is NOT refuted (trellis
competitive, wins at K8/depth4 with freq edges). Design lessons for the decisive real-LM run:
iso-compute MUST be tokens/moves (not candidates); headline arm = freq or verifier-masked (not support);
deeper substrate with a CHEAP solvability oracle. A CI-grade exec/token-parity sweep is the proper next run.

## Phase 6 — PLAN2 corrected run: synthesis + verification (CI-grade exec-parity depth sweep)
Ran the PLAN2 lattice sweep (24 cells: D∈{4,8,16,24} × K∈{8,16,32} × p∈{0.7,0.6}, N=16, seeds 1–8,
24 inst/depth, paired bootstrap n=10000) plus two real-Φ confirm cells (Countdown depth-6, Algebra
mock). Raw per-cell in `outputs/lattice_sweep/results2.json` and `outputs/confirm/results.json`.
Three independent Verify passes were run before writing the headline.

### VERDICT: claim NOT supported (claim_supported = NO). Two of three Verify lines returned ok=false.

**Claim under test:** on the honest moves/exec axis, with a freq (and verifier-masked) headline and
adequate beam, global decode beats compute-matched selection, AND the margin GROWS with depth.

**1) Controlling failure — contamination (Verify "oracle/substrate" ok=false).** The fair generator
memoizes good/bad classification and seeds its RNG by `canon(state)=(s,r)`, which OMITS the
per-instance target T. `run_mechanism2._run_cell` (line 173) builds ONE generator per
(depth,K,p,seed) cell and shares it across all 24 instances — different T, overlapping (s,r) canons
(all share (0,D) + interior nodes). The first instance to populate an (s,r) fixes weights+seed under
the WRONG T for every later instance there. Classification depends on T via `solvable`, so `savi_freq`
is biased downward. The verifier reproduced every published number bit-for-bit under the buggy path,
then re-ran cells with a clean per-instance generator: **signs FLIP** — D8/K8 D1_exec −0.0625→+0.052;
D16/K8 −0.177→+0.292; D24/K32 +0.276→+0.396 (savi_freq 0.745→1.000). The shipped sweep is
contaminated; cache.json must be deleted and the sweep re-run.

**2) Depth sub-claim false even in contaminated data (Verify "determinism+depth" ok=false).** D1_exec
is non-monotone in depth in EVERY (K,p) slice: rises to a D16 peak then DECAYS at D24; at K=8 it is
negative and worsening for D≥8. Beam width K — not depth — gates the sign (24 cells: 16 CI-positive,
5 CI-negative all at K=8 depth≥8, 3 spanning zero at deep K=16). Determinism PASS (bit-identical
cache-free repro of D8/K16/p0.7) and the iso_exec-decays-with-depth control PASS
(K32/p0.7: 0.807→0.766→0.536→0.469).

**3) Honest-axis accounting PASS (Verify "exec-parity" ok=true).** Exec-parity holds across all 4608
paired instances; exec counted for EVERY rollout (not stopped at first win); headline numerator is
savi_freq/savi_verifier_on (not support); no oracle injection. The methodology is honest — the bug is
in the generator wiring, not the accounting.

### Contaminated headline numbers (reported for completeness, NOT trusted)
- D1_exec p0.7 K32: D4 +0.193[+0.141,+0.250], D8 +0.234[+0.177,+0.297], D16 +0.464[+0.391,+0.531],
  D24 +0.276[+0.214,+0.339]. K16: D16 +0.089[+0.031,+0.146], D24 +0.036[−0.031,+0.104] (∋0). K8:
  D8 −0.062, D16 −0.161, D24 −0.125 (CIs exclude 0 on the negative side).
- D_verif p0.7 (oracle mask): rises monotonically — D4 +0.193 → D24 +0.531..+0.568[hi 0.635]; shape
  clean but levels on the contaminated iso_exec baseline.
- D_merge p0.7 K32: D4 +0.000, D8 +0.406, D16 +0.698[+0.630,+0.760], D24 +0.495; merge contribution
  grows with depth at adequate beam.
- Axis-artifact cell D16/K8/p0.6: iso_candidates 0.677 ≫ iso_exec 0.484 > savi_freq 0.380 (n=192) —
  PLAN1 candidate-parity bug reproduced directly.

### Real-Φ confirm (unaffected by the lattice-canon bug; single mid-depth points)
- Countdown (p0.7,K16,N16,maxd6,48 flags): savi_freq 0.875 < iso_exec 0.958, D1_exec
  −0.083[−0.167,−0.021] — trellis does NOT beat exec-matched selection at depth 6; artifact
  reproduced (iso_cand−iso_exec +0.042[0.0,0.104]).
- Algebra (mock, p0.7,K8,N8,maxd8, n=5 shrunk from 8): all arms 0.0, D1_exec 0.0 — mock step-gen is
  the binding constraint; exec-parity holds end-to-end but no ranking signal.

### Required before re-claiming
(i) Fix the leak: canon=(s,r,T) OR fresh generator per instance in _run_cell (T is constant within a
decode, so intra-decode Φ-merge is unaffected). (ii) Delete cache.json; re-run the full sweep;
regenerate outputs/lattice_sweep/results2.json + RESULTS2.md. (iii) Regression test: same
(depth,seed,p), different T, one shared generator must equal a private per-instance generator.
(iv) Restate the headline as K-gated and non-monotone; report all 24 cells with CIs.

## Status: PLAN2 sweep BLOCKED on a generator-contamination bug. Headline written as claim_supported=NO.
Deliverables RESULTS2.md + results2.json (top-level synthesis block; raw per_cell untouched in
outputs/) record the failure and the fix-list. The directional expectation (clean re-run flips the
negative cells positive) is from a handful of verifier re-run cells, NOT a CI-grade result in hand.

## Phase 7 — CONTAMINATION FIX + CLEAN RE-RUN (main session) ✅ verdict FLIPS to SUPPORTED
Applied the fix (all four required items): (i) `LatticeDomain.canon` → `(s,r,T)` (one line; intra-decode
Φ-merge unchanged since T constant within a decode); (ii) deleted `outputs/lattice_sweep/cache.json` and
re-ran the full 8-seed harness clean; (iii) added `tests/test_lattice_canon_T_regression.py` (shared
generator, two instances same (s,r) diff T → argmax move differs: add 3 vs add 1; the leak would force
them equal); (iv) restated the headline (below) — and it is NOT K-gated/non-monotone once clean.
Two independent clean re-runs AGREE:
- **Fixed 8-seed harness** (AUTHORITATIVE, `outputs/lattice_sweep/results2.json`), D1_exec p0.7:
  D4 +0.000(all K, null/saturated); D8 +0.089[+0.052,+0.130](all K); D16 K8 +0.281[+0.214,+0.349] / K16
  +0.255 / K32 +0.245; D24 K8 +0.437[+0.365,+0.510] / K16 +0.458 / K32 +0.438. savi_freq 1.0 at D8,
  0.92–1.0 at D24; iso_exec decays 1.0→0.91→0.68–0.76→0.48–0.56.
- **6-seed per-instance-gen probe** (`clean_rerun.py`, immune to sharing): D8 +0.15/+0.17, D16
  +0.31/+0.38, D24 +0.39/+0.42 — same direction, monotone, CI>0.
**Verdict: SUPPORTED on the lattice.** On the honest moves/exec axis with a freq (or verifier-masked)
headline, global decode beats compute-matched selection at every depth ≥ 8 (CI excl 0), margin rises
monotonically with depth; the PLAN2 negatives were ENTIRELY the contamination (predicted sign-flips
landed). Beam width K barely matters once clean — **depth is the driver** (the contaminated data's
K-gating was an artifact). D_merge>0 at depth≥8 (Φ-merge contributes, smaller lever than the graded edge
+ axis). Tests: 19 pass on the touched suite (incl. the new regression + 2 canon assertions updated to
the 3-tuple). SCOPE unchanged: synthetic Viterbi lattice + oracle-defined competence (NOT a real LM);
real-Φ confirm cells stay null/slightly-negative at the shallow depths they reach (Countdown eff. depth
~3, where the lattice D4 cell is also null) — real-domain confirmation needs a DEEP cheap-oracle domain;
real-LM end-to-end still gated on the emission line. See RESULTS2.md banner + results2.json
`corrected_clean_rerun`.
