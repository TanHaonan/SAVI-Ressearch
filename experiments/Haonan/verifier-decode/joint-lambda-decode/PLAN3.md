# PLAN3 — Joint λ>0 decode: does verifier-weighted global decode over decoupled emission beat iso-token selection, deep enough that depth matters?

**Status: SPEC (planning only; not yet executed).** This is the joint milestone where the two
lines meet at the `sample()` interface. It is the experiment both lines independently concluded is
the missing one (see `../SAVI_INTEGRATION_READING.md` §5 and the emission line's
`countdown-decode/VERDICT.md` "next: open λ>0"). Verifier line owns the decoder + verifier + the
new instruments; the emission line owns the trained generator (which already exists for Countdown).

## 0. The one-sentence question

On a real, executable-judgment task **deep enough that depth matters**, does a **λ>0
(verifier-weighted) global trellis decode** over a **decoupled-trained emission front-end** beat
**greedy** and **iso-token best-of-many** — and is the win attributable to (a) the verifier-weighted
graded edge, (b) the Φ-merge, and (c) the decoupled emission's multi-peak spread *surviving into
per-step decode*?

This closes the exact seam where both lines stopped:
- Emission line (`countdown-decode`): **λ=0** emission-only trellis is non-degenerate but **loses
  to iso-token best-of-many** (savi 0.438 @1158 tok vs best_of_64 0.812 @446 tok; paired −0.375).
  Its own next step is "open λ>0". And its Countdown is **depth-3** (4 numbers) — too shallow.
- Verifier line (`mechanism-recombination`): on the honest moves axis with a graded/verifier edge,
  global decode **beats** compute-matched selection and the **margin rises with depth** — but only
  CI-grade on a **synthetic lattice**; the one real-Φ cell (Countdown depth ~3) was −0.083.

The shared diagnosis: λ=0/support-only loses; the verifier (λ>0)/graded edge is load-bearing; the
win needs depth; iso-compute must be tokens. PLAN3 tests that diagnosis on a real model.

## 1. Hypotheses (falsifiable; prereg thresholds in §9)

- **H1 (λ>0 beats iso-token selection).** At matched token budget, `savi(λ>0, freq)` pass@1 >
  `best_of_many` pass@1 on a deep Countdown set. Prediction: the λ=0 negative (−0.375) moves to
  ≥ 0 and turns positive as depth grows.
- **H2 (margin rises with depth).** `D1_token(depth)` is increasing over depth ∈ {3,4,5,6}; it is
  ≤0 at depth 3 (replicating the existing result) and >0 (CI excluding 0) at the deepest tier.
- **H3 (multi-peak survives into decode — the marginal-vs-path risk).** The decoupled generator's
  **in-trellis** mean K_eff (distinct canonical successors per expanded node) is > 1 and > the
  coupled/one-hot generator's; and H1 holds for decoupled but **not** for coupled (decoupled
  emission is necessary for the λ>0 win). If K_eff ≈ 1 in-trellis (output too sharp,
  `marginal-vs-path`), H1 cannot hold and PLAN3 fails for a stated, measured reason.
- **H4 (Φ-merge contributes).** `savi(λ>0,freq)` > `beam_no_merge(λ>0,freq)` at the same K/N/token
  budget (merge is load-bearing, not just beam). Expected positive but the smaller lever.
- **H5 (the verifier signal is calibrated — memo condition v).** The graded edge / any value used
  is calibrated against Countdown's **exact backward oracle** `reachable`; reliability-curve ECE
  is reported. (Condition v is ⬜ in the scorecard; this is its first real test.)

## 2. Domain — deeper Countdown (primary), with rationale

**Countdown with k ∈ {4,5,6} numbers → derivation depth {3,4,5}.** Why:
- It is `decode_core`'s native domain and the emission line **already trained** decoupled/coupled
  generators on it (`countdown-decode/core.make_real_sampler`, frozen Qwen3-4B + LoRA). So the real
  backend exists; PLAN3 is mostly decoder-side work.
- Φ = sorted value-multiset + target: **strictly shrinks each step** (no canon-invariant productive
  moves → avoids the algebra failure `algebra-decode/FINDING_phi_mismatch.md`) and **genuinely
  aliases** (commutative op orders merge → the diamonds are real; measured 89→17/layer at depth 3).
- It has an **exact backward oracle** `countdown.reachable` for the verifier mask (λ→∞) and for H5
  calibration ground truth.

**Excluded:** algebra (Φ canon-invariant on productive moves → trellis degenerates); the synthetic
lattice (not a real-LM task — it was the mechanism probe, its job is done).

**Compute caveat (the binding constraint on depth):** `reachable` is exponential in remaining
numbers, and the λ>0 mask calls it per sampled successor (≈ K·N·depth times). At k=6 this is the
likely ceiling for a first run; **memoize `reachable` by canon** and cap at k=6 (depth 5). If k=6
is too slow, the depth curve {3,4,5} still tests H2. A second, cheaper-oracle real domain is a
follow-up, not part of this run.

## 3. Arms (the λ × edge_mode matrix + baselines)

All consume the same `sample(state, N, τ, seed) → candidates` backend. Leaf-check = terminal single
value == target (Countdown's existing exact check). `decode_core` already supports every arm.

| arm | decode_core call | role |
|---|---|---|
| greedy | `greedy` | floor |
| self_consistency | N chains, majority canonical terminal | emission/vote floor (= λ=0 yardstick) |
| best_of_many (iso-token) | `best_of_k` drawn to a **token** budget | **headline baseline** |
| savi λ=0 support | `savi(edge="support", verifier=False)` | no-signal floor (replicate the loss) |
| savi λ=0 freq | `savi(edge="freq", verifier=False)` | emission-only graded (the prior result) |
| **savi λ>0 freq** | `savi(edge="freq", verifier=True)` | **the headline arm** |
| beam_no_merge λ>0 freq | `arms_ext.beam_no_merge(verifier=True, edge="freq")` | Φ-ablation (H4) |
| oracle / ceiling | `oracle` (= `reachable(s0)`) | headroom numerator (=1 on solvable set) |

**Model-load-bearing line (unchanged):** the verifier at λ>0 is the per-step **legality + still-
solvable mask** (`reachable`) plus the **exact leaf check**; it is NOT a full solver that emits the
answer. `reachable` is the ceiling/mask, never an arm that proposes moves. (Countdown's per-step
mask IS informative here — unlike algebra where it was vacuous — which is part of why depth helps.)

## 3b. The generalization probe is RUNNABLE NOW (folded into this plan)

Checked 2026-06-29: this is **not** a gated emission-line dependency — we can run it here.
- **GPU:** 8× A100-40GB present, mostly idle.
- **Trained adapters exist on disk** (no re-train needed for the probe):
  `contrib-savi/experiments/Haonan/decouple-training/countdown-decode/outputs/adapter_decoupled_s0/` and
  `…/adapter_coupled_s0/` (LoRA `adapter_model.safetensors` + `adapter_config.json`).
- **Entry points (existing code):** `countdown-decode/core.real_backend.load_countdown_model(
  adapter_path, device="cuda")` (frozen Qwen3-4B bf16, local_files_only, + `PeftModel.from_pretrained`)
  → `core.make_real_sampler(model, tok, domain, "cuda")` → the `sample(state,N,τ,seed)` closure
  decode_core consumes. Instance generator supports k via `{"set":"generated","k":…}`.
- **Probe = Phase 0 of execution** (runs early, in parallel with the CPU Tier-A build; does NOT block
  on emission-line action). It loads each adapter, generates solvable k∈{4,5,6} Countdown states,
  samples N step continuations per state, and reports **parse rate, move legality, and in-trellis
  K_eff at k=4 (in-distribution) vs k=6 (OOD)** — the disambiguation in §13.3. Decision rule:
  K_eff healthy at k=6 → proceed straight to Tier B at depth; K_eff collapses at k=6 but healthy at
  k=4 → OOD, run the short k∈{4,5,6} LoRA SFT (code exists: `make_data.py` + the SFT script) then
  Tier B; K_eff≈1 even at k=4 → the genuine `marginal-vs-path` kill-criterion (redirect to emission).

## 4. Emission backend — two tiers

- **Tier A (CPU, controllable now): profile-matched decoupled mock.** A `sample()` backend that
  reproduces the emission line's *measured* decoupled profile (in-trellis K_eff, abstention rate,
  per-step sharpness from `state-emission`/`sampling-diversity`), plus a coupled (one-hot, K_eff≈1)
  variant. Purpose: build + debug the **deeper-Countdown harness, the iso-token protocol, the K_eff
  instrument, the depth curve, and the value-calibration instrument** with no GPU, and pre-register
  thresholds. NOT the science (a mock, clearly labeled) — it is the plumbing + freeze step.
- **Tier B (GPU, decisive): the real decoupled model.** Swap in the emission line's trained
  `make_real_sampler` (decoupled and coupled) behind the identical interface. Same harness, same
  arms, same axis. This produces the real-LM number. Gated only on GPU availability + re-pointing
  the backend; the model already exists.

## 5. Iso-compute protocol (tokens / forward-passes — never candidates)

- Budget ledger per arm: `tokens` (Σ generated continuation tokens) + `forwards` + `exec`
  (`apply` + `reachable` calls). The **headline axis is tokens.**
- `best_of_many` is run as a **token-budget sweep** (reuse `countdown-decode/iso_compute.py` shape):
  draw rollouts until cumulative tokens ≥ the savi arm's tokens for that instance; report pass@1 at
  matched budget AND the full pass@1-vs-token curve.
- Headline comparison: `savi(λ>0,freq)` vs `best_of_many` **at equal tokens**, paired bootstrap.
- Report the headroom decomposition (per `DECODING_MODEL.md` §4): greedy → SC (+emission/vote) →
  `savi(λ>0)` → oracle-best-of-K; H1 = savi(λ>0) recovers the slice **only the verifier can claim**.

## 6. Instruments (new this milestone)

1. **In-trellis K_eff** (H3, the marginal-vs-path probe): at each expanded node, count distinct
   `canon(s')` among the N parsed successors; report mean over nodes, decoupled vs coupled. This is
   condition iii measured **at the actual decode nodes**, not the answer-position logit.
2. **Depth curve** (H2): D1_token(depth) over k ∈ {4,5,6}.
3. **Value calibration** (H5, condition v): reliability curve + ECE of the freq edge (and/or any
   learned value) against `reachable` ground truth, per layer.
4. **Φ-merge ablation** (H4): `beam_no_merge` at λ>0.
5. Carry over the fixes: canon fully identifies the instance; visited-set marks only the beam-kept
   frontier; determinism (process-stable seeds); finiteness guard.

## 7. Memo conditions tested alongside (close the ⬜ rows while we're here)

- **E6a** (canonical prompting loses no info): accuracy with full surface history vs canonical-state
  re-render, within 2–3 pts. The whole `render(S)` rests on this and it is untested.
- **ii on Countdown** (Φ false-merge audit): run the dedicated audit on Countdown (the 0/122 is M2
  algebra; the scorecard marks Countdown ii ⬜).
- **E6c** (DP-backtracking helps): `savi` (merge + backpointers) vs plain beam **without merge** at
  matched budget — the *actual* E6c ablation (not just "depth matters"). H4's `beam_no_merge` is
  exactly this instrument.
- **v** (backward-value calibration): H5 above.

## 8. Validity gates (must pass or the cell is instrument-invalid, not a null)

- **V1** selectable headroom non-empty on the deep set (solvable-but-greedy-fails > 0).
- **V2** in-trellis K_eff(decoupled) > K_eff(coupled) and > 1 (else nothing to decode — and H3 has
  already failed, which is itself the finding).
- **V3** oracle-best-of-K > best_of_many at iso-token (there is headroom selection can't already get).
- **Φ works**: merge ratio > 1 and false-merge audit clean on the chosen depth (the algebra lesson).

## 9. Prereg thresholds (DRAFT — freeze on Tier A before the Tier B decisive run)

- δ(H1): `savi(λ>0,freq) − best_of_many` ≥ +3 pts at iso-token, deepest tier, CI excluding 0.
- H2: D1_token strictly increasing across depth; ≤0 at k=4, >0 (CI excl 0) at k=6.
- H3: K_eff(decoupled, in-trellis) ≥ 1.5 and > K_eff(coupled); H1 holds for decoupled, not coupled.
- H4/E6c: `savi − beam_no_merge` > 0, CI excluding 0.
- H5/v: ECE reported (descriptive, no pass/fail).
Paired bootstrap n=10000, Holm across H1/H4. All δ frozen before Tier B.

## 10. Ownership / seam

- **Verifier line builds + runs:** the deeper-Countdown harness, the λ-sweep arm matrix, the
  iso-token protocol, the K_eff / depth / calibration / merge-ablation instruments, the Tier-A mock,
  the prereg freeze, Tier-A validation. (decode_core stays FROZEN; new code in this dir.)
- **Emission line provides (mostly exists):** the trained decoupled + coupled Countdown
  `make_real_sampler` behind `sample()`. The only new emission-side ask is confirming the trained
  model generalizes to k=5–6 numbers (it was trained on k=4); if not, a short SFT extension on a
  k∈{4,5,6} mixture. This is the one real dependency to flag.
- Findings → this dir's `VERDICT.md` + `results.json`; master reads them for scorecard rows
  (end-to-end, v, ii-Countdown, E6a, E6c). Master-owned ledgers/scorecard not edited here.

## 11. Deliverables

`domain_countdown_deep.py` (k∈{4,5,6} instances + headroom + reachable memo), the harness
`run_joint.py` (arms + iso-token + instruments + audits), Tier-A mock backend, tests (incl. reachable
memo correctness, K_eff, iso-token accounting, determinism), `PREREG.md` (frozen), `VERDICT.md`,
`results.json`, `EXECUTION_LOG.md`.

## 12. What this settles / does not

**Settles (if H1–H4 hold on Tier B):** the memo's headline — global decode beats greedy AND
iso-token selection at equal compute, on a real model in a real verifiable domain, with the margin
rising with depth — *correctly attributed* (decoupled emission makes candidates comparable; the
verifier-weighted global path picks the right one; depth is the driver). Plus four ⬜ memo rows
(E6a, ii-Countdown, E6c, v).

**Does NOT settle:** generality beyond Countdown (one domain; a second deep cheap-oracle domain is
the follow-up); free-prose (non-structured) emission (the `L_commit` lever only works on a parseable
slot); and whether the decoupled model trained on k=4 transfers to k=6 (the emission-side dependency).

**Kill criterion (stated up front):** if in-trellis K_eff(decoupled) ≈ 1 (H3 fails — the real
model's per-step output is too sharp, the `marginal-vs-path` risk realized), then λ>0 cannot help
either and the honest verdict is "decoupled training does not produce enough live branching at decode
time for the trellis to exploit" — a real, publishable negative that redirects effort to the emission
objective, not the decoder.

## 13. Decisions (RESOLVED)

1. **Depth ceiling = k=6 (depth 5) first.** Depth curve k ∈ {4,5,6}. k=7 only if `reachable`
   memoization proves cheap enough — not for the first run.
2. **Tier A first, then Tier B.** Tier A (CPU mock) freezes the prereg + de-risks the harness at ~no
   cost; Tier B (real model, GPU) is the decisive run.
3. **Emission OOD dependency — resolved by a generalization probe, NOT a blocker now.** The
   decoupled/coupled Countdown model was SFT'd on **k=4**; k=6 is **out-of-distribution**. This only
   bites at Tier B (Tier A is model-free). Protocol:
   - After Tier A, before Tier B, run a cheap (~1 GPU-hr) **generalization probe** of the real model
     on k=5 and k=6 states: measure **parse rate, move legality, and in-trellis K_eff**.
   - **Disambiguation (critical for interpretability):** measure K_eff at **k=4 (in-distribution)
     AND k=6 (OOD)** in the probe. If K_eff is healthy at k=4 but collapses at k=6 → it is **OOD**
     (fixable). If K_eff ≈ 1 **even at k=4** → that is the genuine `marginal-vs-path` finding (the H3
     kill criterion fires for the right reason — redirect to the emission objective, not the decoder).
   - **Branch:** probe OK → proceed to Tier B. Probe fails (OOD collapse / low parse) → emission line
     runs a short **LoRA SFT on a k∈{4,5,6} mixture**, then Tier B. This is the one emission-side
     dependency; flag it now so the k-mixture model is ready when Tier A completes.
