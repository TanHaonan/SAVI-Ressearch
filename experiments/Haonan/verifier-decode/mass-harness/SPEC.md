# SPEC — math harness around the 1/mass criterion (real-model Countdown)

## Why this design
The lattice showed recombination is real at the PATH level but not categorical at
the INSTANCE level, because the additive lattice has many correct paths (total
correct mass moderate even when each path is rare). The categorical
("effect-not-compute") win needs a substrate where the correct-solution SET is
globally sparse, so `1/mass ≈ 1/q`. Countdown is the right real-model substrate:
canonical mergeable state `canon=(sorted_values, target)`, exact checker + exact
solvability oracle, a real LLM generator (Qwen3-4B+LoRA), and a learned value.

## Mass is empirical here
With an LLM generator P_p is not analytic, so we estimate
`mass = P(one rollout correct)` by sampling. For the interesting (sparse)
instances mass is tiny, so we do not need it precisely — we need the operational
fact **"best-of-K_big samples 0 correct chains on an oracle-solvable instance"**
(i.e. `1/mass ≳ K_big`). That is selection's compute wall at the tested budget.

## Stages
1. **Mass / sparse selection.** Per instance: confirm `oracle(inst)` solvable;
   sample `K_big` full chains from the real model; `mass_hat = #correct/K_big`;
   `chain_solve = mass_hat>0`. **Sparse subset = solvable ∧ ¬chain_solve**
   (selection fails at budget K_big though a solution exists).
2. **Decode arms** over the merged Countdown trellis (per-step sampling + merge):
   - `exact`  : `savi(verifier=True)` — exact-oracle mask. Upper bound: *can*
     stitching reach a correct path at all (vs a knowledge failure where the model
     never even samples the needed steps)?
   - `learned`: `savi_value(V_learned, λ)` — realistic verifier.
   - `freq`   : `savi(verifier=False)` — no verifier (lower bound).
3. **Metrics** (node identity = `canon(state)`; edge = canonical transition):
   - solved; **only_decode** = solved ∧ ¬chain_solve (decode solves where
     best-of-K_big fails);
   - **novel** = decode's canon-state-sequence ∉ any sampled chain's;
   - **stitch** = novel ∧ every canonical transition covered by some chain
     (selection sampled all fragments, never assembled them).
   Aggregate over all instances and over the **sparse subset**.

## Headline test
On the sparse subset (best-of-K_big fails, solution exists): does decode solve,
and are the solutions stitched? `exact` answers "is recombination *possible* here"
(commitment vs knowledge failure); `learned` answers "does a realistic verifier
capture it." A categorical win = decode solves instances where best-of-K_big
finds 0 correct, via stitched paths.

## Run
Domain `CountdownDomain`, builtin instance set (28; 16 headroom + 8 greedy + 4
unsolvable→filtered). K_big=64 chains, decode K=8 N=8 τ=1.0, max_depth=k−1.
Report tokens per arm (token-matched context). Validate metrics (mock domain,
no GPU) + 2-instance GPU smoke before the full run.
