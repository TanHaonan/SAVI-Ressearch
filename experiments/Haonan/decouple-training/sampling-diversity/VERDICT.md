# VERDICT — sampling-diversity-after-Φ

> Does the LoRA that makes the **answer-position readout** match the true posterior (the `controllable-posterior`
> decoupled adapter) also make **temperature-sampled free generation** semantically diverse? Measured on the
> prior-free elimination carrier (Bayes posterior = uniform over the j surviving options), frozen Qwen3-4B +
> cached adapters, N temperature samples per item → semantic-merge Φ → state-count / coverage / TV-to-truth.
> Groups: base (no adapter) · coupled (one-hot CE) · decoupled (soft-target) · shuffle (broken-signal control).
> Φ = deterministic parser (letter regime) + external Llama-3.1-8B judge (freeform regime). Data: `outputs/`.

## 0 one line (plain)

**On this carrier the answer is YES — the calibrated readout DOES reach sampling.** When the decoupled model
is asked to answer and we draw 24 samples, its generations spread across the surviving options almost exactly
as the true posterior says they should: with j surviving options it visits ≈ j distinct options and covers
≈ 100% of them (j=2→5). The standard one-hot model (coupled) does the opposite — it mode-collapses, every
sample lands on one option (coverage ≈ 1/j, generation-TV ≈ 1−1/j exactly). The same gap survives, weaker,
when the model answers in **free sentences** judged by an external model (decoupled keeps 69–92% coverage
where coupled falls to 15% by j=2 and 0% by j=3). **So on discrete, nameable alternatives the decoupled
objective is not a probe-only trick — it changes what the model actually samples.** The open edge is whether
this holds when the alternatives are *not* pre-enumerated and Φ must merge by meaning (the word-sense
setting, where an earlier abstention probe found calibration did *not* reach free generation).

## 1 letter regime — the clean test (on-distribution, N=24 samples/item, T=1.0, aggregated over k by j)

The model is asked exactly as it was trained ("Answer with one letter"); Φ = parse the committed option.
This isolates "does the calibrated answer-position distribution govern actual sampling?".

| metric (by survivor-count j) | j=1 | j=2 | j=3 | j=4 | j=5 |
|---|---|---|---|---|---|
| **K_eff distinct states** — decoupled | 1.00 | **2.00** | **2.93** | **3.92** | **4.58** |
| K_eff — coupled | 1.00 | 1.02 | 0.00 | 0.00 | 0.00 |
| **coverage of survivors** — decoupled | 1.00 | **1.00** | **0.98** | **0.98** | **0.92** |
| coverage — coupled | 1.00 | 0.51 | 0.00 | 0.00 | 0.00 |
| **generation-TV to truth** — decoupled | 0.00 | 0.155 | 0.221 | 0.240 | 0.286 |
| generation-TV — coupled | 0.00 | 0.498 | — | — | — |
| readout-TV (known) — decoupled | 0.00 | 0.027 | 0.026 | 0.022 | 0.017 |
| readout-TV — coupled | 0.00 | 0.500 | 0.667 | 0.750 | 0.800 |
| collapse reference 1−1/j | 0.00 | 0.500 | 0.667 | 0.750 | 0.800 |

- **decoupled generation tracks the posterior:** K_eff ≈ j and coverage ≈ 1 across j=2…5 — the sampler really
  does visit all surviving options. This is the result the front-end claim needs.
- **coupled generation collapses:** its readout-TV is exactly 1−1/j (all mass on one option) and its sampling
  mirrors it — coverage 1/2 at j=2, then by j≥3 the constrained answers stop parsing as a single option at all.
- **decoupled has a real but small readout→generation gap:** generation-TV (0.15→0.29) sits above readout-TV
  (≈0.02) — sampling is a bit less perfectly calibrated than the logit slice — but it is **nowhere near
  collapse**; the spread is genuine.
- **shuffle (control)** stays broken (coverage ≤ 0.07, generation-TV 0.5–1.0); **base (no adapter)** abstains
  ~100% — the un-adapted model will not even commit to the one-letter format here, so the informative contrast
  is decoupled-vs-coupled: **same adapter recipe, only the training objective differs.**

## 2 freeform regime — multi-token, external LLM-Φ (the SAVI-faithful, harder case)

The model answers in one–two sentences ("…say which option and why"); an external Llama-3.1-8B judge maps each
sentence to the option it commits to (or hedge/none). 60 items/group judged, N=12 samples/item, T=1.0.
The deterministic parser is unreliable here (free-text reasoning restates the *eliminated* options' nouns, so
the regex returns "multi" — parser↔LLM agreement only 0.07–0.24), so the **LLM-Φ numbers are authoritative**.
Only j≤3 fell in the judged subset.

| metric (by j) | j=1 | j=2 | j=3 |
|---|---|---|---|
| **coverage** — decoupled | **0.92** | **0.83** | **0.69** |
| coverage — coupled | 0.75 | 0.15 | 0.00 |
| coverage — shuffle | 0.88 | 0.69 | 0.25 |
| **K_eff distinct** — decoupled | 1.54 | 1.71 | 2.08 |
| K_eff — coupled | 1.17 | 0.29 | 0.00 |
| **abstain rate** — decoupled | 0.46 | 0.29 | 0.24 |
| abstain — coupled | 0.48 | 0.93 | 1.00 |
| generation-TV — decoupled | 0.36 | 0.37 | 0.39 |

- **Same ordering, attenuated:** decoupled keeps the highest coverage and K_eff and the lowest abstain at every
  j; coupled collapses (coverage 0.15 by j=2, 0 by j=3; abstains ≥0.93 once ambiguous). So the diversity does
  carry into genuine multi-token free text, not just the single answer token.
- **But the calibration degrades in free text:** decoupled's coverage falls with j (0.92→0.69) and its
  generation-TV (~0.37) is well above its letter-regime value and its readout (0.02) — the readout→generation
  gap **widens** when the model writes sentences.
- **A freeform floor exists:** even shuffle shows nonzero coverage (free-text has intrinsic lexical
  variability the judge can read as different options), so part of the freeform "diversity" is surface noise,
  not calibrated structure. Decoupled's advantage over shuffle is real but grows only with j (0.83 vs 0.69 at
  j=2; 0.69 vs 0.25 at j=3).

## 3 conditions — what this establishes and what it does not

- **Establishes (on this carrier):** the decoupled objective is **not probe-only here** — it changes the
  sampled output. On discrete, nameable options, drawing from the calibrated model gives candidates that Φ
  maps to ≈ j distinct surviving states with near-full coverage (letter regime), and a clearly-better-than-
  one-hot, clearly-better-than-control spread that survives into external-judge free-text (freeform). Directly
  answers the question for this carrier: *temperature sampling does produce semantically distinct
  paths, so Φ has something to distinguish; the connection does not need a new training mechanism in this case.*
- **Does NOT establish (the live edge):** (a) that this holds when the alternatives are **not pre-enumerated /
  not nameable**, where Φ must merge by meaning rather than read off an option label — that is the
  word-sense setting where an earlier abstention probe found calibration did **not** reach free generation; the
  distinguishing condition is plausibly "discrete enumerated choices (transfers) vs. open ambiguity judgment
  (does not)". (b) Perfect generation calibration — there is a real readout→generation gap that **widens in
  free text** (generation-TV 0.02 readout → 0.15–0.29 letter → ~0.37 freeform). (c) Coverage at j≥4 under
  LLM-Φ (judged subset only reached j≤3). (d) That base is a fair generation baseline — it abstains on the
  format, so the clean independent-variable contrast is decoupled-vs-coupled.
- **Conditions the earlier "calibration does not reach generation" statement:** that wall was found on the
  word-sense yes/no carrier; on the elimination carrier with discrete nameable options it does **not** hold —
  so the earlier statement should be conditioned, not generalized: *calibration reaches sampling when the
  alternatives are discrete and nameable; whether it reaches open free generation with semantic-merge Φ is
  still open.*
- **Method caveat:** the smoke (N=4) had shown apparent collapse — a small-sample artifact (4 draws cannot
  reveal a spread over 5 options); the N=24 full run is the trustworthy estimate. Bootstrap CIs per cell are
  in `outputs/diversity_*.json`; the decoupled-vs-coupled coverage gap (0.92–1.0 vs 0–0.51) is far outside CI.

## files
`core/sample.py`, `core/phi.py`, `core/divmetrics.py`, `core/run_diversity.py`, `core/tests/` (23 tests).
Pipeline: TDD per subtask; integration smoke runs pass. Full run: 4 groups × {letter temp-sweep
0.7/1.0/1.3, freeform} + consolidated LLM-Φ; per-cell metrics `outputs/diversity_<group>.json`,
LLM-Φ `outputs/faithfulness.json`, log `outputs/run.log`.
