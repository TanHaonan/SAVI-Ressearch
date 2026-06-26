# core/ — self-contained minimal scripts

Latent-state consistency decoding: read a per-slot answer distribution from a
frozen language model (the "node emission"), build cross-slot consistency
constraints (the "edges"), assemble a factor graph, and decode it globally
(greedy / marginal / viterbi) against an exactly-known gold assignment.

This directory is self-contained. All cross-module helpers are vendored under
`_deps/`, and every script loads them from there — no manual import-path edits,
no dependency on any sibling directory.

## Two-step reproduction

From inside this directory:

```bash
# 1) build the carrier dataset -> data/items.json
python gen_data.py

# 2) run one configuration end-to-end (train adapter -> read emission ->
#    build edges -> decode -> score with bootstrap CIs -> outputs/run_*.json)
python run.py --emission decoupled --edges self,oracle,shuffle
```

Then aggregate any run JSONs into the summary tables:

```bash
python analyze.py
```

`run.py --emission {base,coupled,decoupled}` selects the readout target; `base`
skips adapter training. Add `cheap` to `--edges` to also use a small, separate
edge-builder model. The optional elicitation sweep is `python elicit_probe.py`.

Model loading uses the Hugging Face cache via `HF_HOME` if set, otherwise the
transformers default cache. Set `HF_HOME` to point at a local cache if needed.

## Files

- `gen_data.py` — generates the multi-slot consistency carrier (T slots, k options
  per slot, same/different cross-slot constraints, one planted local-vs-global
  conflict per item, gold = exact joint MAP); writes `data/items.json`.
- `nodes.py` — node-emission readout + training targets: read the candidate-option
  distribution at a frozen model's answer position; `decoupled` (calibrated soft
  target) / `coupled` (one-hot) objectives, plus a fluency KL anchor.
- `edges.py` — build per-constraint (k,k) log-potential matrices from four sources
  (oracle / self / cheap / shuffle).
- `decoders.py` — assemble unary + edges into a factor graph and decode three ways
  (greedy / marginal = sum-product / viterbi = max-product).
- `run.py` — one full run: train a LoRA adapter, read the held-out emission, build
  edges, decode, and score by joint exact-match with item-bootstrap CIs.
- `analyze.py` — aggregate multiple run outputs into four summary tables + decisions.
- `elicit_probe.py` — elicitation sweep: query wording (M0/M1/M2) x separate models;
  measures AUC of the consistent-vs-inconsistent pairwise judgment, then decode-confirm.
- `tests/` — unit tests (gen / nodes / edges / decoders / integration / elicit).

## `_deps/` — vendored helpers

These files are imported by the scripts above and were copied in so the directory
runs standalone:

- `_deps/oracle.py` — verified exact chain factor-graph core (MAP / marginals / energy).
- `_deps/graph.py` — thin factor-graph wrapper over `oracle.py`.
- `_deps/metrics.py` — accuracy / exact-match / feasibility-split / bootstrap-CI helpers.
- `_deps/core.py` — answer-position option readout + decoupled/coupled losses + fluency KL anchor.
- `_deps/common.py` — shared model-loading + yes/no and letter scoring helpers.
- `_deps/frontier.py` — cross-tokenizer model loader for the separate edge-builder models.

Importing `nodes`, `edges`, `decoders`, `gen_data`, `metrics` does not load any
model or touch the GPU; model loading inside `run.py` / `elicit_probe.py` is lazy.
