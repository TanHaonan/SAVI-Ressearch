# core — reproduce the execution-selection probe

Two harness files are copied here verbatim at relocation (after the 7B run finishes):

- `expc_code_harness.py` — generation recipe, evalplus sandbox, base/hidden-test grading,
  paired bootstrap, the repair-arm baselines.
- `expc_code_sel_harness.py` — the selection arms (best-of-K, value, value_indep,
  consensus, oracle), the per-candidate logging, and the validity gates. Imports its
  sibling `expc_code_harness`; keep both in this directory.

## Environment

```
pip install evalplus transformers torch numpy
export HF_HOME=/path/to/huggingface-cache
export TOKENIZERS_PARALLELISM=false
```

Models from the HuggingFace cache: `bigcode/starcoderbase-{1b,7b}`, `Qwen/Qwen2.5-1.5B`.
HumanEval+ is loaded through `evalplus`.

## Run (two steps)

```
# CPU sanity (unit tests)
python -m pytest tests/test_expc_code_sel.py -q

# GPU: the selection ladder on one model size (one cell ≈ 100 min on a 40 GB GPU)
python core/expc_code_sel_harness.py --stage15 --forward bigcode/starcoderbase-1b --device cuda:0 --label run_1b
```

Raw per-run outputs (logs, partial JSONL, full per-task dumps) are reproducible and are not
committed; the distilled headline numbers are in `../results.json`. The raw archives from
the original runs remain in the program repo where they were produced.
