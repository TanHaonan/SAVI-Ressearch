"""Model load + answer-position scoring helpers, vendored so this directory runs on its own.

The probe reads, at the assistant generation point, the model's distribution over a small set of
answer tokens. `chat_prefix_ids` builds the token ids up to (and including) the assistant generation
prompt, so the next token is the answer. Defaults to Qwen3-4B with thinking disabled.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

# Qwen/Qwen3-4B run with thinking DISABLED so it answers directly. Adjust the cache dir / model id
# for your environment. cache_dir falls back to the transformers default when HF_HUB_CACHE is unset.
CH = os.environ.get("HF_HUB_CACHE")
DEFAULT_MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")


def load(name=DEFAULT_MODEL, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True, cache_dir=CH)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, local_files_only=True, cache_dir=CH).eval().to(device)
    return m, tok


def chat_prefix_ids(tok, conv, device):
    """Token ids up to (and including) the assistant generation prompt — next token is the answer.
    `conv` is either a user string (single turn) or a full list of {role,content} messages.
    Thinking is disabled (Qwen3 templates) so the answer token comes immediately; harmless on
    templates without that flag."""
    msgs = [{"role": "user", "content": conv}] if isinstance(conv, str) else list(conv)
    kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=False)
    try:
        ids = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        ids = tok.apply_chat_template(msgs, **kw)
    if hasattr(ids, "input_ids"):          # BatchEncoding (transformers 5.x)
        ids = ids["input_ids"]
    if ids.dim() == 1:
        ids = ids[None]
    return ids.to(device)


def boot_ci(values, n_boot=2000, seed=0):
    """Item-level bootstrap (items are independent here) — 2.5/50/97.5 percentiles of the mean."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [float(np.percentile(means, q)) for q in (2.5, 50, 97.5)]
