"""Cross-tokenizer model loader for the cheap edge-builder models.

Only the model-loading helper is needed here: each builder model uses its own
tokenizer + BOS (cross-tokenizer by design), frozen and eval-only. The dataset
and reranking machinery of the original tool is intentionally omitted; nothing in
this package calls it.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Model cache: honour HF_HOME if set, else fall back to the default cache (cache_dir=None).
CH = os.environ.get("HF_HOME") or None


def load(name, device):
    """Load a frozen causal-LM + its own tokenizer (eval-only). Cross-tokenizer by design."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True, cache_dir=CH)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, local_files_only=True, cache_dir=CH).eval().to(device)
    return m, tok
