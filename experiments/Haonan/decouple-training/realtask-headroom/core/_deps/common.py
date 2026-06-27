"""Shared machinery for the consistency-enforcement probe: chat model load, answer scoring,
metrics. Frozen instruct model, eval-only. Defaults to Qwen3-4B-Instruct-2507.

The probe reads, at the assistant generation point, the model's probability over a small set
of answer strings ("yes"/"no", or "A"/"B"). Scores are length-normalised log-probs summed over
each answer's tokens with leading-space and capitalisation variants merged by log-sum-exp, so
the read-out is robust to the tokenizer and chat template.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

# NOTE: the chosen Qwen3-4B-Instruct-2507 weights are not fetchable here (LFS throttled, mirror
# down). Qwen/Qwen3-4B is fully cached and is the same architecture/size; we run it with thinking
# DISABLED so it answers directly like the instruct variant. Swap via --model when the network allows.
CH = os.path.join(os.environ["HF_HOME"], "hub")
DEFAULT_MODEL = "Qwen/Qwen3-4B"

YES = ["yes", "Yes", " yes", " Yes"]
NO = ["no", "No", " no", " No"]


def load(name=DEFAULT_MODEL, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True, cache_dir=CH)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, local_files_only=True, cache_dir=CH).eval().to(device)
    return m, tok


def chat_prefix_ids(tok, conv, device):
    """Token ids up to (and including) the assistant generation prompt — next token is the answer.
    `conv` is either a user string (single turn) or a full list of {role,content} messages
    (multi-turn). Thinking is disabled (Qwen3 templates) so the answer token comes immediately;
    harmless on templates without that flag."""
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


@torch.no_grad()
def _cont_logprob(model, prefix_ids, answer_ids, device):
    """Sum log P of answer_ids continuing prefix_ids (teacher-forced), in one forward pass."""
    a = torch.tensor(answer_ids, device=device, dtype=torch.long)[None]
    seq = torch.cat([prefix_ids, a], dim=1)
    lp = F.log_softmax(model(seq).logits.float(), dim=-1)
    L = prefix_ids.shape[1]
    idx = torch.arange(L - 1, L - 1 + len(answer_ids), device=device)
    return float(lp[0, idx, a[0]].sum())


@torch.no_grad()
def answer_logprobs(model, tok, user, device, variant_sets):
    """For a chat prompt, return a log-prob per answer class, each = log-sum-exp over its surface
    variants. variant_sets: dict class_name -> list[str]."""
    prefix = chat_prefix_ids(tok, user, device)
    out = {}
    for cls, variants in variant_sets.items():
        lps = []
        for v in variants:
            aid = tok.encode(v, add_special_tokens=False)
            if not aid:
                continue
            lps.append(_cont_logprob(model, prefix, aid, device))
        out[cls] = float(torch.logsumexp(torch.tensor(lps), dim=0)) if lps else float("-inf")
    return out


def p_yes(model, tok, user, device):
    """P(yes) normalised against P(no) for a yes/no chat question."""
    lp = answer_logprobs(model, tok, user, device, {"yes": YES, "no": NO})
    a, b = lp["yes"], lp["no"]
    m = max(a, b)
    return math.exp(a - m) / (math.exp(a - m) + math.exp(b - m))


def p_letter(model, tok, user, device, letters=("A", "B")):
    """P over single-letter multiple-choice answers (normalised across the given letters)."""
    vs = {L: [L, " " + L] for L in letters}
    lp = answer_logprobs(model, tok, user, device, vs)
    m = max(lp.values())
    z = sum(math.exp(lp[L] - m) for L in letters)
    return {L: math.exp(lp[L] - m) / z for L in letters}


@torch.no_grad()
def generate(model, tok, conv, max_new_tokens=90):
    """Greedy free-text continuation for a conversation (str or msg list). Returns the new text."""
    ids = chat_prefix_ids(tok, conv, model.device if hasattr(model, "device") else "cuda")
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def phi(a, b):
    """Mean-square-contingency correlation of two binary error indicators (matches frontier.py)."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n11 = (a & b).sum(); n10 = (a & ~b).sum(); n01 = (~a & b).sum(); n00 = (~a & ~b).sum()
    den = math.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return float((n11 * n00 - n10 * n01) / den) if den > 0 else float("nan")


def boot_ci(values, n_boot=2000, seed=0):
    """Item-level bootstrap (items are independent here) — 2.5/50/97.5 percentiles of the mean."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [float(np.percentile(means, q)) for q in (2.5, 50, 97.5)]
