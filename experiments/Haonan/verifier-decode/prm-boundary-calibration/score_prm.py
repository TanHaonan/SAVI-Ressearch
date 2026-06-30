"""Per-step scorer for Qwen2.5-Math-PRM-7B over ProcessBench.

The published checkpoint ships custom remote modeling code written for an older
transformers; it NaNs / breaks under transformers 5.x (legacy DynamicCache +
attention-mask API). We reconstruct the model faithfully from native parts:
the maintained `Qwen2Model` backbone + the PRM's exact 2-layer head
(`score = Sequential(Linear(h,h), ReLU(), Linear(h,2))`, keys score.0/score.2),
loaded directly from the safetensors. Per-step reward = positive-class softmax
probability at each <extra_0> separator (one trailing sep per step).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen2Model

import data as D

MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
SEP = "<extra_0>"


def load_prm(device, dtype=torch.bfloat16):
    # No trust_remote_code: config.json is model_type=qwen2, so we get the native
    # Qwen2Config/tokenizer (the remote auto_map classes break under transformers 5.x).
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)  # native Qwen2Config
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = tok.eos_token_id
    # native backbone: loads model.* weights (score.* / lm_head.weight ignored)
    backbone = Qwen2Model.from_pretrained(
        MODEL, config=cfg, torch_dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    h = cfg.hidden_size
    head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))
    sd = {}
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snap = glob.glob(os.path.join(
        hf_home, "hub/models--Qwen--Qwen2.5-Math-PRM-7B/snapshots/*/*.safetensors"))
    for f in snap:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("score."):
                    sd[k[len("score."):]] = sf.get_tensor(k)
    missing, unexpected = head.load_state_dict(sd, strict=False)
    assert not missing, f"head missing keys: {missing}"
    head = head.to(device, dtype).eval()
    sep_id = tok.encode(SEP)[0]
    return tok, backbone, head, sep_id


@torch.no_grad()
def score_record(backbone, head, tok, sep_id, problem, steps, max_len, device):
    content = SEP.join(steps) + SEP
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": content},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tok.encode(text, return_tensors="pt")
    truncated = ids.shape[1] > max_len
    if truncated:
        ids = ids[:, :max_len]
    ids = ids.to(device)
    hidden = backbone(input_ids=ids, use_cache=False).last_hidden_state
    logits = head(hidden).float()
    probs = logits.softmax(dim=-1)
    sep_pos = (ids[0] == sep_id).nonzero(as_tuple=True)[0]
    rewards = probs[0, sep_pos, 1].cpu().tolist()
    return rewards, truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=D.CONFIGS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = D.load_processbench(a.configs, limit_per_config=a.limit)
    print(f"loaded {len(recs)} records from {a.configs}", flush=True)
    tok, backbone, head, sep_id = load_prm(a.device)
    print("model ready", flush=True)

    rewards_by_id, n_trunc, n_mismatch = {}, 0, 0
    t0 = time.time()
    for i, r in enumerate(recs):
        rw, trunc = score_record(backbone, head, tok, sep_id, r["problem"],
                                 r["steps"], a.max_len, a.device)
        n_trunc += int(trunc)
        if len(rw) != len(r["steps"]):
            n_mismatch += 1
            rw = (rw + [float("nan")] * len(r["steps"]))[: len(r["steps"])]
        rewards_by_id[r["id"]] = rw
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(recs)} ({(time.time()-t0)/(i+1)*1000:.0f} ms/rec)", flush=True)

    json.dump(rewards_by_id, open(a.out, "w"))
    print(f"wrote {len(rewards_by_id)} -> {a.out} | truncated={n_trunc} "
          f"len_mismatch={n_mismatch} | {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
