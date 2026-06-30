"""LM-as-step-judge (generative critic) over ProcessBench — calibration control.

ProcessBench critic protocol: show the full solution, let the model reason, then
parse a structured first-error verdict. We convert that single prediction to
per-step pseudo-rewards (0 at the predicted first-error step, 1 elsewhere) so the
PRM metrics pipeline applies unchanged at θ=0.5:
  - good step the critic flags as the error  -> FP (and its distance to the true
    boundary is measurable, directly comparable to the PRM's FP-by-distance);
  - true first-error step the critic misses   -> FN.
Generation is batched; the set is subsampled (logged) since there is no vllm.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import data as D

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = ("You are a careful mathematics grader. You are given a competition math "
          "problem and a numbered step-by-step solution. Check the solution one "
          "step at a time and find the FIRST step that contains a mathematical "
          "error (a wrong calculation, invalid deduction, or false claim).")
INSTR = ("Reason briefly about the steps, then finish with a single final line in "
         "exactly this format:\n"
         "FIRST_ERROR: X\n"
         "where X is the number of the first incorrect step, or NONE if every "
         "step is correct.")

_PAT = re.compile(r"FIRST_ERROR:\s*(NONE|\d+)", re.IGNORECASE)


def build_prompt(tok, problem, steps):
    body = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    user = f"Problem:\n{problem}\n\nSolution:\n{body}\n\n{INSTR}"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_pred(text, nsteps):
    m = None
    for m in _PAT.finditer(text):
        pass  # take the last occurrence (the final answer line)
    if m is None:
        return None
    g = m.group(1).upper()
    if g == "NONE":
        return -1
    k = int(g) - 1
    return k if 0 <= k < nsteps else -1


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new):
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def labeled_count(r):
    return len(r["steps"]) if r["label"] == -1 else r["label"] + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=D.CONFIGS)
    ap.add_argument("--limit", type=int, default=None, help="per-config record cap (subsample)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = D.load_processbench(a.configs, limit_per_config=a.limit)
    if a.limit is not None:
        print(f"SUBSAMPLE: {a.limit}/config -> {len(recs)} solutions (not full set)", flush=True)
    print(f"{len(recs)} solutions to judge", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).to(a.device).eval()
    print("model ready", flush=True)

    # sort by length to reduce padding waste; remember original order
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]["steps"]))
    rewards_by_id, preds = {}, {}
    n_unparsed = 0
    t0 = time.time()
    for b in range(0, len(order), a.batch):
        idxs = order[b: b + a.batch]
        prompts = [build_prompt(tok, recs[i]["problem"], recs[i]["steps"]) for i in idxs]
        outs = generate_batch(model, tok, prompts, a.max_new)
        for i, text in zip(idxs, outs):
            r = recs[i]
            pk = parse_pred(text, len(r["steps"]))
            if pk is None:
                n_unparsed += 1
                pk = -1
            preds[r["id"]] = pk
            rw = [float("nan")] * len(r["steps"])
            for j in range(labeled_count(r)):
                rw[j] = 0.0 if j == pk else 1.0
            rewards_by_id[r["id"]] = rw
        done = b + len(idxs)
        print(f"  {done}/{len(recs)} ({(time.time()-t0)/done:.1f} s/sol)", flush=True)

    json.dump({"rewards": rewards_by_id, "preds": preds}, open(a.out, "w"))
    print(f"wrote {len(rewards_by_id)} -> {a.out} | unparsed={n_unparsed} | "
          f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
