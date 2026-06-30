"""Plain SFT (causal-LM cross-entropy) of a frozen Qwen3-4B + LoRA(r16) on the COUNTDOWN
move-emission task. Trains the assistant to emit one move per state in the fixed format
``MOVE: <move text>`` (countdown move text = ``"<a> <symbol> <b>"``); the LM loss is
masked to the COMPLETION tokens only (prompt tokens carry label -100). The trained
adapter is the ``real_*`` emission backend the trellis decoder consumes (PREREG sec 3 /
DECODING_MODEL sec 4); decoupled vs coupled is purely a property of the jsonl supervision,
not of this script -- the same plain SFT trains either.

This is the countdown analogue of ``algebra-decode/core/train_algebra.py``: the
data->tensor->labelmask path is domain-AGNOSTIC plain SFT over ``{prompt, completion}``
jsonl, so the only countdown-specific dependency is the prompt builder it imports from
``real_backend`` (``build_move_prompt``), which now resolves to the COUNTDOWN backend (the
system instruction teaches the countdown combine format). The completion-only label mask
and the optional fluency KL anchor are unchanged.

Reuses the carrier conventions (``controllable-posterior/core/run.py`` + ``common.py``):
frozen backbone, LoRA r16 on the same target_modules, bfloat16, ``HF_HUB_CACHE`` +
``local_files_only``, AdamW lr 1e-4, ~6 epochs, bs 8, seed 0, and the optional fluency KL
anchor (``kl_to_base`` on a few generic sentences, default on, low weight).

The data->tensor->labelmask path (``read_jsonl`` / ``encode_row`` / ``collate``) is pure and
GPU-free so the CPU wiring test exercises it without instantiating the 4B model.

CLI
---
    python train_sft.py --data data/sft_decoupled.jsonl \
        --out outputs/adapter_countdown_decoupled_s0 \
        --epochs 6 --lr 1e-4 --rank 16 --seed 0 [--model Qwen/Qwen3-4B] \
        [--max_steps N] [--max_items M]   # the last two bound a cheap 1-epoch smoke

Output adapter dir layout (``--out``, a standard PEFT save_pretrained dir):
    <out>/adapter_config.json          # LoRA config (r, alpha, target_modules, base model id)
    <out>/adapter_model.safetensors    # the LoRA delta weights (the only trained params)
    <out>/train_meta.json              # this script's run record (cfg, n_rows, final loss)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

# SINGLE source of truth for the move prompt: the trainer wraps each row's ``prompt``
# (= raw render(S)) with the SAME chat builder the real backend uses at inference, so the
# prompt tokens the LoRA is trained on are byte-identical to what it sees at decode time.
# This resolves to the COUNTDOWN real_backend (system instruction = the countdown combine
# format), so train and inference share the exact ``MOVE:`` slot.
from real_backend import build_move_prompt

HERE = Path(__file__).resolve().parent

# Cache dir + default model id, same source of truth as the carrier's common.py.
CH = os.environ.get("HF_HUB_CACHE")
DEFAULT_MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")

# LoRA targets identical to run.py (the carrier's frozen-backbone LoRA config).
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# The fixed completion prefix every row carries; the supervised slot is everything after.
MOVE_PREFIX = "MOVE: "

# Generic sentences for the fluency KL anchor (kept identical to common.GENERAL so the
# anchor is the same one the carrier uses; only a few are needed, low weight).
GENERAL = [
    "The sun rose slowly over the quiet harbor as the fishing boats returned.",
    "She opened the old notebook and began to write down everything she remembered.",
    "Economists disagree about the long-term effects of the new trade policy.",
    "After the rain stopped, the children ran outside to play in the puddles.",
    "He studied the map carefully before deciding which trail to take.",
    "A gentle breeze carried the scent of pine through the open window.",
]


# ---------------------------------------------------------------------------
# Data -> tensor -> label-mask path (pure, GPU-free; the CPU test exercises this).
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Read an SFT jsonl into a list of ``{"prompt": str, "completion": str}`` dicts.

    Blank lines are skipped. Each row must carry non-empty ``prompt`` and ``completion``
    string fields and the completion must start with the ``MOVE: `` prefix (the fixed
    emission slot); anything else is a datagen bug and raises rather than training silently
    on a malformed slot.
    """
    rows = []
    for ln, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not (isinstance(obj.get("prompt"), str) and obj["prompt"]):
            raise ValueError(f"{path}:{ln}: missing/empty 'prompt'")
        if not (isinstance(obj.get("completion"), str) and obj["completion"]):
            raise ValueError(f"{path}:{ln}: missing/empty 'completion'")
        if not obj["completion"].startswith(MOVE_PREFIX):
            raise ValueError(
                f"{path}:{ln}: completion must start with {MOVE_PREFIX!r}, "
                f"got {obj['completion']!r}")
        rows.append({"prompt": obj["prompt"], "completion": obj["completion"]})
    return rows


def _prompt_prefix_ids(tok, prompt):
    """Token ids up to (and including) the assistant generation prompt -- the next token
    is the start of the completion. ``prompt`` is the raw ``render(S)``; we wrap it with
    ``real_backend.build_move_prompt`` (system + user, identical to inference) so the
    masked prefix matches the decode-time prompt token-for-token. Returns a python list."""
    msgs = build_move_prompt(prompt)
    kw = dict(add_generation_prompt=True)
    try:
        out = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        out = tok.apply_chat_template(msgs, **kw)
    if hasattr(out, "input_ids"):            # BatchEncoding (transformers 5.x)
        out = out["input_ids"]
    if out and isinstance(out[0], list):     # batched-by-one shape guard
        out = out[0]
    return list(out)


def encode_row(tok, row, max_len=None):
    """Encode one ``{prompt, completion}`` row into ``(input_ids, labels)`` (python lists).

    Layout: ``[ chat-prefix(prompt) ] [ completion tokens ] [ eos ]``. Loss is masked to
    the COMPLETION span only -- every prompt/chat-prefix token gets label -100, the
    completion tokens and the final eos carry their own id as the label (so the model is
    trained to emit the move text AND to stop). The ``MOVE: `` boundary needs no special
    handling: it lives inside the completion span, so it is supervised like the rest of the
    move text (we WANT the model to produce the prefix).

    ``max_len`` (optional) right-truncates ``input_ids``/``labels`` together; for these
    short single-move rows it is effectively a no-op guard.
    """
    prefix = _prompt_prefix_ids(tok, row["prompt"])
    comp = tok.encode(row["completion"], add_special_tokens=False)
    eos = tok.eos_token_id
    comp_with_eos = comp + ([eos] if eos is not None else [])
    input_ids = prefix + comp_with_eos
    labels = [-100] * len(prefix) + list(comp_with_eos)
    if max_len is not None and len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return input_ids, labels


def collate(batch, pad_id):
    """Right-pad a list of ``(input_ids, labels)`` into batched tensors.

    Returns ``dict(input_ids, attention_mask, labels)`` with int64 tensors. Padded
    positions get ``pad_id`` in ``input_ids``, 0 in ``attention_mask``, and -100 in
    ``labels`` (so padding never contributes to the loss, exactly like the prompt mask)."""
    maxlen = max(len(x[0]) for x in batch)
    ii, am, lab = [], [], []
    for input_ids, labels in batch:
        pad = maxlen - len(input_ids)
        ii.append(input_ids + [pad_id] * pad)
        am.append([1] * len(input_ids) + [0] * pad)
        lab.append(labels + [-100] * pad)
    return {
        "input_ids": torch.tensor(ii, dtype=torch.long),
        "attention_mask": torch.tensor(am, dtype=torch.long),
        "labels": torch.tensor(lab, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Fluency KL anchor (same shape as common.kl_to_base; optional, low weight).
# ---------------------------------------------------------------------------

def kl_to_base(model, tok, sents, dev):
    """KL(base next-token dist || adapter) on generic text -- keeps the general LM intact.
    The base distribution is read with the LoRA adapter disabled (frozen backbone)."""
    import torch.nn.functional as F
    tot = 0.0
    for s in sents:
        ids = tok(s, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad(), model.disable_adapter():
            base = F.softmax(model(ids).logits[0].float(), -1)
        adpt = F.log_softmax(model(ids).logits[0].float(), -1)
        tot = tot + F.kl_div(adpt, base, reduction="batchmean")
    return tot / len(sents)


# ---------------------------------------------------------------------------
# Train loop (GPU; not exercised by the CPU test).
# ---------------------------------------------------------------------------

def _build_model(model_id, rank, dev):
    """Frozen backbone + LoRA(r=rank) on LORA_TARGET_MODULES, bf16, offline cache.
    Mirrors run.py's load + the gptqmodel shim so peft imports cleanly."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft.import_utils as _piu
    _piu.is_gptqmodel_available = lambda *x, **k: False
    try:
        import peft.tuners.lora.gptq as _pg
        _pg.is_gptqmodel_available = lambda *x, **k: False
    except Exception:
        pass
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True, cache_dir=CH)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, local_files_only=True, cache_dir=CH).to(dev)
    lcfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                      task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES)
    model = get_peft_model(base, lcfg)
    assert all((not p.requires_grad) for n, p in model.named_parameters()
               if "lora" not in n), "backbone must be frozen"
    return model, tok


def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.data)
    if args.max_items is not None:
        rows = rows[:args.max_items]
    if not rows:
        raise ValueError(f"no rows read from {args.data}")

    model, tok = _build_model(args.model, args.rank, dev)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    encoded = [encode_row(tok, r, max_len=args.max_len) for r in rows]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    print(f"[train_sft] {len(encoded)} rows | model={args.model} | rank={args.rank} "
          f"| bs={args.bs} | lr={args.lr} | epochs={args.epochs} | dev={dev}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    step = 0
    last_loss = float("nan")
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(encoded), generator=g).tolist()
        run, seen = 0.0, 0
        for bstart in range(0, len(perm), args.bs):
            idx = perm[bstart:bstart + args.bs]
            batch = collate([encoded[i] for i in idx], pad_id)
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            if args.lam_kl > 0:
                (args.lam_kl * kl_to_base(model, tok, GENERAL[:4], dev)).backward()
            opt.step()
            opt.zero_grad()
            run += float(loss)
            seen += 1
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        last_loss = run / max(seen, 1)
        ips = seen / (time.time() - t0 + 1e-9)
        print(f"  ep{ep}: loss={last_loss:.4f} steps={seen} steps/s={ips:.2f}", flush=True)
        if args.max_steps is not None and step >= args.max_steps:
            print(f"  [smoke] hit --max_steps={args.max_steps}, stopping", flush=True)
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    meta = dict(model=args.model, rank=args.rank, epochs=args.epochs, lr=args.lr,
                bs=args.bs, seed=args.seed, lam_kl=args.lam_kl, n_rows=len(rows),
                max_steps=args.max_steps, max_items=args.max_items,
                final_loss=float(last_loss), data=str(args.data))
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[train_sft] saved adapter -> {out_dir} (final_loss={last_loss:.4f})",
          flush=True)
    return meta


def build_argparser():
    ap = argparse.ArgumentParser(description="Plain SFT of LoRA-on-Qwen3-4B for countdown "
                                             "move emission (completion-only loss).")
    ap.add_argument("--data", required=True, help="jsonl of {prompt, completion} rows")
    ap.add_argument("--out", required=True, help="output adapter dir (PEFT save_pretrained)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="base model id (Qwen/Qwen3-4B or Qwen/Qwen3-4B-Instruct-2507)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam_kl", type=float, default=0.05,
                    help="fluency KL anchor weight (default on, low; 0 disables)")
    ap.add_argument("--max_len", type=int, default=512,
                    help="right-truncation guard on input_ids+labels")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap total optimizer steps (cheap 1-epoch smoke)")
    ap.add_argument("--max_items", type=int, default=None,
                    help="cap number of jsonl rows read (cheap smoke)")
    return ap


def main():
    train(build_argparser().parse_args())


if __name__ == "__main__":
    main()
