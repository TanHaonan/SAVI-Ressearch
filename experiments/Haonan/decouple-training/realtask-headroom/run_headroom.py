"""Domain-headroom scout (base model only, no adapter).

On recognized math tasks (GSM8K, MATH-500) it asks two questions about the BASE model:
  1. Is temperature sampling degenerate (collapses to one answer)?  -> k_eff_distinct, top_answer_mass
  2. Is there real SELECTABLE headroom for a global decoder / self-consistency to lift accuracy?
       selectable headroom  = oracle best-of-N accuracy - greedy accuracy   (gold is among the N samples)
       self-consistency gain = majority-vote accuracy   - greedy accuracy

This is a parallel probe, NOT the main calibration experiment, and it gives the iso-compute baseline.

GSM8K integer match is the clean primary signal. MATH-500 \\boxed equivalence is approximate
exact-match (see core/answer_phi), so its accuracy numbers are a lower bound.
"""
from __future__ import annotations

import argparse


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="gsm8k,math500",
                   help="comma list of: gsm8k, math500")
    p.add_argument("--n_items_gsm8k", type=int, default=200, help="num GSM8K problems")
    p.add_argument("--n_items_math500", type=int, default=100, help="num MATH-500 problems")
    p.add_argument("--n_items", type=int, default=None,
                   help="override BOTH per-dataset item counts (smoke convenience)")
    p.add_argument("--n", type=int, default=16, help="num temperature samples per problem")
    p.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
    p.add_argument("--top_p", type=float, default=0.95, help="nucleus top-p for sampling")
    p.add_argument("--max_new_tokens", type=int, default=320)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=None, help="default: <script_dir>/outputs")
    return p


def main():
    args = build_parser().parse_args()

    # --- heavy imports only after argparse, so --help is instant -------------------------------
    import json
    import os
    import sys
    import time

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    import numpy as np
    import torch
    from tqdm import tqdm

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    # vendored loader + bootstrap (copied under core/_deps so this dir is self-contained)
    DEPS = os.path.normpath(os.path.join(HERE, "core", "_deps"))
    sys.path.insert(0, DEPS)
    import common as C  # noqa: E402
    from cp_metrics import boot_ci  # noqa: E402

    from core.answer_phi import (  # noqa: E402
        gsm8k_gold,
        math500_gold,
        extract_numeric,
        extract_boxed,
        normalize_num,
    )

    outdir = args.outdir or os.path.join(HERE, "outputs")
    os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(outdir, "headroom.log")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logpath, "a") as f:
            f.write(line + "\n")

    n_g = args.n_items if args.n_items is not None else args.n_items_gsm8k
    n_m = args.n_items if args.n_items is not None else args.n_items_math500

    log(f"START scout model={args.model} datasets={args.datasets} "
        f"n={args.n} temp={args.temp} top_p={args.top_p} max_new={args.max_new_tokens} "
        f"n_gsm8k={n_g} n_math500={n_m} seed={args.seed}")

    # --- datasets (offline, fault-tolerant) ----------------------------------------------------
    import datasets as hfds

    PROMPT = "Solve the problem. Think step by step, then give the final answer.\n\n"

    def load_gsm8k():
        d = hfds.load_dataset("gsm8k", "main", split="test",
                              download_mode="reuse_dataset_if_exists")
        items = []
        for r in d.select(range(min(n_g, len(d)))):
            items.append({
                "problem": r["question"],
                "gold": gsm8k_gold(r["answer"]),
            })
        return items

    def load_math500():
        d = hfds.load_dataset("HuggingFaceH4/math-500", split="test",
                              download_mode="reuse_dataset_if_exists")
        items = []
        for r in d.select(range(min(n_m, len(d)))):
            items.append({
                "problem": r["problem"],
                "gold": math500_gold(r["answer"]),
            })
        return items

    # per-dataset answer extractor + equality
    def gsm8k_parse(text):
        return extract_numeric(text)

    def gsm8k_eq(a, b):
        if a is None or b is None:
            return False
        return normalize_num(a) == normalize_num(b)

    def math500_parse(text):
        return extract_boxed(text)

    def math500_eq(a, b):
        if a is None or b is None:
            return False
        return a == b  # already normalized by extract_boxed / math500_gold

    SPECS = {
        "gsm8k": dict(loader=load_gsm8k, parse=gsm8k_parse, eq=gsm8k_eq, n=n_g),
        "math500": dict(loader=load_math500, parse=math500_parse, eq=math500_eq, n=n_m),
    }

    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    loaded = {}
    for name in wanted:
        if name not in SPECS:
            log(f"WARN unknown dataset '{name}', skipping")
            continue
        try:
            items = SPECS[name]["loader"]()
            loaded[name] = items
            log(f"LOADED {name}: {len(items)} items "
                f"(gold parse-rate {sum(it['gold'] is not None for it in items)}/{len(items)})")
        except Exception as e:  # noqa: BLE001 - intentionally continue with whatever loaded
            log(f"ERROR loading {name}: {type(e).__name__}: {str(e)[:200]} -- continuing")

    if not loaded:
        log("FATAL no datasets loaded; nothing to do.")
        return

    # --- model ---------------------------------------------------------------------------------
    log(f"loading model {args.model} on {args.device} (bf16, eval, no adapter)")
    t0 = time.time()
    model, tok = C.load(args.model, device=args.device)
    log(f"model loaded in {time.time()-t0:.1f}s")

    @torch.no_grad()
    def gen(prompt, do_sample, n_return):
        ids = C.chat_prefix_ids(tok, PROMPT + prompt, args.device)
        kw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.eos_token_id)
        if do_sample:
            kw.update(do_sample=True, temperature=args.temp, top_p=args.top_p,
                      num_return_sequences=n_return)
        else:
            kw.update(do_sample=False)
        out = model.generate(ids, **kw)
        L = ids.shape[1]
        return [tok.decode(out[i, L:], skip_special_tokens=True).strip()
                for i in range(out.shape[0])]

    # --- per dataset ---------------------------------------------------------------------------
    for name, items in loaded.items():
        spec = SPECS[name]
        parse, eq = spec["parse"], spec["eq"]
        log(f"=== {name}: scoring {len(items)} items, N={args.n} samples each ===")

        per = []          # per-problem metric dicts
        raw = []          # raw samples for recompute
        tok_count = 0
        t_start = time.time()

        for it in tqdm(items, desc=name):
            torch.manual_seed(args.seed)
            prob, gold = it["problem"], it["gold"]

            greedy_txt = gen(prob, do_sample=False, n_return=1)[0]
            sample_txts = gen(prob, do_sample=True, n_return=args.n)
            tok_count += args.max_new_tokens * (1 + args.n)  # upper-bound budget proxy

            greedy_ans = parse(greedy_txt)
            parsed = [parse(t) for t in sample_txts]
            parsed_ok = [a for a in parsed if a is not None]

            # modal answer over parsed samples
            counts = {}
            for a in parsed_ok:
                counts[a] = counts.get(a, 0) + 1
            if counts:
                modal = max(counts, key=lambda k: (counts[k], k))
                top_mass = counts[modal] / len(parsed)          # fraction over ALL N
                sc_correct = eq(modal, gold)
            else:
                modal = None
                top_mass = 0.0
                sc_correct = False

            greedy_correct = eq(greedy_ans, gold)
            oracle_correct = any(eq(a, gold) for a in parsed_ok)
            k_eff = len(set(parsed_ok))
            parse_rate = len(parsed_ok) / len(parsed) if parsed else 0.0

            per.append(dict(
                greedy_correct=greedy_correct,
                selfconsistency_correct=sc_correct,
                oracle_bestofN_correct=oracle_correct,
                k_eff_distinct=k_eff,
                top_answer_mass=top_mass,
                parse_rate=parse_rate,
            ))
            raw.append(dict(
                problem=prob, gold=gold, greedy=greedy_txt, greedy_ans=greedy_ans,
                samples=sample_txts, parsed=parsed, modal=modal,
            ))

        # --- aggregate -------------------------------------------------------------------------
        def col(k):
            return np.array([float(p[k]) for p in per])

        greedy_acc = float(col("greedy_correct").mean())
        sc_acc = float(col("selfconsistency_correct").mean())
        oracle_acc = float(col("oracle_bestofN_correct").mean())
        mean_keff = float(col("k_eff_distinct").mean())
        mean_topmass = float(col("top_answer_mass").mean())
        mean_parse = float(col("parse_rate").mean())

        # headroom, with item-level bootstrap CI on the per-item differences
        sel_diff = col("oracle_bestofN_correct") - col("greedy_correct")
        sc_diff = col("selfconsistency_correct") - col("greedy_correct")
        sel_head = float(sel_diff.mean())
        sc_head = float(sc_diff.mean())
        sel_ci = boot_ci(sel_diff.tolist(), seed=args.seed)
        sc_ci = boot_ci(sc_diff.tolist(), seed=args.seed)

        elapsed = time.time() - t_start
        thr = tok_count / elapsed if elapsed > 0 else 0.0

        summary = dict(
            dataset=name, model=args.model, n_items=len(items), n_samples=args.n,
            temp=args.temp, top_p=args.top_p, max_new_tokens=args.max_new_tokens, seed=args.seed,
            greedy_acc=greedy_acc, selfconsistency_acc=sc_acc, oracle_bestofN_acc=oracle_acc,
            mean_k_eff=mean_keff, mean_top_answer_mass=mean_topmass, mean_parse_rate=mean_parse,
            selectable_headroom=sel_head, selectable_headroom_ci=sel_ci,
            selfconsistency_gain=sc_head, selfconsistency_gain_ci=sc_ci,
            elapsed_sec=elapsed, approx_tokens=tok_count, throughput_tok_per_s=thr,
        )

        with open(os.path.join(outdir, f"headroom_{name}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(outdir, f"samples_{name}.json"), "w") as f:
            json.dump(raw, f, indent=2)

        log(f"RESULT {name}: greedy={greedy_acc:.3f} self-consistency={sc_acc:.3f} "
            f"oracle-bestof{args.n}={oracle_acc:.3f} | SELECTABLE-headroom={sel_head:+.3f} "
            f"CI[{sel_ci[0]:+.3f},{sel_ci[2]:+.3f}] sc-gain={sc_head:+.3f} "
            f"CI[{sc_ci[0]:+.3f},{sc_ci[2]:+.3f}] | k_eff={mean_keff:.2f} "
            f"top_mass={mean_topmass:.2f} parse={mean_parse:.2f} | "
            f"{elapsed:.0f}s {thr:.0f} tok/s")

    log("DONE")


if __name__ == "__main__":
    main()
