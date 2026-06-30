"""Token-matched (iso-compute) check: does savi (λ=0 trellis) still beat SELECTION when the
selection baseline is given savi's TOKEN budget (not just 16 candidates)?

savi spends ~1150 tokens/instance (step-expansion over the beam). best_of_16 spends ~110.
The honest axis (sister line: 'iso-compute must be tokens not candidates') is to scale
best_of_k up to savi's token budget and compare. We run savi (K=8,N=16) and best_of_k at a
sweep of K on the SAME instances/backend, recording pass@1 + mean tokens, so savi can be
placed on the selection pass@1-vs-tokens curve.

Usage: CUDA_VISIBLE_DEVICES=g python iso_compute.py --adapter <dir> --instances eval.jsonl \
         --ks 16,64,128 --limit 20 --offset 0 --out outputs/iso/<tag>.json
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
import core  # noqa: E402
from core import metrics  # noqa: E402
from core.run_decode import _result_pass1, _budget_dict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--instances", default="data/eval.jsonl")
    ap.add_argument("--ks", default="16,64,128")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--savi_k", type=int, default=8)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]

    from core.run_decode import load_held_out
    insts = load_held_out(a.instances)[a.offset:a.offset + a.limit]
    domain = core.CountdownDomain()
    model, tok = core.load_countdown_model(a.adapter, device="cuda")
    emit = core.make_real_sampler(model, tok, domain, "cuda")

    recs = []
    for i, inst in enumerate(insts):
        di = {"numbers": inst.numbers, "target": inst.target, "id": inst.id}
        sv = core.savi(domain, emit, di, K=a.savi_k, N=a.n, edge_mode="freq", tau=1.0,
                       seed=a.seed, verifier=False, max_depth=a.max_depth)
        rec = {"id": inst.id,
               "savi": _result_pass1(domain, di, inst.target, sv),
               "savi_tokens": sv.budget.tokens}
        for K in ks:
            bk = core.best_of_k(domain, emit, di, K=K, tau=1.0, seed=a.seed)
            rec[f"bok{K}"] = _result_pass1(domain, di, inst.target, bk)
            rec[f"bok{K}_tokens"] = bk.budget.tokens
        recs.append(rec)
        print(f"[{i+1}/{len(insts)}] {inst.id}: savi={rec['savi']}(tok{rec['savi_tokens']}) "
              + " ".join(f"bok{K}={rec[f'bok{K}']}(tok{rec[f'bok{K}_tokens']})" for K in ks),
              flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(recs, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
