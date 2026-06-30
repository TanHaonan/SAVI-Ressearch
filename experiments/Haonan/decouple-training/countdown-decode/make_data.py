"""Generate countdown SFT data + a solvable held-out eval set (CPU, reproducible).

Train and eval instances are DISJOINT by seed:
    train = generated seed 0  -> data/sft_{coupled,decoupled}.jsonl (solvable only; datagen
            rolls solve_one so only solvable instances yield states)
    eval  = generated seed 7  -> data/eval.jsonl of {numbers,target,id}, filtered to SOLVABLE
            (so oracle=1 and there is a correct answer to find).

Usage:
    python make_data.py --train_n 400 --eval_n 120 --k 4 --m 4
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
import core  # noqa: E402
from gen_sft_data import gen  # noqa: E402
from _deps.instances import label_instance  # noqa: E402


def _solvable(insts):
    return [it for it in insts if label_instance(it)["solvable"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=400)
    ap.add_argument("--eval_n", type=int, default=120)
    ap.add_argument("--k", type=int, default=4, help="numbers per instance")
    ap.add_argument("--m", type=int, default=4, help="max decoupled moves per state")
    ap.add_argument("--target_range", default="10-100")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.target_range.split("-"))

    train = _solvable(core.load_instances(
        {"set": "generated", "seed": 0, "n": a.train_n, "k": a.k, "target_range": [lo, hi]}))
    summary = gen(train, str(_ROOT), m_decoupled=a.m, seed=0)
    print("SFT (train) summary:", summary)

    ev = _solvable(core.load_instances(
        {"set": "generated", "seed": 7, "n": a.eval_n, "k": a.k, "target_range": [lo, hi]}))
    eval_path = _ROOT / "data" / "eval.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as fh:
        for it in ev:
            fh.write(json.dumps({"numbers": list(map(int, it.numbers)),
                                 "target": int(it.target), "id": it.id}) + "\n")
    print(f"eval: {len(ev)} solvable instances -> {eval_path}")


if __name__ == "__main__":
    main()
