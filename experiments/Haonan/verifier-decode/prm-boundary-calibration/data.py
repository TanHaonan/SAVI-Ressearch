"""ProcessBench loader — reads local JSON, no network / no `datasets` builder.

ProcessBench record: {id, generator, problem, steps:list[str],
final_answer_correct:bool, label:int}. `label` = 0-based index of the first
erroneous step, or -1 if every step is correct (step-level, not final-answer).
"""
from __future__ import annotations
import glob
import json
import os

CONFIGS = ["gsm8k", "math", "olympiadbench", "omnimath"]

# Resolve the HF cache root the same way download_assets.sh does (HF_HOME, else
# ~/.cache/huggingface). Override PROCESSBENCH_DIR for a manual dataset copy.
_HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Both a manual copy and the HF hub snapshot hold the same gsm8k.json etc.
_SEARCH_DIRS = [
    os.environ.get("PROCESSBENCH_DIR", "datasets/Qwen__ProcessBench"),
    os.path.join(_HF_HOME, "hub/datasets--Qwen--ProcessBench/snapshots"),
]


def _find_json(config: str) -> str:
    for d in _SEARCH_DIRS:
        hits = glob.glob(os.path.join(d, "**", f"{config}.json"), recursive=True)
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(f"{config}.json not found under {_SEARCH_DIRS}")


def load_processbench(configs=CONFIGS, limit_per_config=None):
    """Return a flat list of normalized records (adds 'config')."""
    recs = []
    for c in configs:
        data = json.load(open(_find_json(c)))
        if limit_per_config is not None:
            data = data[:limit_per_config]
        for r in data:
            recs.append({
                "config": c,
                "id": f"{c}:{r['id']}",
                "problem": r["problem"],
                "steps": list(r["steps"]),
                "label": int(r["label"]),
                "final_answer_correct": r.get("final_answer_correct"),
            })
    return recs
