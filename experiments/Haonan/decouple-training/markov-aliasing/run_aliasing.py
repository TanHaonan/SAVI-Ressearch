# markov-aliasing/run_aliasing.py
"""Semantic-aliasing / Markov probe.

QUESTION: for the toy elimination carrier, two different textual histories that map to the SAME
canonical state — does the model give the SAME next-step distribution over the answer letters?
Near-zero divergence ~= the Markov / trellis precondition (history collapses to the canonical state).
And the claim under test: does DECOUPLED training REDUCE that aliasing vs one-hot (COUPLED)?

THE TWO HISTORY VIEWS (same survivors, same target, same option letters — only the textual history
differs), already present per item in the original carrier:
  * prompt_stated (oracle) — directly names the surviving options.
  * prompt_clue   (self)   — gives only the elimination clues; survivors must be inferred.
For each item we read the answer-position option distribution under BOTH views (two forward passes,
NO generation, all under torch.no_grad) and score:
  * markov-js(stated, clue)  — symmetric, 0 iff identical (the Markov ideal). HEADLINE.
  * markov-kl(stated || clue) — directional companion.
  * readout-TV(stated, truth) and readout-TV(clue, truth) — sanity (how well each view tracks the
    known true posterior; via cp_metrics.tv).

CAVEAT (the clean-signal note): stated vs clue differ slightly in inferential DIFFICULTY (not a pure
paraphrase), so the ABSOLUTE stated-clue divergence is not a clean aliasing number on its own. The
clean signal is the CROSS-GROUP comparison — is decoupled's mean markov-js SMALLER than coupled's
and base's? That cross-group gap is what answers "does decoupled reduce semantic aliasing".

REUSE (read-only, not rebuilt here):
  * items  = controllable-posterior/data/items.json (has prompt_stated/prompt_clue/letters/target/...).
  * adapters = controllable-posterior/outputs/adapter_oracle_{coupled,decoupled,shuffle}_s0; base=none.
  * load + readout: state-emission/core/_deps/{common.py,cp_core.py,cp_metrics.py} (vendored here).
  * markov math: state-emission/core/state_metrics.py (markov_js/markov_kl, already unit-tested).

Smoke:
  CUDA_VISIBLE_DEVICES=2 python run_aliasing.py --groups base,decoupled --ks 3 --per_cell 2
Full (background):
  CUDA_VISIBLE_DEVICES=2 nohup python run_aliasing.py \
      --groups base,coupled,decoupled,shuffle --ks 2,3,4,5 --per_cell 10 > outputs/aliasing.stdout 2>&1 &
"""
import argparse
import importlib.util as ilu
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:                      # progress bar optional; degrade to identity
    def tqdm(x, **k):
        return x

HERE = Path(__file__).resolve().parent
CORE = HERE / "core"
DEPS = CORE / "_deps"
OUT = HERE / "outputs"

DEFAULT_CP = str(HERE.parent / "controllable-posterior")


def by_path(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


# ---- reused infra (importlib-by-path; no package install). numpy/torch/re only at import time;
# transformers/peft load lazily inside load_backbone, AFTER argparse, so --help stays import-light. ----
C = by_path("dep_common", DEPS / "common.py")        # load, chat_prefix_ids, DEFAULT_MODEL, CH
cpcore = by_path("cp_core", DEPS / "cp_core.py")     # option_logits, letter_ids
cpm = by_path("cp_metrics", DEPS / "cp_metrics.py")  # tv, boot_ci
al = by_path("aliasing", CORE / "aliasing.py")       # dist_dict, pair_divergence (thin wrappers)


# How the original adapters were produced (NOT run here): the decoupled-vs-coupled contrast
# reuses the cached oracle-mode seed-0 LoRAs from the controllable-posterior tree. base = no adapter.
def cp_groups(cp_root):
    cp = Path(cp_root)
    return {
        "base": "base",
        "coupled": str(cp / "outputs/adapter_oracle_coupled_s0"),
        "decoupled": str(cp / "outputs/adapter_oracle_decoupled_s0"),
        "shuffle": str(cp / "outputs/adapter_oracle_shuffle_s0"),
    }


def log_line(msg):
    """One human-readable line, both to stdout and appended to outputs/aliasing.log."""
    OUT.mkdir(exist_ok=True)
    with open(OUT / "aliasing.log", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def load_items(items_path, ks, per_cell, max_items=None):
    """Held-out subset of the ORIGINAL carrier: split=='test', k in ks, up to per_cell items per
    (k,j) cell, deterministically ordered by id. Each item already carries prompt_stated/prompt_clue."""
    path = Path(items_path)
    if not path.exists():
        raise FileNotFoundError(f"items missing: {path}")
    items = json.loads(path.read_text())
    cell_items = defaultdict(list)
    for it in items:
        if it["split"] != "test" or it["k"] not in ks:
            continue
        cell_items[(it["k"], it["j"])].append(it)
    sub = []
    for cell in sorted(cell_items):
        sub.extend(sorted(cell_items[cell], key=lambda x: x["id"])[:per_cell])
    if max_items is not None:
        sub = sub[:max_items]
    if not sub:
        raise ValueError(f"empty item subset (ks={ks}, per_cell={per_cell}); nothing to measure")
    return sub


def target_np(item):
    """True posterior as a numpy vector aligned to item['letters']."""
    return np.array([item["target"][L] for L in item["letters"]], dtype=float)


# ----------------------------------------------------------- model loading
def load_backbone(adapter, dev):
    """Qwen3-4B (bf16, eval, frozen) + (for non-'base') the cached LoRA. gptq monkeypatch is applied
    BEFORE importing peft (mirrors state-emission/run_state_emission.load_backbone)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft.import_utils as _piu
    _piu.is_gptqmodel_available = lambda *x, **k: False
    try:
        import peft.tuners.lora.gptq as _pg
        _pg.is_gptqmodel_available = lambda *x, **k: False
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL, local_files_only=True, cache_dir=C.CH)
    model = AutoModelForCausalLM.from_pretrained(
        C.DEFAULT_MODEL, dtype=torch.bfloat16, local_files_only=True, cache_dir=C.CH).to(dev)
    if adapter != "base":
        ap = Path(adapter)
        if not ap.exists():
            raise FileNotFoundError(f"adapter dir missing: {adapter} (expected a cached oracle LoRA)")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()                          # eval-only; backbone frozen, no training
    return model, tok


# ------------------------------------------------------------- per-item probe
def item_aliasing(model, tok, item, dev):
    """Two forward passes (stated view, clue view), NO generation, under no_grad. Returns the per-item
    aliasing row: markov-js/kl(stated,clue) + each view's readout-TV to the known true posterior."""
    k = item["k"]
    letters = item["letters"]
    lids = cpcore.letter_ids(tok, k)
    t = target_np(item)
    with torch.no_grad():
        lg_stated = cpcore.option_logits(model, tok, item["prompt_stated"], k, lids, dev)
        lg_clue = cpcore.option_logits(model, tok, item["prompt_clue"], k, lids, dev)
    p_stated = al.dist_dict(lg_stated, letters)      # softmax -> {letter: prob}
    p_clue = al.dist_dict(lg_clue, letters)
    div = al.pair_divergence(p_stated, p_clue)       # {'js','kl'} via the tested upstream markov math
    tv_stated = cpm.tv([p_stated[L] for L in letters], t)
    tv_clue = cpm.tv([p_clue[L] for L in letters], t)
    return dict(
        id=item["id"], k=k, j=item["j"],
        js=float(div["js"]), kl=float(div["kl"]),
        readout_tv_stated=float(tv_stated), readout_tv_clue=float(tv_clue),
    )


# --------------------------------------------------------------- aggregation
AGG_KEYS = ["js", "kl", "readout_tv_stated", "readout_tv_clue"]


def cell_label(k, j):
    return f"k{k}_j{j}"


def aggregate(rows, seed):
    """Per (k,j) cell AND a group-level 'all' cell: {metric: {mean, ci, n}} with item-level bootstrap
    CI (items are independent). Every metric here is always finite (no abstention path)."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["k"], r["j"])].append(r)

    def agg_rows(rs):
        out = {}
        for key in AGG_KEYS:
            vals = [r[key] for r in rs if np.isfinite(r[key])]
            if not vals:
                out[key] = dict(mean=float("nan"), ci=[float("nan")] * 3, n=0)
            else:
                out[key] = dict(mean=float(np.mean(vals)),
                                ci=cpm.boot_ci(vals, seed=seed), n=len(vals))
        return out

    by_cell = {cell_label(*cell): agg_rows(cells[cell]) for cell in sorted(cells)}
    by_cell["all"] = agg_rows(rows)
    return by_cell


# ------------------------------------------------------------------- driver
def run_group(group, adapter, items, dev, args):
    """Load the model fresh for one group, probe every item (2 fwd passes each), aggregate, write the
    group's JSON + one summary log line. Frees the model + empties cache before returning."""
    log_line(f"[{args.tag}] === group {group}: loading model (adapter={adapter}) ===")
    model, tok = load_backbone(adapter, dev)

    rows = []
    t0 = time.time()
    for it in tqdm(items, desc=group):
        rows.append(item_aliasing(model, tok, it, dev))
    dt = max(time.time() - t0, 1e-9)

    cells = aggregate(rows, args.seed)
    allc = cells["all"]
    blob = dict(
        group=group, adapter=adapter, mode="aliasing",
        items_path=args.items, ks=args.ks, per_cell=args.per_cell,
        n_items=len(items), seed=args.seed,
        caveat=("stated vs clue differ slightly in inferential difficulty (not a pure paraphrase); "
                "the clean aliasing signal is the CROSS-GROUP comparison of mean markov-js, not the "
                "absolute value"),
        cells=cells,
    )
    (OUT / f"aliasing_{group}.json").write_text(json.dumps(blob, indent=2))
    log_line(
        f"[{args.tag}] {group:>9} ALL | "
        f"mean_js={allc['js']['mean']:.4f} ci=[{allc['js']['ci'][0]:.4f},{allc['js']['ci'][2]:.4f}] "
        f"mean_kl={allc['kl']['mean']:.4f} "
        f"readout_tv_stated={allc['readout_tv_stated']['mean']:.3f} "
        f"readout_tv_clue={allc['readout_tv_clue']['mean']:.3f} "
        f"(n={allc['js']['n']}, {len(items) / dt:.2f} items/s)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()         # free before the next group


def main():
    ap = argparse.ArgumentParser(description="semantic-aliasing / Markov probe (stated-vs-clue readout divergence by group)")
    ap.add_argument("--groups", default="base,coupled,decoupled,shuffle",
                    help="comma list from base,coupled,decoupled,shuffle")
    ap.add_argument("--ks", default="2,3,4,5", help="comma list of option-counts to include")
    ap.add_argument("--per_cell", type=int, default=10, help="items per (k,j) cell")
    ap.add_argument("--cp_root", default=DEFAULT_CP,
                    help="root holding data/items.json + outputs/adapter_oracle_<group>_s0")
    ap.add_argument("--items", default=None,
                    help="path to items.json (default: <cp_root>/data/items.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max_items", type=int, default=None, help="optional global item cap (smoke)")
    args = ap.parse_args()

    args.groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    args.ks = [int(x) for x in args.ks.split(",") if x.strip()]
    if args.items is None:
        args.items = str(Path(args.cp_root) / "data" / "items.json")

    # everything below touches torch device / loads models — AFTER argparse on purpose, so --help
    # prints without importing transformers/peft or allocating a device.
    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    groups = cp_groups(args.cp_root)
    for g in args.groups:
        if g not in groups:
            raise ValueError(f"unknown group '{g}' (choose from {list(groups)})")

    items = load_items(args.items, args.ks, args.per_cell, args.max_items)
    log_line(f"[{args.tag}] ==== semantic-aliasing run: groups={args.groups} ks={args.ks} "
             f"per_cell={args.per_cell} n_items={len(items)} dev={dev} items={args.items} "
             f"seed={args.seed} ====")
    log_line(f"[{args.tag}] CAVEAT: stated-vs-clue differ in inferential difficulty (not a pure "
             f"paraphrase); read the CROSS-GROUP mean-markov-js gap (decoupled < coupled < base?), "
             f"not the absolute value.")
    t0 = time.time()

    for g in args.groups:
        run_group(g, groups[g], items, dev, args)

    log_line(f"[{args.tag}] DONE in {time.time() - t0:.1f}s | "
             f"aliasing_<group>.json written to {OUT}")


if __name__ == "__main__":
    main()
