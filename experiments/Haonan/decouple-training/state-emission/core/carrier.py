"""Free-text carrier: the controllable-posterior elimination puzzle with TWO extra free-text prompt
views that carry the SAME known posterior (uniform over the j survivors).

Why: the single-letter answer variant makes Phi trivial. This carrier keeps the *meaning* of the
options (random nouns) but asks the model to commit in free text, so Phi must merge by meaning — and
because we constructed each item, the true posterior is still known exactly (uniform over j survivors).

Two regimes share one target:
  - prompt_b1: free reasoning + a deterministic `COMMIT: <thing>` slot -> Phi is deterministic/low-noise.
  - prompt_b2: fully free answer (no slot) -> commitment is in surface paraphrase, Phi merges by meaning.

The legacy training views (prompt_stated oracle / prompt_clue self) and the target / survivors /
letters / nouns / split semantics are kept BYTE-IDENTICAL to controllable-posterior/core/gen_data.py
so the existing controllable-posterior/core/run.py trains on data/items.json UNCHANGED. We import that
module by path and call its make_item to guarantee the shared fields cannot drift, then attach the two
new prompt views. This file is model-free: pure functions + a data-writing main().
"""
import argparse
import importlib.util as ilu
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
# controllable-posterior/core/gen_data.py is the sibling carrier we extend (NOT edited).
_GEN_DATA = (HERE.parent.parent / "controllable-posterior" / "core" / "gen_data.py")


def _bp(name, path):
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_GD = _bp("cp_gen_data", _GEN_DATA)


def _elimination_setup(it):
    """Reconstruct the elimination setup sentence from the base item, IDENTICAL to gen_data's
    prompt_clue stem (option list + the 'It is not the X.' clues), without the trailing
    'Answer with one letter.' instruction — the free-text views append their own instruction."""
    k, j = it["k"], it["j"]
    letters, nouns, survivors = it["letters"], it["nouns"], it["survivors"]
    opts = "; ".join(f"{letters[i]}) the {nouns[i]}" for i in range(k))
    elim_nouns = [nouns[i] for i in range(k) if letters[i] not in survivors]
    clues = " ".join(f"It is not the {n}." for n in elim_nouns)
    base = f"One prize is hidden behind exactly one of these {k} options: {opts}."
    return f"{base} {clues}".strip()


def make_item(rng, k, j, idx):
    """Delegate to gen_data.make_item for the shared elimination structure (target / survivors /
    letters / nouns / prompt_stated / prompt_clue stay IDENTICAL), then add prompt_b1 / prompt_b2
    with the SAME target. The clue order inside prompt_b1/prompt_b2 is deterministic (elimination
    order) and is only a wording detail — it does not affect target/survivors."""
    it = _GD.make_item(rng, k, j, idx)
    setup = _elimination_setup(it)
    it["prompt_b1"] = (
        f"{setup} Explain in one or two sentences which thing hides the prize, then end with a "
        f"line exactly: `COMMIT: <the thing>`."
    )
    it["prompt_b2"] = (
        f"{setup} Say which thing hides the prize and why, in one or two short sentences."
    )
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="2,3,4,5")
    ap.add_argument("--per_kj", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    items = []
    for k in [int(x) for x in a.ks.split(",")]:
        for j in range(1, k + 1):
            for idx in range(a.per_kj):
                it = make_item(rng, k, j, idx)
                r = rng.random()
                it["split"] = "train" if r < 0.6 else ("val" if r < 0.75 else "test")
                items.append(it)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "items.json").write_text(json.dumps(items, indent=2))
    print("items:", len(items), "| split:", dict(Counter(it["split"] for it in items)))
    print("by (k,j):", {f"{k},{j}": n
                        for (k, j), n in sorted(Counter((it["k"], it["j"]) for it in items).items())})


if __name__ == "__main__":
    main()
