"""Truth-controllable carrier. k content-free options (random nouns -> letters A..); a set of elimination
clues removes (k-j) of them; the Bayes-optimal posterior over options (uniform prior, hard eliminations) is
EXACTLY uniform over the j survivors. Determinate = j1 (one-hot), ambiguous = j>=2 (uniform-over-j). Two
prompt views per item, same target: prompt_stated (survivors named = oracle-forced) and prompt_clue
(eliminations only = self-applied)."""
import argparse, json, random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDEF"
NOUNS = ["apple", "train", "cloud", "river", "candle", "ladder", "violin", "pebble", "helmet", "orchard",
         "magnet", "saddle", "lantern", "compass", "feather", "kettle", "anchor", "bramble", "marble",
         "puzzle", "trumpet", "glacier", "beacon", "thicket", "quilt", "harbor", "willow", "cobweb",
         "domino", "sparrow"]


def make_item(rng, k, j, idx):
    nouns = rng.sample(NOUNS, k)
    letters = list(LETTERS[:k])
    surv_ix = sorted(rng.sample(range(k), j))
    elim_ix = [i for i in range(k) if i not in surv_ix]
    opts = "; ".join(f"{letters[i]}) the {nouns[i]}" for i in range(k))
    clues = [f"It is not the {nouns[i]}." for i in elim_ix]
    rng.shuffle(clues)
    surv_nouns = ", ".join(f"the {nouns[i]}" for i in surv_ix)
    base = f"One prize is hidden behind exactly one of these {k} options: {opts}."
    prompt_clue = f"{base} {' '.join(clues)} Which option hides the prize? Answer with one letter."
    prompt_stated = (f"{base} The prize is equally likely behind any of: {surv_nouns}. "
                     f"Which option hides the prize? Answer with one letter.")
    target = {letters[i]: (1.0 / j if i in surv_ix else 0.0) for i in range(k)}
    return dict(id=f"k{k}_j{j}_{idx}", k=k, j=j, letters=letters, nouns=nouns,
                survivors=[letters[i] for i in surv_ix], target=target,
                prompt_clue=prompt_clue, prompt_stated=prompt_stated)


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
    print("by (k,j):", {f"{k},{j}": n for (k, j), n in sorted(Counter((it["k"], it["j"]) for it in items).items())})


if __name__ == "__main__":
    main()
