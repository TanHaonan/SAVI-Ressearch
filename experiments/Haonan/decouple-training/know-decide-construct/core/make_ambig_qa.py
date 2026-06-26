"""Ambiguous yes/no items: a NEUTRAL scene + the sense-A question -> genuinely 50/50 (the scene doesn't
tell you whether the word means sense a or b). Target p(yes)=0.5. The external source = the enumerated
two senses (we KNOW both are valid here).
Input: core/data/polysemy_pairs.json. Output: core/data/ambiguous_qa.json (already shipped here).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

NEUTRAL = [
    "We were talking about the {w} earlier today.",
    "She mentioned the {w} in passing during the meeting.",
    "The {w} came up again in our conversation yesterday.",
    "He left a short note that just said: the {w}.",
]


def main():
    pairs = json.loads((DATA / "polysemy_pairs.json").read_text())
    items = []
    for p in pairs:
        for k, tpl in enumerate(NEUTRAL):
            items.append(dict(id=f"{p['word']}_ambqa{k}", word=p["word"],
                              scene=tpl.format(w=p["word"]), question=p["question_a"],
                              gloss_a=p["gloss_a"], gloss_b=p["gloss_b"], target_pyes=0.5))
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "ambiguous_qa.json").write_text(json.dumps(items, indent=2))
    print(f"wrote {len(items)} ambiguous yes/no items ({len(pairs)} words x {len(NEUTRAL)} neutral scenes)")


if __name__ == "__main__":
    main()
