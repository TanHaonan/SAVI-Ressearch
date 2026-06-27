# markov-aliasing — semantic-aliasing / Markov probe

**Question.** For the toy elimination carrier, two different textual histories that map to the
**same canonical state** (same surviving options, same true posterior, same option letters) — does
the model give the **same next-step distribution** over the answer letters? If the divergence is
near zero, the history has collapsed to the canonical state, which is the precondition a trellis /
Markov decoder needs. This probes the "reduce semantic aliasing" requirement: one of the metrics
relevant to whether the model's next-step distribution depends only on the canonical state.

**The two history views** (already present per item in the original carrier; they differ ONLY in the
textual history, not in survivors / target / letters):

- `prompt_stated` (oracle) — directly names the surviving options.
- `prompt_clue`   (self)   — gives only the elimination clues; the survivors must be inferred.

For each item we read the answer-position option distribution under **both** views (two forward
passes, **no generation**, all under `torch.no_grad`) and score:

- **markov-js(stated, clue)** — symmetric, 0 iff the two distributions are identical (the Markov
  ideal). This is the **headline** aliasing scalar.
- **markov-kl(stated ‖ clue)** — directional companion.
- **readout-TV(stated, truth)** and **readout-TV(clue, truth)** — sanity: how far each view's
  readout is from the known true posterior.

## CAVEAT (read this before reading any number)

`prompt_stated` and `prompt_clue` are **not a pure paraphrase**: the clue view requires an extra
inferential step (work out the survivors from the clues), so it is genuinely a bit harder. That
means the **absolute** stated-clue divergence is not a clean aliasing number on its own — part of it
is difficulty, not aliasing.

The clean signal is the **cross-group comparison**: is **decoupled's** mean markov-js **smaller than
coupled's and base's**? Difficulty is shared across groups (same items, same two prompts), so a
group that shrinks the stated-clue gap relative to the others is the one reducing aliasing. Read the
gap, not the level.

## Layout

- `core/aliasing.py` — thin wrappers: `dist_dict(logits, letters)` (softmax → `{letter: prob}`) and
  `pair_divergence(p_stated, p_clue)` → `{js, kl}`. The Markov math itself is single-sourced from
  the vendored `state_metrics.markov_js/markov_kl` (already unit-tested).
- `core/state_metrics.py`, `core/_deps/{common,cp_core,cp_metrics}.py` — vendored read-only copies of
  the reused load / readout / metric helpers.
- `tests/test_aliasing.py` — TDD for the two wrappers (pmf validity, js==0 on identical, js>0 on
  disjoint mass, a hand-checked ln 2 case, exact agreement with the upstream metric).
- `run_aliasing.py` — argparse-first driver: per group, load model fresh, two forward passes per
  item, aggregate per `(k,j)` cell + an `all` cell with item-level bootstrap CI, write
  `outputs/aliasing_<group>.json` + a summary line to `outputs/aliasing.log`.

## Reuse (read-only, not rebuilt)

- items   = `controllable-posterior/data/items.json` (carries `prompt_stated`/`prompt_clue`).
- adapters = `controllable-posterior/outputs/adapter_oracle_{coupled,decoupled,shuffle}_s0`; base = none.

## Run

```bash
# tests
python -m pytest tests -v

# smoke (pipeline proof)
CUDA_VISIBLE_DEVICES=2 python run_aliasing.py --groups base,decoupled --ks 3 --per_cell 2

# full (background)
CUDA_VISIBLE_DEVICES=2 nohup python run_aliasing.py \
    --groups base,coupled,decoupled,shuffle --ks 2,3,4,5 --per_cell 10 \
    > outputs/aliasing.stdout 2>&1 &
```

Outputs land in `outputs/aliasing_<group>.json` (small, committed) and `outputs/aliasing.log`
(human-readable, gitignored). **What to look for when it finishes:** mean markov-js for **decoupled
< coupled** and **decoupled < base** — i.e. decoupled training reduces semantic aliasing.
