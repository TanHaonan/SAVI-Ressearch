"""Frozen-backbone features + same-model baselines for ONE model, in a single load. Run per model.

Per item, per sense s in {a,b}, build the reverse sequence [assertion_s + ' '][scene] and run the frozen
backbone once with hidden states. We pool ONLY the SCENE token positions (scene-only pooling: excludes
the gloss/assertion tokens, matching what the crude reverse scores and blocking a gloss-token leak):
mean over scene tokens and last scene token, at every layer. We also store NO-SCENE control features
(pool the assertion tokens of [assertion only]) to detect a gloss-surface cheat.

Also, the FORWARD answering pass: chat-template [scene + question], one forward, pool the hidden state
at the answer position (last token) at every layer -> tests whether the correct answer is decodable from
the very pass the model answers (and fails) on. Plus the crude reverse sum-logprob (rev_a, rev_b) and the
LOCAL forward correctness P(yes|context).

Saves data/feats_<short>.pt with float16 tensors (sense order = a,b):
  rev_mean,rev_last,ns_mean,ns_last : [N,2,L+1,H]   fwd_hidden : [N,L+1,H]
  meta: id, word, split, gold, cued, local_correct, p_local, rev_a, rev_b, rev_correct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import common as C   # local: frozen chat-model load + yes/no answer scoring

HERE = Path(__file__).resolve().parent
ANS = "Answer with only 'yes' or 'no'."


def assertion(item, sense):
    g = item["gloss_a"] if sense == "a" else item["gloss_b"]
    return f'In this passage, the word "{item["word"]}" refers to {g}.'


@torch.no_grad()
def _rev_config(model, tok, item, sense, device):
    pre = tok.encode(assertion(item, sense) + " ", add_special_tokens=True)
    cont = tok.encode(item["scene"], add_special_tokens=False)
    ids = torch.tensor(pre + cont, device=device)[None]
    out = model(ids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, 0)[:, 0].float()            # [L+1,T,H]
    L = len(pre); sc = hs[:, L:L + len(cont), :]
    mean_sc, last_sc = sc.mean(1), sc[:, -1, :]
    lp = F.log_softmax(out.logits[0].float(), -1)
    idx = torch.arange(L - 1, L - 1 + len(cont), device=device)
    rev = float(lp[idx, torch.tensor(cont, device=device)].sum())
    out2 = model(torch.tensor(pre, device=device)[None], output_hidden_states=True)
    hs2 = torch.stack(out2.hidden_states, 0)[:, 0].float()          # [L+1,Tpre,H]
    mean_ns, last_ns = hs2[:, 1:, :].mean(1), hs2[:, -1, :]
    return mean_sc.half().cpu(), last_sc.half().cpu(), mean_ns.half().cpu(), last_ns.half().cpu(), rev


@torch.no_grad()
def _forward_hidden(model, tok, item, device):
    ids = C.chat_prefix_ids(tok, f"{item['scene']} {item['question']} {ANS}", device)
    out = model(ids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, 0)[:, 0, -1, :].float()     # [L+1,H] at answer position
    return hs.half().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--corpus", default="data/corpus.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    short = a.model.split("/")[-1].replace("-", "_").lower()[:16]
    items = json.loads((HERE / a.corpus).read_text())

    m, tok = C.load(a.model, dev)
    rm, rl, nm, nl, fh, meta = [], [], [], [], [], []
    for i, it in enumerate(items):
        cfg = {s: _rev_config(m, tok, it, s, dev) for s in ("a", "b")}
        rm.append(torch.stack([cfg["a"][0], cfg["b"][0]]))
        rl.append(torch.stack([cfg["a"][1], cfg["b"][1]]))
        nm.append(torch.stack([cfg["a"][2], cfg["b"][2]]))
        nl.append(torch.stack([cfg["a"][3], cfg["b"][3]]))
        fh.append(_forward_hidden(m, tok, it, dev))
        rev_a, rev_b = cfg["a"][4], cfg["b"][4]
        p = C.p_yes(m, tok, f"{it['scene']} {it['question']} {ANS}", dev)
        gold_yes = it["gold"] == "yes"
        pred_sense = "a" if rev_a >= rev_b else "b"
        meta.append(dict(id=it["id"], word=it["word"], split=it["split"], gold=it["gold"], cued=it["cued"],
                         local_correct=bool((p > 0.5) == gold_yes), p_local=float(p),
                         rev_a=rev_a, rev_b=rev_b, rev_correct=bool(pred_sense == it["cued"])))
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)}")

    out = Path(a.out) if a.out else HERE / f"data/feats_{short}.pt"
    torch.save(dict(rev_mean=torch.stack(rm), rev_last=torch.stack(rl),
                    ns_mean=torch.stack(nm), ns_last=torch.stack(nl),
                    fwd_hidden=torch.stack(fh), meta=meta, model=a.model), out)
    print(f"saved {out}  N={len(items)}  rev_shape={tuple(torch.stack(rm).shape)}")


if __name__ == "__main__":
    main()
