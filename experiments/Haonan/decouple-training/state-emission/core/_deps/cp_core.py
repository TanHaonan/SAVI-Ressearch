"""Multi-option answer-position readout + decoupled/coupled objectives + fluency KL anchor.
Mode-agnostic: the caller passes the prompt string (item['prompt_stated'] for oracle, item['prompt_clue']
for self)."""
import importlib.util as ilu
from pathlib import Path
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDEF"

def _by_path(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m

# Model load + answer-position scoring helpers (chat_prefix_ids, DEFAULT_MODEL, CH, boot_ci),
# vendored locally so this directory runs on its own.
C = _by_path("common", HERE / "common.py")

GENERAL = [
    "The sun rose slowly over the quiet harbor as the fishing boats returned.",
    "She opened the old notebook and began to write down everything she remembered.",
    "Economists disagree about the long-term effects of the new trade policy.",
    "After the rain stopped, the children ran outside to play in the puddles.",
    "He studied the map carefully before deciding which trail to take.",
    "A gentle breeze carried the scent of pine through the open window.",
    "Researchers published their findings in a peer-reviewed journal last month.",
    "By the time the movie ended, almost everyone in the theater was asleep.",
]

def letter_ids(tok, k):
    out = []
    for ch in LETTERS[:k]:
        s = set()
        for v in (ch, " " + ch):
            e = tok.encode(v, add_special_tokens=False)
            if e:
                s.add(e[0])
        out.append(sorted(s))
    return out

def option_logits(model, tok, prompt, k, lids, dev):
    """Grad-friendly: tensor[k] of per-option logits at the answer position."""
    ids = C.chat_prefix_ids(tok, prompt, dev)
    logits = model(ids).logits[0, -1].float()
    return torch.stack([torch.logsumexp(logits[torch.tensor(lids[i], device=dev)], 0) for i in range(k)])

def target_vec(item, dev):
    return torch.tensor([item["target"][L] for L in item["letters"]], device=dev)

def loss_decoupled(lg, t):
    """KL(target || softmax(lg)) — soft target, does not over-sharpen on ambiguous (t is the known posterior)."""
    return F.kl_div(F.log_softmax(lg, 0), t, reduction="sum")

def loss_coupled(lg, t):
    """Hard cross-entropy to the single argmax survivor — the standard one-hot objective; over-collapses."""
    label = int(torch.argmax(t))
    return F.cross_entropy(lg[None], torch.tensor([label], device=lg.device))

def kl_to_base(model, tok, sents, dev):
    """Fluency anchor: KL(base next-token dist || adapter) on generic text — keeps general LM intact."""
    tot = 0.0
    for s in sents:
        ids = tok(s, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad(), model.disable_adapter():
            base = F.softmax(model(ids).logits[0].float(), -1)
        adpt = F.log_softmax(model(ids).logits[0].float(), -1)
        tot = tot + F.kl_div(adpt, base, reduction="batchmean")
    return tot / len(sents)
