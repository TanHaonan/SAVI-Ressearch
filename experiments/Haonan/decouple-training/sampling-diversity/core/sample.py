# sampling-diversity/core/sample.py
"""Temperature-sample free generation from base/base+adapter on the controllable-posterior carrier.
Two regimes: 'letter' (on-distribution, the original 'Answer with one letter.' prompt) and 'freeform'
(multi-token sentence, where Phi has real merging work). The deterministic RNG core is isolated for tests;
the model-bound generate path is exercised only in the integration smoke."""
import importlib.util as ilu
from pathlib import Path
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DEPS = HERE / "_deps"               # vendored deps -> this dir is self-contained (no repo-relative paths)
def by_path(name, p):
    s = ilu.spec_from_file_location(name, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m
C = by_path("ce_common", DEPS / "common.py")

LETTER_TAIL = "Answer with one letter."

def letter_prompt(item, mode):
    return item["prompt_stated"] if mode == "oracle" else item["prompt_clue"]

def freeform_prompt(item, mode):
    base = (item["prompt_stated"] if mode == "oracle" else item["prompt_clue"]).replace(LETTER_TAIL, "").strip()
    return base + " Say which option hides the prize and why, in one or two short sentences."

def multinomial_sample(logits, n, temperature, seed):
    """Pure RNG core: draw n indices from softmax(logits / T). Deterministic given seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = F.softmax(logits.float().cpu() / temperature, dim=-1)
    return torch.multinomial(p, n, replacement=True, generator=g).tolist()

@torch.no_grad()
def sample_continuations(model, tok, prompt, dev, n, temperature, max_new, seed):
    """N independent temperature samples of the continuation after `prompt`. Returns list[str]."""
    ids = C.chat_prefix_ids(tok, prompt, dev)            # [1, L]
    torch.manual_seed(seed)
    batch = ids.expand(n, ids.shape[1]).contiguous()     # identical prompts, independent samples
    out = model.generate(batch, do_sample=True, temperature=temperature, top_p=1.0,
                         max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    return [tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True).strip() for i in range(n)]
