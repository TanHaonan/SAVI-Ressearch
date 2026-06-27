"""Candidate generator behind the shared SAVI interface: sample(state, N, temperature, seed) -> [texts].
The decode side sees only candidate texts and may resample; generator logits are never returned.
`render_state` is the single domain seam: it maps a carrier state dict to a prompt string."""
import importlib.util as ilu
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def _bp(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


C = _bp("dep_common", HERE / "_deps/common.py")


def multinomial_sample(logits, n, temperature, seed):
    """Pure RNG core (unit-tested): n draws from softmax(logits/T). Deterministic given seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = F.softmax(logits.float().cpu() / temperature, dim=-1)
    return torch.multinomial(p, n, replacement=True, generator=g).tolist()


class Generator(Protocol):
    """Shared contract the decode line consumes: a state and sampling controls in, candidate texts out."""
    def sample(self, state, N, temperature, seed) -> list: ...


class AdapterGenerator:
    """Frozen backbone (+ optional cached LoRA) behind sample(). render_state injects the domain prompt."""
    def __init__(self, model, tok, dev, render_state, max_new=48):
        self.model, self.tok, self.dev = model, tok, dev
        self.render_state, self.max_new = render_state, max_new

    @torch.no_grad()
    def sample(self, state, N, temperature, seed) -> list:
        prompt = self.render_state(state)
        ids = C.chat_prefix_ids(self.tok, prompt, self.dev)
        torch.manual_seed(seed)
        batch = ids.expand(N, ids.shape[1]).contiguous()
        out = self.model.generate(batch, do_sample=True, temperature=temperature, top_p=1.0,
                                  max_new_tokens=self.max_new, pad_token_id=self.tok.eos_token_id)
        return [self.tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True).strip() for i in range(N)]


class StubGenerator:
    """Model-free generator for tests: returns scripted texts, records the rendered state."""
    def __init__(self, render_state, scripted):
        self.render_state, self.scripted = render_state, scripted

    def sample(self, state, N, temperature, seed) -> list:
        self.render_state(state)
        return list(self.scripted[:N])
