"""Node-emission readout + objectives. Reuses the calibrated readout helpers (_deps/core.py) for the
answer-position option readout, the decoupled/coupled losses, and the fluency KL anchor. The emission
target for a slot is the renormalized local-bias unary (temperature tau): keeps the gold option's mass
alive (decoupled) instead of collapsing to the locally-favoured option (coupled).

Reuse confirmed against the readout API:
  option_logits(model, tok, prompt, k, lids, dev) -> tensor[k]
  letter_ids(tok, k), loss_decoupled(lg, t), loss_coupled(lg, t), kl_to_base(model, tok, sents, dev),
  GENERAL (list[str]), C (the shared model-loading + scoring helpers).
Importing this module loads only the vendored helper modules (function definitions and env vars); it
does NOT load any model or touch the GPU.
"""
import importlib.util as ilu
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
DEPS = HERE / "_deps"


def _bp(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


core = _bp("dep_core", DEPS / "core.py")

# Re-exports (reuse, do not reimplement).
C = core.C
GENERAL = core.GENERAL
letter_ids = core.letter_ids
loss_decoupled = core.loss_decoupled
loss_coupled = core.loss_coupled
kl_to_base = core.kl_to_base


def slot_logits(model, tok, prompt, k, lids, dev):
    """tensor[k] of per-option logits at the answer position (grad-friendly). Reuse core.option_logits."""
    return core.option_logits(model, tok, prompt, k, lids, dev)


def emission_target(local_unary_row, tau=1.0):
    """Softmax(local_bias / tau) -> a calibrated distribution that keeps every option (incl. gold) > 0.
    This is the soft target the decoupled objective drives toward; coupled would instead push to its argmax.
    Returns a torch.float32 tensor that sums to 1."""
    z = np.asarray(local_unary_row, float) / max(tau, 1e-6)
    z = z - z.max()
    p = np.exp(z); p = p / p.sum()
    return torch.tensor(p, dtype=torch.float32)
