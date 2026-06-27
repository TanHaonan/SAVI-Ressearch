"""Commit-slot calibration — the genuinely-new code of the generation-level lever.

The split this addresses: the decoupled adapter is calibrated when read at the answer position
(readout-TV ~0.02) but NOT when it freely generates a `COMMIT: <thing>` line
(calibration-TV 0.31-0.52, ~1/3 abstain). This module adds ONE training term that
directly calibrates the token emitted in the COMMIT slot — over the OPTION NOUNS, which
is exactly the surface `phi_b1` reads downstream.

Three pieces, mirroring controllable-posterior/core/core.py but at the commit slot:
  - noun_ids   : like letter_ids, but the first-token ids of each option noun (merging
                 the "<noun>" and " <noun>" variants);
  - commit_logits : build prompt_b1 + a fixed reasoning template + " COMMIT:", forward,
                 read logsumexp over each noun's first-token ids at the last position
                 (grad-friendly — used in training);
  - loss_commit : KL(target ‖ softmax(commit_lg)), the SAME form as loss_decoupled.
"""
import importlib.util as ilu
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

# Default fixed reasoning preamble inserted between prompt_b1's assistant-generation point and the
# " COMMIT:" slot. The eval SAMPLES reasoning freely; this is one fixed template (see PREREG risk).
DEFAULT_TEMPLATE = " Let me weigh the remaining clues."


def _by_path(name, path):
    s = ilu.spec_from_file_location(name, str(path))
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# chat_prefix_ids lives next to the trainer's core; resolve it lazily so the unit tests (which never
# build commit_logits) import this module without needing transformers / the model deps.
_CP_COMMON = (HERE.parent.parent / "controllable-posterior" / "core" / "common.py")


def _chat_prefix_ids(tok, prompt, dev):
    C = _by_path("cp_common", _CP_COMMON)
    return C.chat_prefix_ids(tok, prompt, dev)


def noun_ids(tok, nouns):
    """For each option noun, the first-token ids of BOTH "<noun>" and " <noun>" (merged, de-duped,
    sorted). Analogous to core.letter_ids but over the option nouns instead of the letters A..F."""
    out = []
    for n in nouns:
        s = set()
        for v in (n, " " + n):
            e = tok.encode(v, add_special_tokens=False)
            if e:
                s.add(int(e[0]))
        out.append(sorted(s))
    return out


def commit_logits(model, tok, prompt_b1, reasoning_template, nouns, noun_id_lists, dev):
    """Grad-friendly tensor[k]: the per-noun logit at the COMMIT slot.

    ids = chat_prefix_ids(prompt_b1) ++ encode(reasoning_template + " COMMIT:"); forward; take the
    next-token logits at the last position; for option i, logsumexp over its noun first-token ids.
    This is the distribution the model would draw the committed noun from, given a fixed preamble.
    """
    prefix = _chat_prefix_ids(tok, prompt_b1, dev)               # [1, L]
    tail = tok.encode(reasoning_template + " COMMIT:", add_special_tokens=False)
    tail_ids = torch.tensor(tail, device=dev, dtype=prefix.dtype)[None]
    ids = torch.cat([prefix, tail_ids], dim=1)
    logits = model(ids).logits[0, -1].float()
    return torch.stack([
        torch.logsumexp(logits[torch.tensor(noun_id_lists[i], device=dev)], 0)
        for i in range(len(noun_id_lists))
    ])


def loss_commit(commit_lg, target_vec):
    """KL(target ‖ softmax(commit_lg)) at the commit slot over the option nouns — the SAME form as
    core.loss_decoupled, just read at the COMMIT slot instead of the answer position."""
    return F.kl_div(F.log_softmax(commit_lg, 0), target_vec, reduction="sum")
