"""Thin custom helpers for the semantic-aliasing / Markov probe.

The question: for the toy elimination carrier, two textual histories that map to the SAME canonical
state (same survivors, same target, same option letters) — does the model give the SAME next-step
distribution over the answer letters? Near-zero divergence ~= the trellis/Markov precondition holds
(history collapses to the canonical state). The two histories per item are:
  * prompt_stated (oracle) — directly names the surviving options;
  * prompt_clue   (self)   — gives only the elimination clues; survivors must be inferred.
So the divergence between the answer-position option distribution under the two views is a direct
(proxy) measure of semantic aliasing.

This module owns ONLY the thin glue. All Markov math lives in the vendored
core/state_metrics.py (markov_js / markov_kl, already unit-tested); pair_divergence
just routes through it. Keeping the heavy math single-sourced is deliberate — do not re-implement it.

CAVEAT (documented at the call sites and in the run header): stated vs clue differ slightly in
inferential difficulty (not a pure paraphrase), so the absolute stated-clue divergence is NOT a clean
aliasing number on its own. The clean signal is the CROSS-GROUP comparison — is decoupled's mean
stated-clue divergence smaller than coupled's and base's?
"""
import importlib.util as ilu
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent


def _by_path(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


# Single-source the (already-tested) Markov JS/KL from the vendored state_metrics module.
_sm = _by_path("se_state_metrics", _HERE / "state_metrics.py")


def dist_dict(logits_tensor, letters):
    """Softmax the per-option logits into a next-step distribution NAMED by the item's own letters.

    logits_tensor: 1-D tensor of length len(letters) (one logit per option, e.g. from
                   cp_core.option_logits). letters: the item's option letters, e.g. ['A','B','C'].
    Returns {letter: prob}: a valid pmf over the letters (sums to 1, aligned to `letters`), the
    form markov_js / markov_kl consume.
    """
    t = torch.as_tensor(logits_tensor).detach().float().reshape(-1)
    if t.numel() != len(letters):
        raise ValueError(
            f"dist_dict: {t.numel()} logits but {len(letters)} letters ({letters}) — must match")
    p = torch.softmax(t, 0).tolist()
    return {L: float(p[i]) for i, L in enumerate(letters)}


def pair_divergence(p_stated_dict, p_clue_dict):
    """Divergence between the two history views' next-step distributions over the SAME letters.

    Thin wrapper: routes to the upstream, already-tested markov_js / markov_kl. Returns
      {'js': markov_js(stated, clue), 'kl': markov_kl(stated, clue)}  (both in nats).
    js is symmetric and 0 iff the two distributions are identical (the Markov ideal); kl is the
    directional KL(stated || clue). Larger = more semantic aliasing between the two histories.
    """
    return {
        "js": float(_sm.markov_js(p_stated_dict, p_clue_dict)),
        "kl": float(_sm.markov_kl(p_stated_dict, p_clue_dict)),
    }
