"""Sanity tests. Run: python -m pytest tests -q"""
import os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import know_decide as K


def test_q_sense_identity():
    """Design identity: gold=='yes' iff cued==q_sense (so DECIDE-correct reduces to sense-correct)."""
    blob = K.load_blob(); meta = blob["meta"]
    for m in meta:
        qs = K.q_sense_of(m["id"], m["cued"], m["gold"])
        assert (m["gold"] == "yes") == (m["cued"] == qs)


def test_decide_correct_is_sense_correct():
    """answer (yes iff KNOW_sense==q_sense) is correct iff KNOW_sense==cued — regardless of q_sense (no leak)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        cued = int(rng.integers(2)); gold = "yes" if rng.random() < 0.5 else "no"
        qs = 0 if K.q_sense_of("w_a", cued, gold) == "a" else 1
        gold_yes = 1 if gold == "yes" else 0
        # note: q_sense_of ignores the id for {a,b}-suffix-as-cued; recompute gold_yes consistently
        gold_yes = int(cued == qs)
        logits = torch.tensor([[rng.standard_normal(), rng.standard_normal()]])
        p_yes = float(K.decide(logits, [qs])[0])
        ans = int(p_yes > 0.5)
        know_sense = int(logits.argmax(1)[0])
        assert (ans == gold_yes) == (know_sense == cued) or gold_yes != int(cued == qs)


def test_auc_and_ece_bounds():
    s = np.array([0.1, 0.4, 0.35, 0.8]); y = np.array([0, 0, 1, 1])
    assert 0.0 <= K._auc(s, y) <= 1.0
    p = np.array([0.9, 0.1, 0.8, 0.2]); g = np.array([1, 0, 1, 0])
    assert 0.0 <= K.ece(p, g) <= 1.0
    assert K.ece(np.array([1.0, 0.0]), np.array([1, 0])) < 1e-6   # perfectly calibrated hard preds


def test_temperature_preserves_argmax():
    """DECIDE temperature is monotone -> argmax/AUC invariant (the decoupling guarantee)."""
    rng = np.random.default_rng(1)
    logits = torch.tensor(rng.standard_normal((20, 2)), dtype=torch.float)
    q = list(rng.integers(0, 2, 20))
    a1 = (K.decide(logits, q, T=0.5) > 0.5).int()
    a2 = (K.decide(logits, q, T=4.0) > 0.5).int()
    assert torch.equal(a1, a2)
