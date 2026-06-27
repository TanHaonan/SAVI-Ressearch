"""TDD for core/aliasing.py — the thin custom helpers of the semantic-aliasing / Markov probe.

The heavy math (markov_js / markov_kl) is already unit-tested upstream in
state-emission/core/state_metrics.py; here we only test the two thin wrappers:

  * dist_dict(logits_tensor, letters) -> {letter: prob}: a softmax that names its bins by the
    item's own option letters. Must be a valid pmf (sums to 1) and aligned to `letters`.
  * pair_divergence(p_stated_dict, p_clue_dict) -> {'js':, 'kl':}: routes the two next-step
    distributions through state_metrics.markov_js / markov_kl. Must be 0 when the two histories
    give the SAME distribution (the Markov ideal) and > 0 when they put mass on different letters
    (measured aliasing).
"""
import importlib.util as ilu
import math
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "core"


def _by_path(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


al = _by_path("aliasing", CORE / "aliasing.py")


# ---------------------------------------------------------------- dist_dict
def test_dist_dict_sums_to_one_and_aligns_to_letters():
    logits = torch.tensor([2.0, 1.0, 0.0])
    letters = ["A", "B", "C"]
    d = al.dist_dict(logits, letters)
    assert set(d.keys()) == set(letters)            # one bin per letter, named by letter
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-6)
    # softmax is monotone in the logits: higher logit -> higher prob, in letter order
    assert d["A"] > d["B"] > d["C"] > 0.0


def test_dist_dict_matches_manual_softmax():
    logits = torch.tensor([1.0, 3.0])
    letters = ["A", "B"]
    d = al.dist_dict(logits, letters)
    p = torch.softmax(logits, 0).tolist()
    assert math.isclose(d["A"], p[0], abs_tol=1e-6)
    assert math.isclose(d["B"], p[1], abs_tol=1e-6)


def test_dist_dict_uniform_logits_give_uniform():
    logits = torch.zeros(4)
    letters = ["A", "B", "C", "D"]
    d = al.dist_dict(logits, letters)
    for L in letters:
        assert math.isclose(d[L], 0.25, abs_tol=1e-6)


def test_dist_dict_length_mismatch_raises():
    with pytest.raises((ValueError, AssertionError)):
        al.dist_dict(torch.tensor([1.0, 2.0, 3.0]), ["A", "B"])


# ------------------------------------------------------------ pair_divergence
def test_pair_divergence_zero_when_identical():
    """Markov ideal: two histories -> SAME next-step distribution -> zero divergence."""
    p = {"A": 0.7, "B": 0.2, "C": 0.1}
    out = al.pair_divergence(p, dict(p))
    assert math.isclose(out["js"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["kl"], 0.0, abs_tol=1e-9)


def test_pair_divergence_positive_on_different_mass():
    """Aliasing: the two views put mass on different letters -> strictly positive js and kl."""
    p = {"A": 0.9, "B": 0.05, "C": 0.05}
    q = {"A": 0.05, "B": 0.9, "C": 0.05}
    out = al.pair_divergence(p, q)
    assert out["js"] > 0.0
    assert out["kl"] > 0.0


def test_pair_divergence_js_symmetric_kl_not():
    """js is symmetric (a Markov-precondition score); kl is directional (stated || clue)."""
    p = {"A": 0.8, "B": 0.2}
    q = {"A": 0.3, "B": 0.7}
    js_pq = al.pair_divergence(p, q)["js"]
    js_qp = al.pair_divergence(q, p)["js"]
    assert math.isclose(js_pq, js_qp, abs_tol=1e-9)
    kl_pq = al.pair_divergence(p, q)["kl"]
    kl_qp = al.pair_divergence(q, p)["kl"]
    assert not math.isclose(kl_pq, kl_qp, abs_tol=1e-6)


def test_pair_divergence_hand_case():
    """Hand-checked JS for a maximally-disjoint 2-letter pair.
    p=(1,0), q=(0,1): mix=(.5,.5); JS = 0.5*KL(p||mix)+0.5*KL(q||mix)
       = 0.5*ln(1/.5) + 0.5*ln(1/.5) = ln 2 (nats)."""
    p = {"A": 1.0, "B": 0.0}
    q = {"A": 0.0, "B": 1.0}
    out = al.pair_divergence(p, q)
    assert math.isclose(out["js"], math.log(2.0), abs_tol=1e-6)


def test_pair_divergence_routes_through_state_metrics():
    """pair_divergence must agree exactly with the upstream (already-tested) markov_js/markov_kl,
    proving it is a thin wrapper and inherits their correctness."""
    sm = _by_path("se_state_metrics", CORE / "state_metrics.py")
    p = {"A": 0.6, "B": 0.3, "C": 0.1}
    q = {"A": 0.2, "B": 0.5, "C": 0.3}
    out = al.pair_divergence(p, q)
    assert math.isclose(out["js"], sm.markov_js(p, q), rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(out["kl"], sm.markov_kl(p, q), rel_tol=1e-12, abs_tol=1e-12)


# -------------------------------------------------------- end-to-end thin glue
def test_dist_dict_then_pair_divergence_zero_for_same_logits():
    """The two real call sites composed: identical logits under both views -> zero aliasing."""
    logits = torch.tensor([0.5, 2.0, -1.0])
    letters = ["A", "B", "C"]
    p = al.dist_dict(logits, letters)
    q = al.dist_dict(logits.clone(), letters)
    out = al.pair_divergence(p, q)
    assert math.isclose(out["js"], 0.0, abs_tol=1e-9)
