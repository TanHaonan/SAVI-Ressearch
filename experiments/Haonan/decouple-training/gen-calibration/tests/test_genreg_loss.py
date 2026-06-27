"""TDD for core/genreg_loss.py — the genuinely-new code of the commit-slot lever.

These tests are deterministic and load NO model. They cover:
  - noun_ids: shapes/non-empty, merges "<noun>" and " <noun>" first-token ids;
  - loss_commit: == 0 when commit logits already match a uniform target in logit space,
                 > 0 otherwise; same KL form as loss_decoupled;
  - DEFAULT_TEMPLATE is the documented reasoning preamble.
"""
import importlib.util as ilu
import math
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
GL = HERE.parent / "core" / "genreg_loss.py"


def _load(name, path):
    s = ilu.spec_from_file_location(name, str(path))
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


gl = _load("genreg_loss", GL)


# ---- a tiny deterministic fake tokenizer (no model, no HF download) --------
class FakeTok:
    """Maps each whole string to a stable list of fake ids, one id per word. The first id of
    "noun" and of " noun" differ (leading-space variant), so noun_ids must merge BOTH variants.
    encode(...) just splits on spaces and hashes each token into a small id space."""
    VOCAB = 4096

    def _id(self, word):
        return (hash(word) % (self.VOCAB - 1)) + 1   # never 0

    def encode(self, s, add_special_tokens=False):
        toks = s.split(" ")
        # an empty leading split element (from a leading space) becomes a distinct space marker token
        ids = []
        for i, t in enumerate(toks):
            if t == "" and i == 0:
                ids.append(self._id("<sp>"))
            elif t == "":
                continue
            else:
                ids.append(self._id(t))
        return ids


@pytest.fixture
def tok():
    return FakeTok()


# ------------------------------------------------------------ noun_ids ------
def test_noun_ids_shapes_nonempty(tok):
    nouns = ["lantern", "cobweb", "harbor"]
    nid = gl.noun_ids(tok, nouns)
    assert isinstance(nid, list) and len(nid) == len(nouns)
    for ids in nid:
        assert isinstance(ids, list)
        assert len(ids) >= 1                     # at least one first-token id per noun
        assert all(isinstance(x, int) for x in ids)


def test_noun_ids_merges_space_variant(tok):
    """Must include the first-token id of BOTH "<noun>" and " <noun>"."""
    nouns = ["lantern"]
    nid = gl.noun_ids(tok, nouns)[0]
    plain_first = tok.encode("lantern", add_special_tokens=False)[0]
    space_first = tok.encode(" lantern", add_special_tokens=False)[0]
    assert plain_first in nid
    assert space_first in nid
    # the two variants differ for this fake tok -> set has both, de-duplicated and sorted
    assert nid == sorted(set(nid))


def test_noun_ids_dedup_when_variants_collide():
    """If a tokenizer yields the same first id for both variants, noun_ids must not duplicate it."""
    class CollideTok:
        def encode(self, s, add_special_tokens=False):
            return [7, 8, 9]                      # identical regardless of leading space
    nid = gl.noun_ids(CollideTok(), ["x"])[0]
    assert nid == [7]


# ----------------------------------------------------------- loss_commit ----
def test_loss_commit_zero_when_matches_uniform_target():
    """KL(target ‖ softmax(commit_lg)) == 0 when commit_lg is uniform (logits equal) and target is uniform."""
    k = 4
    commit_lg = torch.zeros(k)                    # equal logits -> uniform softmax
    target = torch.full((k,), 1.0 / k)
    loss = gl.loss_commit(commit_lg, target)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_loss_commit_zero_when_matches_partial_survivor_target():
    """Uniform-over-survivors target (some zeros): loss==0 iff softmax(commit_lg) equals it.
    Build commit_lg so its softmax is exactly [0.5,0,0.5,0] via large negative logits on zeros."""
    target = torch.tensor([0.5, 0.0, 0.5, 0.0])
    NEG = -1e4
    commit_lg = torch.tensor([0.0, NEG, 0.0, NEG])
    loss = gl.loss_commit(commit_lg, target)
    assert float(loss) == pytest.approx(0.0, abs=1e-4)


def test_loss_commit_positive_when_mismatch():
    k = 4
    target = torch.full((k,), 1.0 / k)
    commit_lg = torch.tensor([3.0, 0.0, 0.0, 0.0])   # peaked on option 0, target uniform
    loss = gl.loss_commit(commit_lg, target)
    assert float(loss) > 0.0


def test_loss_commit_matches_kl_form():
    """loss_commit must be the SAME KL form as loss_decoupled: KL(target ‖ softmax(lg))."""
    k = 3
    torch.manual_seed(0)
    commit_lg = torch.randn(k)
    target = torch.tensor([0.5, 0.5, 0.0])
    import torch.nn.functional as F
    expect = F.kl_div(F.log_softmax(commit_lg, 0), target, reduction="sum")
    got = gl.loss_commit(commit_lg, target)
    assert float(got) == pytest.approx(float(expect), abs=1e-6)


def test_loss_commit_gradient_flows():
    """The commit loss must be differentiable wrt the commit logits (used in training)."""
    k = 3
    commit_lg = torch.zeros(k, requires_grad=True)
    target = torch.tensor([0.5, 0.5, 0.0])
    loss = gl.loss_commit(commit_lg, target)
    loss.backward()
    assert commit_lg.grad is not None
    assert torch.isfinite(commit_lg.grad).all()


# --------------------------------------------------------- default template -
def test_default_template():
    assert gl.DEFAULT_TEMPLATE == " Let me weigh the remaining clues."
