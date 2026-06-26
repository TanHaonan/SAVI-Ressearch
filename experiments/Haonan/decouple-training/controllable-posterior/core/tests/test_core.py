import importlib.util as ilu
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
spec = ilu.spec_from_file_location("core", HERE.parent / "core.py")
core = ilu.module_from_spec(spec); spec.loader.exec_module(core)

def test_decoupled_zero_at_match():
    t = torch.tensor([0.5, 0.5, 0.0])
    lg = torch.log(t + 1e-9)                       # softmax(lg) ~= t
    assert core.loss_decoupled(lg, t).item() < 1e-3
    lg2 = torch.tensor([5.0, 0.0, 0.0])            # collapsed
    assert core.loss_decoupled(lg2, t).item() > 0.3

def test_coupled_is_ce_to_argmax():
    t = torch.tensor([0.0, 0.7, 0.3]); lg = torch.tensor([1.0, 2.0, 0.5])
    import torch.nn.functional as F
    expect = F.cross_entropy(lg[None], torch.tensor([1]))
    assert abs(core.loss_coupled(lg, t).item() - expect.item()) < 1e-5

def test_target_vec_matches_dict():
    item = {"letters": ["A", "B", "C"], "target": {"A": 0.0, "B": 0.5, "C": 0.5}}
    v = core.target_vec(item, "cpu")
    assert torch.allclose(v, torch.tensor([0.0, 0.5, 0.5]))

def test_letter_ids_dedup():
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return {"A": [10], " A": [10], "B": [11], " B": [12]}[s]
    ids = core.letter_ids(Tok(), 2)
    assert ids[0] == [10] and sorted(ids[1]) == [11, 12]

def test_option_logits_shape_and_differentiable():
    # FAKE tokenizer: encode() feeds letter_ids; apply_chat_template() feeds C.chat_prefix_ids.
    # No 4B load, CPU only.
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return {"A": [3], " A": [3], "B": [5], " B": [6], "C": [9], " C": [9]}[s]
        def apply_chat_template(self, msgs, add_generation_prompt=True,
                                return_tensors="pt", return_dict=False, **kw):
            return torch.tensor([[1, 2, 4]])           # 1 x T LongTensor

    T, V = 3, 12
    leaf = torch.randn(1, T, V, requires_grad=True)    # leaf the grad must reach

    class Out:
        def __init__(self, logits): self.logits = logits
    def fake_model(ids):
        return Out(leaf)                               # logits independent of ids, but a leaf

    tok = Tok()
    lids = core.letter_ids(tok, 3)                     # [[3], [5, 6], [9]]
    out = core.option_logits(fake_model, tok, "prompt", 3, lids, "cpu")
    assert out.shape == (3,)
    assert out.requires_grad
    out.sum().backward()
    assert leaf.grad is not None and leaf.grad.abs().sum() > 0   # logsumexp preserves grad path
