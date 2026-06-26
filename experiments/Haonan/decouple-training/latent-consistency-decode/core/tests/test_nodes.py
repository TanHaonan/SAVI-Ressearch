import importlib.util as ilu
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
spec = ilu.spec_from_file_location("nodes", HERE.parent / "nodes.py")
N = ilu.module_from_spec(spec); spec.loader.exec_module(N)


def test_emission_target_keeps_gold_mass():
    # local bias favours option 1 but gold is 0; target must still give option 0 positive mass
    local = np.array([2.0, 5.0, 2.0]); t = N.emission_target(local, tau=1.0)
    assert abs(t.sum() - 1.0) < 1e-6 and t[0] > 0.02


def test_decoupled_zero_at_match():
    t = torch.tensor([0.5, 0.3, 0.2]); lg = torch.log(t + 1e-9)
    assert N.loss_decoupled(lg, t).item() < 1e-3
    assert N.loss_decoupled(torch.tensor([8.0, 0.0, 0.0]), t).item() > 0.3


def test_coupled_is_ce_to_argmax():
    import torch.nn.functional as F
    t = torch.tensor([0.0, 0.7, 0.3]); lg = torch.tensor([1.0, 2.0, 0.5])
    exp = F.cross_entropy(lg[None], torch.tensor([1]))
    assert abs(N.loss_coupled(lg, t).item() - exp.item()) < 1e-5


def test_letter_ids_present():
    assert hasattr(N, "letter_ids") and hasattr(N, "slot_logits")
