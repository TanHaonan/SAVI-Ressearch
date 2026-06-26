import importlib.util as ilu, json, random
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
spec = ilu.spec_from_file_location("run", HERE.parent / "run.py")
R = ilu.module_from_spec(spec); spec.loader.exec_module(R)
gspec = ilu.spec_from_file_location("gen", HERE.parent / "gen_data.py")
gen = ilu.module_from_spec(gspec); gspec.loader.exec_module(gen)

def test_evaluate_finite_on_tiny(monkeypatch):
    # a fake model whose answer-position logits are deterministic -> evaluate() must return finite TV
    rng = random.Random(0)
    items = [gen.make_item(rng, 3, 2, i) for i in range(4)]
    res = R.evaluate_with(lambda prompt, k: torch.zeros(k), items, "prompt_stated", device="cpu")
    assert all(torch.isfinite(torch.tensor(c["tv_mean"])) for c in res.values())
