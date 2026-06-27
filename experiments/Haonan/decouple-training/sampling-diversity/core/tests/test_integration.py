# sampling-diversity/core/tests/test_integration.py
import importlib.util as ilu
from pathlib import Path
HERE = Path(__file__).resolve().parent
def _load(name, fn):
    s = ilu.spec_from_file_location(name, HERE.parent / fn); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m
phi = _load("phi", "phi.py"); M = _load("divmetrics", "divmetrics.py")

ITEM = {"letters": ["A", "B", "C"], "nouns": ["apple", "river", "cloud"], "survivors": ["A", "B", "C"],
        "target": {"A": 1/3, "B": 1/3, "C": 1/3}, "k": 3, "j": 3}

def _states(texts): return [phi.phi_parse(t, ITEM) for t in texts]

def test_collapsed_generation_cell():
    texts = ["option A"] * 9
    st = _states(texts)
    assert M.k_eff_distinct(st, ITEM["letters"]) == 1
    assert abs(M.coverage(st, ITEM["survivors"]) - 1/3) < 1e-9

def test_spread_generation_cell():
    texts = ["the apple", "the river", "the cloud"] * 3
    st = _states(texts)
    assert M.k_eff_distinct(st, ITEM["letters"]) == 3
    assert abs(M.coverage(st, ITEM["survivors"]) - 1.0) < 1e-9
    assert M.gen_state_tv(st, ITEM["target"], ITEM["letters"]) < 1e-9
