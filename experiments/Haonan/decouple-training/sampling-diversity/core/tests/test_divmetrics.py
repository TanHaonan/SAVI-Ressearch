# sampling-diversity/core/tests/test_divmetrics.py
import importlib.util as ilu
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = ilu.spec_from_file_location("divmetrics", HERE.parent / "divmetrics.py")
M = ilu.module_from_spec(D); D.loader.exec_module(M)

LETTERS = ["A", "B", "C", "D"]; SURV = ["A", "B", "C"]              # j=3 survivors
TARGET = {"A": 1/3, "B": 1/3, "C": 1/3, "D": 0.0}

def test_collapse_one_state():
    st = ["A"] * 12
    assert M.k_eff_distinct(st, LETTERS) == 1
    assert abs(M.coverage(st, SURV) - 1/3) < 1e-9
    assert abs(M.gen_state_tv(st, TARGET, LETTERS) - (1 - 1/3)) < 1e-9   # all mass on A vs uniform-3

def test_full_spread():
    st = ["A", "B", "C"] * 4
    assert M.k_eff_distinct(st, LETTERS) == 3
    assert abs(M.coverage(st, SURV) - 1.0) < 1e-9
    assert M.gen_state_tv(st, TARGET, LETTERS) < 1e-9
    assert abs(M.k_eff_entropy(st, LETTERS) - 3.0) < 1e-6

def test_eliminated_and_abstain():
    st = ["A", "D", "hedge", "none"]                                 # D is eliminated (not a survivor)
    assert abs(M.eliminated_mass(st, SURV, LETTERS) - 1/4) < 1e-9    # only D counts as committed-but-wrong
    assert abs(M.abstain_rate(st) - 2/4) < 1e-9                      # hedge+none
    assert M.distinct_text(["x", "x", "y"]) == 2
