import importlib.util as ilu
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = ilu.spec_from_file_location("metrics", HERE.parent / "metrics.py")
M = ilu.module_from_spec(spec); spec.loader.exec_module(M)

def test_tv_basic():
    assert abs(M.tv([0.5, 0.5, 0.0], [0.5, 0.5, 0.0])) < 1e-12
    assert abs(M.tv([1.0, 0.0], [0.0, 1.0]) - 1.0) < 1e-12

def test_survivor_mass_and_uniformity():
    surv = np.array([True, True, False])
    assert abs(M.survivor_mass([0.4, 0.4, 0.2], surv) - 0.8) < 1e-9
    assert M.within_uniformity_tv([0.4, 0.4, 0.2], surv) < 1e-9          # uniform among survivors
    assert M.within_uniformity_tv([0.7, 0.1, 0.2], surv) > 0.2          # collapsed

def test_kl_zero_at_match():
    assert abs(M.kl([0.5, 0.5], [0.5, 0.5])) < 1e-9

def test_boot_ci_deterministic():
    a = M.boot_ci([0.1, 0.2, 0.3, 0.4], seed=0)
    b = M.boot_ci([0.1, 0.2, 0.3, 0.4], seed=0)
    assert a == b and a[0] <= a[1] <= a[2]
