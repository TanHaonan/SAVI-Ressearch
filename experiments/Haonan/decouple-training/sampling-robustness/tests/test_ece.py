"""TDD for core/ece.py — the genuinely-new custom code of the robustness probe.

reliability_curve(confidences, correct, n_bins) -> list of (mean_conf, acc, count) per occupied bin.
ece(confidences, correct, n_bins) -> sum_k (count_k/N) * |acc_k - conf_k|.

Hand cases:
  perfectly-calibrated      -> ECE 0
  all-confident-all-wrong   -> ECE ~1
"""
import importlib.util as ilu
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ECE = HERE.parent / "core" / "ece.py"


def _by_path(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


ece_mod = _by_path("ece", ECE)


# --------------------------------------------------------------- ece() hand cases
def test_ece_perfectly_calibrated_is_zero():
    # In each confidence band the empirical accuracy EQUALS the confidence:
    # conf 0.0 -> 0% correct, conf 1.0 -> 100% correct. ECE must be exactly 0.
    confidences = [0.0] * 10 + [1.0] * 10
    correct = [0] * 10 + [1] * 10
    assert ece_mod.ece(confidences, correct, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_calibrated_midband_is_zero():
    # A 0.5-confidence band that is right exactly half the time is perfectly calibrated.
    confidences = [0.5] * 10
    correct = [1, 0] * 5
    assert ece_mod.ece(confidences, correct, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_all_confident_all_wrong_is_about_one():
    # Maximally confident (1.0) yet always wrong -> |acc - conf| = 1 in the top bin.
    confidences = [1.0] * 20
    correct = [0] * 20
    assert ece_mod.ece(confidences, correct, n_bins=10) == pytest.approx(1.0, abs=1e-9)


def test_ece_all_confident_all_right_is_zero():
    confidences = [1.0] * 20
    correct = [1] * 20
    assert ece_mod.ece(confidences, correct, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_known_intermediate_value():
    # One bin: 4 samples at conf 0.9, 2 correct -> acc 0.5. |0.5 - 0.9| = 0.4, weight 1.0 -> ECE 0.4.
    confidences = [0.9, 0.9, 0.9, 0.9]
    correct = [1, 1, 0, 0]
    assert ece_mod.ece(confidences, correct, n_bins=10) == pytest.approx(0.4, abs=1e-9)


def test_ece_weights_bins_by_count():
    # Bin A (conf 0.95): 8 samples, all correct -> |1.0-0.95|=0.05.
    # Bin B (conf 0.05): 2 samples, all wrong  -> |0.0-0.05|=0.05.
    # Both gaps 0.05 -> weighted mean 0.05 regardless of the count split.
    confidences = [0.95] * 8 + [0.05] * 2
    correct = [1] * 8 + [0] * 2
    assert ece_mod.ece(confidences, correct, n_bins=10) == pytest.approx(0.05, abs=1e-9)


def test_ece_empty_is_nan():
    assert np.isnan(ece_mod.ece([], [], n_bins=10))


# ----------------------------------------------------------- reliability_curve()
def test_reliability_curve_only_occupied_bins():
    confidences = [0.05, 0.95, 0.95]
    correct = [0, 1, 0]
    curve = ece_mod.reliability_curve(confidences, correct, n_bins=10)
    # exactly two occupied bins
    assert len(curve) == 2
    counts = sorted(c for (_, _, c) in curve)
    assert counts == [1, 2]


def test_reliability_curve_values():
    # bin [0.9,1.0]: confs 0.9,1.0 -> mean_conf 0.95, accs 1,0 -> acc 0.5, count 2.
    confidences = [0.9, 1.0]
    correct = [1, 0]
    curve = ece_mod.reliability_curve(confidences, correct, n_bins=10)
    assert len(curve) == 1
    mean_conf, acc, count = curve[0]
    assert mean_conf == pytest.approx(0.95, abs=1e-9)
    assert acc == pytest.approx(0.5, abs=1e-9)
    assert count == 2


def test_reliability_curve_total_count_preserved():
    rng = np.random.default_rng(0)
    confidences = rng.random(100).tolist()
    correct = rng.integers(0, 2, 100).tolist()
    curve = ece_mod.reliability_curve(confidences, correct, n_bins=10)
    assert sum(c for (_, _, c) in curve) == 100


def test_ece_matches_reliability_curve_definition():
    # ece() must equal sum_k (count_k/N)*|acc_k - mean_conf_k| over the curve.
    rng = np.random.default_rng(1)
    confidences = rng.random(200).tolist()
    correct = rng.integers(0, 2, 200).tolist()
    curve = ece_mod.reliability_curve(confidences, correct, n_bins=10)
    n = len(confidences)
    expected = sum((c / n) * abs(acc - mc) for (mc, acc, c) in curve)
    assert ece_mod.ece(confidences, correct, n_bins=10) == pytest.approx(expected, abs=1e-12)


def test_boundary_one_lands_in_top_bin():
    # conf exactly 1.0 must fall in the last bin, not overflow.
    curve = ece_mod.reliability_curve([1.0], [1], n_bins=10)
    assert len(curve) == 1
    assert curve[0][2] == 1


def test_boundary_zero_lands_in_bottom_bin():
    curve = ece_mod.reliability_curve([0.0], [0], n_bins=10)
    assert len(curve) == 1
    assert curve[0][2] == 1
