"""Tests for the pairwise edge builder. No model is loaded: build_edges takes an injected
score_yes_no closure, so a trivial lambda stands in for the 4B/cheap scorer. The oracle source
(build_edges_from_truth) is tested directly rather than through a fragile stateful scorer."""
import importlib.util as ilu, random
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = ilu.spec_from_file_location("edges", HERE.parent / "edges.py")
E = ilu.module_from_spec(spec); spec.loader.exec_module(E)
gspec = ilu.spec_from_file_location("gen", HERE.parent / "gen_data.py")
gen = ilu.module_from_spec(gspec); gspec.loader.exec_module(gen)


def _item():
    return gen.make_item(random.Random(0), 3, 3, 2, 0)


def test_build_edges_shapes():
    it = _item()
    Ms = E.build_edges(it, score_yes_no=lambda p: 0.0)
    assert len(Ms) == len(it["constraints"])
    assert all(np.array(M).shape == (it["k"], it["k"]) for (_, _, M) in Ms)
    # edge identities (i,j) must match the constraints, in order
    assert [(i, j) for (i, j, _) in Ms] == [(e["i"], e["j"]) for e in it["constraints"]]


def test_oracle_recovered_when_scorer_is_truth():
    it = _item()
    Ms = E.build_edges_from_truth(it)
    assert len(Ms) == len(it["constraints"])
    for (i, j, M) in Ms:
        rel = next(e["relation"] for e in it["constraints"] if (e["i"], e["j"]) == (i, j))
        M = np.array(M)
        assert M.shape == (it["k"], it["k"])
        off = M[~np.eye(it["k"], dtype=bool)]
        if rel == "same":
            # 'same' => diagonal allowed (0), off-diagonal penalised => diag outscores off
            assert M.diagonal().mean() > off.mean()
        else:
            # 'different' => diagonal forbidden => off-diagonal outscores diag
            assert M.diagonal().mean() < off.mean()


def test_shuffle_is_deterministic_permutation():
    it = _item()
    a = E.shuffle_edges([it], seed=0)
    b = E.shuffle_edges([it], seed=0)
    assert E.edges_equal(a, b)


def test_shuffle_breaks_signal_in_general():
    # With several distinct items, shuffle should NOT reproduce each item's own oracle edges.
    rng = random.Random(3)
    items = [gen.make_item(rng, 3, 3, 2, i) for i in range(8)]
    truth = [E.build_edges_from_truth(it) for it in items]
    shuf = E.shuffle_edges(items, seed=0)
    # at least one item gets edges different from its own oracle (the control genuinely breaks signal)
    assert any(not E.edges_equal([t], [s]) for t, s in zip(truth, shuf))


def test_constant_scorer_is_flat():
    it = _item()
    Ms = E.build_edges(it, score_yes_no=lambda p: 1.234)
    for (_, _, M) in Ms:
        M = np.array(M)
        assert np.allclose(M, M.flat[0])
        assert np.allclose(M, 1.234)


def test_edge_phi_range():
    a = np.array([1, 0, 1, 0, 1])
    b = np.array([1, 0, 0, 0, 1])
    p = E.edge_phi(a, b)
    assert -1.0 <= p <= 1.0
