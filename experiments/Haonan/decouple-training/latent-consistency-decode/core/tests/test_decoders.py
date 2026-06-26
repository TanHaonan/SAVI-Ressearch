import importlib.util as ilu, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_deps"))
spec = ilu.spec_from_file_location("dec", HERE.parent / "decoders.py")
D = ilu.module_from_spec(spec); spec.loader.exec_module(D)
import oracle


def _graph(theta, edges):
    return D.assemble(np.array(theta, float), edges, k=np.array(theta).shape[1])


def test_viterbi_matches_brute():
    theta = [[0.0, 1.0], [0.0, 0.9]]
    edges = [(0, 1, np.array([[0.0, oracle.HARD_PEN], [oracle.HARD_PEN, 0.0]]))]  # must be equal
    g = _graph(theta, edges)
    assert D.decode_viterbi(g) == list(oracle.brute_force(g.to_instance())["map"])


def test_marginal_matches_brute_mpm():
    theta = [[0.2, 0.1, 0.0], [0.0, 0.15, 0.1]]
    edges = []
    g = _graph(theta, edges)
    assert D.decode_marginal(g) == D.brute_mpm(g)


def test_marginal_equals_viterbi_when_sharp():
    theta = [[10.0, 0.0], [0.0, 10.0]]; edges = []
    g = _graph(theta, edges)
    assert D.decode_marginal(g) == D.decode_viterbi(g)


def test_greedy_ignores_edges():
    theta = [[0.0, 1.0], [1.0, 0.0]]
    g0 = _graph(theta, [])
    g1 = _graph(theta, [(0, 1, np.array([[0.0, oracle.HARD_PEN], [oracle.HARD_PEN, 0.0]]))])
    assert D.decode_greedy(g0) == D.decode_greedy(g1)


def test_edge_flips_local_greedy():
    # slot0 local prefers 1, slot1 local prefers 1; a 'different' edge should move the joint off (1,1)
    theta = [[0.0, 0.5], [0.0, 0.5]]
    diff = np.array([[0.0, 0.0], [0.0, oracle.HARD_PEN]])   # (1,1) forbidden
    g = _graph(theta, [(0, 1, diff)])
    assert D.decode_marginal(g) != [1, 1]
