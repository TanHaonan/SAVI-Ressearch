import importlib.util as ilu, os, random, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_deps"))
G = ilu.spec_from_file_location("gen", HERE.parent / "gen_data.py")
gen = ilu.module_from_spec(G); G.loader.exec_module(gen)
import oracle  # noqa


def _items(T, k, C, n=8):
    rng = random.Random(0)
    return [gen.make_item(rng, T, k, C, i) for i in range(n)]


def test_gold_is_unique_exact_map():
    # NOTE (oracle-API adaptation): oracle.exact() does NOT return a "map_unique"
    # flag, and it requires every long-range edge to be a hub edge (i==0). The
    # generator restricts constraint pairs to (adjacent | hub) so every Instance
    # is valid for BOTH oracle.exact and oracle.brute_force. Uniqueness is
    # detected via gen.is_unique_map (strict top-vs-runnerup energy gap on the
    # brute-force enumeration). Correctness is pinned by exact==brute_force.
    for T in (3, 4):
        for k in (3, 4):
            for it in _items(T, k, C=2):
                inst = gen.oracle_instance(it)          # oracle unary + oracle edges
                ex = oracle.exact(inst); bf = oracle.brute_force(inst)
                assert list(ex["map"]) == list(bf["map"])      # solver agrees with brute
                assert list(ex["map"]) == it["gold"]           # gold == MAP
                assert gen.is_unique_map(inst, it["gold"]) is True  # unique optimum


def test_reverse_bite_present():
    # per-slot argmax of the LOCAL-BIAS unary must disagree with gold somewhere
    for it in _items(4, 4, C=2, n=16):
        local = np.argmax(np.array(it["local_unary"]), axis=1).tolist()
        assert local != it["gold"]


def test_constraint_statements_name_their_slots():
    for it in _items(4, 3, C=3):
        for e in it["constraints"]:
            s = e["text"].lower()
            assert it["slot_names"][e["i"]].lower() in s
            assert it["slot_names"][e["j"]].lower() in s
            assert e["relation"] in ("same", "different")


def test_options_distinct_per_slot():
    for it in _items(4, 4, C=2):
        for opts in it["options"]:
            assert len(set(opts)) == it["k"]


def test_compat_matrix_matches_relation():
    for it in _items(3, 3, C=2):
        for e in it["constraints"]:
            M = np.array(e["compat"])               # (k,k) hard log-potential
            # 'same' => 0 on diagonal, penalty off; 'different' => opposite
            diag_ok = np.allclose(np.diag(M), 0.0)
            assert diag_ok == (e["relation"] == "same")


def test_split_disjoint():
    rng = random.Random(1)
    ids = [gen.make_item(rng, 3, 3, 2, i)["id"] for i in range(40)]
    assert len(ids) == len(set(ids))
