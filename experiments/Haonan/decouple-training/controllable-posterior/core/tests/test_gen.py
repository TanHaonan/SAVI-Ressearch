import importlib.util as ilu, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
G = ilu.spec_from_file_location("gen", HERE.parent / "gen_data.py")
gen = ilu.module_from_spec(G); G.loader.exec_module(gen)

def _items(k, j, n=8):
    rng = random.Random(0)
    return [gen.make_item(rng, k, j, i) for i in range(n)]

def test_target_is_uniform_over_survivors():
    for k in (2, 3, 4, 5):
        for j in range(1, k + 1):
            for it in _items(k, j):
                t = it["target"]
                assert abs(sum(t.values()) - 1.0) < 1e-9
                surv = [L for L in it["letters"] if t[L] > 0]
                assert len(surv) == j == len(it["survivors"])
                for L in surv:
                    assert abs(t[L] - 1.0 / j) < 1e-9

def test_determinate_is_one_hot():
    for it in _items(4, 1):
        assert max(it["target"].values()) == 1.0

def test_clue_negates_exactly_the_eliminated_nouns():
    for it in _items(5, 2):
        # surviving nouns must NOT be negated; eliminated nouns MUST be negated
        for L, noun in zip(it["letters"], it["nouns"]):
            negated = f"not the {noun}" in it["prompt_clue"]
            assert negated == (it["target"][L] == 0.0)

def test_stated_names_exactly_survivors():
    for it in _items(5, 3):
        for L, noun in zip(it["letters"], it["nouns"]):
            named = noun in it["prompt_stated"].split("equally likely")[-1]
            assert named == (it["target"][L] > 0.0)

def test_options_are_distinct_and_prior_free():
    for it in _items(5, 2):
        assert len(set(it["nouns"])) == it["k"]   # distinct nouns

def test_split_disjoint():
    rng = random.Random(1)
    ids = [gen.make_item(rng, 3, 2, i)["id"] for i in range(50)]
    assert len(ids) == len(set(ids))
