"""TDD tests for the elicitation probe. No model is loaded: the elicitation methods are pure
prompt-string functions, the ground-truth consistency helper is pure logic, and consistency_auc
is exercised with fake scorers (a perfect oracle and a constant). This file is the ONLY place that
knows about elicit_probe's internals from the test side; elicit_probe.py must not import from here."""
import importlib.util as ilu, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = ilu.spec_from_file_location("elicit_probe", HERE.parent / "elicit_probe.py")
EP = ilu.module_from_spec(spec); spec.loader.exec_module(EP)
gspec = ilu.spec_from_file_location("gen", HERE.parent / "gen_data.py")
gen = ilu.module_from_spec(gspec); gspec.loader.exec_module(gen)


def _item():
    return gen.make_item(random.Random(0), 3, 3, 2, 0)


def _diff_constraint(item):
    return next(e for e in item["constraints"] if e["relation"] == "different")


def _same_constraint(item):
    return next((e for e in item["constraints"] if e["relation"] == "same"), None)


def test_M1_different_prompt_shape():
    it = _item()
    e = _diff_constraint(it)
    a, b = 0, 1                                   # a != b on a 'different' constraint
    p = EP.prompt_M1(it, e, a, b)
    na = it["options"][e["i"]][a]
    nb = it["options"][e["j"]][b]
    assert na in p and nb in p                    # both nouns named
    assert "different" in p.lower()               # rule stated as identity, not "kind"
    assert "kind" not in p.lower()                # the ambiguous word is gone
    assert "yes" in p.lower() and "no" in p.lower()   # asks yes/no


def test_M1_same_prompt_shape():
    it = _item()
    e = _same_constraint(it)
    if e is None:                                 # this draw had no 'same' constraint; build one synthetically
        e = dict(it["constraints"][0]); e["relation"] = "same"
    a, b = 0, 0
    p = EP.prompt_M1(it, e, a, b)
    assert "same" in p.lower()
    assert "kind" not in p.lower()
    assert "yes" in p.lower() and "no" in p.lower()


def test_ground_truth_consistency_all_pairs():
    # different => consistent iff a != b ; same => consistent iff a == b. Test every (a,b) for k=3.
    k = 3
    for a in range(k):
        for b in range(k):
            assert EP.is_consistent("different", a, b) == (a != b)
            assert EP.is_consistent("same", a, b) == (a == b)


def test_auc_perfect_scorer_is_one():
    rng = random.Random(0)
    items = [gen.make_item(rng, 3, 3, 2, i) for i in range(4)]

    def perfect(prompt):
        # the perfect scorer reads the encoded truth off the prompt tag the harness attaches
        return EP._truth_of(prompt)

    auc, mc, mi, n = EP.consistency_auc(perfect, items, EP._tagged(EP.prompt_M1))
    assert auc == 1.0
    assert mc > mi
    assert n > 0


def test_auc_constant_scorer_is_half():
    rng = random.Random(0)
    items = [gen.make_item(rng, 3, 3, 2, i) for i in range(4)]
    auc, mc, mi, n = EP.consistency_auc(lambda p: 7.0, items, EP._tagged(EP.prompt_M1))
    assert auc == 0.5
    assert mc == mi == 7.0


def test_M2_has_two_examples_plus_question():
    it = _item()
    e = _diff_constraint(it)
    p = EP.prompt_M2(it, e, 0, 1)
    # exactly two worked-example answer lines, then the live question (no trailing "Answer:" baked in)
    assert p.count("Answer: yes") + p.count("Answer: no") == 2
    # the live M1 query is appended
    assert "kind" not in p.lower()
    assert "yes or no" in p.lower() or ("yes" in p.lower() and "no" in p.lower())


def test_M0_reproduces_edges_query():
    # M0 must be byte-identical to edges._query (the baseline to reproduce).
    espec = ilu.spec_from_file_location("edges", HERE.parent / "edges.py")
    E = ilu.module_from_spec(espec); espec.loader.exec_module(E)
    it = _item()
    e = it["constraints"][0]
    assert EP.prompt_M0(it, e, 0, 1) == E._query(it, e, 0, 1)
