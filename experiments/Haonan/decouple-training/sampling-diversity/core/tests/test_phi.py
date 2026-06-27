# sampling-diversity/core/tests/test_phi.py
import importlib.util as ilu
from pathlib import Path
HERE = Path(__file__).resolve().parent
P = ilu.spec_from_file_location("phi", HERE.parent / "phi.py")
phi = ilu.module_from_spec(P); P.loader.exec_module(phi)

ITEM = {"letters": ["A", "B", "C"], "nouns": ["apple", "river", "cloud"], "survivors": ["A", "C"]}

def test_parse_letter():       assert phi.phi_parse("I think option C.", ITEM) == "C"
def test_parse_noun():         assert phi.phi_parse("It is hidden behind the apple.", ITEM) == "A"
def test_parse_multi():        assert phi.phi_parse("Could be the apple or the cloud.", ITEM) == "multi"
def test_parse_hedge():        assert phi.phi_parse("It is ambiguous; either could be right.", ITEM) == "hedge"
def test_parse_none():         assert phi.phi_parse("The weather is nice today.", ITEM) == "none"
def test_llm_classes_cover():  assert set("ABC").issubset(set(phi.phi_llm_classes(ITEM))) and "H" in phi.phi_llm_classes(ITEM)

# Regression: lowercase committed letters in answer-labeled contexts must parse, the bare
# UPPERCASE label still parses, and the English article "a"/"A" must NOT count as option A.
def test_parse_lowercase_answer_label():     assert phi.phi_parse("My answer: b", ITEM) == "B"
def test_parse_the_answer_is_lower():        assert phi.phi_parse("So the answer is c).", ITEM) == "C"
def test_parse_article_a_not_committed():    assert phi.phi_parse("It could be a different trick.", ITEM) == "none"
def test_parse_sentence_initial_A_article(): assert phi.phi_parse("A clue eliminated the others, so it stays open.", ITEM) == "none"
def test_parse_bare_uppercase_label():       assert phi.phi_parse("B.", ITEM) == "B"
