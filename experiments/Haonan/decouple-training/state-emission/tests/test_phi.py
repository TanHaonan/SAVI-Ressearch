# tests/test_phi.py
"""Unit tests for the free-text carrier (core/carrier.py) and the semantic-merge Phi (core/phi.py).

These observe the two model-free modules externally (no model load):
  - carrier.make_item must keep the controllable-posterior elimination semantics
    (target uniform over the j survivors, survivors/letters/nouns/prompt_stated/prompt_clue)
    IDENTICAL while ADDING two free-text prompt views (prompt_b1 with a deterministic COMMIT slot,
    prompt_b2 fully free).
  - phi_b1 is the deterministic structured-commit map: ONLY the `COMMIT: <noun>` line decides the
    committed letter. Eliminated nouns mentioned in the free reasoning must be ignored. It returns the
    survivor's letter, or 'hedge' / 'none' / 'multi'.
  - audit_false_merge flags a judged/parsed state that is a letter NOT in the item's known
    survivor set (false commit), since each item is constructed with ground truth.
The free-text external-reading surface (phi_b2_llm_prompt / phi_b2_llm_classes) is checked for shape
only; the external model itself is invoked by the integration pipeline, not here.
"""
import importlib.util as ilu
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bp(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


CAR = bp("carrier", HERE.parent / "core/carrier.py")
PHI = bp("phi", HERE.parent / "core/phi.py")


def _item():
    return CAR.make_item(random.Random(0), k=3, j=2, idx=0)


def _surv_noun(it, which=0):
    """The noun string of the `which`-th survivor letter."""
    surv = it["survivors"][which]
    return it["nouns"][it["letters"].index(surv)]


def _elim_noun(it, which=0):
    return [n for L, n in zip(it["letters"], it["nouns"]) if L not in it["survivors"]][which]


# ---- carrier ----
def test_carrier_target_valid():
    it = _item()
    assert abs(sum(it["target"].values()) - 1.0) < 1e-9
    assert len(it["survivors"]) == 2
    assert "prompt_b1" in it and "prompt_b2" in it


def test_carrier_keeps_legacy_views():
    """run.py trains on prompt_stated / prompt_clue unchanged — they must still be present."""
    it = _item()
    for key in ("prompt_stated", "prompt_clue", "letters", "nouns", "survivors", "target", "k", "j", "id"):
        assert key in it
    # target is uniform over the j survivors, 0 elsewhere
    j = it["j"]
    for L in it["letters"]:
        if L in it["survivors"]:
            assert abs(it["target"][L] - 1.0 / j) < 1e-9
        else:
            assert it["target"][L] == 0.0


def test_carrier_b1_has_commit_instruction():
    it = _item()
    assert "COMMIT:" in it["prompt_b1"]
    assert "COMMIT:" not in it["prompt_b2"]


# ---- phi_b1 ----
def test_b1_commit_ignores_eliminated_mentions():
    it = _item()
    surv0 = it["survivors"][0]
    noun0 = _surv_noun(it, 0)
    elim = _elim_noun(it, 0)
    text = f"It's not the {elim}, that was ruled out. COMMIT: the {noun0}"
    assert PHI.phi_b1(text, it) == surv0


def test_b1_commit_negation_in_reasoning_does_not_flip():
    """A reasoning sentence like 'It's not the keys' must not flip the committed state."""
    it = _item()
    surv0 = it["survivors"][0]
    noun0 = _surv_noun(it, 0)
    surv1_noun = _surv_noun(it, 1)
    text = (f"It's not the {surv1_noun}, I ruled that out earlier even though it survived. "
            f"COMMIT: {noun0}")
    assert PHI.phi_b1(text, it) == surv0


def test_b1_commit_surface_variants():
    """Article / plural / adjective-prefixed minor variants of the committed noun still map."""
    it = _item()
    surv0 = it["survivors"][0]
    noun0 = _surv_noun(it, 0)
    for surface in (f"the {noun0}", f"{noun0}s", f"a shiny {noun0}", f"  THE {noun0.upper()}  "):
        text = f"Reasoning here.\nCOMMIT: {surface}"
        assert PHI.phi_b1(text, it) == surv0, surface


def test_b1_hedge():
    it = _item()
    assert PHI.phi_b1("I cannot tell, it's ambiguous.", it) == "hedge"


def test_b1_hedge_in_commit_line():
    it = _item()
    assert PHI.phi_b1("Some reasoning.\nCOMMIT: cannot tell, it's ambiguous", it) == "hedge"


def test_b1_none_when_no_commit_line():
    it = _item()
    noun0 = _surv_noun(it, 0)
    # mentions a survivor noun in reasoning but never commits -> must be 'none', not the letter
    assert PHI.phi_b1(f"I think it might be the {noun0}, but I won't say.", it) == "none"


def test_b1_multi_when_two_survivors_committed():
    it = _item()
    n0 = _surv_noun(it, 0)
    n1 = _surv_noun(it, 1)
    text = f"Reasoning.\nCOMMIT: the {n0} and the {n1}"
    assert PHI.phi_b1(text, it) == "multi"


def test_b1_commit_to_eliminated_noun_is_a_letter_not_survivor():
    """If the model commits to an eliminated thing, phi_b1 returns that (wrong) letter so the
    downstream audit can flag a false commit. (It is a valid letter, just not a survivor.)"""
    it = _item()
    elim_noun = _elim_noun(it, 0)
    elim_letter = it["nouns"].index(elim_noun)
    elim_letter = it["letters"][elim_letter]
    text = f"Reasoning.\nCOMMIT: the {elim_noun}"
    assert PHI.phi_b1(text, it) == elim_letter
    assert elim_letter not in it["survivors"]


# ---- free-text external-reading surface (shape only) ----
def test_b2_llm_classes_cover_letters_plus_hedge_none():
    it = _item()
    classes = PHI.phi_b2_llm_classes(it)
    assert set(it["letters"]).issubset(set(classes))
    assert "H" in classes and "Z" in classes


def test_b2_llm_prompt_mentions_options_and_text():
    it = _item()
    prompt = PHI.phi_b2_llm_prompt("I think it is the apple.", it)
    assert "I think it is the apple." in prompt
    for n in it["nouns"]:
        assert n in prompt


def test_b2_parse_is_weak_secondary():
    it = _item()
    surv0 = it["survivors"][0]
    noun0 = _surv_noun(it, 0)
    # a clear free-text commitment to one survivor noun should be recoverable by the weak parser
    assert PHI.phi_b2_parse(f"The prize hides behind the {noun0}.", it) == surv0


# ---- audit ----
def test_audit_detects_false_merge():
    it = _item()
    wrong = [L for L in it["letters"] if L not in it["survivors"]][0]
    assert PHI.audit_false_merge(it, wrong)["false_commit"] is True


def test_audit_survivor_is_not_false_commit():
    it = _item()
    right = it["survivors"][0]
    assert PHI.audit_false_merge(it, right)["false_commit"] is False


def test_audit_abstain_is_not_false_commit():
    it = _item()
    for s in ("hedge", "none", "multi"):
        assert PHI.audit_false_merge(it, s)["false_commit"] is False
