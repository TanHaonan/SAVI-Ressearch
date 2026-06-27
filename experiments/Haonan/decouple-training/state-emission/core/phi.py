"""Phi: map a free-text answer to a committed semantic state (an option letter, or an abstention).

This module is model-free (pure functions) so it is unit-testable; the external independent-model
reading for the free-text regime is invoked by the integration pipeline (which builds the classifier
from phi_b2_llm_prompt / phi_b2_llm_classes and the decorrelated model), never here.

Two regimes:
  - structured-commit (`b1`, deterministic, low noise): phi_b1 reads ONLY the `COMMIT: <thing>` line.
    Free reasoning above it may mention eliminated nouns ("It's not the keys") — that must NOT flip the
    committed state. Returns the survivor/option letter, or 'hedge' / 'none' / 'multi'.
  - free-text (`b2`, free paraphrase, Phi must merge by meaning): phi_b2_llm_* is the authoritative
    decorrelated reading surface (one class per letter + H hedge + Z none, mirroring sampling-diversity/phi.py);
    phi_b2_parse is a WEAK deterministic secondary used only for agreement reporting, never authoritative.

audit_false_merge compares a judged/parsed state to the item's KNOWN survivor set (each item is
constructed) and flags a false commit when the state is a letter that is not a survivor.
"""
import re

# Hedge / refusal cues (mirrors sampling-diversity/phi.py HEDGE, kept in sync intentionally).
HEDGE = re.compile(
    r"\b(cannot tell|can't tell|cannot determine|can't determine|either one|either could|both|"
    r"ambiguous|not sure|unclear|undecided|depends|any of them|impossible to tell|don't know|"
    r"do not know)\b",
    re.I,
)

# The deterministic structured-commit slot. The committed thing is everything AFTER `COMMIT:` up to the end
# of that physical line (or end of text). We do NOT anchor `COMMIT` to line start: the prompt asks the
# model to "end with a line exactly: COMMIT: <thing>", but models often write it inline after a
# sentence ("... ruled out. COMMIT: the cobweb"). Because the body begins after the colon, any
# eliminated nouns mentioned in the reasoning BEFORE `COMMIT:` are never part of the captured body.
# `[^\n]*` (not `.*?`) keeps the full remainder of the line; the trailing strip() trims whitespace.
_COMMIT_LINE = re.compile(r"COMMIT\s*:\s*(?P<body>[^\n]*)", re.I)


def _commit_body(text):
    """Return the content of the LAST `COMMIT:` line, or None if there is no commit line.

    Using the last commit line is deliberate: if the model restates COMMIT (e.g. a corrected final
    line) the final commitment wins; the free reasoning above the slot is never consulted."""
    matches = list(_COMMIT_LINE.finditer(text))
    if not matches:
        return None
    return (matches[-1].group("body") or "").strip()


def _noun_variants_in(span, nouns):
    """Letters whose noun appears in `span`, allowing minor surface variants:
    article ('the X' / 'a X'), simple plural ('Xs' / 'Xes'), and adjective-prefixed ('shiny X').
    We match each noun as a whole word with an optional trailing plural; articles/adjectives are
    simply other words in the span and need no special handling because we scan word-boundaried."""
    found = set()
    for i, n in enumerate(nouns):
        # whole-word noun with optional simple plural suffix
        if re.search(rf"\b{re.escape(n)}(?:s|es)?\b", span, re.I):
            found.add(i)
    return found


def phi_b1(text, item):
    """Deterministic structured-commit map. ONLY the COMMIT line decides.

    Returns:
      - the committed option letter (in item['letters']) when exactly one option's noun is on the
        commit line (it may be an eliminated option's letter — the downstream audit catches that);
      - 'hedge' when the commit line (or, absent a commit line is handled below) expresses hedging;
      - 'multi' when two or more DIFFERENT options' nouns are on the commit line;
      - 'none' when there is no commit line, or the commit line names no option and does not hedge.
    """
    letters, nouns = item["letters"], item["nouns"]
    body = _commit_body(text)
    if body is None:
        # No COMMIT line at all: nothing was committed in the slot. An explicit hedge/refusal in the
        # answer still counts as a 'hedge' (the model declined rather than committed); otherwise the
        # missing slot is 'none'. Eliminated nouns merely mentioned in reasoning never become a letter.
        return "hedge" if HEDGE.search(text) else "none"
    if HEDGE.search(body):
        return "hedge"
    found = _noun_variants_in(body, nouns)
    if len(found) == 1:
        return letters[next(iter(found))]
    if len(found) > 1:
        return "multi"
    # Commit line present but names no option noun and does not hedge -> nothing usable committed.
    return "none"


def phi_b2_llm_classes(item):
    """Variant token sets for the external-LLM free-text reading: one class per option letter + H(hedge) +
    Z(none). Mirrors sampling-diversity/phi.phi_llm_classes."""
    vs = {L: [L, " " + L] for L in item["letters"]}
    vs["H"] = ["H", " H"]
    vs["Z"] = ["Z", " Z"]
    return vs


def phi_b2_llm_prompt(text, item):
    """Prompt for the decorrelated external model to classify a free-text answer into one option letter
    / H / Z. Mirrors sampling-diversity/phi.phi_llm_prompt."""
    opts = "; ".join(f"{L}) the {n}" for L, n in zip(item["letters"], item["nouns"]))
    return ("A model was asked which option hides a prize. The options are: " + opts + ".\n"
            f'The model answered: "{text}"\n\n'
            "Classify which single option the model commits to with ONE letter from "
            f"{','.join(item['letters'])}. If it hedges/says ambiguous, answer H. "
            "If it commits to nothing, answer Z. Answer with only one letter.")


def phi_b2_parse(text, item):
    """Weak deterministic secondary for the free-text regime (committed-noun heuristic). Used ONLY for
    agreement reporting against the authoritative external-model reading, never as the free-text state itself.

    Unlike phi_b1 there is no commit slot, so this scans the whole free answer for option nouns /
    explicit letter labels (same surface cues as sampling-diversity/phi.phi_parse) and hedge cues.
    Deliberately conservative: returns 'multi' if more than one option is named."""
    letters, nouns = item["letters"], item["nouns"]
    t = text.strip()
    found = set()
    for i, L in enumerate(letters):
        if (re.search(rf"\boption\s+\(?{L}\b", t, re.I)
                or re.search(rf"\b(?:answer|choice|letter)\s*[:=]\s*\(?{L}\b", t, re.I)
                or re.search(rf"\bthe\s+answer\s+is\s+\(?{L}\b", t, re.I)
                or re.search(rf"(?<![A-Za-z]){L}(?=[\).:,]|$)", t)
                or re.search(rf"\b{re.escape(nouns[i])}(?:s|es)?\b", t, re.I)):
            found.add(L)
    if len(found) == 1:
        return next(iter(found))
    if len(found) > 1:
        return "multi"
    if HEDGE.search(t):
        return "hedge"
    return "none"


def audit_false_merge(item, judged_state):
    """Soundness audit against constructed ground truth.

    Each item is built with a known survivor set, so any committed LETTER that is NOT a survivor is a
    'false commit' (Phi merged the answer onto a wrong option). Abstentions ('hedge'/'none'/'multi')
    are not false commits. A correct commit lands on a survivor letter.

    Returns at least {'false_commit': bool}; also reports whether the state is a valid letter, a
    survivor, or an abstention, to ease aggregate false-merge / false-split reporting downstream.
    """
    letters = item["letters"]
    survivors = set(item["survivors"])
    is_letter = judged_state in letters
    is_survivor = judged_state in survivors
    is_abstain = judged_state in ("hedge", "none", "multi")
    false_commit = bool(is_letter and not is_survivor)
    return {
        "false_commit": false_commit,
        "is_letter": is_letter,
        "is_survivor": is_survivor,
        "is_abstain": is_abstain,
        "judged_state": judged_state,
    }
