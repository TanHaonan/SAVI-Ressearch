# sampling-diversity/core/phi.py
"""Phi: map a free-text continuation to a committed option-state. Primary = deterministic parser.
Secondary = an external LLM (built by run_diversity via C.answer_logprobs) as a decorrelated cross-check,
mirroring native-abstain/judge.py. This module stays model-free (pure functions) so it is unit-testable."""
import re

HEDGE = re.compile(r"\b(cannot tell|can't tell|cannot determine|either one|either could|both|ambiguous|"
                   r"not sure|unclear|depends|any of them|impossible to tell)\b", re.I)

def phi_parse(text, item):
    letters, nouns = item["letters"], item["nouns"]
    t = text.strip()
    found = set()
    for i, L in enumerate(letters):
        if (re.search(rf"\boption\s+\(?{L}\b", t, re.I)                       # "option B" / "option (b)"
                or re.search(rf"\b(?:answer|choice|letter)\s*[:=]\s*\(?{L}\b", t, re.I)  # "answer: b", "choice = C"
                or re.search(rf"\bthe\s+answer\s+is\s+\(?{L}\b", t, re.I)      # "the answer is c"
                or re.search(rf"(?<![A-Za-z]){L}(?=[\).:,]|$)", t)            # bare UPPERCASE label: "A)", "B.", "C," or "B"<EOL>
                or re.search(rf"\b{re.escape(nouns[i])}\b", t, re.I)):        # the option's distinctive noun
            found.add(L)
    if len(found) == 1:
        return next(iter(found))
    if len(found) > 1:
        return "multi"
    if HEDGE.search(t):
        return "hedge"
    return "none"

def phi_llm_classes(item):
    """Variant sets for the external-LLM Phi: one class per option letter + H(hedge) + Z(none)."""
    vs = {L: [L, " " + L] for L in item["letters"]}
    vs["H"] = ["H", " H"]; vs["Z"] = ["Z", " Z"]
    return vs

def phi_llm_prompt(text, item):
    opts = "; ".join(f"{L}) the {n}" for L, n in zip(item["letters"], item["nouns"]))
    return ("A model was asked which option hides a prize. The options are: " + opts + ".\n"
            f'The model answered: "{text}"\n\n'
            "Classify which single option the model commits to with ONE letter from "
            f"{','.join(item['letters'])}. If it hedges/says ambiguous, answer H. "
            "If it commits to nothing, answer Z. Answer with only one letter.")
