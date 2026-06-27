"""Deterministic answer extractors (the answer-Phi) for the real-task headroom scout.

Two task families:
  - GSM8K: gold is the integer after '#### '; the model's answer is a number we pull from its
    free generation. Integer match is clean and is the primary signal.
  - MATH-500: gold is a (LaTeX) string; the model's answer is the last \\boxed{...}. Comparison is
    approximate exact-match after light normalization (whitespace / $ / \\left\\right / trailing
    zeros). This catches most but not all mathematically-equivalent forms, so MATH-500 accuracy is
    a lower bound, not a clean number.

Nothing here loads a model or dataset; all functions are pure and unit-tested.
"""
from __future__ import annotations

import re

# ----------------------------------------------------------------------------- numbers (GSM8K)

# A signed number with optional thousands separators and an optional decimal part.
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_num(s):
    """Canonical string form of a numeric answer for comparison.

    Strips '$', commas and surrounding whitespace; collapses an int-valued float to its integer
    ('18.0' -> '18') and drops trailing zeros in a real decimal ('3.50' -> '3.5'). Returns the
    cleaned token unchanged if it is not parseable as a number."""
    if s is None:
        return None
    t = str(s).strip().replace(",", "").replace("$", "").strip()
    if t == "":
        return None
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return t
    if "." in t:
        f = float(t)
        if f == int(f):
            return str(int(f))
        # drop trailing zeros but keep the value (e.g. 3.50 -> 3.5)
        return ("%f" % f).rstrip("0").rstrip(".")
    return str(int(t))


def _clean_number_token(tok):
    return normalize_num(tok)


def gsm8k_gold(answer_field):
    """The integer answer after the final '####' marker in a GSM8K answer field."""
    if answer_field is None:
        return None
    if "####" in answer_field:
        tail = answer_field.split("####")[-1]
    else:
        tail = answer_field
    m = _NUM_RE.search(tail)
    return _clean_number_token(m.group(0)) if m else None


def extract_numeric(text):
    """The model's final numeric answer from a free generation, normalized (or None).

    Preference order: an explicit '#### X' marker; an 'the answer is X' phrase; the last
    \\boxed{X} that contains a number; otherwise the last number in the text."""
    if text is None:
        return None

    # 1. '#### X'
    if "####" in text:
        m = _NUM_RE.search(text.split("####")[-1])
        if m:
            return _clean_number_token(m.group(0))

    # 2. 'the answer is X' / 'answer: X' (take the last such phrase, then the first number after)
    ans_phrases = list(
        re.finditer(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*", text, re.IGNORECASE)
    )
    if ans_phrases:
        after = text[ans_phrases[-1].end():]
        m = _NUM_RE.search(after)
        if m:
            return _clean_number_token(m.group(0))

    # 3. last \boxed{...} containing a number
    boxed = extract_boxed(text)
    if boxed is not None:
        m = _NUM_RE.search(boxed)
        if m:
            return _clean_number_token(m.group(0))

    # 4. fallback: last number anywhere
    nums = _NUM_RE.findall(text)
    if nums:
        return _clean_number_token(nums[-1])

    return None


# ----------------------------------------------------------------------------- boxed (MATH-500)


def _last_boxed(text):
    """Content of the last '\\boxed{...}' with balanced braces, or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None  # unbalanced


def _normalize_latex(s):
    """Light normalization for approximate exact-match of LaTeX answers."""
    if s is None:
        return None
    t = s.strip()
    # strip a wrapping $...$
    t = t.strip("$").strip()
    # drop spacing / sizing commands that don't change the value
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = t.replace("\\text{ }", " ")
    # remove a trailing period and collapse whitespace
    t = t.rstrip(".")
    t = re.sub(r"\s+", "", t)
    # numeric? canonicalize trailing zeros
    n = normalize_num(t)
    if n is not None and re.fullmatch(r"-?\d+(?:\.\d+)?", n):
        return n
    return t


def extract_boxed(text):
    """Content of the last '\\boxed{...}' in `text`, lightly normalized, or None."""
    if text is None:
        return None
    inner = _last_boxed(text)
    if inner is None:
        return None
    return _normalize_latex(inner)


def math500_gold(answer_field):
    """Normalized gold answer for a MATH-500 item (a LaTeX/string answer)."""
    if answer_field is None:
        return None
    return _normalize_latex(str(answer_field))
