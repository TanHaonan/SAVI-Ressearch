"""TDD hand cases for the answer extractors. Run: pytest tests/test_answer_phi.py -q"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.answer_phi import (
    gsm8k_gold,
    extract_numeric,
    math500_gold,
    extract_boxed,
    normalize_num,
)


# ---- gsm8k_gold: integer after '####' ----
def test_gsm8k_gold_basic():
    assert gsm8k_gold("She makes ... = 18.\n#### 18") == "18"


def test_gsm8k_gold_with_commas():
    assert gsm8k_gold("...\n#### 1,024") == "1024"


def test_gsm8k_gold_negative():
    assert gsm8k_gold("...\n#### -5") == "-5"


# ---- extract_numeric: model's final numeric answer from free text ----
def test_extract_numeric_answer_is_phrase_with_comma():
    assert extract_numeric("First A, then B, so the answer is 1,024.") == "1024"


def test_extract_numeric_hash_marker():
    assert extract_numeric("work work\n#### 42") == "42"


def test_extract_numeric_boxed():
    assert extract_numeric("therefore \\boxed{42}") == "42"


def test_extract_numeric_dollar_and_commas():
    assert extract_numeric("The total is $1,200 in the end.") == "1200"


def test_extract_numeric_last_number_fallback():
    assert extract_numeric("step gives 7 then 13") == "13"


def test_extract_numeric_none_when_no_number():
    assert extract_numeric("there is no numeric answer here") is None


def test_extract_numeric_decimal():
    assert extract_numeric("the answer is 3.5") == "3.5"


# ---- math500_gold / extract_boxed ----
def test_math500_gold_passthrough_normalized():
    assert math500_gold("42") == "42"


def test_extract_boxed_basic():
    assert extract_boxed("so \\boxed{42}") == "42"


def test_extract_boxed_last_wins():
    assert extract_boxed("\\boxed{1} ... final \\boxed{7}") == "7"


def test_extract_boxed_nested_braces():
    assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_boxed_none():
    assert extract_boxed("no box at all") is None


# ---- normalize_num ----
def test_normalize_num_strips_commas_dollar():
    assert normalize_num("$1,024") == "1024"


def test_normalize_num_trailing_zeros_decimal():
    # 3.50 and 3.5 should compare equal
    assert normalize_num("3.50") == normalize_num("3.5")


def test_normalize_num_int_float_equiv():
    assert normalize_num("18.0") == normalize_num("18")
