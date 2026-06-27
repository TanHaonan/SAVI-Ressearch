# sampling-diversity/core/tests/test_sample.py
import importlib.util as ilu
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
S = ilu.spec_from_file_location("sample", HERE.parent / "sample.py")
sample = ilu.module_from_spec(S); S.loader.exec_module(sample)

def _entropy(idx, k):
    c = np.bincount(idx, minlength=k).astype(float); p = c / c.sum()
    p = p[p > 0]; return float(-(p * np.log(p)).sum())

def test_multinomial_sample_deterministic():
    lg = torch.tensor([2.0, 1.0, 0.0, -1.0])
    a = sample.multinomial_sample(lg, 50, 1.0, seed=0)
    b = sample.multinomial_sample(lg, 50, 1.0, seed=0)
    assert a == b and len(a) == 50

def test_temperature_increases_entropy():
    lg = torch.tensor([4.0, 1.0, 0.0, -2.0])
    hot = _entropy(sample.multinomial_sample(lg, 4000, 1.5, seed=1), 4)
    cold = _entropy(sample.multinomial_sample(lg, 4000, 0.5, seed=1), 4)
    assert hot >= cold + 0.1

def test_freeform_strips_letter_tail():
    item = {"prompt_stated": "Foo bar. Which option hides the prize? Answer with one letter.",
            "prompt_clue":   "Foo clue. Which option hides the prize? Answer with one letter."}
    ff = sample.freeform_prompt(item, "oracle")
    assert "Answer with one letter" not in ff and len(ff) > 0

def test_letter_prompt_preserves_tail():
    item = {"prompt_stated": "X. Answer with one letter.", "prompt_clue": "Y. Answer with one letter."}
    assert "Answer with one letter" in sample.letter_prompt(item, "oracle")
