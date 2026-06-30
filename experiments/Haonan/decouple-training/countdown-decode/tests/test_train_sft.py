"""L0 guard for the countdown SFT trainer: the masked PREFIX the LoRA is trained on must
be token-identical to the prompt the real backend feeds at inference (both route through
``real_backend.build_move_prompt``). A divergence here silently destroys the SFT signal.
Also a fake-tokenizer completion-only mask sanity check (no model, CPU).

Run:  HF_HUB_CACHE=<your local HF hub cache> pytest tests/test_train_sft.py
"""
import importlib.util as ilu
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import core  # noqa: E402  -- sets up _deps/core on sys.path so train_sft's
            # ``from real_backend import build_move_prompt`` resolves at exec time.

_TRAIN = _ROOT / "core" / "train_sft.py"


def _load_train():
    spec = ilu.spec_from_file_location("train_sft", str(_TRAIN))
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TA = _load_train()


class _FakeTok:
    """Minimal whitespace tokenizer; eos id 0. apply_chat_template concatenates ALL turns
    (build_move_prompt returns [system, user])."""
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self):
        self._vocab = {}

    def _id(self, w):
        return self._vocab.setdefault(w, len(self._vocab) + 1)

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.split()]

    def apply_chat_template(self, msgs, add_generation_prompt=True, enable_thinking=False):
        body = " ".join(m["content"] for m in msgs).split()
        return [self._id(w) for w in ["<user>"] + body + ["<assistant>"]]


def test_completion_only_mask_fake_tokenizer():
    tok = _FakeTok()
    row = {"prompt": "numbers: 3, 7, 8 | target: 24 | propose ONE next operation",
           "completion": "MOVE: 3 + 7"}
    prefix = TA._prompt_prefix_ids(tok, row["prompt"])
    input_ids, labels = TA.encode_row(tok, row)
    n = len(prefix)
    assert labels[:n] == [-100] * n                      # prompt fully masked
    assert labels[n:] == input_ids[n:]                   # completion supervised (teacher forcing)
    comp = tok.encode(row["completion"], add_special_tokens=False)
    assert input_ids[n:] == comp + [tok.eos_token_id]    # completion + eos
    assert input_ids[:n] == prefix


_CH = os.environ.get("HF_HUB_CACHE")
_MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")


def test_train_prompt_matches_inference_prompt():
    """GUARD: trainer prefix == real backend inference prompt, token-for-token, same state."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True, cache_dir=_CH)
    except Exception as e:
        pytest.skip(f"Qwen tokenizer unavailable offline: {e}")
    from real_backend import build_move_prompt, _prompt_ids
    for rendered in ["numbers: 3, 7, 8 | target: 24 | propose ONE next operation",
                     "numbers: 2, 9 | target: 18 | propose ONE next operation"]:
        train_prefix = TA._prompt_prefix_ids(tok, rendered)
        infer_ids = _prompt_ids(tok, build_move_prompt(rendered), "cpu")
        assert list(train_prefix) == infer_ids[0].tolist()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
