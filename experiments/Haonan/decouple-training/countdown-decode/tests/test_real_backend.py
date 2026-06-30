"""L0 (CPU, no GPU, no model) tests for the COUNTDOWN real emission backend.

The actual model load + ``model.generate`` is exercised later on GPU (L1). Here we
prove the GPU-FREE logic the backend layers on top of the model:

  * ``extract_move_text`` -- the ``MOVE:`` line regex: extraction, LAST-occurrence
    rule, missing-line -> "", prefix stripping (so ``parse_move`` sees ``"3 + 7"``).
  * the ``MOVE:`` prompt/format constants are coherent (the literal training and
    inference must share) and ``build_move_prompt`` embeds the rendered state.
  * the chain rollout LOOP logic: with the step-generation primitive STUBBED to
    yield fixed countdown move texts, the rollout stops at ``is_goal`` and at the
    per-state cap (``min(STEP_CAP, #values - 1)`` -- enough combines to reduce the
    value multiset to one value), joins with ';', stops on the first
    unparseable/illegal step, and the joined chain is accepted by a REAL
    ``CountdownDomain.parse_chain`` on a real instance.

We test the rollout by building the sampler with a fake model whose ``.generate``
returns canned token ids decoded by a fake tokenizer -- the cleanest seam that still
exercises the real extraction + rollout code paths end to end without torch's
generation. The 4B model is NEVER loaded.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import core  # noqa: E402
from core import (  # noqa: E402
    extract_move_text,
    build_move_prompt,
    make_real_sampler,
    MOVE_LINE_PREFIX,
    MOVE_PROMPT_SYSTEM,
    STEP_CAP,
)


# ---------------------------------------------------------------------------
# extract_move_text: the MOVE: line regex
# ---------------------------------------------------------------------------

def test_extract_basic():
    assert extract_move_text(f"{MOVE_LINE_PREFIX} 3 + 7") == "3 + 7"


def test_extract_strips_prefix_and_whitespace():
    # parse_move must see "3 + 7", not "MOVE: 3 + 7", and no stray surrounding spaces.
    assert extract_move_text("  MOVE:   2 * 9  ") == "2 * 9"


def test_extract_last_occurrence():
    # A chatty completion that re-states the format earlier must not shadow the
    # final answer: the LAST MOVE: line wins.
    text = "MOVE: <a> <op> <b>\nthinking...\nMOVE: 5 / 8"
    assert extract_move_text(text) == "5 / 8"


def test_extract_missing_line_is_empty():
    # No MOVE: line -> "" (a non-move parse_move rejects; we never invent a move).
    assert extract_move_text("the answer is 10, obviously") == ""
    assert extract_move_text("") == ""


def test_extract_non_string_is_empty():
    assert extract_move_text(None) == ""
    assert extract_move_text(42) == ""


def test_extract_mid_line_prefix_ignored():
    # Mid-line "MOVE:" (not at line start) is ignored; only an own-line MOVE: counts.
    assert extract_move_text("I will do MOVE: nothing here\nMOVE: 4 + 6") == "4 + 6"


def test_extract_garbage_after_prefix_passes_through():
    # Garbage after MOVE: is returned verbatim (parse_move is the legality gate,
    # NOT this extractor) -- the decoder filters it, the backend does not pre-filter.
    assert extract_move_text("MOVE: floomp the widget") == "floomp the widget"


# ---------------------------------------------------------------------------
# The MOVE: prompt/format constants are coherent
# ---------------------------------------------------------------------------

def test_prompt_constants_share_the_move_literal():
    # Training (a datagen completion "MOVE: <a> <op> <b>") and inference must agree on
    # the literal; the system prompt and the extractor both key on MOVE_LINE_PREFIX.
    assert MOVE_LINE_PREFIX == "MOVE:"
    assert MOVE_LINE_PREFIX in MOVE_PROMPT_SYSTEM


def test_build_move_prompt_embeds_state_and_format():
    rendered = "numbers: 3, 7, 8 | target: 24 | propose ONE next operation"
    msgs = build_move_prompt(rendered)
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == MOVE_PROMPT_SYSTEM
    assert msgs[1]["role"] == "user"
    assert rendered in msgs[1]["content"]
    assert MOVE_LINE_PREFIX in msgs[1]["content"]


def test_few_shot_examples_round_trip_through_extract():
    # The few-shot example completions must themselves parse back to a move text via
    # the SAME extractor (training/inference contract is self-consistent).
    assert extract_move_text(f"{MOVE_LINE_PREFIX} 3 + 7") == "3 + 7"
    assert extract_move_text(f"{MOVE_LINE_PREFIX} 2 * 9") == "2 * 9"


def test_datagen_completion_round_trips_through_extract():
    """The gen_sft_data SFT completion ("MOVE: <a> <op> <b>") must round-trip through the
    inference-side extractor EXACTLY (training and inference agree on the literal MOVE:
    slot). This is the load-bearing cross-module coordination guard.
    """
    import gen_sft_data  # sibling module; its _COMPLETION_PREFIX shares the literal
    # The literals agree on the "MOVE:" token (datagen adds a trailing space).
    assert gen_sft_data._COMPLETION_PREFIX.startswith(MOVE_LINE_PREFIX)
    for mv in ("3 + 7", "2 * 9", "4 - 6", "5 / 8", "1 + 1"):
        completion = gen_sft_data._completion(mv)
        assert extract_move_text(completion) == mv


# ---------------------------------------------------------------------------
# Fakes: a model.generate / tokenizer pair that returns CANNED completions
# ---------------------------------------------------------------------------

class _FakeTok:
    """Tokenizer stub: chat template -> a 1-token prompt; decode -> a queued string.

    ``generate`` (the fake model) returns rows whose first ``prompt_len`` ids are the
    prompt and the rest is an INDEX into the queued-completion list; ``decode`` maps
    a row's tail back to the queued completion string. This drives the real
    ``extract_move_text`` over canned model text without any torch generation.
    """

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self._completions = []  # filled by the fake model per call

    def apply_chat_template(self, messages, **kw):
        import torch  # local: only the test environment needs torch present
        # A fixed 1-token prompt (content irrelevant to the rollout logic under test).
        return torch.tensor([[7]])

    def decode(self, ids, skip_special_tokens=True):
        # The fake model encodes the completion index in the single tail id.
        idx = int(ids[-1].item())
        return self._completions[idx]


class _FakeModel:
    """Model stub: ``.generate`` returns one row per queued completion text.

    The queue is a list of lists: each call to ``generate`` pops the next batch of
    completion texts (one per ``num_return_sequences``). The returned tensor's rows
    carry [prompt_id, completion_index] so the fake tokenizer can recover the text.
    """

    def __init__(self, completion_batches):
        self._batches = list(completion_batches)
        self._tok = None  # set by the test so generate can stash texts for decode

    def bind_tok(self, tok):
        self._tok = tok

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        import torch
        n = kwargs.get("num_return_sequences", 1)
        batch = self._batches.pop(0)
        assert len(batch) == n, f"stub batch size {len(batch)} != requested {n}"
        # Register this batch's texts with the tokenizer and emit index-carrying rows.
        base = len(self._tok._completions)
        self._tok._completions.extend(batch)
        prompt_len = input_ids.shape[1]
        rows = []
        for i in range(n):
            row = list(input_ids[0].tolist()) + [base + i]
            rows.append(row)
        return torch.tensor(rows)


def _fake_pair(completion_batches):
    torch = pytest.importorskip("torch")  # noqa: F841
    tok = _FakeTok()
    model = _FakeModel(completion_batches)
    model.bind_tok(tok)
    return model, tok


def _inst(domain, numbers, target):
    return domain.initial_state({"numbers": numbers, "target": target, "id": "t"})


# ---------------------------------------------------------------------------
# step mode: N raw move texts, MOVE: prefix stripped
# ---------------------------------------------------------------------------

def test_step_mode_returns_n_stripped_texts():
    domain = core.CountdownDomain()
    s0 = _inst(domain, (3, 7, 8), 24)
    # SAMPLING path (T>0): one generate of num_return_sequences=N=3 -> three canned
    # completions. (Step mode draws N DISTINCT samples only when sampling; the savi /
    # best_of_k arms call it at tau=1.0.)
    model, tok = _fake_pair([[f"{MOVE_LINE_PREFIX} 3 + 7",
                              f"{MOVE_LINE_PREFIX} 8 * 9",
                              "garbage with no move line"]])
    sample = make_real_sampler(model, tok, domain, device="cpu")
    out = sample(s0, 3, 1.0, 0, "step")
    # Raw texts, prefix stripped; the illegal one passes through as "" (parse_move,
    # inside the decoder, is the filter -- the backend does not pre-filter).
    assert out == ["3 + 7", "8 * 9", ""]

    # GREEDY path (T=0): generate ONE sequence and replicate to N -- never ask the model
    # for N identical greedy beams (the do_sample=False, N>1 guard).
    model2, tok2 = _fake_pair([[f"{MOVE_LINE_PREFIX} 3 + 7"]])
    sample2 = make_real_sampler(model2, tok2, domain, device="cpu")
    assert sample2(s0, 3, 0.0, 0, "step") == ["3 + 7", "3 + 7", "3 + 7"]


# ---------------------------------------------------------------------------
# chain mode: rollout loop -- stops at is_goal, joins with ';', parse_chain accepts
# ---------------------------------------------------------------------------

def test_chain_rollout_stops_at_goal_and_parse_chain_accepts():
    domain = core.CountdownDomain()
    # numbers (2,3,4) target 10: 2*3=6 -> {4,6}; 4+6=10 (goal). After goal the loop
    # must STOP and never consume the trailing junk batch.
    s0 = _inst(domain, (2, 3, 4), 10)
    batches = [
        [f"{MOVE_LINE_PREFIX} 2 * 3"],   # step 0 -> {4, 6}
        [f"{MOVE_LINE_PREFIX} 4 + 6"],   # step 1 -> 10 (is_goal True)
        ["should never be read"],         # step 2 (must NOT be consumed)
    ]
    model, tok = _fake_pair(batches)
    sample = make_real_sampler(model, tok, domain, device="cpu")
    out = sample(s0, 1, 0.0, 0, "chain")
    assert len(out) == 1
    chain = out[0]
    assert chain == "2 * 3;4 + 6"
    # The joined chain is a LEGAL chain on the real domain and reaches the goal.
    moves = domain.parse_chain(chain, s0)
    assert moves is not None
    cur = s0
    for m in moves:
        cur = domain.apply(cur, m)
    assert domain.is_goal(cur)
    # The post-goal batch was not consumed (loop stopped at is_goal).
    assert len(model._batches) == 1


def test_chain_rollout_stops_on_illegal_step():
    domain = core.CountdownDomain()
    s0 = _inst(domain, (2, 3, 4), 10)
    # First step legal; second step is garbage (parse_move -> None) -> rollout stops,
    # chain is just the first move.
    batches = [
        [f"{MOVE_LINE_PREFIX} 2 * 3"],             # legal -> {4, 6}
        ["MOVE: this is not a real move"],          # parse_move -> None -> STOP
    ]
    model, tok = _fake_pair(batches)
    sample = make_real_sampler(model, tok, domain, device="cpu")
    out = sample(s0, 1, 0.0, 0, "chain")
    assert out == ["2 * 3"]
    assert domain.parse_chain("2 * 3", s0) is not None


def test_chain_rollout_stops_on_illegal_op_text():
    domain = core.CountdownDomain()
    s0 = _inst(domain, (2, 3, 4), 10)
    # A well-FORMED op text that is ILLEGAL at the state (operand 99 not present):
    # parse_move -> None -> rollout stops at the first step (empty chain).
    batches = [["MOVE: 99 + 1"]]
    model, tok = _fake_pair(batches)
    sample = make_real_sampler(model, tok, domain, device="cpu")
    out = sample(s0, 1, 0.0, 0, "chain")
    assert out == [""]


def test_chain_rollout_empty_when_first_step_illegal():
    domain = core.CountdownDomain()
    s0 = _inst(domain, (2, 3, 4), 10)
    batches = [["MOVE: nonsense"]]  # first step unparseable -> empty chain
    model, tok = _fake_pair(batches)
    sample = make_real_sampler(model, tok, domain, device="cpu")
    out = sample(s0, 1, 0.0, 0, "chain")
    assert out == [""]
    # An empty chain is NOT a legal chain (parse_chain('') is None) -- matches the
    # mock sampler's terminal convention; the decoder treats it as a failed arm.
    assert domain.parse_chain("", s0) is None


def test_chain_rollout_honors_value_count_cap():
    domain = core.CountdownDomain()
    # 4-number instance, target deliberately unreachable by this sequence (99): the
    # rollout reduces the multiset to ONE value in exactly (#values - 1) = 3 combines,
    # then STOPS at the per-state cap (no goal, no further combine possible). This is
    # the countdown analogue of the algebra STEP_CAP guard -- the cap here is
    # min(STEP_CAP, #values - 1), and #values - 1 binds for a 4-number state.
    s0 = _inst(domain, (1, 2, 3, 4), 99)
    n_values = len(s0.values)
    assert n_values - 1 < STEP_CAP  # the value-count cap (3) is the binding one here
    batches = [
        [f"{MOVE_LINE_PREFIX} 1 + 2"],   # {1,2,3,4} -> {3,3,4}
        [f"{MOVE_LINE_PREFIX} 3 + 3"],   # {3,3,4}   -> {4,6}
        [f"{MOVE_LINE_PREFIX} 4 + 6"],   # {4,6}     -> {10}  (single value, not goal)
        ["should never be read"],         # 4th step must NOT be requested (cap reached)
    ]
    model, tok = _fake_pair(batches)
    sample = make_real_sampler(model, tok, domain, device="cpu")
    out = sample(s0, 1, 0.0, 0, "chain")
    moves_text = out[0].split(";")
    # Exactly (#values - 1) combines emitted: reduced to one value, then stopped.
    assert len(moves_text) == n_values - 1 == 3
    assert moves_text == ["1 + 2", "3 + 3", "4 + 6"]
    # The cap stopped the loop: only 3 batches consumed, the 4th is untouched.
    assert len(model._batches) == 1
    # The emitted chain is a legal chain on the real domain (reduces to one value).
    assert domain.parse_chain(out[0], s0) is not None


def test_chain_mode_n_independent_rollouts():
    domain = core.CountdownDomain()
    s0 = _inst(domain, (2, 3, 4), 10)
    # N=2 rollouts, each a single solving chain of length 2 (4 step-batches total).
    one = [[f"{MOVE_LINE_PREFIX} 2 * 3"], [f"{MOVE_LINE_PREFIX} 4 + 6"]]
    batches = one + one  # rollout 0 then rollout 1
    model, tok = _fake_pair(batches)
    sample = make_real_sampler(model, tok, domain, device="cpu")
    out = sample(s0, 2, 0.0, 0, "chain")
    assert out == ["2 * 3;4 + 6", "2 * 3;4 + 6"]


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_n_zero_returns_empty():
    domain = core.CountdownDomain()
    s0 = _inst(domain, (2, 3, 4), 10)
    model, tok = _fake_pair([])
    sample = make_real_sampler(model, tok, domain, device="cpu")
    assert sample(s0, 0, 0.0, 0, "step") == []
    assert sample(s0, 0, 0.0, 0, "chain") == []


def test_unknown_mode_raises():
    domain = core.CountdownDomain()
    s0 = _inst(domain, (2, 3, 4), 10)
    model, tok = _fake_pair([])
    sample = make_real_sampler(model, tok, domain, device="cpu")
    with pytest.raises(ValueError):
        sample(s0, 1, 0.0, 0, "bogus")


def test_real_backends_in_mock_sampler_still_raise():
    # The mock sampler's real_* path is unchanged (the real backend lives in its own
    # module); the documented NotImplementedError contract still holds there.
    domain = core.CountdownDomain()
    s0 = _inst(domain, (2, 3, 4), 10)
    with pytest.raises(NotImplementedError):
        core.sample(s0, 4, 1.0, 0, "real_decoupled", "step")
