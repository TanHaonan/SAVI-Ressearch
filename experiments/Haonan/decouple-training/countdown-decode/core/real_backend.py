"""Real emission backend: trained Qwen3-4B(+LoRA) -> countdown move texts.

This fills the ``real_onehot`` / ``real_decoupled`` hole the mock sampler leaves
(``sampler.sample`` raises ``NotImplementedError("emission line dependency")`` for the
``real_*`` backends). It implements the COUNTDOWN analogue of algebra-decode's
``real_backend``: a factory that returns a ``decode_core``-compatible
``sample(state, N, temperature, seed, mode)`` closure driven by a real model, plus the
model/LoRA loader.

Countdown is the decoder's NATIVE domain. A state is a multiset of available exact
values + a fixed target; a move COMBINES two of the available values into one. The move
text is ``"<a> <symbol> <b>"`` (three SPACE-delimited tokens, ``a <= b`` canonical,
``symbol in {'+','-','*','/'}``), e.g. ``"3 + 7"``. The decoder reads candidate TEXTS
only (never logits): from a canonical state ``S`` the model PROPOSES a move as text and
``CountdownDomain.parse_move`` filters illegals. So the model never enumerates the move
set; it writes ``MOVE: <a> <op> <b>`` and we extract the move text (the decoder's
``parse_move`` is the legality gate -- we do NOT pre-filter, we only strip the ``MOVE:``
prefix so ``parse_move`` sees ``"3 + 7"`` not ``"MOVE: 3 + 7"``).

Two modes (same contract as the mock):
    step  : N single-move texts (one ``generate`` of ``num_return_sequences=N``).
    chain : N ';'-joined chains, each an independent T-by-T rollout of the step skill
            (greedy = ``do_sample=False`` when temperature==0), advancing the real
            ``CountdownDomain`` until is_goal / a step cap (enough to reduce the value
            multiset to one value) / an unparseable-or-illegal step.

THE ``MOVE:`` PROMPT IS A MODULE CONSTANT (``MOVE_PROMPT_SYSTEM`` / ``MOVE_LINE_PREFIX``
/ ``build_move_prompt``) so training (a datagen completion ``"MOVE: <a> <op> <b>"``) and
inference agree EXACTLY on the literal ``MOVE:`` slot. Any datagen row schema MUST emit
its completion as ``MOVE_LINE_PREFIX + " " + <move text>``.

GPU NOTE: ``load_countdown_model`` is the ONLY place a model/tokenizer is touched, and
the import of ``torch`` / ``transformers`` / ``peft`` is deferred INSIDE the functions
so importing this module is CPU-only and free (the L0 tests import it without a GPU).
"""

import hashlib
import re

from _deps.countdown_domain import CountdownDomain


def _mix_seed(seed, state, domain):
    """Per-node sampling seed: fold ``canon(state)`` into the base ``seed``.

    The decoder reuses the SAME base seed at every trellis node (decode_core passes the
    one decode seed to every ``sample`` call), so without mixing the state in, ``set_seed``
    would reset the global RNG identically at each node and the per-node sampling noise
    would be correlated across the trellis. Folding ``canon(state)`` gives each distinct
    state its own independent RNG stream -- matching the mock sampler's ``_digest_seed``
    behaviour. Process-stable (sha256, not Python's salted ``hash``)."""
    key = str(domain.canon(state))
    h = hashlib.sha256(f"{int(seed)}|{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# THE move-format prompt (LOAD-BEARING constant -- coordinate with any datagen)
# ---------------------------------------------------------------------------

# The literal token the model must emit before the move, and that we extract on.
# A datagen completion MUST be exactly ``MOVE_LINE_PREFIX + " " + <move text>``.
MOVE_LINE_PREFIX = "MOVE:"

# Chain step cap: a countdown chain never needs more combines than (#values - 1) to
# reduce the multiset to a single value, so the rollout caps at the smaller of STEP_CAP
# and the per-state value count (default ceiling 8 ~ the initial value count).
STEP_CAP = 8

# Max new tokens per single-move generation (a move text is short: "3 + 7").
MAX_NEW_TOKENS = 24

# System instruction + few-shot move-format examples. This is the FIXED contract
# between training (the SFT completion format) and inference (what we parse back).
# The examples teach ONLY the OUTPUT FORMAT (last line ``MOVE: <a> <op> <b>``) and the
# move vocabulary -- they are not goal-directed demonstrations.
MOVE_PROMPT_SYSTEM = (
    "You are playing Countdown: reach the target by combining numbers one step at "
    "a time.\n"
    "Propose exactly ONE legal operation: combine TWO of the available numbers with "
    "one of + - * or /, consuming both and producing one new number.\n"
    "Write the operation as three space-separated tokens '<a> <op> <b>' with the "
    "smaller operand first (a <= b); '-' is the larger minus the smaller, '/' is the "
    "larger divided by the smaller.\n"
    "Reply with the operation and NOTHING else. The LAST line must be exactly:\n"
    f"  {MOVE_LINE_PREFIX} <a> <op> <b>\n"
    "Examples:\n"
    f"  for  numbers: 3, 7, 8  ->  {MOVE_LINE_PREFIX} 3 + 7\n"
    f"  for  numbers: 2, 9     ->  {MOVE_LINE_PREFIX} 2 * 9"
)

# The user-turn template: the canonical state rendering (domain.render) verbatim.
# (domain.render already says "propose ONE next operation".)
MOVE_PROMPT_USER = "{rendered}\nReply with one line: " + MOVE_LINE_PREFIX + " <a> <op> <b>"

# Regex extracting the move text after the LAST ``MOVE:`` occurrence on its line.
# Tolerates leading whitespace and an optional space after the colon; captures the
# rest of the line. ``re.MULTILINE`` so ``^``/``$`` are per-line; ``findall`` ->
# last match for the "last occurrence" rule.
_MOVE_RE = re.compile(rf"^\s*{re.escape(MOVE_LINE_PREFIX)}\s*(.*)$", re.MULTILINE)


def build_move_prompt(rendered):
    """Return the chat ``messages`` list for one step from a rendered state string.

    ``rendered`` is ``CountdownDomain.render(state)``. The structure (system =
    ``MOVE_PROMPT_SYSTEM``, user = state + the ``MOVE:`` instruction) is the FIXED
    contract any datagen must mirror for its prompt side.
    """
    return [
        {"role": "system", "content": MOVE_PROMPT_SYSTEM},
        {"role": "user", "content": MOVE_PROMPT_USER.format(rendered=rendered)},
    ]


def extract_move_text(generated):
    """Extract the move text from a model completion (last ``MOVE:`` line).

    Returns the stripped text AFTER the ``MOVE:`` prefix of the LAST matching line
    (so a chatty model that re-states the format earlier does not shadow its final
    answer). If no ``MOVE:`` line is present, returns ``""`` -- a non-move that
    ``parse_move`` rejects, i.e. the illegal/garbage slot the decoder filters out
    (we do NOT invent a move). The ``MOVE:`` prefix itself is stripped so
    ``parse_move`` sees ``"3 + 7"`` not ``"MOVE: 3 + 7"``.
    """
    if not isinstance(generated, str):
        return ""
    matches = _MOVE_RE.findall(generated)
    if not matches:
        return ""
    return matches[-1].strip()


# ---------------------------------------------------------------------------
# Model + LoRA loader (the ONLY GPU touch; deferred imports keep import-time CPU)
# ---------------------------------------------------------------------------

def load_countdown_model(adapter_path_or_None, device="cuda"):
    """Load frozen Qwen3-4B (bf16, local_files_only) and optionally a LoRA adapter.

    Mirrors the ``controllable-posterior`` convention (``core/common.py`` +
    ``run.py``): ``DEFAULT_MODEL`` / ``HF_HUB_CACHE`` from the environment,
    ``dtype=torch.bfloat16``, ``local_files_only=True``, and the same
    gptqmodel-availability monkeypatch peft needs in this environment. When
    ``adapter_path_or_None`` is a path, the saved PEFT LoRA is loaded on top of the
    frozen base (``PeftModel.from_pretrained``); when ``None`` the bare instruct
    base is returned (the ``base`` arm).

    Returns ``(model, tok)`` both ready for ``.generate``. Heavy imports are local
    so importing ``real_backend`` stays CPU-only -- nothing loads until called.
    """
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cache_dir = os.environ.get("HF_HUB_CACHE")
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, local_files_only=True, cache_dir=cache_dir
    ).to(device)

    if adapter_path_or_None is not None:
        # Same peft gptqmodel workaround as controllable-posterior/run.py.
        import peft.import_utils as _piu
        _piu.is_gptqmodel_available = lambda *x, **k: False
        try:
            import peft.tuners.lora.gptq as _pg
            _pg.is_gptqmodel_available = lambda *x, **k: False
        except Exception:
            pass
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path_or_None))

    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Chat prompt -> input ids (deferred-torch helper, shared by step generation)
# ---------------------------------------------------------------------------

def _prompt_ids(tok, messages, device):
    """Token ids up to (and including) the assistant generation prompt.

    Thinking is disabled (Qwen3 templates) so the move comes immediately; harmless
    on templates without that flag (mirrors ``common.chat_prefix_ids``).
    """
    kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=False)
    try:
        ids = tok.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        ids = tok.apply_chat_template(messages, **kw)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids.dim() == 1:
        ids = ids[None]
    return ids.to(device)


# ---------------------------------------------------------------------------
# The factory: model -> decode_core-compatible sampler
# ---------------------------------------------------------------------------

def make_real_sampler(model, tok, domain, device):
    """Return ``sample(state, N, temperature, seed, mode) -> list[str]`` over a real model.

    This is the ``real_*`` emission backend the decoder consumes. The closure binds
    the model/tokenizer/domain so its signature matches the mock sampler's adapted
    form (the ``backend`` is baked in -- there is one backend per loaded adapter:
    pass the coupled adapter for ``real_onehot``, the decoupled adapter for
    ``real_decoupled``).

    step  : one ``model.generate`` with ``num_return_sequences=N`` from the
            ``MOVE:`` prompt; ``do_sample = temperature > 0``; seed via
            ``transformers.set_seed``. Returns N raw move texts (``MOVE:`` prefix
            stripped; ``parse_move`` -- inside the decoder -- is the legality filter,
            we do NOT pre-filter).
    chain : N independent rollouts. Each rolls the step skill forward T-by-T,
            applying ``parse_move``/``apply`` until ``is_goal`` / the per-state step
            cap (``min(STEP_CAP, #values - 1)`` -- enough to reduce to one value) / an
            unparseable-or-illegal step, then joins the EMITTED move texts with ';' to
            match ``parse_chain``. Greedy (temperature==0) uses ``do_sample=False``.
    """
    import torch

    def _generate_step_texts(state, n, temperature, seed):
        """Low-level: n raw move texts from ONE generate at ``state``."""
        messages = build_move_prompt(domain.render(state))
        input_ids = _prompt_ids(tok, messages, device)
        do_sample = float(temperature) > 0.0
        # do_sample=False is deterministic: ask for ONE sequence and (if N>1) replicate,
        # rather than have generate return N identical greedy beams.
        n_seqs = n if do_sample else 1
        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=n_seqs,
            do_sample=do_sample,
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        # Seed reproducibly via the GLOBAL torch RNG that generate() samples from, mixing
        # canon(state) so each trellis node draws from an INDEPENDENT stream (the decoder
        # reuses one base seed at every node). A per-call ``generator=`` kwarg is rejected
        # by transformers as an unused model arg, so it must NOT be passed; set_seed is the
        # supported reproducible-sampling path.
        from transformers import set_seed
        set_seed(_mix_seed(seed, state, domain))
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
        # Decode ONLY the newly generated continuation (strip the prompt prefix).
        prompt_len = input_ids.shape[1]
        texts = []
        for row in out:
            completion = tok.decode(row[prompt_len:], skip_special_tokens=True)
            texts.append(extract_move_text(completion))
        if not do_sample and len(texts) < n:           # greedy: replicate the one move
            texts = (texts or [""]) * n
        return texts[:n]

    def _step_cap(state):
        """Combines needed to reduce ``state`` to one value, capped at STEP_CAP.

        A countdown chain never needs more than (#values - 1) combines; STEP_CAP is the
        overall ceiling. At a terminal (single-value) state the cap is 0 (no rollout).
        """
        n_vals = len(state.values)
        return min(STEP_CAP, max(0, n_vals - 1))

    def _rollout_chain(state, temperature, seed):
        """One chain: roll the step skill forward to goal / cap / dead-end.

        Returns the ';'-joined emitted move texts (possibly empty if the first step
        is already unparseable). Greedy when temperature==0 (do_sample=False, one
        sequence). Each step's seed is offset by the step index for fresh sampling.
        """
        cur = state
        moves_text = []
        cap = _step_cap(state)
        for t in range(cap):
            if domain.is_goal(cur):
                break
            step_seed = (int(seed) * 1000003 + t) & 0x7FFFFFFF
            cand = _generate_step_texts(cur, 1, temperature, step_seed)
            text = cand[0] if cand else ""
            move = domain.parse_move(text, cur)
            if move is None:
                break  # unparseable / illegal step -> stop the rollout
            cur = domain.apply(cur, move)
            moves_text.append(text)
        return ";".join(moves_text)

    def sample(state, N, temperature, seed, mode):
        if N <= 0:
            return []
        if mode == "step":
            return _generate_step_texts(state, N, temperature, seed)
        if mode == "chain":
            # Each chain is its own rollout with seed offset by its index.
            return [_rollout_chain(state, temperature, (int(seed) + i) & 0x7FFFFFFF)
                    for i in range(N)]
        raise ValueError(f"unknown mode: {mode!r}")

    return sample
