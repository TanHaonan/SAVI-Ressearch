"""Real-LLM prompt-conditioning coverage probe (the faithful ~1x mechanism).

WHAT THIS IS (and what the mock could NOT model)
------------------------------------------------
The correlated-coverage mock (`../corr_sampler.py`, MECHANISM.md s3) realises chain
diversity by drawing ``N`` single-op candidates PER STEP and reweighting an empirical
``P_emission`` -- a ``~N x`` generated-token tax (MECHANISM.md s3.1: "step-mode S' is
~Nx more expensive per realized chain"). That is the mock's fatal leak: it cannot be the
cheap mechanism, because every committed step is backed by N sampled candidates that are
mostly discarded.

This probe builds the BACKUP route of MECHANISM.md s3.2 / PREREG s7 -- the ONLY ~1x
mechanism -- on the REAL model: draw each chain cheaply in CHAIN-mode (ONE forward per
step, the model emits the diversified chain DIRECTLY), and condition diversity through
the PROMPT by appending an avoid-list of the canonical states / move-sequences the
earlier chains already walked. Generated-continuation tokens are unchanged vs i.i.d.
best-of-K; only cheap PREFILL tokens grow with the chain index. So:

  * HEADLINE iso-token axis  = iso-GENERATION-token  (Sigma generated continuation
    tokens). The conditioned arm matches i.i.d. by construction at ~1x (no N-candidate
    tax): one forward per committed step, exactly like i.i.d.
  * SENSITIVITY iso-token axis = iso-TOTAL-token (generation + PREFILL), where prefill
    grows with chain index because chain i carries the avoid-list of chains 1..i-1.

TWO ARMS (same rollout code; the ONLY difference is the avoid-list in the prompt)
--------------------------------------------------------------------------------
  iid_bok      : K chains, each rolled out from the BARE problem prompt (empty
                 avoid-list). Coverage = ANY chain reaches the target (domain.is_goal).
  conditioned  : chain i rolled out from the problem prompt AUGMENTED with the avoid-list
                 of canonical states (and the move-sequences) produced by chains 1..i-1.
                 ONE forward per step -- NOT N candidates per step.

NO-CONDITION EQUIVALENCE (V2 analogue): with an empty avoid-list the conditioned rollout
issues byte-identical prompts and the SAME per-chain seed schedule as ``iid_bok``, so the
two arms produce identical chains. ``rollout_one_chain(..., avoid=[])`` is literally the
i.i.d. code path; ``run_arms`` builds ``iid_bok`` by calling the SAME rollout with
``avoid=[]`` at every chain. (See ``tests/test_realprobe.py::test_no_condition_equiv``.)

ONE-FORWARD-PER-STEP (the anti-leak invariant): ``rollout_one_chain`` calls
``_generate_one_move`` exactly ONCE per committed step, with ``num_return_sequences=1``.
It NEVER draws N>1 candidates per step. The instrumented generator
(``CountingGenerator``) records ``forwards`` and ``candidates_per_forward``; a test
asserts ``candidates_per_forward == 1`` for every forward of the conditioned arm.

REUSE: this module imports the countdown-decode ``core`` package (the SAME
``load_countdown_model`` / ``CountdownDomain`` / ``MOVE_*`` constants / ``build_move_prompt``
/ ``extract_move_text`` the production real backend uses). It does NOT touch
``decode_core`` and does NOT touch the sibling ``correlated-coverage`` files.

GPU note: ``load_probe_model`` defers to ``core.load_countdown_model`` (the one GPU
touch). Importing this module is CPU-only; nothing loads until called. The avoid-list
prompt rendering, token accounting, the funnel diagnostic, and the arm driver are all
GPU-free and unit-tested on CPU with a fake generator.
"""

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Boot the countdown-decode ``core`` package (the substrate we reuse verbatim).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
# realprobe/ -> correlated-coverage/ -> decouple-training/ -> .../countdown-decode/
_CD_ROOT = _HERE.parent.parent / "countdown-decode"  # dir holding the ``core`` package
if str(_CD_ROOT) not in sys.path:
    sys.path.insert(0, str(_CD_ROOT))

import core  # noqa: E402  (path setup must precede the import)
from core import (  # noqa: E402
    CountdownDomain,
    build_move_prompt,
    extract_move_text,
    MOVE_PROMPT_SYSTEM,
    MOVE_PROMPT_USER,
    MOVE_LINE_PREFIX,
    STEP_CAP,
    MAX_NEW_TOKENS,
    load_countdown_model,
    load_instances,
    instance_dict,
)
from _deps.countdown import _fmt, canon as _canon  # noqa: E402

DEFAULT_ADAPTER = str(
    (_CD_ROOT / "outputs" / "adapter_decoupled_s0").resolve()
)


# ===========================================================================
# Per-chain seeding (process-stable; chain i in both arms shares a seed)
# ===========================================================================

def chain_seed(base_seed, inst_id, i):
    """Deterministic, process-stable per-chain seed = f(base_seed, inst_id, i).

    Mirrors MECHANISM.md s2.2(B): chain i in the conditioned arm and chain i in
    ``iid_bok`` share a seed (so the empty-avoid equivalence is byte-identical), seeds
    are independent across i, and reruns reproduce. SHA-256 (not Python's salted
    ``hash``) so it is stable across processes / PYTHONHASHSEED.
    """
    payload = f"{int(base_seed)}:{inst_id}:{int(i)}".encode("utf-8")
    h = hashlib.sha256(payload).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFF


# ===========================================================================
# The avoid-list -> prompt rendering (the cheap conditioning context)
# ===========================================================================

def _fmt_state_values(canon_key):
    """Render a ``canon`` key ``(sorted values tuple, target)`` as a compact value list.

    ``canon_key`` is what ``CountdownDomain.canon(state)`` returns. We render only the
    value multiset (the identity that matters for repulsion -- the target is fixed for an
    instance), e.g. ``(Fraction(2), Fraction(6), Fraction(8))`` -> ``"2, 6, 8"``.
    """
    vals, _target = canon_key
    return ", ".join(_fmt(v) for v in vals)


def render_avoid_block(avoid):
    """Render the avoid-list into a short natural-language conditioning block.

    ``avoid`` is a list of per-chain records, each a dict with:
        "moves" : the ';'-joined move-text of an earlier chain (may be "").
        "canons": list of canon keys (the canonical states that chain passed through).

    Returns "" for an empty avoid-list (so the empty-avoid prompt is byte-identical to
    the bare problem prompt -> the no-condition equivalence). Otherwise returns a block
    naming the move-sequences AND the canonical intermediate value-multisets already
    tried, instructing the model to reach the target by a DIFFERENT route.

    This is the CHEAP conditioning context (PREREG s7 BACKUP route): it adds PREFILL
    tokens only; it does NOT change the generated-continuation accounting.
    """
    if not avoid:
        return ""
    lines = [
        "You have already tried the following solution attempts; they are USED UP. "
        "Reach the target by a DIFFERENT route -- do NOT repeat any of these move "
        "sequences and try to avoid passing through the same intermediate number sets:"
    ]
    for idx, rec in enumerate(avoid, start=1):
        moves = rec.get("moves", "")
        canons = rec.get("canons", [])
        # The move-sequence already tried (the surface path).
        if moves:
            lines.append(f"  attempt {idx}: {moves}")
        else:
            lines.append(f"  attempt {idx}: (no valid move)")
        # The canonical intermediate value-multisets that attempt passed through.
        if canons:
            inter = " ; ".join(_fmt_state_values(c) for c in canons)
            lines.append(f"    intermediate number sets: {inter}")
    return "\n".join(lines)


def build_conditioned_messages(rendered, avoid):
    """Chat ``messages`` for one step, with the avoid-list folded into the SYSTEM turn.

    ``rendered`` is ``CountdownDomain.render(state)`` (the current state). ``avoid`` is
    the avoid-list (see ``render_avoid_block``). With an EMPTY avoid-list this returns
    EXACTLY ``core.build_move_prompt(rendered)`` (same system, same user) -> the
    no-condition equivalence is byte-identical at the prompt level.

    With a non-empty avoid-list the avoid block is appended to the system content (so the
    user turn -- the per-step state -- is untouched and the model still answers with one
    ``MOVE:`` line). Conditioning lives in the PREFILL; generation is unchanged.
    """
    block = render_avoid_block(avoid)
    if not block:
        return build_move_prompt(rendered)
    system = MOVE_PROMPT_SYSTEM + "\n" + block
    user = MOVE_PROMPT_USER.format(rendered=rendered)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ===========================================================================
# Generation backends (real model + a CPU fake, behind one tiny interface)
# ===========================================================================

class CountingGenerator:
    """Wrap a raw generation function and instrument the one-forward-per-step invariant.

    ``raw_generate(messages, do_sample, temperature, seed) -> (text, n_gen_tokens,
    n_prompt_tokens)`` returns ONE completion's move text plus the number of newly
    generated tokens and the number of prompt (prefill) tokens.

    The wrapper:
      * counts ``forwards`` (one per ``generate`` call),
      * records ``candidates_per_forward`` (always 1 here -- the anti-leak invariant),
      * accumulates ``gen_tokens`` (generation, the HEADLINE axis) and ``prefill_tokens``
        (the SENSITIVITY axis), both as REAL tokenizer token counts.
    """

    def __init__(self, raw_generate):
        self._raw = raw_generate
        self.forwards = 0
        self.candidates_per_forward = []  # one entry per forward; MUST all be 1
        self.gen_tokens = 0
        self.prefill_tokens = 0

    def one_move(self, messages, do_sample, temperature, seed):
        """ONE forward, ONE returned move (num_return_sequences == 1, by contract)."""
        text, n_gen, n_prompt = self._raw(messages, do_sample, temperature, seed)
        self.forwards += 1
        self.candidates_per_forward.append(1)
        self.gen_tokens += int(n_gen)
        self.prefill_tokens += int(n_prompt)
        return text


def make_real_generate(model, tok, device):
    """Return ``raw_generate(messages, do_sample, temperature, seed)`` over the real model.

    ONE ``model.generate`` with ``num_return_sequences=1`` (anti-leak: never N>1). Mirrors
    ``core.real_backend``'s low-level step generation (same chat-template prefill, same
    ``set_seed`` reproducibility, same ``extract_move_text`` parse), but takes the
    ``messages`` from the CALLER so the avoid-list can be injected. Returns
    ``(move_text, n_generated_tokens, n_prompt_tokens)``.
    """
    import torch
    from transformers import set_seed

    def _prompt_ids(messages):
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

    def raw_generate(messages, do_sample, temperature, seed):
        input_ids = _prompt_ids(messages)
        prompt_len = input_ids.shape[1]
        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=1,  # ANTI-LEAK: exactly one candidate per forward
            do_sample=bool(do_sample),
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None
                          else tok.eos_token_id),
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        set_seed(int(seed) & 0x7FFFFFFF)
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
        row = out[0]
        gen_ids = row[prompt_len:]
        completion = tok.decode(gen_ids, skip_special_tokens=True)
        n_gen = int((gen_ids != tok.pad_token_id).sum().item()) if tok.pad_token_id is not None \
            else int(gen_ids.shape[0])
        return extract_move_text(completion), n_gen, int(prompt_len)

    return raw_generate


# ===========================================================================
# The rollout: ONE chain, ONE forward per step (shared by both arms)
# ===========================================================================

@dataclass
class ChainTrace:
    """One rolled-out chain.

    moves_text   : ';'-joined emitted move texts (the chain text; "" if first step fails).
    canons       : canon keys of the SUCCESSOR states the chain committed to (depth 1..d).
    reached_goal : did the realized leaf satisfy domain.is_goal.
    step_records : per committed step: {"to_solvable": bool, "to_dead": bool} -- the
                   narrow-funnel diagnostic (did the step move to a still-solvable or a
                   dead canonical successor).
    n_steps      : number of committed (applied) steps.
    """

    moves_text: str
    canons: list = field(default_factory=list)
    reached_goal: bool = False
    step_records: list = field(default_factory=list)
    n_steps: int = 0


def _step_cap(state):
    """Combines to reduce ``state`` to one value, capped at STEP_CAP (mirrors core)."""
    return min(STEP_CAP, max(0, len(state.values) - 1))


def rollout_one_chain(gen, domain, state, temperature, seed, avoid):
    """Roll ONE chain forward to goal / cap / dead-end -- ONE forward per committed step.

    ``gen`` is a ``CountingGenerator``. At each step we build the per-step messages
    (``build_conditioned_messages`` -> empty ``avoid`` reduces to the bare prompt), call
    ``gen.one_move`` EXACTLY ONCE (never N candidates), parse the returned text with
    ``domain.parse_move`` (the legality gate), and ``apply`` it. The per-step seed is
    offset by the step index (matching ``core.real_backend._rollout_chain``'s schedule)
    so each step draws fresh sampling noise.

    The narrow-funnel diagnostic is recorded per committed step: whether the chosen
    successor is still ``domain.solvable`` (good diversification) or ``dead`` (diffusion
    into the unsolvable region).

    Returns a ``ChainTrace``. With ``avoid=[]`` this is the i.i.d. code path.
    """
    do_sample = float(temperature) > 0.0
    cur = state
    moves_text = []
    canons = []
    step_records = []
    cap = _step_cap(state)
    for t in range(cap):
        if domain.is_goal(cur):
            break
        rendered = domain.render(cur)
        messages = build_conditioned_messages(rendered, avoid)
        step_seed = (int(seed) * 1000003 + t) & 0x7FFFFFFF
        text = gen.one_move(messages, do_sample, temperature, step_seed)
        move = domain.parse_move(text, cur)
        if move is None:
            break  # unparseable / illegal -> chain dies here (a miss that cost tokens)
        nxt = domain.apply(cur, move)
        moves_text.append(text)
        canons.append(domain.canon(nxt))
        # Funnel: is the committed successor still solvable, or a dead end?
        if domain.is_goal(nxt):
            to_solvable, to_dead = True, False
        else:
            solvable = domain.solvable(nxt)
            to_solvable, to_dead = bool(solvable), (not solvable)
        step_records.append({"to_solvable": to_solvable, "to_dead": to_dead})
        cur = nxt
    return ChainTrace(
        moves_text=";".join(moves_text),
        canons=canons,
        reached_goal=domain.is_goal(cur),
        step_records=step_records,
        n_steps=len(moves_text),
    )


# ===========================================================================
# Token accounting (Budget split-proxy + real tokenizer gen/prefill)
# ===========================================================================

def chain_split_tokens(moves_text):
    """``len(moves_text.split())`` -- the decode_core ``Budget.tokens`` proxy.

    Reported alongside the real tokenizer counts so this probe's ledger lines up with the
    sibling chain-mode arms (decode_core counts a chain text by ``split()``). NOTE: the
    ';' separator glues adjacent op tokens, so a depth-d chain is ``2d+1`` split-tokens,
    not ``3d`` -- this is purely the proxy; the HEADLINE axis is the real ``gen_tokens``.
    """
    return len(moves_text.split())


# ===========================================================================
# The arm driver: iid_bok and conditioned over one instance
# ===========================================================================

@dataclass
class ArmResult:
    """One arm over one instance.

    coverage      : ANY of the K chains reached the goal (the headline statistic).
    n_goal        : how many of the K chains reached the goal.
    chains        : list[ChainTrace] (K of them).
    gen_tokens    : Sigma real generated tokens over the K chains (HEADLINE iso-token).
    prefill_tokens: Sigma real prefill tokens over the K chains (grows with chain index
                    in the conditioned arm; the SENSITIVITY axis adds this to gen_tokens).
    forwards      : total forwards (== Sigma committed steps; ONE per step).
    cands_per_fwd : the per-forward candidate counts (MUST all be 1: anti-leak proof).
    split_tokens  : Sigma chain_split_tokens (the decode_core Budget proxy).
    distinct_leaf : number of DISTINCT canonical leaf states across the K chains.
    distinct_canon: number of DISTINCT canonical states across ALL committed steps.
    funnel        : {"to_solvable", "to_dead", "n_steps"} aggregated over the K chains.
    """

    coverage: bool
    n_goal: int
    chains: list
    gen_tokens: int
    prefill_tokens: int
    forwards: int
    cands_per_fwd: list
    split_tokens: int
    distinct_leaf: int
    distinct_canon: int
    funnel: dict


def _funnel_of(chains):
    to_solvable = sum(r["to_solvable"] for c in chains for r in c.step_records)
    to_dead = sum(r["to_dead"] for c in chains for r in c.step_records)
    n_steps = sum(len(c.step_records) for c in chains)
    return {"to_solvable": to_solvable, "to_dead": to_dead, "n_steps": n_steps}


def _leaf_canon(chain):
    """The canon key of a chain's realized leaf (last committed successor), or None."""
    return chain.canons[-1] if chain.canons else None


def run_arm(gen_factory, domain, inst, K, temperature, base_seed, conditioned):
    """Run ONE arm (iid_bok if ``conditioned`` is False, conditioned if True).

    ``gen_factory()`` returns a FRESH ``CountingGenerator`` (so its counters belong to
    this arm). Chains are drawn SEQUENTIALLY; for the conditioned arm the avoid-list grows
    by one record (the just-finished chain's moves + canons) after each chain, and is
    rendered into chain i's prompt. For ``iid_bok`` the avoid-list stays EMPTY at every
    chain (so each chain's prompt is the bare problem prompt -> the no-condition path).

    Per-chain seed = ``chain_seed(base_seed, inst.id, i)`` (shared with the other arm at
    the same i, independent across i).
    """
    gen = gen_factory()
    s0 = domain.initial_state(instance_dict(inst))
    avoid = []
    chains = []
    for i in range(K):
        seed_i = chain_seed(base_seed, inst.id, i)
        use_avoid = avoid if conditioned else []
        trace = rollout_one_chain(gen, domain, s0, temperature, seed_i, use_avoid)
        chains.append(trace)
        if conditioned:
            avoid = avoid + [{"moves": trace.moves_text, "canons": list(trace.canons)}]

    n_goal = sum(1 for c in chains if c.reached_goal)
    leaves = {_leaf_canon(c) for c in chains if _leaf_canon(c) is not None}
    all_canons = {ck for c in chains for ck in c.canons}
    return ArmResult(
        coverage=(n_goal > 0),
        n_goal=n_goal,
        chains=chains,
        gen_tokens=gen.gen_tokens,
        prefill_tokens=gen.prefill_tokens,
        forwards=gen.forwards,
        cands_per_fwd=list(gen.candidates_per_forward),
        split_tokens=sum(chain_split_tokens(c.moves_text) for c in chains),
        distinct_leaf=len(leaves),
        distinct_canon=len(all_canons),
        funnel=_funnel_of(chains),
    )


# ===========================================================================
# Convenience: load the real model and build a generator factory
# ===========================================================================

def load_probe_model(adapter=DEFAULT_ADAPTER, device="cuda"):
    """Load the real model (reusing ``core.load_countdown_model``) -> (model, tok).

    ``adapter`` defaults to the decoupled LoRA on disk; pass ``None`` / ``"base"`` for the
    bare instruct base. This is the ONLY GPU touch.
    """
    adapter_path = None if adapter in (None, "base") else adapter
    return load_countdown_model(adapter_path, device=device)


def real_gen_factory(model, tok, device):
    """Return a zero-arg factory making FRESH ``CountingGenerator`` over the real model.

    The raw generate closure is built ONCE (it binds the model); each factory call wraps
    it in a fresh counter so each arm gets its own ledger.
    """
    raw = make_real_generate(model, tok, device)
    return lambda: CountingGenerator(raw)
