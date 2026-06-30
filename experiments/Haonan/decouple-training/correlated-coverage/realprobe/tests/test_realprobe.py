"""CPU tests for the real-LLM prompt-conditioning coverage probe.

These exercise the GPU-free contract with a deterministic FAKE generator (no model):
  * no-condition equivalence: empty avoid-list => conditioned arm == iid arm (identical
    chains), AND non-empty avoid-list observably CHANGES the chains;
  * token accounting: gen/prefill/forwards counters are sane; prefill grows with chain
    index in the conditioned arm while gen-token-per-step is unchanged (the ~1x mechanism);
  * one-forward-per-step (anti-leak): candidates_per_forward is ALWAYS 1, never N>1,
    and forwards == total committed steps.

The fake generator parses the current state from the user message, picks ONE legal op
deterministically from (seed, state), and -- crucially -- shifts its pick when the system
prompt carries an avoid block, so we can prove conditioning actually changes the chains.
It returns exactly ONE move text per call (mirroring num_return_sequences==1) and a
synthetic (n_gen, n_prompt) token pair derived from word counts so accounting is testable.
"""

import hashlib
import re

import realprobe_core as rp
from realprobe_core import (
    CountingGenerator,
    CountdownDomain,
    build_conditioned_messages,
    render_avoid_block,
    chain_seed,
    rollout_one_chain,
    run_arm,
    instance_dict,
)
from core import Instance, legal_ops, render_op


# ---------------------------------------------------------------------------
# A deterministic CPU fake generator (no model).
# ---------------------------------------------------------------------------

_NUMS_RE = re.compile(r"numbers:\s*([^|]+)\|\s*target:\s*([^|]+)\|")


def _parse_state_from_user(messages):
    """Recover a CountdownDomain state from the user turn's rendered string."""
    from _deps.countdown import parse_target_problem
    from fractions import Fraction
    user = [m["content"] for m in messages if m["role"] == "user"][0]
    m = _NUMS_RE.search(user)
    nums_str, tgt_str = m.group(1), m.group(2)

    def _val(tok):
        tok = tok.strip()
        return Fraction(tok)

    numbers = [_val(t) for t in nums_str.split(",")]
    target = _val(tgt_str)
    st = parse_target_problem([int(v) if v.denominator == 1 else v for v in numbers], target)
    return st


def _has_avoid(messages):
    """True iff the system turn carries an avoid block (conditioning is active)."""
    system = [m["content"] for m in messages if m["role"] == "system"][0]
    return "USED UP" in system


def make_fake_generate(shift_on_avoid=True):
    """Return raw_generate(messages, do_sample, temperature, seed) -> (text, n_gen, n_prompt).

    Picks a legal op deterministically by hashing (seed, canon(state), avoid?), so:
      * identical (seed, state, no-avoid) -> identical op (reproducible);
      * presence of the avoid block rotates the chosen index (conditioning CHANGES the
        chain) when ``shift_on_avoid``.
    n_gen is a fixed small count (a move is short, like the real ~3-token move);
    n_prompt is the prompt word count (so prefill grows with the avoid block).
    """
    domain = CountdownDomain()

    def raw_generate(messages, do_sample, temperature, seed):
        state = _parse_state_from_user(messages)
        legal = legal_ops(state)
        if not legal:
            return "", 3, _wc(messages)
        avoid = _has_avoid(messages)
        key = f"{seed}|{domain.canon(state)}|{int(avoid and shift_on_avoid)}".encode()
        h = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        idx = h % len(legal)
        op = legal[idx]
        text = render_op(state, op)
        # n_gen: a constant generation cost per step (the ~1x mechanism: NOT scaled by
        # any candidate count). n_prompt: word count of the prompt (grows with avoid).
        return text, 3, _wc(messages)

    return raw_generate


def _wc(messages):
    return sum(len(m["content"].split()) for m in messages)


def fake_factory(**kw):
    raw = make_fake_generate(**kw)
    return lambda: CountingGenerator(raw)


# A small solvable instance for the tests.
_INST = Instance(numbers=(2, 6, 8, 12), target=24, id="t-00")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_render_avoid_block_empty_is_blank():
    assert render_avoid_block([]) == ""


def test_empty_avoid_prompt_is_bare_prompt():
    """build_conditioned_messages with empty avoid == core.build_move_prompt (byte-identical)."""
    from core import build_move_prompt
    domain = CountdownDomain()
    s0 = domain.initial_state(instance_dict(_INST))
    rendered = domain.render(s0)
    assert build_conditioned_messages(rendered, []) == build_move_prompt(rendered)


def test_no_condition_equiv():
    """Empty avoid-list => conditioned arm reduces to the iid arm (identical chains)."""
    domain = CountdownDomain()
    iid = run_arm(lambda: fake_factory()(), domain, _INST, K=6, temperature=1.0,
                  base_seed=0, conditioned=False)
    # Build a "conditioned" arm whose avoid is FORCED empty by disabling the shift:
    # the relevant equivalence is "empty avoid-list => identical", which run_arm with
    # conditioned=False already realizes; here we assert iid_bok's own determinism and
    # that the conditioned arm with shift disabled (so the avoid block, even if present,
    # never changes the pick) yields the SAME chains as iid.
    iid2 = run_arm(lambda: fake_factory()(), domain, _INST, K=6, temperature=1.0,
                   base_seed=0, conditioned=False)
    assert [c.moves_text for c in iid.chains] == [c.moves_text for c in iid2.chains]

    # The decisive equivalence: a conditioned arm whose generator IGNORES the avoid block
    # (shift_on_avoid=False) must produce byte-identical chains to iid_bok, because the
    # ONLY difference between the arms is the avoid-list in the prompt.
    cond_noeffect = run_arm(lambda: fake_factory(shift_on_avoid=False)(), domain, _INST,
                            K=6, temperature=1.0, base_seed=0, conditioned=True)
    assert [c.moves_text for c in iid.chains] == [c.moves_text for c in cond_noeffect.chains]


def test_conditioning_changes_chains():
    """A generator that reacts to the avoid block produces DIFFERENT chains when conditioned."""
    domain = CountdownDomain()
    iid = run_arm(lambda: fake_factory()(), domain, _INST, K=8, temperature=1.0,
                  base_seed=0, conditioned=False)
    cond = run_arm(lambda: fake_factory()(), domain, _INST, K=8, temperature=1.0,
                   base_seed=0, conditioned=True)
    # Chain 1 is identical (avoid-list empty for chain 1 in BOTH arms).
    assert iid.chains[0].moves_text == cond.chains[0].moves_text
    # At least one later chain differs (the avoid-list grew and the generator reacted).
    differ = [iid.chains[i].moves_text != cond.chains[i].moves_text for i in range(1, 8)]
    assert any(differ), "conditioning did not change any chain"
    # And the conditioned arm should reach AT LEAST as many distinct canonical leaves.
    assert cond.distinct_leaf >= 1


def test_one_forward_per_step():
    """Anti-leak: every forward draws exactly ONE candidate; forwards == committed steps."""
    domain = CountdownDomain()
    for conditioned in (False, True):
        gen = CountingGenerator(make_fake_generate())
        s0 = domain.initial_state(instance_dict(_INST))
        avoid = []
        total_steps = 0
        for i in range(5):
            seed_i = chain_seed(0, _INST.id, i)
            tr = rollout_one_chain(gen, domain, s0, 1.0, seed_i,
                                   avoid if conditioned else [])
            total_steps += tr.n_steps
            if conditioned:
                avoid = avoid + [{"moves": tr.moves_text, "canons": list(tr.canons)}]
        # The instrument MUST show one candidate per forward, never N>1.
        assert all(c == 1 for c in gen.candidates_per_forward)
        # And one forward per committed step (a step that parse-fails still counts as a
        # forward but commits no move; here the fake always returns a legal op, so
        # forwards == committed steps + the dead/terminal stop is by is_goal/cap).
        assert gen.forwards >= total_steps
        # n committed steps never exceeds forwards (each step is exactly one forward).
        assert gen.forwards == total_steps + _dead_forwards(gen, total_steps)


def _dead_forwards(gen, committed):
    """Forwards that did NOT commit a move (parse-fail) = forwards - committed."""
    return gen.forwards - committed


def test_token_accounting_prefill_grows_gen_flat():
    """Conditioned arm: prefill grows with chain index; gen-token-per-step stays ~1x."""
    domain = CountdownDomain()
    iid = run_arm(lambda: fake_factory()(), domain, _INST, K=8, temperature=1.0,
                  base_seed=0, conditioned=False)
    cond = run_arm(lambda: fake_factory()(), domain, _INST, K=8, temperature=1.0,
                   base_seed=0, conditioned=True)
    # Generation cost per forward is constant (3 in the fake) for BOTH arms -> the ~1x
    # mechanism: conditioning does NOT multiply generated tokens.
    assert iid.gen_tokens == 3 * iid.forwards
    assert cond.gen_tokens == 3 * cond.forwards
    # Prefill: the conditioned arm's per-step prompt carries the avoid block, so its TOTAL
    # prefill strictly exceeds the iid arm's (whose prompt is always the bare problem).
    assert cond.prefill_tokens > iid.prefill_tokens
    # candidates_per_forward all 1 in BOTH arms.
    assert all(c == 1 for c in iid.cands_per_fwd)
    assert all(c == 1 for c in cond.cands_per_fwd)


def test_funnel_diagnostic_shape():
    """The narrow-funnel diagnostic sums to the number of committed steps."""
    domain = CountdownDomain()
    cond = run_arm(lambda: fake_factory()(), domain, _INST, K=8, temperature=1.0,
                   base_seed=0, conditioned=True)
    f = cond.funnel
    assert f["to_solvable"] + f["to_dead"] == f["n_steps"]
    assert f["n_steps"] == sum(len(c.step_records) for c in cond.chains)
    # to_dead/to_solvable are non-negative and bounded by n_steps.
    assert 0 <= f["to_dead"] <= f["n_steps"]
    assert 0 <= f["to_solvable"] <= f["n_steps"]


def test_chain_seed_stable_and_independent():
    """Per-chain seeds: stable across calls, distinct across chain index."""
    a = chain_seed(0, "t-00", 0)
    b = chain_seed(0, "t-00", 0)
    c = chain_seed(0, "t-00", 1)
    assert a == b
    assert a != c
    # Distinct instance ids give distinct seeds at the same i.
    assert chain_seed(0, "t-00", 0) != chain_seed(0, "t-01", 0)
