"""Tier-A profile-matched mock emission backends (CPU, no GPU). PLAN3 §4 Tier A.

Tier A is the PLUMBING + PREREG-FREEZE step, explicitly NOT the science: a clearly
labelled mock ``sample(state, N, tau, seed, mode)`` closure that reproduces the
emission line's *qualitative* decoupled/coupled profile so the deeper-Countdown harness,
the iso-token protocol, the K_eff / depth / calibration / merge instruments, and the
validity gates can all be built, debugged, and pre-registered with no GPU.

Two profiles, both built on the SHARED vendored mock sampler (``core.sample``) so they
live on the exact domain/ledger Tier B uses:

* ``decoupled`` (``mock_decoupled``): N step-candidates SPREAD over several distinct
  legal ops -> in-trellis K_eff > 1 (multi-peak survives into per-step decode). Chain
  mode mixes in oracle-witness rollouts so a correct chain appears among draws on
  solvable instances (coverage > 0) — matching the measured decoupled profile
  (coverage ~0.44, K_eff > coupled, merge ratio ~5x).
* ``coupled`` (``mock_onehot``): N copies of ONE collapsed op -> K_eff == 1 (the
  one-hot, marginal-collapsed generator; the H3 contrast / kill-criterion control).

Optional ``abstain`` injects unparseable ("") candidates at a fixed rate to mimic the
real model's <1.0 parse rate (the mocks are otherwise perfectly on-task at parse=1.0);
this exercises the parse-rate instrument and the decoder's None-skip path. Determinism
is process-stable (sha256 of the call args), matching the frozen samplers.
"""

import hashlib
import random

import core_boot as cb
import domain_countdown_deep as dcd

_PROFILE_BACKEND = {"decoupled": "mock_decoupled", "coupled": "mock_onehot"}


def _abstain_rng_stream(seed, state, mode, N, profile):
    """Process-stable per-call float stream for abstention decisions (sha256-seeded)."""
    import random
    vals, target = cb.CountdownDomain().canon(state)
    state_str = ",".join(str(v) for v in vals) + "|" + str(target)
    payload = "::".join([str(seed), state_str, str(mode), str(int(N)), profile, "abstain"])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def make_mock_backend(profile, abstain=0.0):
    """Return a ``sample(state, N, tau, seed, mode) -> list[str]`` Tier-A mock closure.

    ``profile in {"decoupled","coupled"}``. ``abstain`` in [0,1) replaces each candidate
    text with "" (an unparseable slot the decoder filters) independently at that rate, to
    mock the real model's parse rate. ``abstain=0`` reproduces the bare vendored mock.
    """
    if profile not in _PROFILE_BACKEND:
        raise ValueError(f"unknown profile: {profile!r} (valid: decoupled, coupled)")
    backend = _PROFILE_BACKEND[profile]

    def sample(state, N, tau, seed, mode):
        cands = cb.mock_sample(state, N, tau, seed, backend, mode)
        if abstain and cands:
            rng = _abstain_rng_stream(seed, state, mode, N, profile)
            cands = ["" if rng.random() < abstain else c for c in cands]
        return cands

    sample.profile = profile          # tag for instruments / logging
    sample.abstain = float(abstain)
    sample.tier = "A-mock"
    return sample


# ---------------------------------------------------------------------------
# Competence-parameterized decoupled mock (no oracle injection) — the FAITHFUL Tier-A
# decoupled backend. Unlike ``mock_decoupled`` (which injects an oracle witness into
# chain mode so best_of_many is rigged to coverage 1.0), this rolls a competent-but-
# imperfect policy P_p forward: at each step it weights GOOD moves (solvable successor)
# by p and BAD moves by 1-p. So best_of_many coverage = ~p**depth DROPS with depth while
# the verifier mask (lambda>0) prunes the bad branches — exactly the predicted H1/H2
# direction, on a controlled generator. Mirrors mechanism-recombination/gen_fair's P_p
# (no arm gets extra oracle: every arm consumes this same closure).
# ---------------------------------------------------------------------------

def _comp_rng(seed, state, mode, N, tau, p, domain):
    vals, target = domain.canon(state)
    state_str = ",".join(str(v) for v in vals) + "|" + str(target)
    payload = "::".join([str(seed), state_str, str(mode), str(int(N)),
                         f"{float(tau):.6f}", f"{float(p):.4f}", "competence"])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def make_competence_backend(p=0.7, depth_cap=8, abstain=0.0):
    """Return a competence-``p`` decoupled mock ``sample(state,N,tau,seed,mode)`` closure.

    Binds its own ``DeepCountdownDomain`` (private reach memo) to classify moves. ``p`` is
    the per-step probability mass on GOOD (solvable-successor) moves; ``p=1`` is a perfect
    generator, ``p=0.5`` uninformative. No oracle witness is ever injected.
    """
    dom = dcd.DeepCountdownDomain()

    def _weighted_ops(state, rng, tau):
        legal = dom.legal_ops(state)
        if not legal:
            return []
        good = [op for op in legal if dom.solvable(dom.apply(state, op))]
        good_set = set(good)
        n_good, n_bad = len(good), len(legal) - len(good)
        if n_good == 0 or n_bad == 0:
            weights = [1.0] * len(legal)            # all same class -> uniform
        else:
            weights = [p if op in good_set else (1.0 - p) for op in legal]
        return legal, weights

    def _draw_one(state, rng, tau):
        res = _weighted_ops(state, rng, tau)
        if not res:
            return None
        legal, weights = res
        if float(tau) <= 0.0:                       # argmax-weight, deterministic
            best = max(range(len(legal)),
                       key=lambda i: (weights[i], -i))
            return legal[best]
        return rng.choices(legal, weights=weights, k=1)[0]

    def _step_texts(state, N, rng, tau):
        out = []
        for _ in range(N):
            op = _draw_one(state, rng, tau)
            out.append(dom.render_op(state, op) if op is not None else "")
        return out

    def _chain_text(state, rng, tau):
        cur = state
        texts = []
        for _ in range(depth_cap):
            if dom.is_goal(cur) or len(cur.values) <= 1:
                break
            op = _draw_one(cur, rng, tau)
            if op is None:
                break
            texts.append(dom.render_op(cur, op))
            cur = dom.apply(cur, op)
        return ";".join(texts)

    def sample(state, N, tau, seed, mode):
        if N <= 0:
            return []
        rng = _comp_rng(seed, state, mode, N, tau, p, dom)
        if mode == "step":
            cands = _step_texts(state, N, rng, tau)
        elif mode == "chain":
            cands = [_chain_text(state, rng, tau) for _ in range(N)]
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        if abstain and cands:
            cands = ["" if rng.random() < abstain else c for c in cands]
        return cands

    sample.profile = "decoupled"
    sample.kind = "competence"
    sample.p = float(p)
    sample.tier = "A-mock"
    return sample


def make_tier_a_backends(abstain=0.0, decoupled_kind="competence", competence_p=0.7):
    """Both Tier-A profiles ``{"decoupled": closure, "coupled": closure}``.

    ``decoupled_kind="competence"`` (default) uses the faithful competence-p generator
    (best_of_many coverage drops with depth — the honest protocol test). ``"oraclechain"``
    uses the vendored ``mock_decoupled`` (oracle-injected chains; best_of_many rigged to
    1.0 — only for replicating the M1 mock behavior, NOT the headline).
    """
    if decoupled_kind == "competence":
        dec = make_competence_backend(p=competence_p, abstain=abstain)
    elif decoupled_kind == "oraclechain":
        dec = make_mock_backend("decoupled", abstain=abstain)
    else:
        raise ValueError(f"unknown decoupled_kind: {decoupled_kind!r}")
    return {"decoupled": dec, "coupled": make_mock_backend("coupled", abstain=abstain)}
