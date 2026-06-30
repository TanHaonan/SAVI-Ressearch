"""Canonical-state repulsion sampler S' (the core deliverable; PREREG §4, MECHANISM).

This module implements the two headline arms of the correlated-coverage experiment,
both consuming the SAME per-state emission primitive ``sample(state, N, tau, seed,
mode)`` (the mock_backend closure or a real backend with that signature) and the SAME
``domain`` (the ``CountdownDomain`` Φ-merge protocol). ``decode_core`` is FROZEN; nothing
here touches it. The two arms are:

* ``iid_bok(domain, sample, inst, K, tau, seed)`` — draws K i.i.d. chains by STEPWISE
  sampling from the per-state emission ``P_emission(·|S)`` (reconstructed empirically per
  step exactly as ``savi`` builds its layer support, MECHANISM §1.1), with coverage =
  "any of the K chains reaches the goal" and a frozen-``Budget``-compatible token ledger.

* ``corr_sample(domain, sample, inst, K, N, beta, tau, seed)`` — the constructive S':
  K chains drawn STRICTLY SEQUENTIALLY, each step's transition reweighted by the Boltzmann
  repulsion factor ``exp(-β·n(S',t))`` over a per-(canon, depth) chain counter ``n`` built
  from the chains drawn so far (MECHANISM eq. 2). At ``β=0`` the reweight is the identity,
  so this reduces EXACTLY to ``iid_bok`` (the well-posed V2 equivalence; see §2 below).

The β=0 ⇒ i.i.d. equivalence is by construction (MECHANISM §2 route A+B):

1. ``iid_bok`` is itself defined as the STEPWISE i.i.d. limit of S' — both arms run the
   identical ``_draw_one_chain`` routine with the identical per-chain seed schedule
   ``seed_i = _chain_seed(base_seed, inst_id, i)`` and the identical per-step ``sample``
   calls; the ONLY thing β changes is the weight vector fed to the inverse-CDF draw.
2. At ``β=0`` the reweight short-circuits to ``P_emission`` (``_reweight`` returns it
   unchanged), so chain i's law is independent of chains ``1..i-1`` and identical across i.
3. The categorical draw is a deterministic inverse-CDF over the support SORTED BY ``canon``
   (the same total-order discipline ``decode_core._stable_order`` uses), consuming exactly
   one ``rng.random()`` per step from a per-chain ``random.Random(seed_i)``. So β never
   touches the random stream, only the weights — giving byte-identical K-chain path lists
   between ``corr_sample(β=0)`` and ``iid_bok`` for any (instance, seed).

Token accounting (MECHANISM §3): both arms record EVERY ``sample()`` call through the
frozen ``Budget.record_sample`` (``tokens = Σ len(text.split())`` over returned candidates),
so the ledger is byte-comparable to ``decode_core`` and across arms. The step-mode S'
overhead — N candidates per committed step — is therefore IN its ``Budget.tokens`` by
construction; iso-token fairness is read off ``result.budget.tokens`` directly.

The realized chains and the per-(canon, depth) counter are returned in ``Result.detail``
for the joint-diversity instrument (``distinct_canon@K``, MECHANISM §5.3).
"""

import hashlib
import math
import random

from core_boot import Budget, Result


# ---------------------------------------------------------------------------
# Per-chain seed schedule (process-stable; MECHANISM §2.2 route B)
# ---------------------------------------------------------------------------

def _inst_id(inst):
    """A stable string identity for an instance dict (or object) for seed derivation."""
    if isinstance(inst, dict):
        numbers = tuple(inst.get("numbers", ()))
        target = inst.get("target")
        iid = inst.get("id", "")
        return f"{iid}|{numbers}|{target}"
    # Fall back to an Instance-like object.
    numbers = tuple(getattr(inst, "numbers", ()))
    target = getattr(inst, "target", None)
    iid = getattr(inst, "id", "")
    return f"{iid}|{numbers}|{target}"


def _chain_seed(base_seed, inst, i):
    """Deterministic, process-stable per-chain seed ``seed_i = f(base_seed, inst, i)``.

    Built from a SHA-256 digest (NOT Python's salted ``hash()``) so (i) chain i in
    ``corr_sample`` and chain i in ``iid_bok`` share a seed, (ii) seeds are independent
    across i (no accidental chain-to-chain coupling at β=0), and (iii) reruns reproduce
    in any process. MECHANISM §2.2(B).
    """
    payload = f"{int(base_seed)}::{_inst_id(inst)}::{int(i)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Empirical per-step emission  P_emission(S'|S)  (MECHANISM §1.1, eq. 1)
# ---------------------------------------------------------------------------

def _emission_support(domain, sample, state, N, tau, step_seed, budget,
                      charge="step"):
    """Reconstruct the empirical emission over canonical successors of ``state``.

    Issues ONE ``sample(state, N, tau, step_seed, "step")`` call, parses+executes each
    candidate exactly as ``savi`` does, and tallies the surviving candidates by canonical
    successor key.

    ``charge`` controls how the reconstruction's tokens hit ``budget`` (MECHANISM §3):

    * ``charge="step"`` (the ``corr_step`` arm): charge the FULL ``N``-candidate step draw
      through the frozen ``Budget.record_sample`` (``tokens += Σ len(c.split())``), so the
      step-mode ``N×`` tax is in the ledger by construction (MECHANISM §3.1).
    * ``charge="cheap"`` (the ``corr_cheap`` arm): the ``N``-candidate draw is the model's
      OWN prompt-conditioned proposal distribution — under prompt-level conditioning it is
      emitted at ~1× generation cost, like one chain-mode op (MECHANISM §3.2 BACKUP). So we
      charge only the ONE committed op's generation tokens (≈3, matching ``iid_bok``'s
      per-op cost) — done by the caller, which knows the committed op — and we record the
      ``sample_calls``/``candidates``/``exec`` bookkeeping here WITHOUT the N× token tax.
      The explicit, tunable prompt-overhead token charge for the conditioning context is
      added per-chain by the cheap draw routine (it is NOT free; see ``_draw_one_chain_cheap``).

    Returns a list of records ``(sp_key, sp, prob, text)`` over DISTINCT canonical
    successors, SORTED BY ``sp_key`` (the deterministic total order), with ``prob`` the
    empirical fraction (eq. 1) and ``text`` one representative candidate string that lands
    in ``sp_key`` (used by the cheap arm to charge the committed op's ~1× generation cost).
    One representative raw successor ``sp`` is kept per key (the ``sp_cache`` discipline of
    ``savi``). Empty support (terminal / dead / all-parse-fail) -> ``[]``.
    """
    cands = sample(state, N, tau, step_seed, "step")
    if charge == "step":
        budget.record_sample(cands)
    else:
        # Cheap (prompt-conditioning) charge: the proposal is the model's own conditioned
        # distribution, emitted at ~1× generation; do NOT charge the N× step-draw tax to
        # tokens. Keep the sample_calls/candidates bookkeeping honest (one call, N cands),
        # but the headline `tokens` for this step are charged by the caller as ONE op.
        budget.sample_calls += 1
        budget.candidates += len(cands)

    counts = {}
    sp_cache = {}
    text_cache = {}
    for c in cands:
        move = domain.parse_move(c, state)
        if move is None:
            continue  # malformed / illegal: dropped from P_emission (tokens still charged)
        sp = domain.apply(state, move)
        budget.exec += 1
        sp_key = domain.canon(sp)
        counts[sp_key] = counts.get(sp_key, 0) + 1
        if sp_key not in sp_cache:
            sp_cache[sp_key] = sp
            text_cache[sp_key] = c

    total = sum(counts.values())
    if total == 0:
        return []
    # Sort the support by canonical key (matches decode_core._stable_order tie-break);
    # this fixed order is what makes the inverse-CDF draw — and hence β=0 byte-identity —
    # reproducible.
    keys = sorted(counts.keys())
    return [(k, sp_cache[k], counts[k] / total, text_cache[k]) for k in keys]


# ---------------------------------------------------------------------------
# Repulsion reweight  P~(S'|S,t) ∝ P_emission · exp(-β·n(S',t))  (eq. 2)
# ---------------------------------------------------------------------------

def _reweight(support, beta, n_counter, depth):
    """Apply the Boltzmann repulsion reweight to an emission ``support`` at ``depth``.

    ``support`` is the list of ``(sp_key, sp, p_emission, text)`` from ``_emission_support``.
    ``n_counter`` maps ``(sp_key, depth) -> #chains-so-far that passed through sp_key at
    depth``. Returns a list of ``(sp_key, sp, weight, text)`` with ``weight`` the
    RENORMALIZED ``P~`` (eq. 2), computed in LOG-SPACE for numerical safety (MECHANISM §5.2).

    ``β=0`` SHORT-CIRCUITS to the input probabilities unchanged, so the β=0 path is
    literally the i.i.d. code path (MECHANISM §5.2, §2). The support order is preserved
    (already canon-sorted), so the downstream inverse-CDF draw is order-stable.
    """
    if not support:
        return []
    if beta == 0.0:
        # Identity reweight: P~ == P_emission, byte-for-byte (no log/exp roundtrip).
        return [(k, sp, p, txt) for (k, sp, p, txt) in support]

    log_w = []
    for (k, _sp, p, _txt) in support:
        # p > 0 for every record (counts>0 by construction), so log(p) is finite.
        n = n_counter.get((k, depth), 0)
        log_w.append(math.log(p) - beta * n)
    m = max(log_w)
    raw = [math.exp(lw - m) for lw in log_w]
    z = sum(raw)
    if z <= 0.0:
        return []
    return [(support[i][0], support[i][1], raw[i] / z, support[i][3])
            for i in range(len(support))]


def _inverse_cdf_draw(weights_records, u):
    """Inverse-CDF pick from ``weights_records`` (``(key, sp, weight, text)``) given ``u``.

    The support is assumed already in the deterministic canon-sorted order. Consumes a
    SINGLE uniform ``u`` per draw so that — across β values — only the ``weight`` vector
    changes, never the number of random draws consumed (MECHANISM §2.2(B)). Ties (equal
    weight) are resolved by the canon-sorted position, the same total-order discipline as
    ``decode_core._stable_order``. Returns ``(sp_key, sp, text)``.
    """
    acc = 0.0
    for (k, sp, w, txt) in weights_records:
        acc += w
        if u < acc:
            return (k, sp, txt)
    # Floating-point guard: fall through to the last record (cumulative ~1.0).
    last = weights_records[-1]
    return (last[0], last[1], last[3])


# ---------------------------------------------------------------------------
# Draw ONE chain (shared by both arms; β=0 ⇒ the i.i.d. chain)
# ---------------------------------------------------------------------------

def _draw_one_chain(domain, sample, s0, N, beta, tau, chain_seed, n_counter,
                    budget, max_depth):
    """Draw one chain by stepwise sampling from the (reweighted) per-state emission.

    Steps from ``s0``: at each depth ``t`` (1-indexed for the successor), reconstruct the
    empirical emission via ``_emission_support``, reweight by ``exp(-β·n(·,t))`` over the
    SHARED ``n_counter`` (the chains drawn before this one), draw the next canonical state
    by inverse-CDF over a single ``rng.random()``, and advance. Stops at ``is_goal``, a
    dead / terminal state (no parsed feasible successor), or ``max_depth``.

    The per-step ``sample`` seed is the per-chain ``chain_seed`` (the mock/real backend
    mixes ``canon(state)`` into its own digest, so distinct states already get independent
    streams while distinct chains get distinct ``chain_seed`` — MECHANISM §2.2(B)).

    Returns ``(path_keys, reached_goal)`` where ``path_keys`` is the forward-order list of
    canonical successor keys (depth 1..len). The realized leaf is the last visited state.
    Does NOT mutate ``n_counter`` (the caller commits the path after the draw, so chain i
    is drawn from n built strictly from chains < i — the sequential discipline, §1.2).
    """
    rng = random.Random(chain_seed)
    cur = s0
    path_keys = []
    reached = False
    depth = 0
    while True:
        if max_depth is not None and depth >= max_depth:
            break
        if domain.is_goal(cur):
            reached = True
            break
        if len(cur.values) <= 1:
            break  # terminal non-goal (dead leaf): chain dies as a token-consuming miss
        support = _emission_support(domain, sample, cur, N, tau, chain_seed, budget,
                                    charge="step")
        if not support:
            break  # no feasible successor (all parse-fail / no legal op): dead end
        t = depth + 1  # successor sits at depth t
        weights_records = _reweight(support, beta, n_counter, t)
        if not weights_records:
            break
        u = rng.random()
        sp_key, sp, _txt = _inverse_cdf_draw(weights_records, u)
        path_keys.append(sp_key)
        cur = sp
        depth += 1
        if domain.is_goal(cur):
            reached = True
            break
    return path_keys, reached


# ---------------------------------------------------------------------------
# Cheap (prompt-conditioning) per-chain prompt-overhead token model (MECHANISM §3.2 BACKUP)
# ---------------------------------------------------------------------------

def _cheap_prompt_overhead_tokens(n_prior_states, prompt_overhead_per_state,
                                  prompt_overhead_base):
    """Explicit, TUNABLE prompt-overhead token charge for the cheap arm's conditioning
    context (MECHANISM §3.2 BACKUP). It is NOT free.

    The cheap arm conditions chain ``i``'s per-step proposal on the canonical states the
    earlier chains already visited, by listing those states in the prompt ("已走过 canonical
    前缀,请走不同的解法", PREREG §4). That conditioning context is real prompt text and must
    be charged. We model it as a per-chain charge that GROWS with the number of prior
    canonical states the prompt must list:

        overhead(i) = prompt_overhead_base
                      + prompt_overhead_per_state · (#distinct prior canonical states)

    ``#distinct prior canonical states`` is the size of the avoid-list at the time chain
    ``i`` is drawn = the number of distinct ``(canon, depth)`` cells the counter already
    holds (it grows with the chain index ``i`` AND with the depth of the prior chains, since
    deeper chains contribute more cells). ``prompt_overhead_per_state`` (tokens charged per
    listed canonical state) and ``prompt_overhead_base`` (fixed instruction-string cost) are
    the two tunable knobs; both default >0 so conditioning is never free. The charge is added
    to the SAME frozen ``Budget.tokens`` ledger as generation (MECHANISM §3), so iso-token
    fairness across arms reads the conditioning tax directly off ``result.budget.tokens``.
    """
    base = max(0, int(prompt_overhead_base))
    per = max(0, int(prompt_overhead_per_state))
    return base + per * int(n_prior_states)


def _draw_one_chain_cheap(domain, sample, s0, N, beta, tau, chain_seed, n_counter,
                          budget, max_depth):
    """Draw one chain for the CHEAP (prompt-conditioning) arm (MECHANISM §3.2 BACKUP).

    Identical chain DRAW to ``_draw_one_chain`` — it reconstructs the same conditioned
    proposal ``P_emission`` from the same ``N``-candidate step support, applies the same
    ``exp(-β·n(·,t))`` repulsion over the same shared ``n_counter``, and draws the next
    state by the same single-``rng.random()`` inverse-CDF — so at any β the cheap and step
    arms realize byte-identical chains for a given seed (and at β=0 both equal ``iid_bok``).

    The DIFFERENCE is the cost model (the whole point of the cheap arm; MECHANISM §3.2):

    * GENERATION tokens are charged at ~1× chain-mode (like ``iid_bok``): only the ONE
      committed op's text is charged to ``Budget.tokens`` per step (≈3 tokens), NOT the
      ``N``-candidate step-draw tax. The ``N``-candidate draw stands in for the model's own
      prompt-conditioned proposal distribution, charged with ``charge="cheap"`` so it adds
      sample_calls/candidates/exec bookkeeping but NO step tax to ``tokens``.

    Returns ``(path_keys, reached_goal)``. The per-chain prompt-overhead token charge for
    the conditioning context is added by the CALLER (``corr_cheap``), once per chain, since
    it depends on how many prior canonical states the prompt lists (the avoid-list size).
    """
    rng = random.Random(chain_seed)
    cur = s0
    path_keys = []
    reached = False
    depth = 0
    while True:
        if max_depth is not None and depth >= max_depth:
            break
        if domain.is_goal(cur):
            reached = True
            break
        if len(cur.values) <= 1:
            break  # terminal non-goal (dead leaf): chain dies as a token-consuming miss
        support = _emission_support(domain, sample, cur, N, tau, chain_seed, budget,
                                    charge="cheap")
        if not support:
            break  # no feasible successor (all parse-fail / no legal op): dead end
        t = depth + 1  # successor sits at depth t
        weights_records = _reweight(support, beta, n_counter, t)
        if not weights_records:
            break
        u = rng.random()
        sp_key, sp, txt = _inverse_cdf_draw(weights_records, u)
        # ~1× generation charge: only the ONE committed op's tokens (like iid_bok), NOT the
        # N-candidate step-draw tax. This is what makes the cheap arm "draw whole chains at
        # ~1× generation cost" (MECHANISM §3.2 BACKUP).
        budget.tokens += len(txt.split())
        path_keys.append(sp_key)
        cur = sp
        depth += 1
        if domain.is_goal(cur):
            reached = True
            break
    return path_keys, reached


# ---------------------------------------------------------------------------
# Arm: corr_sample (the constructive S' — sequential canonical-state repulsion)
# ---------------------------------------------------------------------------

def corr_sample(domain, sample, inst, K, N, beta, tau, seed, max_depth=None):
    """Sequential canonical-state repulsion S'; coverage = ANY of K chains hits goal.

    K chains are drawn STRICTLY SEQUENTIALLY (MECHANISM §1.2). A per-(canon, depth) chain
    counter ``n`` starts empty (reset per instance). Chain 1 is drawn from the unmodified
    emission (``n`` all zero ⇒ factor ``e^0=1``); after each chain completes its realized
    path is committed to ``n`` (``n(S',t) += 1`` for each canonical state S' it visited at
    depth t), so each later chain is pushed off its predecessors' canonical states, per
    depth. That shared counter is the entire correlation mechanism.

    Each step's transition is reweighted ``P~(S'|S,t) ∝ P_emission(S'|S)·exp(-β·n(S',t))``
    (eq. 2), drawn by inverse-CDF over a per-chain ``random.Random(seed_i)``. ``β=0``
    short-circuits the reweight, reducing this arm EXACTLY to ``iid_bok``.

    Budget: every step issues one ``sample(state, N, tau, seed_i, "step")`` whose tokens
    are charged through the frozen ``Budget.record_sample`` — so S''s step-mode ``N×``
    overhead is inside ``result.budget.tokens`` (MECHANISM §3).

    Returns a ``Result`` with ``ok`` = coverage, ``budget`` the accumulated ledger, and
    ``detail`` carrying the realized per-chain ``path_keys``, per-chain ``reached`` flags,
    and the final ``n`` counter (for the ``distinct_canon@K`` joint-diversity instrument).
    """
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta}")

    budget = Budget()
    s0 = domain.initial_state(inst)

    n_counter = {}          # (sp_key, depth) -> #chains-so-far through sp_key at depth
    chains = []             # per-chain realized path_keys
    reached_flags = []      # per-chain goal-reached flag
    cum_tokens = []         # accumulated Budget.tokens AFTER each chain (the iso-token trace)
    any_goal = False

    for i in range(K):
        seed_i = _chain_seed(seed, inst, i)
        path_keys, reached = _draw_one_chain(
            domain, sample, s0, N, float(beta), tau, seed_i, n_counter,
            budget, max_depth,
        )
        chains.append(path_keys)
        reached_flags.append(reached)
        cum_tokens.append(int(budget.tokens))
        any_goal = any_goal or reached
        # Commit chain i to the shared counter AFTER the draw, so chain i+1 repels it.
        for t, sp_key in enumerate(path_keys, start=1):
            n_counter[(sp_key, t)] = n_counter.get((sp_key, t), 0) + 1

    return Result(
        ok=any_goal,
        path=None,  # coverage existence statistic (no single winning path for S')
        budget=budget,
        best_score=None,
        detail={
            "arm": "corr_step",
            "beta": float(beta),
            "K": int(K),
            "N": int(N),
            "tau": tau,
            "chains": chains,
            "reached": reached_flags,
            "cum_tokens": cum_tokens,
            "n_counter": n_counter,
            "n_reached": sum(1 for r in reached_flags if r),
        },
    )


# Canonical name for the step-mode repulsion arm (PREREG §5: ``corr_step``); ``corr_sample``
# is kept as the historical alias the tests/instruments already import.
corr_step = corr_sample


# ---------------------------------------------------------------------------
# Arm: corr_cheap (prompt-conditioning repulsion at ~1× generation; MECHANISM §3.2 BACKUP)
# ---------------------------------------------------------------------------

def corr_cheap(domain, sample, inst, K, N, beta, tau, seed, max_depth=None,
               prompt_overhead_per_state=1, prompt_overhead_base=0):
    """Cheap (prompt-conditioning) canonical-state repulsion; coverage = ANY of K hits goal.

    The cheap analogue of ``corr_step``: it induces the SAME ``exp(-β·n(S',t))`` repulsion
    over the SAME sequential per-(canon, depth) counter, with the SAME β, so its realized
    chains are byte-identical to ``corr_step``'s (and, at β=0, to ``iid_bok``) for a given
    seed. The DIFFERENCE is the cost model (MECHANISM §3.2 BACKUP), modelling prompt-level
    conditioning of a real LLM:

    * GENERATION tokens are ~1× chain-mode (like ``iid_bok``): each committed step charges
      only the ONE committed op's tokens (≈3), NOT the ``N``-candidate step-draw tax. The
      ``N``-candidate proposal stands in for the model's own prompt-conditioned distribution
      (``charge="cheap"`` records sample_calls/candidates/exec but no step tax to tokens).
    * PROMPT-OVERHEAD tokens are charged EXPLICITLY per chain for the conditioning context
      (the list of prior canonical states to avoid). This is NOT free: it is
      ``prompt_overhead_base + prompt_overhead_per_state · (#distinct prior canonical
      states)`` tokens, charged into the SAME frozen ``Budget.tokens`` ledger before chain
      ``i`` draws (``_cheap_prompt_overhead_tokens``). The avoid-list — and hence the charge
      — GROWS with the chain index ``i`` and with the depth of prior chains (more cells in
      the counter), so a late chain over a deep instance pays the most conditioning tax.

    At β=0 the arm short-circuits conditioning ENTIRELY (no avoid-list is built, no prompt
    overhead is charged, generation is 1× per op), so ``corr_cheap(β=0)`` reduces EXACTLY to
    i.i.d. best-of-K: its realized chains are BYTE-IDENTICAL to ``iid_bok`` at matched ``N``
    (the V2 equivalence, extended to the cheap arm). Its TOKEN ledger is deliberately the
    cheap ~1× one (it charges only the committed op per step, NOT ``iid_bok``'s N× step tax)
    — that cheaper ledger is the whole point of the prompt-conditioning model, so the cheap
    arm's β=0 reference for cost is ``iid_bok`` at ``N=1`` (the 1× generation baseline), while
    its β=0 reference for the CHAIN DISTRIBUTION is ``iid_bok`` at matched ``N``.

    Returns a ``Result`` mirroring ``corr_step`` (``ok`` = coverage, ``budget`` the ledger
    with generation + prompt-overhead tokens, ``detail`` with chains/reached/cum_tokens, the
    final ``n`` counter, and ``prompt_overhead_tokens`` = the total conditioning tax).
    """
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    if prompt_overhead_per_state < 0 or prompt_overhead_base < 0:
        raise ValueError("prompt-overhead knobs must be >= 0")

    budget = Budget()
    s0 = domain.initial_state(inst)

    n_counter = {}          # (sp_key, depth) -> #chains-so-far through sp_key at depth
    chains = []
    reached_flags = []
    cum_tokens = []         # accumulated Budget.tokens AFTER each chain (the iso-token trace)
    any_goal = False
    prompt_overhead_total = 0
    beta_active = float(beta) != 0.0

    for i in range(K):
        seed_i = _chain_seed(seed, inst, i)
        # Charge the per-chain prompt-overhead for the conditioning context BEFORE the draw
        # (the prompt prefix lists the prior canonical states this chain must avoid). At β=0
        # there is no conditioning at all -> no avoid-list -> no overhead (pure i.i.d.).
        if beta_active:
            n_prior_states = len(n_counter)  # #distinct (canon, depth) cells = avoid-list size
            overhead = _cheap_prompt_overhead_tokens(
                n_prior_states, prompt_overhead_per_state, prompt_overhead_base)
            budget.tokens += overhead
            prompt_overhead_total += overhead
        path_keys, reached = _draw_one_chain_cheap(
            domain, sample, s0, N, float(beta), tau, seed_i, n_counter,
            budget, max_depth,
        )
        chains.append(path_keys)
        reached_flags.append(reached)
        cum_tokens.append(int(budget.tokens))
        any_goal = any_goal or reached
        # Commit chain i to the shared counter AFTER the draw, so chain i+1 repels it.
        for t, sp_key in enumerate(path_keys, start=1):
            n_counter[(sp_key, t)] = n_counter.get((sp_key, t), 0) + 1

    return Result(
        ok=any_goal,
        path=None,
        budget=budget,
        best_score=None,
        detail={
            "arm": "corr_cheap",
            "beta": float(beta),
            "K": int(K),
            "N": int(N),
            "tau": tau,
            "chains": chains,
            "reached": reached_flags,
            "cum_tokens": cum_tokens,
            "n_counter": n_counter,
            "n_reached": sum(1 for r in reached_flags if r),
            "prompt_overhead_tokens": int(prompt_overhead_total),
            "prompt_overhead_per_state": int(prompt_overhead_per_state),
            "prompt_overhead_base": int(prompt_overhead_base),
        },
    )


# ---------------------------------------------------------------------------
# Arm: iid_bok (K i.i.d. chains — the headline baseline / β=0 reference)
# ---------------------------------------------------------------------------

def iid_bok(domain, sample, inst, K, tau, seed, N=1, max_depth=None):
    """K i.i.d. stepwise chains; coverage = ANY reaches goal (the Erdős–Rényi baseline).

    Defined as the STEPWISE i.i.d. limit of S' (MECHANISM §2.2 route A): each of K chains
    is generated by stepwise sampling from the per-state emission ``P_emission(·|S)`` with
    the SAME per-chain seed schedule ``seed_i = _chain_seed(seed, inst, i)`` and the SAME
    per-step ``sample`` calls that ``corr_sample`` issues, drawn through the SAME
    ``_draw_one_chain`` routine with ``beta=0`` and an EMPTY counter that is never updated.
    Hence chain i's law is independent of the others and ``corr_sample(β=0) == iid_bok``
    byte-for-byte (the V2 equivalence is by construction).

    ``N`` (candidates per step used to reconstruct ``P_emission``) defaults to 1 so the
    bare i.i.d. baseline costs ``≈ 3d`` tokens per chain (one candidate per committed step,
    MECHANISM §3.1); pass the same ``N`` as ``corr_sample`` to make the two arms draw
    byte-identical chains (the β=0 equivalence test uses matched ``N``). Coverage is
    unaffected by ``N`` for the i.i.d. arm in expectation; ``N`` only sharpens the empirical
    ``P_emission`` reconstruction and scales the token ledger.

    Budget: every step's ``sample`` call is charged through ``Budget.record_sample`` so the
    ledger is byte-comparable to ``corr_sample`` and to ``decode_core`` (MECHANISM §3).

    Returns a ``Result`` mirroring ``corr_sample`` (``ok`` = coverage, ``detail`` with the
    per-chain paths/reached flags for the joint-diversity instrument).
    """
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")

    budget = Budget()
    s0 = domain.initial_state(inst)

    chains = []
    reached_flags = []
    cum_tokens = []     # accumulated Budget.tokens AFTER each chain (the iso-token trace)
    any_goal = False
    empty_counter = {}  # never updated: the K chains are mutually independent

    for i in range(K):
        seed_i = _chain_seed(seed, inst, i)
        path_keys, reached = _draw_one_chain(
            domain, sample, s0, N, 0.0, tau, seed_i, empty_counter,
            budget, max_depth,
        )
        chains.append(path_keys)
        reached_flags.append(reached)
        cum_tokens.append(int(budget.tokens))
        any_goal = any_goal or reached
        # NB: empty_counter is intentionally NOT updated (i.i.d., no cross-chain coupling).

    return Result(
        ok=any_goal,
        path=None,
        budget=budget,
        best_score=None,
        detail={
            "arm": "iid_bok",
            "K": int(K),
            "N": int(N),
            "tau": tau,
            "chains": chains,
            "reached": reached_flags,
            "cum_tokens": cum_tokens,
            "n_reached": sum(1 for r in reached_flags if r),
        },
    )


# ---------------------------------------------------------------------------
# Joint-diversity instrument helper (MECHANISM §5.3) — convenience, used by tests
# ---------------------------------------------------------------------------

def distinct_canon_at_K(chains):
    """``distinct_canon@K`` over a list of per-chain ``path_keys`` (MECHANISM §5.3).

    Returns ``{"per_depth": {t: #distinct canon at depth t}, "leaf": #distinct leaf canon,
    "union": #distinct canon over all depths}``. A joint (ensemble-level) statistic:
    counts the non-redundancy of the K committed samples TOGETHER, which is exactly what
    coverage depends on and what mere marginal entropy does not control.
    """
    per_depth = {}
    for path in chains:
        for t, key in enumerate(path, start=1):
            per_depth.setdefault(t, set()).add(key)
    leaf = {path[-1] for path in chains if path}
    union = set()
    for s in per_depth.values():
        union |= s
    return {
        "per_depth": {t: len(s) for t, s in sorted(per_depth.items())},
        "leaf": len(leaf),
        "union": len(union),
    }
