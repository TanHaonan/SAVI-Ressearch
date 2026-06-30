"""Tier-A competence-p synthetic emission backend (CPU, no GPU). PREREG §6 Tier A.

Tier A is the PLUMBING + PREREG-FREEZE step (PREREG §6): a clearly-labelled mock
``sample(state, N, tau, seed, mode)`` closure with an EXACT, knob-controlled
per-step emission distribution ``P_emission(S'|S)`` over canonical successors, so
the correlated-coverage harness (the canonical-state repulsion sampler S', the
iso-token ledger, the inverted-U β sweep, the joint-diversity instrument, and the
validity gates) can be built, debugged, and pre-registered with no GPU. It is
"plumbing + principle + freeze", not the science (PREREG §6).

Two knobs (PREREG §6, MECHANISM §3.3, §5.2):

* ``p`` — the per-step GOOD-move probability: the TOTAL probability mass placed on
  goal-reachable moves, where "good" means ``domain.solvable(domain.apply(S, op))``
  is True (the move keeps a goal reachable). ``p=1`` is a perfect generator (all
  mass on good moves), ``p=0.5`` uninformative, ``p`` small adversarial. This is
  the verifier line's competence axis, deliberately same-named/same-shaped
  (PREREG §6) so the ``p × β`` heat-map (H5) shares the axis.

* ``collapse`` — the mode-collapse degree in [0, 1]. Within each move class (good /
  bad) the mass is shaped by a geometric profile whose ratio is set by ``collapse``:
  ``collapse=0`` spreads mass UNIFORMLY across the class (multi-peak emission, the
  decoupled profile), ``collapse=1`` piles ALL the class mass on its single leading
  op (one-hot collapse within the class, the coupled profile). The leading op of a
  class is fixed by a deterministic ``canon``-sorted order so the profile is stable.

These two knobs define an EXACT categorical ``P_emission(S'|S)`` over the canonical
successors of ``S`` (``emission_dist`` returns it in closed form). The ``sample()``
closure draws i.i.d. from that exact distribution per the requested mode:

* ``mode="step"``: N i.i.d. single-op TEXTS drawn from ``P_emission(·|S)`` (each
  ``"<a> <symbol> <b>"``, three space-delimited tokens). This is what the repulsion
  sampler S' consumes to reconstruct ``P_emission`` per layer (MECHANISM §1.1).
* ``mode="chain"``: N i.i.d. whole rollouts, each a ``;``-joined op sequence rolled
  forward by drawing one op per step from ``P_emission(·|cur)`` until ``is_goal`` /
  dead / ``depth_cap``. This is the i.i.d. best-of-K reference's emission.

A separate ``make_collapse_backend`` is the degenerate ONE-HOP control: every draw
returns the single leading op (the ``collapse=1`` limit as an explicit, byte-stable
backend — N copies of one op in step mode, N copies of one chain in chain mode).

Determinism (process-stable; MECHANISM §5.2). Python's builtin ``hash()`` is salted
per process, so we seed every RNG from a SHA-256 digest of the call args
(``seed``, ``canon(state)``, ``mode``, ``N``, ``tau``, ``p``, ``collapse``, kind),
matching the frozen vendored sampler's discipline. Same args -> identical output
list, in any process / with any ``PYTHONHASHSEED``.

Edge cases (committed; MECHANISM §5.2):
* A terminal / dead state (no legal ops) emits ``""`` (an unparseable slot the
  decoder filters) in step mode and an empty chain in chain mode.
* If a class is empty (all moves good, or all bad), all the mass goes to the
  non-empty class (no normalization-by-zero); if BOTH classes are empty the state
  is terminal (handled above).
* ``abstain`` in [0, 1) independently replaces each candidate text with ``""`` at
  that rate, to mock the real model's <1.0 parse rate; ``abstain=0`` is the bare
  on-task generator. Dropped candidates still occupy a slot (they cost tokens in
  the harness ledger, per MECHANISM §5.2).
* ``tau`` modulates spread continuously: it is folded into the effective collapse so
  ``tau=0`` is fully greedy (argmax of ``P_emission``, deterministic) and higher
  ``tau`` flattens toward the raw ``collapse`` profile. The exact ``P_emission`` used
  by S' (``emission_dist``) is the ``tau``-default (raw ``collapse``) distribution;
  the ``tau`` knob exists only so the negative-control ``temp_matched_iid`` arm
  (PREREG §5, H4) can be driven on the same closure.
"""

import hashlib
import random

import core_boot as cb


# ---------------------------------------------------------------------------
# Deterministic, process-stable RNG seeding (sha256 of the call args)
# ---------------------------------------------------------------------------

def _digest_seed(seed, state, domain, mode, N, tau, p, collapse, kind):
    """Process-stable integer RNG seed from a SHA-256 digest of all call args.

    ``canon(state)`` is rendered with ``str`` over its (sorted values, target) so
    equivalent multisets digest identically regardless of value order, mirroring the
    vendored ``sampler._digest_seed``.
    """
    vals, target = domain.canon(state)
    state_str = ",".join(str(v) for v in vals) + "|" + str(target)
    payload = "::".join([
        str(seed),
        state_str,
        str(mode),
        str(int(N)),
        f"{float(tau):.6f}",
        f"{float(p):.4f}",
        f"{float(collapse):.4f}",
        str(kind),
    ])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(seed, state, domain, mode, N, tau, p, collapse, kind):
    return random.Random(
        _digest_seed(seed, state, domain, mode, N, tau, p, collapse, kind)
    )


# ---------------------------------------------------------------------------
# Exact emission distribution  P_emission(S' | S)  (the Tier-A closed form)
# ---------------------------------------------------------------------------

def _geom_weights(n, collapse):
    """Geometric within-class weights for ``n`` ops, shaped by ``collapse`` in [0,1].

    ``collapse=0`` -> all weights equal (uniform spread across the class).
    ``collapse=1`` -> all mass on the leading op (one-hot collapse within the class).
    Returns a list of ``n`` non-negative floats summing to 1 (n>=1).
    """
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    c = min(1.0, max(0.0, float(collapse)))
    if c >= 1.0:
        w = [0.0] * n
        w[0] = 1.0
        return w
    # ratio in [0,1): 0 -> uniform shape would need ratio=1; map collapse->ratio so
    # collapse=0 is uniform (ratio->1) and collapse->1 is one-hot (ratio->0).
    ratio = 1.0 - c
    raw = []
    cur = 1.0
    for _ in range(n):
        raw.append(cur)
        cur *= ratio
    total = sum(raw)
    return [r / total for r in raw]


# Per-(domain, canon(state)) solvability + successor-structure cache. ``solvable``
# (= countdown.reachable) runs a fresh-memo DFS on every call, so calling it per op on
# every one of K*N draws is quadratic-ish and dominates runtime. The successor set of a
# state and each successor's solvability are pure functions of ``canon(state)``, so we
# memoize them once per distinct canonical state per domain object. ``canon`` strictly
# shrinks, so this cache is bounded by the number of distinct reachable multisets.
_STRUCT_CACHE = {}


def _struct(state, domain):
    """Cached (good_recs, bad_recs) for ``state``: de-duped successor records split by
    solvability, in the deterministic canon-sorted order. Each record is
    ``(op, sp, sp_key)``. Pure in ``canon(state)`` + domain identity.
    """
    key = (id(domain), domain.canon(state))
    hit = _STRUCT_CACHE.get(key)
    if hit is not None:
        return hit
    legal = domain.legal_ops(state)
    if not legal:
        res = ([], [])
        _STRUCT_CACHE[key] = res
        return res
    recs = []
    for op in legal:
        sp = domain.apply(state, op)
        recs.append((op, sp, domain.canon(sp)))
    # Deterministic order: sort by canonical successor key, then op tuple, so the
    # "leading op" of each class is stable and the draw is reproducible.
    recs.sort(key=lambda r: (str(r[2]), tuple(r[0])))
    good = [r for r in recs if domain.solvable(r[1])]
    bad = [r for r in recs if not domain.solvable(r[1])]
    res = (good, bad)
    _STRUCT_CACHE[key] = res
    return res


def emission_dist(state, domain, p, collapse):
    """Return the EXACT per-step emission over canonical successors of ``state``.

    The closed-form ``P_emission(S'|S)`` (MECHANISM eq. 1) this Tier-A backend draws
    from, as a list of ``(op, sp, sp_key, prob)`` records over the DE-DUPED legal ops
    (one representative op per distinct canonical successor — ``legal_ops`` already
    de-dups by outcome). GOOD ops (solvable successor) carry total mass ``p``, BAD ops
    total mass ``1-p``; within each class the mass follows the ``collapse``-shaped
    geometric profile over ops sorted by ``canon(sp)`` (deterministic).

    Empty support (terminal / dead state) -> ``[]``. If one class is empty its mass is
    given to the other (no division by zero).
    """
    good, bad = _struct(state, domain)
    if not good and not bad:
        return []

    pp = min(1.0, max(0.0, float(p)))
    # Class mass: good gets pp, bad gets 1-pp; reassign if a class is empty.
    if not good:
        mass_good, mass_bad = 0.0, 1.0
    elif not bad:
        mass_good, mass_bad = 1.0, 0.0
    else:
        mass_good, mass_bad = pp, 1.0 - pp

    out = []
    for cls, mass in ((good, mass_good), (bad, mass_bad)):
        if not cls or mass <= 0.0:
            continue
        wts = _geom_weights(len(cls), collapse)
        for (op, sp, sp_key), w in zip(cls, wts):
            prob = mass * w
            if prob > 0.0:
                out.append((op, sp, sp_key, prob))
    # Renormalize defensively (mass split is already exact, but float-safe).
    z = sum(r[3] for r in out)
    if z <= 0.0:
        return []
    return [(op, sp, sp_key, prob / z) for (op, sp, sp_key, prob) in out]


def _tau_collapse(collapse, tau):
    """Fold ``tau`` into an effective collapse so higher tau -> flatter emission.

    ``tau<=0`` -> fully greedy (effective collapse 1.0; argmax handled by the caller).
    Larger ``tau`` reduces the effective collapse toward 0 (more uniform). At the
    sentinel ``tau`` used for the exact P_emission path (``tau=None``) the raw
    ``collapse`` is returned unchanged.
    """
    if tau is None:
        return float(collapse)
    t = float(tau)
    if t <= 0.0:
        return 1.0
    # Multiplicative shrink of collapse toward 0 as tau grows; monotone, bounded.
    return float(collapse) / (1.0 + t)


# ---------------------------------------------------------------------------
# Competence-p backend (the main Tier-A generator)
# ---------------------------------------------------------------------------

def make_competence_backend(p=0.7, collapse=0.0, depth_cap=8, abstain=0.0,
                            domain=None):
    """Return a competence-``p`` mock ``sample(state, N, tau, seed, mode)`` closure.

    Draws i.i.d. from the EXACT ``emission_dist(state, domain, p, eff_collapse)``
    (closed form; MECHANISM §3.3) where ``eff_collapse`` folds in ``tau`` so ``tau=0``
    is greedy/argmax and larger ``tau`` flattens the emission. Knobs:

    * ``p`` — per-step GOOD-move probability (mass on solvable-successor moves).
    * ``collapse`` — within-class mode-collapse degree in [0,1] (0 uniform, 1 one-hot).
    * ``depth_cap`` — max steps per chain in ``mode="chain"``.
    * ``abstain`` — independent rate of replacing a candidate text with ``""``.

    No oracle witness is ever injected; the only way a correct chain appears is by the
    competence policy actually rolling onto the goal (best-of-K coverage therefore
    drops with depth, the honest H1/H2 direction — PREREG §3).
    """
    dom = domain if domain is not None else cb.CountdownDomain()

    def _draw_one(state, rng, tau):
        """Draw one op from P_emission(·|state); tau=0 -> argmax (deterministic)."""
        eff = _tau_collapse(collapse, tau)
        dist = emission_dist(state, dom, p, eff)
        if not dist:
            return None
        if tau is not None and float(tau) <= 0.0:
            # Greedy: the highest-mass canonical successor (ties by canon order, which
            # emission_dist already imposes via its stable sort -> first max wins).
            best = max(range(len(dist)), key=lambda i: dist[i][3])
            return dist[best][0]
        ops = [r[0] for r in dist]
        weights = [r[3] for r in dist]
        return rng.choices(ops, weights=weights, k=1)[0]

    def _step_texts(state, N, rng, tau):
        legal = dom.legal_ops(state)
        if not legal:
            return [""] * N           # terminal / dead state: unparseable slots
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
        rng = _rng(seed, state, dom, mode, N, tau, p, collapse, "competence")
        if mode == "step":
            cands = _step_texts(state, N, rng, tau)
        elif mode == "chain":
            cands = [_chain_text(state, rng, tau) for _ in range(N)]
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        if abstain and cands:
            cands = ["" if rng.random() < abstain else c for c in cands]
        return cands

    sample.kind = "competence"
    sample.profile = "decoupled" if collapse < 1.0 else "coupled"
    sample.p = float(p)
    sample.collapse = float(collapse)
    sample.depth_cap = int(depth_cap)
    sample.abstain = float(abstain)
    sample.tier = "A-mock"
    sample.domain = dom
    # Expose the exact emission for instruments / S' (raw collapse, tau-default).
    sample.emission_dist = lambda state: emission_dist(state, dom, p, collapse)
    return sample


# ---------------------------------------------------------------------------
# One-hot / collapse control backend (the degenerate width-1 generator)
# ---------------------------------------------------------------------------

def make_collapse_backend(p=1.0, depth_cap=8, abstain=0.0, domain=None):
    """Return a ONE-HOP collapse mock ``sample(state, N, tau, seed, mode)`` closure.

    The degenerate control (the ``collapse=1`` limit as an explicit, byte-stable
    backend): every draw returns the SINGLE leading op of ``emission_dist`` (the
    highest-mass canonical successor under the ``collapse=1`` profile). In step mode it
    returns N copies of that one op text; in chain mode it returns N copies of the one
    greedy chain rolled forward by always taking the leading op. ``p`` still picks the
    class (good vs bad) the leading op is drawn from. Independent of ``tau`` and
    ``seed`` except through ``abstain`` (which still injects ``""`` at its rate).

    This is the H3 contrast / kill-criterion control: width-1 emission collapses K_eff
    to 1 so correlated repulsion has nothing to spread over.
    """
    dom = domain if domain is not None else cb.CountdownDomain()

    def _leading_op(state):
        dist = emission_dist(state, dom, p, collapse=1.0)
        if not dist:
            return None
        best = max(range(len(dist)), key=lambda i: dist[i][3])
        return dist[best][0]

    def _step_texts(state, N):
        op = _leading_op(state)
        if op is None:
            return [""] * N
        return [dom.render_op(state, op)] * N

    def _chain_text(state):
        cur = state
        texts = []
        for _ in range(depth_cap):
            if dom.is_goal(cur) or len(cur.values) <= 1:
                break
            op = _leading_op(cur)
            if op is None:
                break
            texts.append(dom.render_op(cur, op))
            cur = dom.apply(cur, op)
        return ";".join(texts)

    def sample(state, N, tau, seed, mode):
        if N <= 0:
            return []
        if mode == "step":
            cands = _step_texts(state, N)
        elif mode == "chain":
            cands = [_chain_text(state)] * N
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        if abstain and cands:
            rng = _rng(seed, state, dom, mode, N, tau, p, 1.0, "collapse")
            cands = ["" if rng.random() < abstain else c for c in cands]
        return cands

    sample.kind = "collapse"
    sample.profile = "coupled"
    sample.p = float(p)
    sample.collapse = 1.0
    sample.depth_cap = int(depth_cap)
    sample.abstain = float(abstain)
    sample.tier = "A-mock"
    sample.domain = dom
    sample.emission_dist = lambda state: emission_dist(state, dom, p, collapse=1.0)
    return sample
