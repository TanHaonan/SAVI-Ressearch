"""Deeper Countdown (k in {4,5,6} -> derivation depth {3,4,5}) for PLAN3.

This is PLAN3 deliverable #1. It is a THIN extension of the shared ``CountdownDomain``
(the seam substrate, see ``core_boot``) that adds exactly the two things the deep,
verifier-weighted decode needs and the base domain lacks:

1.  **A persistent, canon-keyed reachability memo** (``DeepCountdownDomain.solvable``).
    The base ``CountdownDomain.solvable`` is ``countdown.reachable``, which builds a
    FRESH memo on every call (``countdown._reachable_memo(state, {})``). At lambda>0 the
    decoder calls ``domain.solvable(s')`` once PER sampled successor (~K*N*depth times
    per instance), and ``reachable`` is exponential in the remaining value count — the
    binding compute constraint PLAN3 §2 flags for k=6. Here ``solvable`` runs the SAME
    exact backward search but threads ONE memo that lives for the whole run, so every
    distinct canonical subproblem is solved at most once across all successors, all
    instances, and the depth curve. The boolean answer is identical to
    ``countdown.reachable`` (asserted in tests); only the work is shared.

2.  **A k-parameterized solvable instance set with the headroom stratum**
    (``make_deep_instances`` / ``DeepInstanceSet``). PLAN3 evaluates on SOLVABLE
    instances (so the oracle ceiling is 1 and a correct answer exists to find) and reads
    deltas on the HEADROOM stratum (oracle-solvable yet the model-free greedy-myopic
    proxy fails) — the slice where a global decode can beat a myopic forward chain (V1).

Everything else — canon (Phi = (sorted values, target)), apply, is_goal, parse_move,
parse_chain, render, render_op — is inherited UNCHANGED from ``CountdownDomain`` so the
decoder, the trained model's prompt/parse contract, and the token ledger are identical
to the base line (Tier A and Tier B share this object; only the sampler is swapped).
"""

import hashlib
from dataclasses import dataclass

import core_boot as cb

# Vendored labelling helpers (operate on ``Instance`` objects). Importable because
# ``core_boot`` put the emission ``core`` dir on sys.path, which exposes ``_deps``.
from _deps.instances import label_instance, greedy_myopic_outcome  # noqa: E402


# ---------------------------------------------------------------------------
# Domain: deep Countdown with a persistent reachability memo
# ---------------------------------------------------------------------------

class DeepCountdownDomain(cb.CountdownDomain):
    """``CountdownDomain`` whose ``solvable`` shares ONE canon-keyed memo run-wide.

    The memo persists on the instance, so it must be a per-RUN object (one domain per
    harness invocation). The decoder reuses it across every node and instance of the
    depth curve, which is exactly where the k=6 cost would otherwise blow up.
    """

    def __init__(self):
        # canon(state) -> bool. Persists for the lifetime of this domain object.
        self._reach_memo = {}
        self._reach_hits = 0
        self._reach_misses = 0

    # --- the persistent-memo backward oracle (overrides base solvable) -----

    def solvable(self, state):
        """Exact backward reachability with a run-persistent canon memo.

        Returns the SAME boolean as ``countdown.reachable(state)`` (every legal op
        strictly shrinks the multiset, so the search is a finite DAG); the only
        difference is that subproblem results are cached across calls by canonical key.
        """
        return self._reachable_shared(state)

    # Keep the helper name from the base domain consistent with the override.
    def reachable(self, state):
        return self._reachable_shared(state)

    def _reachable_shared(self, state):
        memo = self._reach_memo
        key = self.canon(state)
        cached = memo.get(key, None)
        if cached is not None:
            self._reach_hits += 1
            return cached
        self._reach_misses += 1
        if self.is_goal(state):
            memo[key] = True
            return True
        if len(state.values) == 1:
            # single value but not the target -> dead end.
            memo[key] = False
            return False
        result = False
        for op in self.legal_ops(state):
            # Each op reduces the value count by one -> finite recursion, no cycle guard
            # needed (mirrors countdown._reachable_memo).
            if self._reachable_shared(self.apply(state, op)):
                result = True
                break
        memo[key] = result
        return result

    def exact_solvable(self, state):
        """The EXACT (uncorrupted) backward oracle — used for the ceiling / pass@1 truth
        even when a subclass corrupts ``solvable`` (the decode mask)."""
        return self._reachable_shared(state)

    def memo_stats(self):
        """Diagnostic: persistent-memo size and hit/miss counts."""
        return {
            "size": len(self._reach_memo),
            "hits": self._reach_hits,
            "misses": self._reach_misses,
        }


class NoisyVerifierDomain(DeepCountdownDomain):
    """``DeepCountdownDomain`` whose ``solvable`` MASK is the exact oracle corrupted at
    rate ε (PLAN4 T1.2). ``is_goal`` and ``exact_solvable`` stay exact, so pass@1 ground
    truth and the oracle ceiling are uncorrupted — only the per-step decode mask is noisy.

    The flip decision is DETERMINISTIC per ``canon`` (sha256 of noise_seed|canon), so a
    state's noisy label is fixed within a run (the trellis sees a consistent mask).
    ``mode``: "sym" (flip either way at ε), "fp" (only unsolvable→solvable: lets dead-ends
    back in), "fn" (only solvable→unsolvable: prunes correct paths).
    """

    def __init__(self, epsilon=0.0, mode="sym", noise_seed=12345):
        super().__init__()
        self.epsilon = float(epsilon)
        self.mode = mode
        self.noise_seed = int(noise_seed)

    def solvable(self, state):
        truth = self._reachable_shared(state)   # exact, memoized
        if self.epsilon <= 0.0:
            return truth
        h = hashlib.sha256(f"{self.noise_seed}|{self.canon(state)}".encode("utf-8")).digest()
        u = int.from_bytes(h[:8], "big") / 2.0 ** 64   # deterministic uniform [0,1)
        if self.mode == "sym":
            return (not truth) if u < self.epsilon else truth
        if self.mode == "fp":      # only flip unsolvable -> "solvable"
            return True if (not truth and u < self.epsilon) else truth
        if self.mode == "fn":      # only flip solvable -> "unsolvable"
            return False if (truth and u < self.epsilon) else truth
        raise ValueError(f"unknown noise mode: {self.mode!r}")


def max_depth_for_k(k):
    """Derivation depth for k numbers: exactly ``k - 1`` combines reach a single value."""
    return int(k) - 1


# ---------------------------------------------------------------------------
# Instance set: solvable k-number instances + the headroom stratum
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeepInstance:
    """A labelled solvable deep-Countdown instance.

    ``headroom`` is True iff oracle-solvable AND the model-free greedy-myopic proxy
    fails (the slice a global decode can beat a myopic chain). ``witness_len`` is the
    length of one exact witness path (== depth == k-1 for a full solve).
    """

    numbers: tuple
    target: int
    id: str
    k: int
    depth: int
    headroom: bool
    witness_len: int

    def as_dict(self):
        """The decoder's instance dict (what the arms / ``initial_state`` consume)."""
        return {"numbers": self.numbers, "target": self.target, "id": self.id}


@dataclass(frozen=True)
class DeepInstanceSet:
    """A per-k set of solvable instances with the headroom stratum precomputed."""

    k: int
    depth: int
    instances: tuple  # tuple[DeepInstance]
    headroom_ids: tuple

    def headroom(self):
        return tuple(i for i in self.instances if i.headroom)


def make_deep_instances(k, n, seed, target_range=(10, 100), max_draw=None,
                        require_headroom=False):
    """Return a :class:`DeepInstanceSet` of ``n`` SOLVABLE k-number instances.

    Draws seeded candidates via the shared generator
    (``load_instances({"set":"generated", ...})``), keeps only oracle-SOLVABLE ones
    (oracle ceiling = 1, so a correct answer exists to find), labels each with the
    headroom flag and witness length, and returns the first ``n`` in deterministic
    order. ``max_draw`` caps the candidate pool (defaults to ``40*n`` — solvability is
    common at target in [10,100] so this is ample). With ``require_headroom`` the
    returned set is restricted to the headroom stratum (V1 denominator only).
    """
    if max_draw is None:
        max_draw = 40 * max(1, n)
    cands = cb.load_instances({
        "set": "generated", "seed": int(seed), "n": int(max_draw),
        "k": int(k), "target_range": [int(target_range[0]), int(target_range[1])],
    })
    depth = max_depth_for_k(k)
    out = []
    for inst in cands:
        lab = label_instance(inst)
        if not lab["solvable"]:
            continue
        hr = not greedy_myopic_outcome(inst)
        if require_headroom and not hr:
            continue
        out.append(DeepInstance(
            numbers=tuple(inst.numbers), target=int(inst.target), id=inst.id,
            k=int(k), depth=depth, headroom=bool(hr),
            witness_len=int(lab["witness_len"]) if lab["witness_len"] is not None else depth,
        ))
        if len(out) >= n:
            break
    headroom_ids = tuple(i.id for i in out if i.headroom)
    return DeepInstanceSet(k=int(k), depth=depth, instances=tuple(out),
                           headroom_ids=headroom_ids)


def make_depth_curve(ks, n, seed, target_range=(10, 100), **kw):
    """A :class:`DeepInstanceSet` per k in ``ks`` (the H2 depth curve), keyed by k.

    Each k uses an independent seed (``seed + k``) so the per-depth sets are disjoint
    and reproducible.
    """
    return {int(k): make_deep_instances(k, n, seed + int(k), target_range=target_range, **kw)
            for k in ks}
