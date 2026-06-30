"""Merge-granularity lattice + decoder-only corrupted verifier (SPEC §2.2, §2.3, §2.4).

Two knobs on the integer-sum lattice substrate (``../mechanism-recombination/
domain_lattice.py``), both designed to leave the EMISSION DISTRIBUTION fixed:

* ``MergeLatticeDomain(g)`` — the **redundancy R** knob. A history ``tag`` is carried on
  the state and ``canon`` retains its last ``g`` moves:
    - ``g=0``   -> ``canon=(s,r,T)``            -> maximum merge (current lattice, high R)
    - ``g=inf`` -> ``canon=(s,r,T, full path)``  -> tree, R≈1 (the Countdown chain)
    - ``g=m``   -> last m moves                  -> intermediate
  Provably emission-DISTRIBUTION-invariant: the three moves always yield distinct sums
  (so successor dedup never fires regardless of tag) and competence depends only on
  ``solvable(successor) = f(s,r,T)``. (Realized interior SAMPLES still vary with g via the
  generator's canon-keyed RNG — the faithful meaning of merge-vs-resample, see SPEC V4.)

* ``NoisyMergeLatticeDomain(g, eps, mode, rho, noise_seed)`` — the **decoder-only**
  imperfect verifier. Inherits canon/apply/is_goal from MergeLatticeDomain (the trellis is
  structurally identical); overrides ONLY ``solvable`` (hard mask, arm A3) and adds
  ``value`` (soft probability, arm A4) so both read the SAME corrupted label of one
  imperfect verifier. Corruption is keyed on ``(s,r,T)`` — the observable state features a
  learned value would see — NOT the tag, so at ρ=1 every path through a node reads the same
  (wrong) label (a faithful deterministic state-function verifier). ρ=0 redraws per call
  (decorrelated noise) via an internal counter.

The generator (gen_fair) must be built on the EXACT ``MergeLatticeDomain(g)``; only the
decode arm runs on the noisy domain (SPEC §2.4 — the two oracle handles).
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass, field

# Reuse the lattice constants + instance generation (tag-agnostic) from the substrate.
# Vendored locally under _deps/ (originally ../mechanism-recombination/domain_lattice.py).
_MR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_deps")
if _MR not in sys.path:
    sys.path.insert(0, _MR)

import domain_lattice as _lat  # noqa: E402  (M, make_lattice_instances, lattice_inst_dict)

M = _lat.M
_MIN_V = min(M)
_MAX_V = max(M)


# ---------------------------------------------------------------------------
# State: (s, r, T) + a history tag (the merge-granularity carrier)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeLatticeState:
    """Lattice node ``(s, r)`` with target ``T`` and a history ``tag`` (a move tuple).

    ``tag`` carries up to ``g`` recent moves; ``canon`` reads it so that two
    move-orders reaching the same ``(s,r,T)`` merge iff their retained tags agree.
    ``tag`` does NOT affect ``solvable`` / ``is_goal`` / competence (those read s,r,T).
    """

    s: int
    r: int
    T: int
    tag: tuple = ()

    @property
    def target(self) -> int:
        return self.T


# ---------------------------------------------------------------------------
# MergeLatticeDomain(g) — exact oracle, canon granularity g
# ---------------------------------------------------------------------------

class MergeLatticeDomain:
    """Lattice domain with a tunable merge granularity ``g`` (SPEC §2.2).

    ``g=0`` full merge (canon=(s,r,T)); ``g=math.inf`` no merge (canon keys the full path);
    ``g=m`` keeps the last m moves. Everything except ``initial_state`` / ``apply`` /
    ``canon`` matches the plain lattice (parse/solvable/is_goal read s,r,T only).
    """

    def __init__(self, g):
        # g is a non-negative int OR math.inf. Store as-is.
        if g != math.inf:
            g = int(g)
            if g < 0:
                raise ValueError(f"g must be >= 0 or inf, got {g}")
        self.g = g

    # -- construction -------------------------------------------------------
    def initial_state(self, instance) -> MergeLatticeState:
        s0 = int(instance.get("s0", 0))
        T = int(instance["target"])
        D = int(instance["depth"])
        return MergeLatticeState(s=s0, r=D, T=T, tag=())

    # -- merge key (the R knob) ---------------------------------------------
    def _retain(self, tag: tuple, v: int) -> tuple:
        """New tag after appending move ``v``, truncated to granularity g."""
        if self.g == 0:
            return ()
        full = tag + (int(v),)
        if self.g == math.inf:
            return full
        return full[-self.g:]

    def canon(self, state: MergeLatticeState):
        return (state.s, state.r, state.T, state.tag)

    # -- goal / transition --------------------------------------------------
    def is_goal(self, state: MergeLatticeState) -> bool:
        return state.s == state.T and state.r == 0

    def apply(self, state: MergeLatticeState, move) -> MergeLatticeState:
        v = int(move)
        return MergeLatticeState(s=state.s + v, r=state.r - 1, T=state.T,
                                 tag=self._retain(state.tag, v))

    # -- parsing (legality reads state.r only) ------------------------------
    def parse_move(self, text, state: MergeLatticeState):
        if text is None:
            return None
        parts = str(text).strip().split()
        if len(parts) != 2 or parts[0] != "add":
            return None
        try:
            v = int(parts[1])
        except ValueError:
            return None
        if v not in M:
            return None
        if state.r <= 0:
            return None
        return v

    def parse_chain(self, text, state: MergeLatticeState):
        if text is None:
            return None
        text = str(text).strip()
        if text == "":
            return []
        moves = []
        cur = state
        for tok in text.split(";"):
            tok = tok.strip()
            if tok == "":
                continue
            move = self.parse_move(tok, cur)
            if move is None:
                return None
            moves.append(move)
            cur = self.apply(cur, move)
        return moves

    # -- exact O(1) oracle --------------------------------------------------
    def solvable(self, state: MergeLatticeState) -> bool:
        residual = state.T - state.s
        return (_MIN_V * state.r) <= residual <= (_MAX_V * state.r)

    def render(self, state: MergeLatticeState) -> str:
        return f"sum={state.s} remaining={state.r} target={state.T}"


# ---------------------------------------------------------------------------
# NoisyMergeLatticeDomain — decoder-only imperfect verifier (eps, mode, rho)
# ---------------------------------------------------------------------------

def _u01(*parts) -> float:
    """Process-stable uniform(0,1) from a sha256 of the payload (no Python hash())."""
    payload = "::".join(str(p) for p in parts)
    d = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") / float(1 << 64)


# ---------------------------------------------------------------------------
# Boundary-concentrated corruption profile (from the PRM calibration check)
# ---------------------------------------------------------------------------
#
# The uniform-eps corruption above flips a flat fraction eps of solvable nodes
# regardless of where they sit. The PRM boundary-step calibration check
# (../prm-boundary-calibration/) found that a real PRM *and* an independent LLM
# critic both concentrate their solvable->unsolvable errors (false positives) on
# the steps adjacent to the on/off-path boundary: FP rate by distance-to-boundary
# (1=last-correct .. 5=far) ~ {0.234, 0.156, 0.121, 0.102, 0.066}, a ~3.5x
# gradient that is data-driven (not PRM-specific). _FP_BY_DIST is that measured
# shape; boundary_weights() turns it into per-bin multipliers whose SOLVABLE-state
# -weighted mean is 1, so profile='boundary' redistributes the SAME total eps
# toward the boundary (it tests error *geometry*, not error *amount*).

_FP_BY_DIST = {1: 0.234, 2: 0.156, 3: 0.121, 4: 0.102, 5: 0.066}

# lattice analog of "distance to the first-error step": integer slack to the
# nearest infeasibility edge (margin 0 -> at the edge -> bin 1, the last-correct
# state before things can go wrong; large margin -> deep in the solvable region).


def boundary_bin(state, n_bins: int = 5) -> int:
    residual = state.T - state.s
    margin = min(residual - _MIN_V * state.r, _MAX_V * state.r - residual)
    if margin < 0:
        return 1  # over the edge (unsolvable); unused for the solvable->unsolvable flip
    return min(n_bins, 1 + int(margin))


_WEIGHT_CACHE: dict = {}


def boundary_weights(depth, n: int = 24, n_seeds: int = 8, fp_by_dist=None) -> dict:
    """Mean-preserving per-bin multipliers (solvable-state-weighted mean = 1) at one
    depth, computed over the reachable grid (the state distribution the decoder visits)."""
    from collections import Counter
    fp = dict(fp_by_dist) if fp_by_dist else dict(_FP_BY_DIST)
    key = (int(depth), int(n), int(n_seeds), tuple(sorted(fp.items())))
    if key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[key]
    dom = MergeLatticeDomain(0)
    D = int(depth)
    c = Counter()
    for sd in range(int(n_seeds)):
        for inst in make_lattice_instances(D, n, sd):
            T = int(inst.target)
            for r in range(0, D + 1):
                for s in range(0, _MAX_V * (D - r) + 1):
                    st = MergeLatticeState(s=s, r=r, T=T)
                    if dom.solvable(st):
                        c[boundary_bin(st)] += 1
    tot = sum(c.values()) or 1
    zbar = sum((c[b] / tot) * fp[b] for b in fp)
    w = {b: fp[b] / zbar for b in fp}
    _WEIGHT_CACHE[key] = w
    return w


class NoisyMergeLatticeDomain(MergeLatticeDomain):
    """MergeLatticeDomain whose VERIFIER (``solvable``/``value``) is corrupted at rate eps.

    ``mode``: 'fn' flips only solvable->unsolvable (prune a good node — the killer);
              'fp' flips only unsolvable->solvable (admit a dead node — tolerant);
              'sym' flips either class at rate eps.
    ``rho``:  1 -> corruption is a deterministic function of (s,r,T) (a realistic learned
                   value: every path through the node reads the same label);
              0 -> corruption redrawn per call (decorrelated noise; SPEC Tier-2 / P3).
    The structural methods (canon/apply/is_goal/parse) are the EXACT inherited ones; only
    the verifier signal is corrupted. Build a FRESH instance per arm-run when rho=0 (the
    call counter must not leak across arms).
    """

    def __init__(self, g, eps, mode="fn", rho=1.0, noise_seed=0,
                 value_hi=0.95, value_lo=0.05, profile="uniform", prof_depth=None):
        super().__init__(g)
        if mode not in ("fn", "fp", "sym"):
            raise ValueError(f"unknown mode: {mode!r}")
        if profile not in ("uniform", "boundary"):
            raise ValueError(f"unknown profile: {profile!r}")
        self.eps = float(eps)
        self.mode = mode
        self.rho = float(rho)
        self.noise_seed = int(noise_seed)
        self.value_hi = float(value_hi)
        self.value_lo = float(value_lo)
        self.profile = profile
        # 'boundary': redistribute eps toward the on/off-path boundary (mean-preserving).
        # Weights depend on the realized state distribution, hence on depth.
        if profile == "boundary":
            if prof_depth is None:
                raise ValueError("profile='boundary' requires prof_depth (the run depth)")
            self._weights = boundary_weights(prof_depth)
        else:
            self._weights = None
        self._calls = 0  # rho=0 per-call nonce (deterministic in call order)

    def _truth(self, state) -> bool:
        residual = state.T - state.s
        return (_MIN_V * state.r) <= residual <= (_MAX_V * state.r)

    def _flip_u(self, state) -> float:
        # Keyed on (s,r,T) ONLY (the observable features) so a learned-style verifier is a
        # function of the node, not the path. rho<1 mixes in a per-call nonce.
        base = (state.s, state.r, state.T, self.mode, self.noise_seed)
        if self.rho >= 1.0:
            return _u01(*base)
        if self.rho <= 0.0:
            self._calls += 1
            return _u01(*base, "call", self._calls)
        # Intermediate rho: with prob rho use the stable draw, else a fresh per-call draw.
        self._calls += 1
        gate = _u01(*base, "gate")
        return _u01(*base) if gate < self.rho else _u01(*base, "call", self._calls)

    def _corrupted_label(self, state) -> bool:
        truth = self._truth(state)
        if self.eps <= 0.0:
            return truth
        # boundary profile concentrates the solvable->unsolvable flip (the measured
        # false-positive direction) near the boundary, mean-preserving over solvable
        # states; the unsolvable->solvable rate stays flat.
        eps_eff = self.eps
        if self.profile == "boundary" and truth:
            eps_eff = min(1.0, self.eps * self._weights[boundary_bin(state)])
        u = self._flip_u(state)
        if u >= eps_eff:
            return truth
        if self.mode == "sym":
            return not truth
        if self.mode == "fn":
            return False if truth else truth   # only solvable->unsolvable
        return True if not truth else truth    # 'fp': only unsolvable->solvable

    # hard mask (arm A3 via savi(verifier=True))
    def solvable(self, state) -> bool:
        return self._corrupted_label(state)

    # soft value in (0,1] (arm A4 via savi_value value_fn) — same corrupted label
    def value(self, state) -> float:
        return self.value_hi if self._corrupted_label(state) else self.value_lo


# ---------------------------------------------------------------------------
# Instance generation (reuse the substrate's; tag-agnostic)
# ---------------------------------------------------------------------------

make_lattice_instances = _lat.make_lattice_instances
lattice_inst_dict = _lat.lattice_inst_dict
lattice_enumerate = _lat.lattice_enumerate
