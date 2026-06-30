"""Correctness gates for the merge×noise phase map (SPEC §7).

Covers: the g=0 truncation trap, canon granularity / R↔g, V4(a) emission-weight
invariance, ρ=1 determinism + fn-rate calibration, V5 oracle-handle separation, and the
hard-mask-via-value ≡ savi(verifier=True) equivalence.
"""

import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.dirname(_HERE)
_VD = os.path.dirname(_DIR)
# gen_fair is vendored locally under _deps/ (originally ../mechanism-recombination/gen_fair.py).
_MR = os.path.join(_DIR, "_deps")
for _p in (_DIR, _VD, _MR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import domain_merge as dm  # noqa: E402
import arms_phasemap as ap  # noqa: E402
import gen_fair as gf  # noqa: E402
from decode_core import decode as dc  # noqa: E402


# --------------------------------------------------------------------------
# g knob: the [-0:] trap and canon granularity
# --------------------------------------------------------------------------

def test_g0_tag_stays_empty():
    """g=0 must keep tag () every step (the Python [-0:]==[:] trap)."""
    dom = dm.MergeLatticeDomain(0)
    s = dom.initial_state({"target": 10, "depth": 5})
    for v in (1, 2, 3, 1):
        s = dom.apply(s, v)
        assert s.tag == (), f"g=0 tag leaked: {s.tag}"
    assert dom.canon(s) == (s.s, s.r, s.T, ())


def test_ginf_full_path():
    dom = dm.MergeLatticeDomain(math.inf)
    s = dom.initial_state({"target": 10, "depth": 5})
    moves = (1, 2, 3)
    for v in moves:
        s = dom.apply(s, v)
    assert s.tag == moves


def test_g2_keeps_last_two():
    dom = dm.MergeLatticeDomain(2)
    s = dom.initial_state({"target": 20, "depth": 6})
    for v in (1, 2, 3, 1, 2):
        s = dom.apply(s, v)
    assert s.tag == (1, 2)


def test_merge_collapses_at_g0_not_ginf():
    """Two move-orders to the same (s,r): merge at g=0, distinct at g=inf."""
    inst = {"target": 30, "depth": 6}
    for g, should_merge in [(0, True), (math.inf, False)]:
        dom = dm.MergeLatticeDomain(g)
        s0 = dom.initial_state(inst)
        a = dom.apply(dom.apply(s0, 1), 2)  # +1 then +2
        b = dom.apply(dom.apply(s0, 2), 1)  # +2 then +1  -> same (s=3, r=4)
        assert (a.s, a.r) == (b.s, b.r) == (3, 4)
        if should_merge:
            assert dom.canon(a) == dom.canon(b)
        else:
            assert dom.canon(a) != dom.canon(b)


# --------------------------------------------------------------------------
# V4(a): emission competence weights are g-invariant (distribution fixed)
# --------------------------------------------------------------------------

def test_v4_weights_invariant_across_g():
    """_classify_weights must be identical across g for a fixed (s,r,T)."""
    inst = {"target": 40, "depth": 8}
    p = 0.7
    d0 = dm.MergeLatticeDomain(0)
    di = dm.MergeLatticeDomain(math.inf)
    # walk a few steps; at each state compare the per-move weight vector.
    s0 = d0.initial_state(inst)
    si = di.initial_state(inst)
    for v in (1, 2, 3):
        texts0 = dm.lattice_enumerate(s0)
        textsi = dm.lattice_enumerate(si)
        kept0, w0 = gf._classify_weights(d0, texts0, s0, p)
        kepti, wi = gf._classify_weights(di, textsi, si, p)
        moves0 = [m for (_t, m, _sp) in kept0]
        movesi = [m for (_t, m, _sp) in kepti]
        assert moves0 == movesi
        assert w0 == pytest.approx(wi), f"weights diverged across g at {(s0.s, s0.r)}"
        s0 = d0.apply(s0, v)
        si = di.apply(si, v)


# --------------------------------------------------------------------------
# Noisy verifier: ρ=1 determinism + fn-rate calibration + V5 separation
# --------------------------------------------------------------------------

def _walk_states(dom, inst, vs):
    s = dom.initial_state(inst)
    out = [s]
    for v in vs:
        if s.r <= 0:
            break
        s = dom.apply(s, v)
        out.append(s)
    return out


def test_rho1_deterministic_and_path_independent():
    """ρ=1: same (s,r,T) -> same label, stable across calls and across paths/tags."""
    nd = dm.NoisyMergeLatticeDomain(0, eps=0.3, mode="sym", rho=1.0, noise_seed=0)
    s = dm.MergeLatticeState(s=5, r=4, T=20, tag=(1, 2))
    s_other_path = dm.MergeLatticeState(s=5, r=4, T=20, tag=(3, 1))  # same (s,r,T)
    labels = {nd.solvable(s) for _ in range(20)}
    assert len(labels) == 1, "rho=1 label not stable across calls"
    assert nd.solvable(s) == nd.solvable(s_other_path), "rho=1 leaked path dependence"
    nd2 = dm.NoisyMergeLatticeDomain(math.inf, eps=0.3, mode="sym", rho=1.0, noise_seed=0)
    assert nd.solvable(s) == nd2.solvable(s), "label must not depend on g"


def test_rho1_fn_rate_matches_eps():
    """Over many truly-solvable states, fn mode flips ~eps of them (±tolerance)."""
    eps = 0.2
    nd = dm.NoisyMergeLatticeDomain(0, eps=eps, mode="fn", rho=1.0, noise_seed=1)
    flips = tot = 0
    for T in range(10, 90):
        for s_sum in range(0, T):
            for r in range(1, 12):
                st = dm.MergeLatticeState(s=s_sum, r=r, T=T)
                if not nd._truth(st):
                    continue
                tot += 1
                if nd.solvable(st) is False:
                    flips += 1
    rate = flips / tot
    assert abs(rate - eps) < 0.03, f"fn rate {rate:.3f} far from eps {eps}"


def test_fp_mode_never_prunes_solvable():
    nd = dm.NoisyMergeLatticeDomain(0, eps=0.5, mode="fp", rho=1.0, noise_seed=2)
    for T in range(10, 60):
        for s_sum in range(0, T):
            for r in range(1, 8):
                st = dm.MergeLatticeState(s=s_sum, r=r, T=T)
                if nd._truth(st):
                    assert nd.solvable(st) is True, "fp mode wrongly pruned a solvable node"


def test_v5_corruption_does_not_touch_exact_domain():
    """The exact domain (generator handle) is independent of any noisy domain (decoder)."""
    inst = {"target": 35, "depth": 7}
    exact = dm.MergeLatticeDomain(0)
    noisy = dm.NoisyMergeLatticeDomain(0, eps=0.5, mode="sym", rho=1.0, noise_seed=3)
    # The exact domain's solvable is the closed-form oracle, untouched by any noise.
    for st in _walk_states(exact, inst, (1, 2, 3, 1, 2, 3, 1)):
        assert exact.solvable(st) == (st.r <= (st.T - st.s) <= 3 * st.r)
    # And the noisy domain disagrees with exact on at least one state at this eps.
    disagree = any(noisy.solvable(st) != exact.solvable(st)
                   for st in _walk_states(exact, inst, range(1, 8)))
    assert disagree, "eps=0.5 produced no corruption — noise wiring dead"


def test_eps0_is_exact():
    nd = dm.NoisyMergeLatticeDomain(0, eps=0.0, mode="fn", rho=1.0, noise_seed=0)
    for T in range(10, 50):
        for s_sum in range(0, T):
            for r in range(1, 8):
                st = dm.MergeLatticeState(s=s_sum, r=r, T=T)
                assert nd.solvable(st) == nd._truth(st)
                assert (nd.value(st) > 0.5) == nd._truth(st)


# --------------------------------------------------------------------------
# savi_value: hard-threshold with an EXACT value ≡ savi(verifier=True)
# --------------------------------------------------------------------------

def test_savi_value_hard_exact_equals_verifier_on():
    """savi_value(exact value, hard_thresh=0.5) reaches goal iff savi(verifier=True) does."""
    p, K, N, tau, seed = 0.7, 8, 16, 1.0, 0
    g = 0
    exact = dm.MergeLatticeDomain(g)
    exact_noisy = dm.NoisyMergeLatticeDomain(g, eps=0.0, mode="fn", rho=1.0)  # exact value
    insts = dm.make_lattice_instances(depth=6, n=12, seed=0)
    gen = gf.make_fair_generator(exact, dm.lattice_enumerate, p, 6)
    for inst in insts:
        idict = dm.lattice_inst_dict(inst)
        r_verif = dc.savi(exact, gen, idict, K, N, "freq", tau, seed,
                          verifier=True, max_depth=6)
        r_value = ap.savi_value(exact_noisy, gen, idict, exact_noisy.value, lam=1.0,
                                K=K, N=N, tau=tau, seed=seed, max_depth=6,
                                edge_mode="freq", hard_thresh=0.5)
        assert r_verif.ok == r_value.ok, f"hard-value != verifier_on on {inst.id}"


def _grid_solvable(dom, D, n=24, n_seeds=8):
    """Yield every solvable reachable state over the cell's instance grid (the decoder's
    state distribution)."""
    for sd in range(n_seeds):
        for inst in dm.make_lattice_instances(D, n, sd):
            T = int(inst.target)
            for r in range(0, D + 1):
                for s in range(0, max(dm.M) * (D - r) + 1):
                    st = dm.MergeLatticeState(s=s, r=r, T=T)
                    if dom._truth(st):
                        yield st


def test_boundary_profile_mean_preserving():
    """profile='boundary' redistributes eps toward the boundary but keeps the SAME mean
    fn-rate over the realized solvable-state distribution (tests geometry, not amount)."""
    D, eps = 24, 0.18
    uni = dm.NoisyMergeLatticeDomain(0, eps=eps, mode="fn", rho=1.0, noise_seed=0)
    bnd = dm.NoisyMergeLatticeDomain(0, eps=eps, mode="fn", rho=1.0, noise_seed=0,
                                     profile="boundary", prof_depth=D)
    u_flips = b_flips = tot = 0
    for st in _grid_solvable(uni, D):
        tot += 1
        u_flips += int(uni.solvable(st) is False)
        b_flips += int(bnd.solvable(st) is False)
    u_rate, b_rate = u_flips / tot, b_flips / tot
    assert abs(u_rate - eps) < 0.02, f"uniform fn-rate {u_rate:.3f} != eps"
    assert abs(b_rate - eps) < 0.02, f"boundary fn-rate {b_rate:.3f} not mean-preserving"


def test_boundary_profile_concentrates_at_boundary():
    """Boundary profile flips far MORE solvable nodes at the edge (bin 1) than far (bin 5),
    with a gradient near the measured ~3.5x; uniform is flat across bins."""
    D, eps = 24, 0.18
    bnd = dm.NoisyMergeLatticeDomain(0, eps=eps, mode="fn", rho=1.0, noise_seed=0,
                                     profile="boundary", prof_depth=D)
    uni = dm.NoisyMergeLatticeDomain(0, eps=eps, mode="fn", rho=1.0, noise_seed=0)
    b_flip = {b: [0, 0] for b in range(1, 6)}   # bin -> [flips, total]
    u_flip = {b: [0, 0] for b in range(1, 6)}
    for st in _grid_solvable(bnd, D):
        b = dm.boundary_bin(st)
        b_flip[b][1] += 1; b_flip[b][0] += int(bnd.solvable(st) is False)
        u_flip[b][1] += 1; u_flip[b][0] += int(uni.solvable(st) is False)
    b_rate = {b: b_flip[b][0] / b_flip[b][1] for b in b_flip if b_flip[b][1]}
    u_rate = {b: u_flip[b][0] / u_flip[b][1] for b in u_flip if u_flip[b][1]}
    assert b_rate[1] > b_rate[5], "boundary profile not higher at the edge"
    assert b_rate[1] / b_rate[5] > 2.5, f"gradient {b_rate[1]/b_rate[5]:.2f} too weak"
    # uniform stays ~flat across bins (each ~eps)
    assert max(u_rate.values()) - min(u_rate.values()) < 0.06, "uniform not flat across bins"
