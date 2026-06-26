"""Multi-slot consistency carrier. T slots, each with k content-free options (random nouns -> letters).
Per-slot elimination clues induce a LOCAL bias (the unary); cross-slot constraints ('slot i and slot j hide
the same / a different kind of prize') couple the slots. The local bias is planted so per-slot greedy is
globally inconsistent (reverse-bite), and the unique globally-consistent assignment (gold) = the exact MAP of
(local-bias unary + hard constraint edges). Truth is exact: prior-free options + deterministic constraints.

Per item we emit:
  options[t]        : k distinct nouns for slot t (prior-free)
  letters           : ['A'..]
  slot_names[t]     : human label for slot t (e.g. 'box 1')  -- used in constraint text and queries
  local_unary       : (T,k) log-potential = the LOCAL evidence the node-emission readout should reflect
  constraints       : list of {i, j, relation in {same,different}, text, compat (k,k) hard log-potential}
  gold              : the unique MAP joint assignment (list length T)
  prompt_slot[t]    : NL prompt whose answer-letter distribution is slot t's node emission (local clue only)

ORACLE-API ADAPTATION (vs the plan's pseudocode):
  oracle.Instance.validate() REQUIRES every long-range edge to be a hub edge (i==0); oracle.exact() solves
  by conditioning on the hub x_0. The plan assumed exact() returns a {"map","map_unique"} dict, but the real
  exact()/brute_force() return {"map","map_energy","logZ","marginals","maxmarg","feasible_map"} -- there is
  NO uniqueness flag. So:
    (1) We restrict every cross-slot constraint pair to (adjacent t,t+1) OR (hub 0,j). Adjacent edges fold
        into the chain `adj`; hub edges go into `longrange` with i==0. Every resulting Instance is therefore
        valid for BOTH oracle.exact and oracle.brute_force. (This never starves a (T,k,C) cell: T=3 gives all
        3 pairs, T=4 gives 5 of 6 pairs, and C<=3.)
    (2) Uniqueness is detected by `is_unique_map` -- brute-force enumeration, strict gap between the best and
        runner-up energy. T,k are tiny so brute is exact and cheap, and it is the SAME backend the
        general-pair decoders use, so "gold == MAP" cannot silently drift between gold-check and decode.
"""
import argparse, json, os, random, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DEPS = HERE / "_deps"
# oracle.py (the verified exact core) is vendored in _deps/.
sys.path.insert(0, str(DEPS))
import oracle  # verified exact core
HARD = oracle.HARD_PEN

LETTERS = "ABCDEF"
NOUNS = ["apple", "train", "cloud", "river", "candle", "ladder", "violin", "pebble", "helmet", "orchard",
         "magnet", "saddle", "lantern", "compass", "feather", "kettle", "anchor", "bramble", "marble",
         "puzzle", "trumpet", "glacier", "beacon", "thicket", "quilt", "harbor", "willow", "cobweb"]

# Energy gap (in log-potential units) required between the MAP and the runner-up for the MAP to count as
# UNIQUE. Local bias is a few units; hard penalty is -1e4. A clean, tie-free planted gold separates by at
# least ~`bias`; anything below this small floor is a near-tie we reject and resample.
UNIQUE_GAP = 1e-6


def _compat(relation, k):
    """(k,k) hard log-potential for a cross-slot relation between option-indices."""
    if relation == "same":
        M = np.full((k, k), HARD, float); np.fill_diagonal(M, 0.0)
    else:  # different
        M = np.zeros((k, k), float); np.fill_diagonal(M, HARD)
    return M


def _allowed_pairs(T):
    """Pairs (i,j), i<j, that map to a VALID oracle Instance: adjacent (fold into chain) OR hub i==0."""
    adj = [(t, t + 1) for t in range(T - 1)]
    hub = [(0, j) for j in range(1, T)]
    seen, out = set(), []
    for p in adj + hub:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def is_unique_map(inst, expected):
    """True iff `expected` is THE unique maximizer of inst's energy (strict gap to runner-up).

    Brute-force enumeration (exact, cheap for T<=4,k<=4). Robust to the absence of a uniqueness flag in
    oracle.exact/brute_force: we recompute the top-2 energies and require the best assignment to equal
    `expected` and to beat every other assignment by > UNIQUE_GAP.
    """
    import itertools
    energies = []
    best_x, best_e = None, -np.inf
    for x in itertools.product(range(inst.n), repeat=inst.T):
        e = oracle.energy(inst, list(x))
        energies.append(e)
        if e > best_e:
            best_e, best_x = e, list(x)
    energies = np.array(energies, float)
    if list(best_x) != list(expected):
        return False
    runner_up = np.partition(energies, -2)[-2] if energies.size >= 2 else -np.inf
    return bool(best_e - runner_up > UNIQUE_GAP)


def make_item(rng, T, k, C, idx, bias=3.0, resid=1.0, max_resamples=4000):
    """Build one carrier item: gold = unique MAP of (local-bias unary + hard constraints), with reverse-bite.

    PLANT (the reverse-bite): each slot's local clue peaks at its gold option, EXCEPT one constrained slot
    `flip_slot` whose local peak is bent to a constraint-breaking option `alt` (bias `bias`). The gold option
    of that bent slot keeps a smaller RESIDUAL bias `resid` (0 < resid < bias). Effect:
      - per-slot argmax of the unary still picks `alt` at flip_slot (reverse-bite: greedy != gold), AND
      - among feasible joints, gold (which hits every slot's main peak except flip_slot, plus the residual at
        flip_slot) strictly out-scores the tie competitor that satisfies the constraint by keeping flip_slot at
        `alt` and moving its partner off-peak. Without `resid` that competitor TIES gold, so gold is not the
        unique MAP -- this is why a single-peak-per-slot plant cannot produce a unique-MAP reverse-bite at
        C=1 (an inherent tie), and why the residual is needed.

    The residual also serves the downstream node-emission story: the bent slot's calibrated emission target
    (softmax of its unary row) keeps mass on BOTH the locally-favoured option and the gold option -- the
    "survivor subspace stays alive" property the experiment tests.

    Resampling is a bounded LOOP (not recursion): unsatisfiable constraints, gold!=MAP, non-unique MAP, or a
    vanished reverse-bite all trigger a redraw, up to `max_resamples`; raises if a cell cannot be filled.
    """
    letters = list(LETTERS[:k])
    slot_names = [f"box {t + 1}" for t in range(T)]
    assert 0.0 < resid < bias, "residual bias must be in (0, bias) to preserve reverse-bite + unique MAP"

    for _attempt in range(max_resamples):
        options = [rng.sample(NOUNS, k) for _ in range(T)]

        # 1) sample C cross-slot constraints on distinct VALID slot pairs (adjacent | hub) so the oracle
        #    Instance stays solvable by both exact and brute_force.
        pairs = _allowed_pairs(T)
        rng.shuffle(pairs)
        chosen = pairs[:C]
        relations = [rng.choice(["same", "different"]) for _ in chosen]

        # 2) sample a gold joint that SATISFIES every constraint (rejection on tiny space).
        def satisfies(x):
            for (i, j), rel in zip(chosen, relations):
                if rel == "same" and x[i] != x[j]:
                    return False
                if rel == "different" and x[i] == x[j]:
                    return False
            return True
        gold = None
        for _ in range(2000):
            cand = [rng.randrange(k) for _ in range(T)]
            if satisfies(cand):
                gold = cand; break
        if gold is None:
            continue  # constraints unsatisfiable for this draw -> resample

        # 3) plant the LOCAL bias unary with the residual (see docstring).
        flip_slot = chosen[0][1]
        alt = (gold[flip_slot] + 1) % k                  # a constraint-breaking option at the bent slot
        local_argmax = list(gold)
        local_argmax[flip_slot] = alt
        local_unary = np.zeros((T, k), float)
        for t in range(T):
            local_unary[t, local_argmax[t]] = bias       # main local peak per slot
        local_unary[flip_slot, gold[flip_slot]] = resid  # residual secondary mass on gold at the bent slot

        # 4) constraint records with NL text + compat matrices.
        constraints = []
        for (i, j), rel in zip(chosen, relations):
            kind = "the same kind of prize" if rel == "same" else "different kinds of prize"
            text = f"{slot_names[i].capitalize()} and {slot_names[j]} hide {kind}."
            constraints.append(dict(i=i, j=j, relation=rel, text=text, compat=_compat(rel, k).tolist()))

        # 5) verify: gold == unique MAP of (local_unary + hard constraints), via brute_force (the exact
        #    backend the general-pair decoders also use). oracle.exact must AGREE with brute_force.
        item = dict(id=f"T{T}_k{k}_C{C}_{idx}", T=T, k=k, C=C, letters=letters, slot_names=slot_names,
                    options=options, local_unary=local_unary.tolist(), constraints=constraints, gold=gold)
        inst = oracle_instance(item)
        bf = oracle.brute_force(inst)
        if (list(bf["map"]) != gold) or (not is_unique_map(inst, gold)):
            continue                                     # resample until clean unique-MAP reverse-bite
        # reverse-bite guard: per-slot greedy of the local unary must disagree with gold somewhere.
        if np.argmax(local_unary, axis=1).tolist() == gold:
            continue
        # 6) per-slot readout prompt (local clue only; no cross-slot info -> node emission is LOCAL).
        item["prompt_slot"] = [_slot_prompt(item, t) for t in range(T)]
        return item

    raise RuntimeError(f"make_item: could not build a unique-MAP reverse-bite item for "
                       f"(T={T},k={k},C={C}) in {max_resamples} resamples")


def _slot_prompt(item, t):
    opts = "; ".join(f"{item['letters'][o]}) the {item['options'][t][o]}" for o in range(item["k"]))
    fav = item["letters"][int(np.argmax(np.array(item["local_unary"])[t]))]
    return (f"A prize is hidden behind one of these options in {item['slot_names'][t]}: {opts}. "
            f"A local hint points to option {fav}. Which option hides the prize? Answer with one letter.")


def oracle_instance(item):
    """oracle.Instance with unary = local bias and pairwise = the TRUE constraint compat matrices.

    Adjacent constraints (j == i+1) fold into the chain `adj`; hub constraints (i == 0) go into `longrange`
    with i==0. By construction (see _allowed_pairs) every constraint is one of these two, so the Instance
    passes oracle.Instance.validate() and is solvable by BOTH oracle.exact and oracle.brute_force.
    """
    T, k = item["T"], item["k"]
    theta = np.array(item["local_unary"], float)
    adj = np.zeros((T - 1, k, k), float)         # no chain constraint by default
    longrange = []
    for e in item["constraints"]:
        i, j = e["i"], e["j"]
        M = np.array(e["compat"], float)
        if j == i + 1:
            adj[i] = adj[i] + M                  # adjacent -> fold into chain edge
        else:
            assert i == 0, f"non-adjacent constraint must be a hub edge (0,j): got ({i},{j})"
            longrange.append((i, j, M))
    inst = oracle.Instance(T=T, n=k, theta=theta, adj=adj, longrange=longrange)
    return inst.validate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", default="3,4")
    ap.add_argument("--ks", default="3,4")
    ap.add_argument("--Cs", default="1,2,3")
    ap.add_argument("--per_cell", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    items = []
    for T in [int(x) for x in a.Ts.split(",")]:
        for k in [int(x) for x in a.ks.split(",")]:
            for C in [int(x) for x in a.Cs.split(",")]:
                if C > len(_allowed_pairs(T)):
                    continue
                for idx in range(a.per_cell):
                    it = make_item(rng, T, k, C, idx)
                    r = rng.random()
                    it["split"] = "train" if r < 0.6 else ("val" if r < 0.75 else "test")
                    items.append(it)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "items.json").write_text(json.dumps(items))
    print("items:", len(items), "| split:", dict(Counter(it["split"] for it in items)))
    print("by (T,k,C):", dict(Counter((it["T"], it["k"], it["C"]) for it in items)))


if __name__ == "__main__":
    main()
