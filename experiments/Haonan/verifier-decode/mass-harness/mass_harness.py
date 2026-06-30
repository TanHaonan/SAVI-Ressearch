"""Math harness around the 1/mass criterion (real-model Countdown).

Stage 1: estimate mass = P(one rollout correct) per instance, select the SPARSE
subset (oracle-solvable but best-of-K_big samples 0 correct chains).
Stage 2: on that subset, run soft/exact global decode and ask whether it solves —
via STITCHED paths no rollout produced — what best-of-K_big cannot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPS = os.path.join(_HERE, "_deps")  # vendored core_boot / arms_local / value_model
for _p in (_DEPS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core_boot as cb        # noqa: E402
import arms_local             # noqa: E402


# --------------------------------------------------------------- pure metrics
def canon_seq(domain, s0, moves):
    """Tuple of canonical states visited along `moves` (path identity, merge-aware)."""
    seq = [domain.canon(s0)]
    cur = s0
    for m in moves:
        cur = domain.apply(cur, m)
        seq.append(domain.canon(cur))
    return tuple(seq)


def _edges(seq):
    return list(zip(seq[:-1], seq[1:]))


def analyze(domain, s0, decode_moves, chains_moves):
    """novel = decode's canon path ∉ sampled chains; stitch = novel AND every
    canonical transition covered by some sampled chain (fragments all present)."""
    dseq = canon_seq(domain, s0, decode_moves)
    cseqs, cover = set(), set()
    for ch in chains_moves:
        cs = canon_seq(domain, s0, ch)
        cseqs.add(cs)
        cover.update(_edges(cs))
    dedges = _edges(dseq)
    n_cov = sum(1 for e in dedges if e in cover)
    novel = dseq not in cseqs
    return {"novel": novel, "stitch": bool(novel and n_cov == len(dedges)),
            "n_edges": len(dedges), "n_covered": n_cov}


def _replay(domain, s0, moves):
    cur = s0
    for m in moves:
        cur = domain.apply(cur, m)
    return cur


def chain_parse_correct(domain, s0, text):
    mv = domain.parse_chain(text, s0)
    if mv is None:
        return None
    return list(mv), bool(domain.is_goal(_replay(domain, s0, mv)))


def res_correct(domain, s0, res):
    if not getattr(res, "ok", False) or res.path is None:
        return False
    return bool(domain.is_goal(_replay(domain, s0, res.path)))


# --------------------------------------------------------------- value (optional)
def train_value(log, ks, seed=0):
    try:
        import value_model as vm
        X, y, _ = vm.make_dataset([seed, seed + 1, seed + 2], ks, n_per=200, seed=seed)
        m = vm.ValueModel(capacity="mlp", seed=seed).fit(X, y)
        log(f"value trained on ks={ks} (n={len(y)})")
        return m.value_fn()
    except Exception as e:  # learned arm is optional; exact/freq still run
        log(f"value training skipped: {type(e).__name__}: {str(e)[:150]}")
        return None


# --------------------------------------------------------------- runner
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", default="builtin")
    ap.add_argument("--gen_seed", type=int, default=0)
    ap.add_argument("--gen_n", type=int, default=24)
    ap.add_argument("--gen_k", type=int, default=4)
    ap.add_argument("--Kbig", type=int, default=64, help="selection rollout budget")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lams", default="0.5,1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lams = [float(x) for x in a.lams.split(",") if x.strip()]

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    exact = cb.CountdownDomain()
    if a.set == "builtin":
        insts = cb.load_instances({"set": "builtin"})
    else:
        insts = cb.load_instances({"set": "generated", "seed": a.gen_seed,
                                   "n": a.gen_n, "k": a.gen_k})
    if a.limit:
        insts = insts[: a.limit]
    ks = sorted({len(i.numbers) for i in insts})
    Vlearned = train_value(log, ks, a.seed)

    log(f"loading {a.adapter}")
    model, tok = cb.load_countdown_model(a.adapter, device=a.device)
    emit = cb.make_real_sampler(model, tok, exact, a.device)
    log(f"loaded; {len(insts)} instances, ks={ks}")

    recs = []
    t0 = time.time()
    for idx, inst in enumerate(insts):
        di = cb.instance_dict(inst)
        s0 = exact.initial_state(di)
        md = len(inst.numbers) - 1
        solvable = bool(cb.oracle(exact, di))

        chains_txt = emit(s0, a.Kbig, a.tau, a.seed, "chain")
        chains_moves, n_corr = [], 0
        for t in chains_txt:
            cc = chain_parse_correct(exact, s0, t)
            if cc is None:
                continue
            mv, ok = cc
            chains_moves.append(mv)
            n_corr += int(ok)
        chain_solve = n_corr > 0

        def run_arm(res):
            ok = res_correct(exact, s0, res)
            info = analyze(exact, s0, res.path, chains_moves) if ok else {}
            return {"solved": ok, "only_decode": bool(ok and not chain_solve),
                    "novel": info.get("novel"), "stitch": info.get("stitch"),
                    "n_edges": info.get("n_edges"), "n_covered": info.get("n_covered"),
                    "tokens": int(getattr(res.budget, "tokens", 0) or 0)}

        arms = {}
        arms["exact"] = run_arm(cb.savi(exact, emit, di, K=a.K, N=a.N, edge_mode="freq",
                                        tau=a.tau, seed=a.seed, verifier=True, max_depth=md))
        arms["freq"] = run_arm(cb.savi(exact, emit, di, K=a.K, N=a.N, edge_mode="freq",
                                       tau=a.tau, seed=a.seed, verifier=False, max_depth=md))
        if Vlearned is not None:
            best = None
            for lam in lams:
                r = arms_local.savi_value(exact, emit, di, Vlearned, lam, a.K, a.N, a.tau,
                                          a.seed, md, edge_mode="freq")
                cand = run_arm(r)
                if best is None or (cand["solved"] and not best["solved"]):
                    best = cand
                if cand["solved"]:
                    break
            arms["learned"] = best

        rec = {"id": inst.id, "k": len(inst.numbers), "solvable": solvable,
               "Kbig": a.Kbig, "n_correct_chains": n_corr, "mass_hat": n_corr / a.Kbig,
               "chain_solve": chain_solve, "arms": arms}
        recs.append(rec)
        log(f"[{idx+1}/{len(insts)}] {inst.id} solvable={solvable} "
            f"chains={n_corr}/{a.Kbig} | exact={arms['exact']['solved']} "
            f"learned={arms.get('learned',{}).get('solved')} "
            f"({(time.time()-t0)/(idx+1):.1f}s/inst)")

    # aggregate, esp. over the sparse subset
    def agg(subset):
        n = len(subset) or 1
        out = {"n": len(subset)}
        for arm in ("exact", "learned", "freq"):
            xs = [r["arms"][arm] for r in subset if arm in r["arms"]]
            if not xs:
                continue
            sv = sum(x["solved"] for x in xs)
            od = sum(bool(x["only_decode"]) for x in xs)
            st = sum(bool(x["stitch"]) for x in xs)
            nv = sum(bool(x["novel"]) for x in xs)
            out[arm] = {"solve_rate": sv / len(xs), "only_decode": od / len(xs),
                        "novel_of_solved": (nv / sv if sv else None),
                        "stitch_of_solved": (st / sv if sv else None)}
        return out

    sparse = [r for r in recs if r["solvable"] and not r["chain_solve"]]
    summary = {"all": agg(recs), "sparse_subset": agg(sparse),
               "n_total": len(recs), "n_solvable": sum(r["solvable"] for r in recs),
               "n_sparse": len(sparse)}
    out = {"plan": "mass-harness-countdown",
           "config": {k: getattr(a, k) for k in ("set", "Kbig", "K", "N", "tau",
                                                 "lams", "seed", "gen_n", "gen_k")},
           "summary": summary, "records": recs}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"wrote {a.out} | sparse={len(sparse)} | {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
