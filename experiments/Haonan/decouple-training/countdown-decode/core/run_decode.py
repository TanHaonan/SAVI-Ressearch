"""Arm-matrix harness for the countdown-decode capstone (PREREG sec 1/5/6).

Runs the decode arms over a held-out instance set with ONE emission backend and writes
``outputs/results.json`` + ``outputs/run.log``. The backend is the experiment variable:
``mock_decoupled`` / ``mock_onehot`` are the GPU-free mock emissions (decoupled vs
one-hot training proxies); ``real`` loads a trained Qwen3-4B(+LoRA) adapter via
``real_backend`` (the coupled adapter -> the one-hot arm, the decoupled adapter -> the
decoupled arm). All decode arms consume the SAME ``emit`` closure so the comparison is
within one emission; oracle is sampler-independent.

This is the COUNTDOWN capstone of the decoupled-emission line, pivoting off the algebra
version whose Phi (canon = lhs-rhs scalar class) made productive moves canon-invariant and
the trellis degenerate. Countdown is the decoder's NATIVE domain: a state is a multiset of
values + a target; a move combines two values into one; the Phi key
``canon = (sorted values, target)`` STRICTLY SHRINKS each step (one fewer value), so the
step-trellis is finite, bounded, never recurs, and distinct combine orders reaching the
same multiset are genuinely Phi-merged (real aliasing). The "solution" is just the TARGET
and correctness = ``is_goal`` at the terminal (a single remaining value == target).

Arms (PREREG sec 5 table, all reported with bootstrap CI over instances)
-----------------------------------------------------------------------
  greedy           one temperature-0 chain (floor).
  self_consistency N chains, Phi-merged terminal states, modal canonical terminal
                   (the SC baseline the decoupled+savi arm must beat -- PREREG sec 1).
  savi             lambda=0 freq trellis: savi(K, N, edge_mode="freq", verifier=False,
                   max_depth) -- the main claim.
  best_of_k        K whole chains, ANY reaches goal (the oracle-selection ceiling, the
                   +0.18 ruler -- NOT a decision line).
  oracle           absolute solvability ceiling (the headroom numerator).

pass@1 (greedy / savi / best_of_k) = ``Result.ok`` AND the goal node's terminal is a
single value EQUAL to the target (``metrics.leaf_check`` == ``is_goal`` for countdown).
pass@1 (self_consistency) = the modal canonical terminal state is a goal AND leaf-checks.

Diagnostics (PREREG sec 5, per arm where meaningful, for null interpretation / iso-compute)
------------------------------------------------------------------------------------------
  correct_terminal_coverage : fraction of instances whose N step/chain samples reach the
                              correct terminal at least once (did emission sample the
                              right answer in at all).
  trellis_width_before/after: mean per-instance mean layer width (aliasing / k_eff proxy;
                              after < before == Phi merged distinct paths).
  parse_rate                : legal moves / emitted candidates over an N-step probe at the
                              root (emission staying on-task).
  budget                    : mean decode_core Budget (sample_calls/candidates/tokens/exec)
                              per arm -- the iso-compute ledger.

results.json schema
-------------------
  {"arms": {<arm>: {"pass1": {"mean","ci","n"}, "diagnostics": {...}}},
   "config": {...}, "tag": <str>}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # the experiment root (dir holding the ``core`` package)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import core  # noqa: E402  (path setup must precede the import)
from core import metrics  # noqa: E402  (re-exported via the package below)


# Arms whose pass@1 comes from a decode_core Result + leaf-check, and carry a Budget.
_RESULT_ARMS = ("greedy", "savi", "best_of_k")
_ALL_ARMS = ("greedy", "self_consistency", "savi", "best_of_k", "oracle")


# ---------------------------------------------------------------------------
# Backend / emission closure
# ---------------------------------------------------------------------------

def make_emit(backend, adapter, device, log):
    """Return ``(emit, meta)``: a ``sample(state,N,temperature,seed,mode)`` closure.

    ``backend`` in {"mock_decoupled","mock_onehot","real"}. For the mock backends the
    vendored ``core.sample`` is wrapped with the backend baked in (the adapter form the
    decoder consumes). For ``real`` the adapter is loaded (``adapter == "base"`` -> no
    LoRA, the bare instruct base) and ``make_real_sampler`` builds the closure; the heavy
    model load lives entirely here so the mock path never touches a GPU.
    """
    if backend in ("mock_decoupled", "mock_onehot"):
        emit = (lambda state, N, temperature, seed, mode:
                core.sample(state, N, temperature, seed, backend, mode))
        return emit, {"backend": backend, "adapter": None}
    if backend == "real":
        adapter_path = None if (adapter in (None, "base")) else adapter
        log(f"[backend] loading real model adapter={adapter_path!r} device={device}")
        model, tok = core.load_countdown_model(adapter_path, device=device)
        domain = core.CountdownDomain()
        emit = core.make_real_sampler(model, tok, domain, device)
        return emit, {"backend": backend, "adapter": (adapter or "base")}
    raise ValueError(f"unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def load_held_out(spec_str):
    """Parse ``--instances`` into the vendored ``load_instances`` spec, return Instances.

    Accepted forms (``--instances``):
      "builtin" / "builtin_small"                  -> the 28 curated countdown instances.
      "generated:seed=0,n=16,k=4,target_range=10-100" -> seeded generator args (key=val
                                                       csv; ``target_range`` is a "lo-hi"
                                                       pair).
      a path to a .jsonl of {"numbers","target","id"} -> a held-out file (one obj/line;
                                                       ``numbers`` a list of ints).
    """
    spec_str = spec_str.strip()
    if spec_str in ("builtin", "builtin_small"):
        return core.load_instances({"set": "builtin"})
    if spec_str.startswith("generated"):
        kv = {}
        rest = spec_str[len("generated"):].lstrip(":").strip()
        if rest:
            for part in rest.split(","):
                k, _, v = part.partition("=")
                kv[k.strip()] = v.strip()
        spec = {"set": "generated",
                "seed": int(kv.get("seed", 0)),
                "n": int(kv.get("n", 16)),
                "k": int(kv.get("k", 4))}
        if "target_range" in kv:
            lo, _, hi = kv["target_range"].partition("-")
            spec["target_range"] = [int(lo), int(hi)]
        return core.load_instances(spec)
    p = Path(spec_str)
    if p.exists():
        from core import Instance
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(Instance(numbers=tuple(obj["numbers"]),
                                target=int(obj["target"]),
                                id=obj["id"]))
        return out
    raise ValueError(f"unrecognized --instances spec: {spec_str!r}")


# ---------------------------------------------------------------------------
# Terminal-state reconstruction + per-arm correctness
# ---------------------------------------------------------------------------

def _terminal_of(domain, inst_dict, path):
    """Apply a Result ``path`` (list of moves) from the initial state; return terminal.

    ``path`` is the decode_core forward-order move list. Returns the final ``State`` or
    ``None`` if ``path`` is ``None`` (the arm did not reach a goal).
    """
    if path is None:
        return None
    s = domain.initial_state(inst_dict)
    for move in path:
        s = domain.apply(s, move)
    return s


def _result_pass1(domain, inst_dict, target, result):
    """pass@1 for a Result arm: reached a goal whose single value equals the target."""
    if not result.ok:
        return False
    terminal = _terminal_of(domain, inst_dict, result.path)
    return metrics.leaf_check(target, terminal, domain)


# ---------------------------------------------------------------------------
# self_consistency arm (N chains -> Phi-merged modal terminal)
# ---------------------------------------------------------------------------

def run_self_consistency(domain, emit, inst_dict, target, N, tau, seed):
    """Sample N chains, Phi-merge their terminal states, pass@1 = modal is correct.

    Each chain text is parsed against the initial state and rolled to its terminal
    ``State`` (a chain that does not parse contributes no terminal). The modal canonical
    terminal (``metrics.modal_state``) is the SC answer; it passes iff it is a goal and
    leaf-checks. Returns ``(pass1: bool, detail: dict)`` with the modal share and the
    candidate count for diagnostics. One ``sample`` call (chain mode).
    """
    s0 = domain.initial_state(inst_dict)
    cands = emit(s0, N, tau, seed, "chain")
    terminals = []
    for text in cands:
        moves = domain.parse_chain(text, s0)
        if moves is None:
            continue
        cur = s0
        for m in moves:
            cur = domain.apply(cur, m)
        terminals.append(cur)
    modal, count, share = metrics.modal_state(domain, terminals)
    ok = metrics.leaf_check(target, modal, domain)
    return ok, {"modal_share": share, "modal_count": count,
                "n_chains": len(cands), "n_parsed": len(terminals)}


# ---------------------------------------------------------------------------
# Per-instance diagnostics (coverage / parse-rate -- emission-side probes)
# ---------------------------------------------------------------------------

def _correct_terminal_in_chains(domain, emit, inst_dict, target, N, tau, seed):
    """Did ANY of N sampled chains reach the correct terminal? (coverage probe).

    Independent of which arm wins: it asks whether the emission ever SAMPLES the right
    answer at all (PREREG sec 6 diagnostic (b): low coverage == emission did not bring the
    correct answer in). Uses chain mode (the whole-rollout coverage the arms draw on).
    """
    s0 = domain.initial_state(inst_dict)
    cands = emit(s0, N, tau, seed, "chain")
    for text in cands:
        moves = domain.parse_chain(text, s0)
        if moves is None:
            continue
        cur = s0
        for m in moves:
            cur = domain.apply(cur, m)
        if metrics.leaf_check(target, cur, domain):
            return True
    return False


def _parse_rate_at_root(domain, emit, inst_dict, N, tau, seed):
    """Legal-move fraction of an N-step probe at the ROOT (emission on-task rate).

    Draws N single-move candidates from the initial state and reports the fraction that
    ``parse_move`` accepts as a legal move (PREREG sec 6 diagnostic (c): low parse-rate ==
    emission off-task). Returns ``(rate, n_emitted)``; ``rate`` is ``nan`` if nothing was
    emitted (a terminal/degenerate root).
    """
    s0 = domain.initial_state(inst_dict)
    cands = emit(s0, N, tau, seed, "step")
    if not cands:
        return float("nan"), 0
    legal = sum(1 for c in cands if domain.parse_move(c, s0) is not None)
    return legal / len(cands), len(cands)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _budget_dict(b):
    return {"sample_calls": b.sample_calls, "candidates": b.candidates,
            "tokens": b.tokens, "exec": b.exec}


def _mean_budget(budgets):
    """Mean of a list of Budget dicts (the iso-compute ledger per arm)."""
    if not budgets:
        return {"sample_calls": 0.0, "candidates": 0.0, "tokens": 0.0, "exec": 0.0}
    keys = ("sample_calls", "candidates", "tokens", "exec")
    return {k: float(np.mean([b[k] for b in budgets])) for k in keys}


def _mean_or_nan(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _pass1_block(outcomes, seed):
    """{"mean","ci","n"} for a 0/1 per-instance outcome list."""
    arr = [1.0 if o else 0.0 for o in outcomes]
    return {"mean": (float(np.mean(arr)) if arr else float("nan")),
            "ci": metrics.boot_ci(arr, seed=seed),
            "n": len(arr)}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(instances, emit, backend_meta, *, N, K, tau, max_depth, seed, tag, log):
    """Execute the full arm matrix over ``instances``; return the results dict.

    Per instance: greedy, savi (lambda=0 freq), best_of_k, oracle, self_consistency, and
    the emission-side coverage / parse-rate probes. Aggregates pass@1 mean + bootstrap CI
    and the mean diagnostics/Budget per arm.
    """
    domain = core.CountdownDomain()

    # Per-arm per-instance accumulators.
    pass1 = {a: [] for a in _ALL_ARMS}
    budgets = {a: [] for a in _RESULT_ARMS}
    widths_before = {a: [] for a in ("savi",)}
    widths_after = {a: [] for a in ("savi",)}
    coverage = []        # correct-terminal coverage in the N chain samples
    parse_rates = []     # root parse-rate
    sc_shares = []       # self-consistency modal share

    log(f"[run] tag={tag} backend={backend_meta} N={N} K={K} tau={tau} "
        f"max_depth={max_depth} seed={seed} n_inst={len(instances)}")

    for i, inst in enumerate(instances):
        di = {"numbers": inst.numbers, "target": inst.target, "id": inst.id}
        target = inst.target

        # --- Result arms ---------------------------------------------------
        g = core.greedy(domain, emit, di, seed=seed)
        pass1["greedy"].append(_result_pass1(domain, di, target, g))
        budgets["greedy"].append(_budget_dict(g.budget))

        sv = core.savi(domain, emit, di, K=K, N=N, edge_mode="freq", tau=tau,
                       seed=seed, verifier=False, max_depth=max_depth)
        pass1["savi"].append(_result_pass1(domain, di, target, sv))
        budgets["savi"].append(_budget_dict(sv.budget))
        if sv.trellis_widths_before_merge:
            widths_before["savi"].append(float(np.mean(sv.trellis_widths_before_merge)))
            widths_after["savi"].append(float(np.mean(sv.trellis_widths_after_merge)))

        bk = core.best_of_k(domain, emit, di, K=K, tau=tau, seed=seed)
        pass1["best_of_k"].append(_result_pass1(domain, di, target, bk))
        budgets["best_of_k"].append(_budget_dict(bk.budget))

        # --- oracle (sampler-independent ceiling) --------------------------
        pass1["oracle"].append(bool(core.oracle(domain, di)))

        # --- self-consistency (the line to beat) ---------------------------
        sc_ok, sc_detail = run_self_consistency(domain, emit, di, target, N, tau, seed)
        pass1["self_consistency"].append(sc_ok)
        sc_shares.append(sc_detail["modal_share"])

        # --- emission-side diagnostics -------------------------------------
        coverage.append(_correct_terminal_in_chains(domain, emit, di, target, N, tau, seed))
        pr, _ = _parse_rate_at_root(domain, emit, di, N, tau, seed)
        parse_rates.append(pr)

        log(f"  [{i+1}/{len(instances)}] {inst.id}: greedy={pass1['greedy'][-1]} "
            f"sc={sc_ok} savi={pass1['savi'][-1]} bok={pass1['best_of_k'][-1]} "
            f"cover={coverage[-1]} parse={pr:.2f}")

    # --- assemble per-arm blocks ------------------------------------------
    shared_diag = {
        "correct_terminal_coverage": _mean_or_nan([1.0 if c else 0.0 for c in coverage]),
        "parse_rate": _mean_or_nan(parse_rates),
    }
    arms = {}
    for a in _ALL_ARMS:
        diag = dict(shared_diag)
        if a in _RESULT_ARMS:
            diag["budget"] = _mean_budget(budgets[a])
        if a == "savi":
            diag["trellis_width_before_merge"] = _mean_or_nan(widths_before["savi"])
            diag["trellis_width_after_merge"] = _mean_or_nan(widths_after["savi"])
        if a == "self_consistency":
            diag["modal_share"] = _mean_or_nan(sc_shares)
        arms[a] = {"pass1": _pass1_block(pass1[a], seed), "diagnostics": diag}

    # Per-instance outcomes (aligned by index): enables MERGING split runs and the PAIRED
    # savi-vs-self_consistency comparison (the core claim is paired on the same instances,
    # not a marginal-CI overlap). One record per instance, every arm + the emission probes.
    per_instance = [
        {"id": inst.id,
         "numbers": [int(v) for v in inst.numbers], "target": int(inst.target),
         "greedy": bool(pass1["greedy"][i]),
         "self_consistency": bool(pass1["self_consistency"][i]),
         "savi": bool(pass1["savi"][i]),
         "best_of_k": bool(pass1["best_of_k"][i]),
         "oracle": bool(pass1["oracle"][i]),
         "coverage": bool(coverage[i]),
         "parse_rate": (None if (isinstance(parse_rates[i], float) and np.isnan(parse_rates[i]))
                        else float(parse_rates[i]))}
        for i, inst in enumerate(instances)
    ]

    results = {
        "arms": arms,
        "per_instance": per_instance,
        "config": {
            "backend": backend_meta,
            "N": N, "K": K, "tau": tau, "max_depth": max_depth, "seed": seed,
            "n_instances": len(instances),
            "instance_ids": [inst.id for inst in instances],
        },
        "tag": tag,
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description="countdown-decode arm-matrix harness (PREREG sec 5)")
    ap.add_argument("--backend", required=True,
                    choices=["mock_decoupled", "mock_onehot", "real"])
    ap.add_argument("--adapter", default=None,
                    help="real backend only: LoRA adapter dir, or 'base' for no adapter")
    ap.add_argument("--instances", default="builtin",
                    help="'builtin' | 'generated:seed=S,n=N,k=K[,target_range=lo-hi]' | path.jsonl")
    ap.add_argument("--n", type=int, default=16, help="samples per state (N)")
    ap.add_argument("--k", type=int, default=8, help="beam width K (best_of_k uses 16 by default)")
    ap.add_argument("--best_of_k", type=int, default=16, help="K for the best_of_k arm")
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="real backend only")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None,
                    help="results.json path (default outputs/results[_tag].json)")
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    tag = a.tag or f"{a.backend}" + (f"_{a.adapter}" if a.adapter else "")

    # Default outputs go to ``outputs/``; an explicit ``--out`` redirects BOTH the
    # results json and the run log into that file's directory (so a custom run -- or a
    # test -- keeps its log beside its results instead of polluting the shared dir).
    results_path = Path(a.out) if a.out else (
        _HERE.parent / "outputs" / (f"results_{tag}.json" if a.tag else "results.json"))
    out_dir = results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / (f"run_{tag}.log" if a.tag else "run.log")

    log_fh = log_path.open("w", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    try:
        instances = load_held_out(a.instances)
        log(f"[instances] spec={a.instances!r} -> {len(instances)} instances")
        emit, backend_meta = make_emit(a.backend, a.adapter, a.device, log)

        # best_of_k uses its own K (the +0.18 ceiling ruler); the other arms use --k.
        results = run(
            instances, emit, backend_meta,
            N=a.n, K=a.k, tau=a.tau, max_depth=a.max_depth, seed=a.seed,
            tag=tag, log=log,
        )
        # best_of_k arm re-run at its dedicated K if it differs from the beam K.
        if a.best_of_k != a.k:
            domain = core.CountdownDomain()
            bok_pass, bok_budgets = [], []
            for inst in instances:
                di = {"numbers": inst.numbers, "target": inst.target, "id": inst.id}
                bk = core.best_of_k(domain, emit, di, K=a.best_of_k, tau=a.tau, seed=a.seed)
                bok_pass.append(_result_pass1(domain, di, inst.target, bk))
                bok_budgets.append(_budget_dict(bk.budget))
            results["arms"]["best_of_k"]["pass1"] = _pass1_block(bok_pass, a.seed)
            results["arms"]["best_of_k"]["diagnostics"]["budget"] = _mean_budget(bok_budgets)
            results["arms"]["best_of_k"]["diagnostics"]["K"] = a.best_of_k
            log(f"[best_of_k] re-ran at K={a.best_of_k}: "
                f"mean={results['arms']['best_of_k']['pass1']['mean']:.3f}")

        results["config"]["best_of_k_K"] = a.best_of_k
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log(f"[done] wrote {results_path}")
        for arm in _ALL_ARMS:
            blk = results["arms"][arm]["pass1"]
            log(f"  {arm:16s} pass@1 mean={blk['mean']:.3f} "
                f"ci=[{blk['ci'][0]:.3f},{blk['ci'][2]:.3f}] n={blk['n']}")
    finally:
        log_fh.close()
    return results_path


if __name__ == "__main__":
    main()
