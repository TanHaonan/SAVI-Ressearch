"""Elicitation probe: can a BETTER-WORDED pairwise query let a cheap, separate model supply
usable consistency edges?

Background (the validation that prompted this): the latent-consistency decoder works when fed the TRUE
constraint as edges (greedy joint-EM 0.00 -> marginal 0.88 / viterbi 0.92 on 25 held-out items). The
cheap edge source FAILED: a 1.1B model's pairwise yes/no margins separate consistent from inconsistent
option pairs at AUC ~0.475 (chance), while a 4B model gets ~0.669 on the SAME query. The suspicion: the
current query talks about boxes hiding "the same / a different KIND of prize", which invites a SEMANTIC
category judgment ("is an apple the same kind as a train?") rather than the intended SYMBOLIC identity
check (apple != train). This probe tries three wordings and measures whether an unambiguous, identity-
framed query rescues the cheap models.

Pairwise ground truth (pure symbolic identity, NOT world knowledge):
  under relation "different", the pair (slot_i=noun_a, slot_j=noun_b) is CONSISTENT iff a != b;
  under relation "same",      it is CONSISTENT iff a == b.

Three elicitation methods, each a pure (item, constraint e, option-index a, option-index b) -> str:
  M0 = the CURRENT query (byte-identical to edges._query): uses the constraint's "...same/different
       kind(s) of prize" text. The baseline to reproduce.
  M1 = UNAMBIGUOUS SYMBOLIC REFRAME: state the rule as identity of the two named items, no "kind".
  M2 = FEW-SHOT on M1: prepend two worked examples (one satisfied, one violated, using OTHER nouns not
       in the item), then the M1 query, all in a single user turn.

The probe (a) sweeps models x methods and prints an AUC table, then (b) runs a decode-confirm: build
edges with the best (model, method) for the 25 held-out T=3,k=3,C=2 items using the SAME decoupled
emission as the validation (adapter_decoupled_s0 on Qwen3-4B), assemble + decode marginal/viterbi, and
score joint exact-match vs gold, alongside greedy and oracle references on the same items/emission.

Model-free by construction up to the sweep: prompt_* and the AUC machinery take an injected
score_yes_no closure, so the unit tests exercise everything without a GPU.
"""
import argparse, importlib.util as ilu, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEPS = HERE / "_deps"
if str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))


def _bp(name, p):
    s = ilu.spec_from_file_location(name, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m


edges = _bp("lcd_edges", HERE / "edges.py")
decoders = _bp("lcd_decoders", HERE / "decoders.py")
gen = _bp("lcd_gen", HERE / "gen_data.py")

# Nouns for M2's worked examples. Chosen to NOT collide with item nouns where possible; if a collision
# is unavoidable for a given item the examples still teach the rule (the live query uses the item nouns).
_FEWSHOT_NOUNS = ["acorn", "button", "dragon", "feather", "kettle", "marble"]


# =====================================================================================
# Ground truth (pure symbolic identity)
# =====================================================================================
def is_consistent(relation, a, b):
    """CONSISTENT iff: 'different' => a != b ; 'same' => a == b. Pure index identity, no world knowledge."""
    if relation == "different":
        return a != b
    if relation == "same":
        return a == b
    raise ValueError(f"unknown relation {relation!r}")


# =====================================================================================
# Three elicitation methods (pure prompt-string functions)
# =====================================================================================
def prompt_M0(item, e, a, b):
    """M0 = the CURRENT query, byte-identical to edges._query (the baseline to reproduce)."""
    return edges._query(item, e, a, b)


def prompt_M1(item, e, a, b):
    """M1 = unambiguous SYMBOLIC reframe: state the rule as identity of the two named items, no 'kind'.
    The model only needs to check whether the two named strings are different or identical."""
    ni = item["options"][e["i"]][a]
    nj = item["options"][e["j"]][b]
    si, sj = item["slot_names"][e["i"]], item["slot_names"][e["j"]]
    if e["relation"] == "different":
        rule = f"Rule: {si} and {sj} must hold two DIFFERENT items."
    else:
        rule = f"Rule: {si} and {sj} must hold the SAME item."
    return (f"{rule}\n"
            f"{si} holds: the {ni}\n"
            f"{sj} holds: the {nj}\n"
            f"Is the rule satisfied? Answer yes or no.")


def _fewshot_pair(relation, satisfied, n1, n2):
    """One worked M1-style example (generic slots/nouns) ending in 'Answer: yes' / 'Answer: no'.
    `satisfied` says whether this example's named items satisfy the rule (-> the gold answer)."""
    if relation == "different":
        rule = "Rule: box A and box B must hold two DIFFERENT items."
    else:
        rule = "Rule: box A and box B must hold the SAME item."
    ans = "yes" if satisfied else "no"
    return (f"{rule}\n"
            f"box A holds: the {n1}\n"
            f"box B holds: the {n2}\n"
            f"Is the rule satisfied? Answer yes or no.\n"
            f"Answer: {ans}")


def prompt_M2(item, e, a, b):
    """M2 = few-shot on M1: two worked examples (one satisfied, one violated, using OTHER nouns), then
    the live M1 query. Single user turn. The two examples teach the symbolic rule for THIS relation."""
    rel = e["relation"]
    n = _FEWSHOT_NOUNS
    if rel == "different":
        # satisfied: two different nouns ; violated: the same noun twice
        ex_ok = _fewshot_pair(rel, True, n[0], n[1])
        ex_bad = _fewshot_pair(rel, False, n[2], n[2])
    else:  # same
        # satisfied: the same noun twice ; violated: two different nouns
        ex_ok = _fewshot_pair(rel, True, n[3], n[3])
        ex_bad = _fewshot_pair(rel, False, n[4], n[5])
    live = prompt_M1(item, e, a, b)
    return f"{ex_ok}\n\n{ex_bad}\n\n{live}"


# =====================================================================================
# AUC over all (constraint, a, b) pairs, split by ground-truth consistency
# =====================================================================================
def _auc_from_margins(consistent, inconsistent):
    """Fraction of (consistent, inconsistent) margin pairs with consistent>inconsistent; ties = 0.5.
    This is the Mann-Whitney / ROC-AUC statistic. Empty either side -> NaN."""
    c = np.asarray(consistent, float); i = np.asarray(inconsistent, float)
    if c.size == 0 or i.size == 0:
        return float("nan")
    wins = 0.0
    for cv in c:
        wins += np.sum(cv > i) + 0.5 * np.sum(cv == i)
    return float(wins / (c.size * i.size))


def consistency_auc(score_yes_no, items, method_fn):
    """Over all constraints and all (a,b) option pairs in `items`, split margins by ground-truth
    consistency and compute AUC + class means.

    score_yes_no(prompt) -> float : the logp(yes)-logp(no) margin for that prompt (real: a model
        closure; tests: a fake). method_fn(item, e, a, b) -> prompt string.
    Returns (auc, mean_consistent, mean_inconsistent, n_pairs)."""
    cons, inco = [], []
    for it in items:
        k = it["k"]
        for e in it["constraints"]:
            for a in range(k):
                for b in range(k):
                    m = float(score_yes_no(method_fn(it, e, a, b)))
                    if is_consistent(e["relation"], a, b):
                        cons.append(m)
                    else:
                        inco.append(m)
    auc = _auc_from_margins(cons, inco)
    mc = float(np.mean(cons)) if cons else float("nan")
    mi = float(np.mean(inco)) if inco else float("nan")
    return auc, mc, mi, len(cons) + len(inco)


# =====================================================================================
# Test-only helpers: let a fake scorer read the ground truth off the prompt string.
# The real model scorers IGNORE the tag (they only read the natural-language query). _tagged wraps a
# method so the prompt carries an encoded truth marker; _truth_of decodes it. Used ONLY by tests.
# =====================================================================================
_TAG = " TRUTH="


def _tagged(method_fn):
    """Wrap a method so the returned prompt has a hidden ground-truth tag appended (tests only)."""
    def f(item, e, a, b):
        base = method_fn(item, e, a, b)
        t = 1 if is_consistent(e["relation"], a, b) else -1
        return f"{base}{_TAG}{t}"
    return f


def _truth_of(prompt):
    """Decode the hidden ground-truth tag from a _tagged prompt: +1 consistent, -1 inconsistent."""
    return float(prompt.rsplit(_TAG, 1)[1])


METHODS = {"M0": prompt_M0, "M1": prompt_M1, "M2": prompt_M2}


# =====================================================================================
# The model sweep + decode-confirm (this is the deliverable; not exercised by the unit tests)
# =====================================================================================
def _cached(score_fn):
    """Wrap a model scorer in a per-prompt dict cache (build_edges issues C*k*k forwards per item)."""
    cache = {}

    def f(prompt):
        if prompt not in cache:
            cache[prompt] = float(score_fn(prompt))
        return cache[prompt]
    return f


def _sweep(models, sweep_items, dev):
    """For each (model, method) print AUC + class means; return a nested dict results[model][method]."""
    import torch
    C = _bp("lcd_nodes", HERE / "nodes.py").C
    frontier = _bp("frontier", DEPS / "frontier.py")

    results = {}
    for spec in models:
        name = spec["name"]
        loader = spec["loader"]                  # "frontier" (cross-tokenizer) or "C" (Qwen self)
        print(f"\n[sweep] loading {name} via {loader} ...", flush=True)
        t0 = time.time()
        if loader == "C":
            model, tok = C.load(name, dev)
        else:
            model, tok = frontier.load(name, dev)
        raw = edges.make_score_yes_no(model, tok, dev)
        score = _cached(raw)
        results[name] = {}
        for mkey, mfn in METHODS.items():
            auc, mc, mi, n = consistency_auc(score, sweep_items, mfn)
            results[name][mkey] = dict(auc=auc, mean_consistent=mc, mean_inconsistent=mi, n=n)
            print(f"    {name:38s} {mkey}: AUC={auc:.3f}  "
                  f"mean_consistent={mc:+.3f}  mean_inconsistent={mi:+.3f}  (n={n})", flush=True)
        del model, tok, raw, score
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"    [{name}] done in {time.time()-t0:.1f}s", flush=True)
    return results


def _load_decoupled_qwen(dev):
    """Load the validation's decoupled adapter onto Qwen3-4B (the SAME emission the validation used).
    Returns (model, tok, slot_logits, letter_ids-dict-for-k3)."""
    import torch
    nodes = _bp("lcd_nodes", HERE / "nodes.py")
    C = nodes.C
    adapter_dir = HERE / "outputs" / "adapter_decoupled_s0"
    if not adapter_dir.exists():
        raise SystemExit(f"missing decoupled adapter at {adapter_dir}")
    # mirror run.py's peft<->gptq stub + reload pattern
    import peft.import_utils as _piu
    _piu.is_gptqmodel_available = lambda *x, **k: False
    try:
        import peft.tuners.lora.gptq as _pg
        _pg.is_gptqmodel_available = lambda *x, **k: False
    except Exception:
        pass
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL, local_files_only=True, cache_dir=C.CH)
    backbone = AutoModelForCausalLM.from_pretrained(
        C.DEFAULT_MODEL, dtype=torch.bfloat16, local_files_only=True, cache_dir=C.CH).to(dev)
    model = PeftModel.from_pretrained(backbone, str(adapter_dir)).eval()
    lids = nodes.letter_ids(tok, 3)
    return model, tok, nodes.slot_logits, lids


def _theta_for(item, emit_np):
    """Per-slot unary = log_softmax(emission logits). Mirrors run._theta_for."""
    T, k = item["T"], item["k"]
    theta = np.zeros((T, k), float)
    for t in range(T):
        lg = np.asarray(emit_np(item, t), float).reshape(-1)
        m = lg.max()
        logZ = m + np.log(np.exp(lg - m).sum())
        theta[t] = lg - logZ
    return theta


def _joint_em(preds, golds):
    return float(np.mean([1.0 if list(p) == list(g) else 0.0 for p, g in zip(preds, golds)]))


def _decode_confirm(test_items, method_key, model_spec, dev):
    """Build edges for the held-out items with (model_spec, method), using the decoupled-Qwen emission;
    assemble + decode marginal/viterbi; score joint-EM vs gold. Greedy and oracle on the SAME emission
    are references. Returns a dict of joint-EM numbers."""
    import torch
    C = _bp("lcd_nodes", HERE / "nodes.py").C
    frontier = _bp("frontier", DEPS / "frontier.py")
    method_fn = METHODS[method_key]

    # 1) emission: the decoupled adapter on Qwen3-4B (same as the validation) -> theta per item.
    qmodel, qtok, slot_logits, lids = _load_decoupled_qwen(dev)

    def emit_np(item, t):
        with torch.no_grad():
            return slot_logits(qmodel, qtok, item["prompt_slot"][t], item["k"], lids, dev) \
                .detach().float().cpu().numpy()

    thetas = [_theta_for(it, emit_np) for it in test_items]
    golds = [it["gold"] for it in test_items]
    del qmodel
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # 2) the edge-builder model (the swept source) scores the chosen method's queries (cached).
    if model_spec["loader"] == "C":
        emodel, etok = C.load(model_spec["name"], dev)
    else:
        emodel, etok = frontier.load(model_spec["name"], dev)
    raw = edges.make_score_yes_no(emodel, etok, dev)
    score = _cached(raw)

    # build edges per item using the chosen elicitation method (NOT edges._query, which is M0 only).
    src_edges = []
    for it in test_items:
        k = it["k"]
        eg = []
        for e in it["constraints"]:
            M = np.zeros((k, k), float)
            for a in range(k):
                for b in range(k):
                    M[a, b] = score(method_fn(it, e, a, b))
            eg.append((e["i"], e["j"], M))
        src_edges.append(eg)
    del emodel, etok, raw, score
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    oracle_edges = [edges.build_edges_from_truth(it) for it in test_items]

    # 3) assemble + decode.
    src_graphs = [decoders.assemble(thetas[i], src_edges[i], k=test_items[i]["k"]) for i in range(len(test_items))]
    ora_graphs = [decoders.assemble(thetas[i], oracle_edges[i], k=test_items[i]["k"]) for i in range(len(test_items))]

    greedy = [decoders.decode_greedy(g) for g in src_graphs]   # greedy ignores edges -> same for both
    src_marg = [decoders.decode_marginal(g) for g in src_graphs]
    src_vit = [decoders.decode_viterbi(g) for g in src_graphs]
    ora_marg = [decoders.decode_marginal(g) for g in ora_graphs]
    ora_vit = [decoders.decode_viterbi(g) for g in ora_graphs]

    return dict(
        model=model_spec["name"], method=method_key, n=len(test_items),
        greedy=_joint_em(greedy, golds),
        source_marginal=_joint_em(src_marg, golds),
        source_viterbi=_joint_em(src_vit, golds),
        oracle_marginal=_joint_em(ora_marg, golds),
        oracle_viterbi=_joint_em(ora_vit, golds),
    )


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sweep", type=int, default=16, help="items (fixed seed) for the AUC sweep")
    ap.add_argument("--sweep_seed", type=int, default=0)
    ap.add_argument("--decode_threshold", type=float, default=0.85,
                    help="also decode-confirm any cell with AUC >= this")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sweep items: ~16 from make_item at a fixed seed (mix of relations; T=3,k=3,C=2 to match the validation).
    import random
    rng = random.Random(a.sweep_seed)
    sweep_items = [gen.make_item(rng, 3, 3, 2, i) for i in range(a.n_sweep)]
    n_diff = sum(1 for it in sweep_items for e in it["constraints"] if e["relation"] == "different")
    n_same = sum(1 for it in sweep_items for e in it["constraints"] if e["relation"] == "same")
    print(f"[probe] {len(sweep_items)} sweep items; constraints: {n_diff} different / {n_same} same", flush=True)

    # models to sweep: three decorrelated (cross-tokenizer via frontier.load) + Qwen3-4B reference (self).
    models = [
        {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "loader": "frontier"},
        {"name": "meta-llama/Llama-3.2-3B-Instruct", "loader": "frontier"},
        {"name": "meta-llama/Llama-3.1-8B-Instruct", "loader": "frontier"},
        {"name": "Qwen/Qwen3-4B", "loader": "C"},   # self / correlated reference only
    ]
    by_name = {m["name"]: m for m in models}

    sweep = _sweep(models, sweep_items, dev)

    # AUC table.
    print("\n=== AUC table (rows=models, cols=M0/M1/M2) ===", flush=True)
    print(f"{'model':40s} {'M0':>8s} {'M1':>8s} {'M2':>8s}", flush=True)
    for m in models:
        row = sweep[m["name"]]
        print(f"{m['name']:40s} "
              f"{row['M0']['auc']:8.3f} {row['M1']['auc']:8.3f} {row['M2']['auc']:8.3f}", flush=True)

    # pick decode-confirm cells: the single best (model, method) by AUC, plus any cell AUC >= threshold.
    cells = []
    best_key, best_auc = None, -1.0
    for mname, row in sweep.items():
        for mkey, v in row.items():
            au = v["auc"]
            if np.isfinite(au) and au > best_auc:
                best_auc, best_key = au, (mname, mkey)
            if np.isfinite(au) and au >= a.decode_threshold:
                cells.append((mname, mkey))
    if best_key and best_key not in cells:
        cells.append(best_key)
    # de-dup preserving order
    seen = set(); cells = [c for c in cells if not (c in seen or seen.add(c))]
    print(f"\n[probe] best cell: {best_key} (AUC={best_auc:.3f}); "
          f"decode-confirm cells: {cells}", flush=True)

    # held-out test items: split=='test', T=3,k=3,C=2 (exactly the validation's 25).
    all_items = json.loads((HERE / "data" / "items.json").read_text())
    test_items = [it for it in all_items
                  if it.get("split") == "test" and it["T"] == 3 and it["k"] == 3 and it["C"] == 2]
    print(f"[probe] decode-confirm on {len(test_items)} held-out T3k3C2 test items", flush=True)

    confirms = []
    for (mname, mkey) in cells:
        print(f"\n[decode-confirm] model={mname} method={mkey} ...", flush=True)
        dc = _decode_confirm(test_items, mkey, by_name[mname], dev)
        confirms.append(dc)
        print(f"    greedy={dc['greedy']:.3f}  source_marginal={dc['source_marginal']:.3f}  "
              f"source_viterbi={dc['source_viterbi']:.3f}  | oracle_marginal={dc['oracle_marginal']:.3f}  "
              f"oracle_viterbi={dc['oracle_viterbi']:.3f}", flush=True)

    out = dict(
        sweep=sweep,
        sweep_config=dict(n_sweep=len(sweep_items), seed=a.sweep_seed,
                          n_diff=n_diff, n_same=n_same, models=[m["name"] for m in models]),
        best=dict(model=best_key[0], method=best_key[1], auc=best_auc) if best_key else None,
        decode_threshold=a.decode_threshold,
        decode_confirm=confirms,
        n_test=len(test_items),
    )
    (HERE / "outputs").mkdir(exist_ok=True)
    out_path = HERE / "outputs" / "elicit_probe.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    print("AUC table:", flush=True)
    print(f"{'model':40s} {'M0':>8s} {'M1':>8s} {'M2':>8s}", flush=True)
    for m in models:
        row = sweep[m["name"]]
        print(f"{m['name']:40s} "
              f"{row['M0']['auc']:8.3f} {row['M1']['auc']:8.3f} {row['M2']['auc']:8.3f}", flush=True)
    print(f"\nbest cell: {best_key} AUC={best_auc:.3f}", flush=True)
    print("decode-confirm (joint exact-match vs gold; oracle ref ~0.88 marginal / 0.92 viterbi, "
          "greedy 0.00):", flush=True)
    for dc in confirms:
        print(f"  {dc['model']:38s} {dc['method']}: greedy={dc['greedy']:.3f} "
              f"src_marg={dc['source_marginal']:.3f} src_vit={dc['source_viterbi']:.3f} "
              f"oracle_marg={dc['oracle_marginal']:.3f} oracle_vit={dc['oracle_viterbi']:.3f}", flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
