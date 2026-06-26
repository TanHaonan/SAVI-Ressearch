"""Latent-state consistency decoding: one run for --emission {base,coupled,decoupled}.

What this does, in plain terms:
  - For `coupled`/`decoupled` we train a tiny LoRA adapter on a FROZEN Qwen3-4B so the per-slot
    answer-letter distribution (the "node emission") matches a target: the calibrated soft target
    (decoupled, keeps the correct option alive) or the one-hot gold option (coupled, collapses it).
    A fluency KL anchor keeps the base LM intact. `base` skips training (no adapter).
  - Then, on the held-out (test) split, we read each slot's emission, build cross-slot consistency
    edges from one of four sources {self, cheap, oracle, shuffle}, assemble a factor graph per item,
    and decode it three ways {greedy, marginal, viterbi}. Each (edge source x decoder) cell is scored
    against the exactly-known gold (joint exact-match, per-slot accuracy, feasibility-vs-accuracy
    split) with an item-bootstrap CI. We also measure edge phi (builder error vs the Qwen3-4B-self
    error) for the decorrelation check.

The eval core `evaluate_with(...)` takes injected closures (emit_fn, score_yes_no[_by_source]) so it
runs with no model at all (the integration test passes fakes); the real run passes closures over
nodes.slot_logits (emission) and edges.make_score_yes_no (self/cheap).

Components consumed:
  gen_data.py  -> data/items.json (item schema)
  nodes.py     -> slot_logits, emission_target, loss_decoupled, loss_coupled, kl_to_base, letter_ids, C, GENERAL
  edges.py     -> build_edges, build_edges_from_truth, shuffle_edges, make_score_yes_no, edge_phi
  decoders.py  -> assemble, decode_greedy, decode_marginal, decode_viterbi
  _deps/metrics.py -> exact_match, accuracy, feasibility_vs_accuracy_split, _bootstrap_ci

Implementation notes:
  (1) shuffle_edges is called PER (T,k,C) CELL, not on the mixed held-out set, so the donor always has
      the same shape and the shuffled signal is uniformly misleading (never degenerates to "no edges").
  (2) the self/cheap score closures are wrapped in a per-prompt dict cache (build_edges issues C*k*k
      forwards per item; caching by the exact prompt string keeps the cheap source to minutes).
  (3) edge_phi returns 0.0 on a degenerate denominator (handled in edges.py); aggregated as-is.
  (4) viterbi AND marginal joint-EM are reported separately for every cell.
  (5) greedy joint-EM is ~0 by construction (the planted conflict is one bent slot per item) -- the floor.
"""
import argparse, importlib.util as ilu, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEPS = HERE / "_deps"
# oracle.py / SemanticGraph are vendored in _deps/.
if str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))


def _bp(name, p):
    s = ilu.spec_from_file_location(name, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


# --- pipeline modules (function defs + env vars only; NO model load at import time) ---
nodes = _bp("lcd_nodes", HERE / "nodes.py")
edges = _bp("lcd_edges", HERE / "edges.py")
decoders = _bp("lcd_decoders", HERE / "decoders.py")
metrics = _bp("dep_metrics", DEPS / "metrics.py")

C = nodes.C                          # shared model-loading + scoring helpers (load, boot_ci, DEFAULT_MODEL, CH, ...)

ALL_SOURCES = ("self", "cheap", "oracle", "shuffle")
DECODERS = ("greedy", "marginal", "viterbi")
_DECODE_FN = {"greedy": decoders.decode_greedy,
              "marginal": decoders.decode_marginal,
              "viterbi": decoders.decode_viterbi}


# =====================================================================================
# Shared evaluation core (model-agnostic; the integration test calls this with fakes)
# =====================================================================================
def _cell_key(item):
    return (item["T"], item["k"], item["C"])


def _theta_for(item, emit_fn):
    """Per-slot unary log-potentials = log_softmax(emission logits). emit_fn(item, t) -> logits[k]."""
    T, k = item["T"], item["k"]
    theta = np.zeros((T, k), float)
    for t in range(T):
        lg = np.asarray(emit_fn(item, t), float).reshape(-1)
        if lg.shape[0] != k:
            raise ValueError(f"emit_fn returned {lg.shape[0]} logits for slot {t}, expected k={k}")
        if not np.all(np.isfinite(lg)):
            raise ValueError(f"emit_fn returned non-finite logits for item {item['id']} slot {t}")
        m = lg.max()
        logZ = m + np.log(np.exp(lg - m).sum())
        theta[t] = lg - logZ                       # log-softmax (valid log-probs)
    return theta


def _edges_for_source(items, src, score_yes_no_by_source, seed):
    """Return a list (aligned with `items`) of edge lists [(i,j,M)] for the given source.

    shuffle is computed PER (T,k,C) CELL: items are grouped by cell, shuffle_edges is run within each
    group (so the donor always shares the shape), then results are scattered back to item order. This
    keeps the shuffled control uniformly misleading instead of handing ~half the items a flat donor.
    """
    n = len(items)
    if src == "oracle":
        return [edges.build_edges_from_truth(it) for it in items]
    if src == "shuffle":
        out = [None] * n
        by_cell = defaultdict(list)                 # cell -> list of (orig_index, item)
        for idx, it in enumerate(items):
            by_cell[_cell_key(it)].append((idx, it))
        for cell, group in by_cell.items():
            grp_items = [it for (_idx, it) in group]
            # deterministic per-cell seed so the same run is reproducible
            cell_seed = seed + (hash(cell) % 100003)
            shuffled = edges.shuffle_edges(grp_items, seed=cell_seed)
            for (orig_idx, _it), eg in zip(group, shuffled):
                out[orig_idx] = eg
        return out
    if src in ("self", "cheap"):
        if score_yes_no_by_source is None or src not in score_yes_no_by_source:
            raise ValueError(f"no score_yes_no closure provided for edge source '{src}'")
        f = score_yes_no_by_source[src]
        return [edges.build_edges(it, f) for it in items]
    raise ValueError(f"unknown edge source: {src}")


def _builder_error_vec(items, edge_lists):
    """Boolean per-(item,constraint) error: does the builder's argmax-consistent pair disagree with the
    oracle compat sign? Used for the edge decorrelation check. For each constraint cell we compare 'is the gold (a,b)=
    (gold[i],gold[j]) the max-margin pair under the builder matrix' vs 'under the oracle matrix'."""
    errs = []
    for it, eg in zip(items, edge_lists):
        truth = {(e["i"], e["j"]): np.array(e["compat"], float) for e in it["constraints"]}
        gold = it["gold"]
        for (i, j, M) in eg:
            M = np.asarray(M, float)
            To = truth.get((i, j))
            if To is None:
                continue
            # the builder "errs" on this constraint if its preferred compatibility for the gold pair
            # disagrees with the oracle: oracle says gold pair is allowed (compat==0, the max); builder
            # errs if gold (a,b) is NOT among its argmax cells.
            gi, gj = gold[i], gold[j]
            builder_ok = (M[gi, gj] >= M.max() - 1e-9)
            errs.append(not builder_ok)
    return np.array(errs, bool)


def evaluate_with(items, emit_fn, score_yes_no=None, score_yes_no_by_source=None,
                  sources=("self", "oracle", "shuffle"), seed=0, ci_seed=0):
    """Decode every held-out item under every (edge source x decoder) and score vs gold.

    Args:
      items: list of carrier items (the held-out / test split).
      emit_fn(item, t) -> logits[k]: the node emission (real: nodes.slot_logits; test: a fake).
      score_yes_no(prompt) -> float: a single yes/no margin closure (used for self AND cheap if a
          per-source dict is not given; the integration test passes this).
      score_yes_no_by_source: {"self": fn, "cheap": fn} for the real run (cheap is a different model).
      sources: which edge sources to evaluate (default self/oracle/shuffle; add "cheap" when available).
      seed: per-cell shuffle seed.
      ci_seed: bootstrap seed.

    Returns: {src: {decoder: {joint_em, joint_em_ci, acc, acc_ci, feas_split, n}}, ...,
              "_phi": {...}, "_elicitation": int, "_n_items": int}
    Every metric is finite or the function raises (never emits a silent NaN).
    """
    if not items:
        raise ValueError("evaluate_with: empty item set (empty cell)")

    # resolve the yes/no closures per source
    by_src = dict(score_yes_no_by_source) if score_yes_no_by_source else {}
    if score_yes_no is not None:
        by_src.setdefault("self", score_yes_no)
        by_src.setdefault("cheap", score_yes_no)

    # 1) per-item unary (node emission) -> theta and the SemanticGraph topology is shared across edge
    #    sources (only the edge matrices change). Precompute theta once.
    thetas = [_theta_for(it, emit_fn) for it in items]
    golds = [it["gold"] for it in items]

    # 2) build the edge lists for each requested source (+ always build "self" for the phi reference if
    #    a self scorer exists, since the decorrelation check compares a builder against the self builder).
    elicitation = 0
    edge_lists = {}
    for src in sources:
        edge_lists[src] = _edges_for_source(items, src, by_src, seed)
        if src in ("self", "cheap"):
            # one yes/no query per (constraint, a, b) cell per item
            elicitation += sum(it["C"] * it["k"] * it["k"] for it in items)

    # 3) decode + score each (source x decoder) cell.
    result = {}
    for src in sources:
        result[src] = {}
        # assemble per-item graphs for this source (reused across decoders)
        graphs = [decoders.assemble(thetas[i], edge_lists[src][i], k=items[i]["k"])
                  for i in range(len(items))]
        for dec in DECODERS:
            fn = _DECODE_FN[dec]
            preds = [fn(g) for g in graphs]
            # per-item joint-EM and per-item per-slot accuracy (paired -> CI)
            em_vec = np.array([1.0 if list(p) == list(g) else 0.0 for p, g in zip(preds, golds)], float)
            acc_vec = np.array([float(np.mean(np.asarray(p) == np.asarray(g)))
                                for p, g in zip(preds, golds)], float)
            joint_em = float(em_vec.mean())
            acc = float(acc_vec.mean())
            if not (np.isfinite(joint_em) and np.isfinite(acc)):
                raise FloatingPointError(f"non-finite metric for {src}/{dec}")
            feas = metrics.feasibility_vs_accuracy_split(preds, graphs, golds)
            result[src][dec] = dict(
                joint_em=joint_em,
                joint_em_ci=C.boot_ci(em_vec.tolist(), seed=ci_seed),     # [2.5,50,97.5]
                acc=acc,
                acc_ci=C.boot_ci(acc_vec.tolist(), seed=ci_seed),
                feas_split=feas,
                n=len(items),
                em_vec=em_vec.tolist(),                                    # kept for paired CIs in analyze
            )

    # 4) edge decorrelation: builder error vs the self builder error. Only computable when "self"
    #    edges were built; compares each available builder source against self.
    phi_out = {}
    if "self" in edge_lists:
        self_err = _builder_error_vec(items, edge_lists["self"])
        for src in ("cheap", "self"):
            if src in edge_lists:
                src_err = _builder_error_vec(items, edge_lists[src])
                if src_err.shape == self_err.shape and src_err.size:
                    phi_out[src] = float(edges.edge_phi(src_err, self_err))
                else:
                    phi_out[src] = 0.0

    result["_phi"] = phi_out
    result["_elicitation"] = int(elicitation)
    result["_n_items"] = len(items)
    return result


# =====================================================================================
# Real run (model side): train the adapter, save+reload, then call evaluate_with.
# =====================================================================================
def _apply_peft_gptq_patch():
    """peft<->optimum gptq availability check raises when probed; we use no quantization, so stub it
    False BEFORE importing the LoRA tuner."""
    import peft.import_utils as _piu
    _piu.is_gptqmodel_available = lambda *x, **k: False
    try:
        import peft.tuners.lora.gptq as _pg
        _pg.is_gptqmodel_available = lambda *x, **k: False
    except Exception:
        pass


def _load_model(emission, rank, dev):
    """Load frozen Qwen3-4B. For base: no adapter (frozen model). For coupled/decoupled: + LoRA(rank)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL, local_files_only=True, cache_dir=C.CH)
    backbone = AutoModelForCausalLM.from_pretrained(
        C.DEFAULT_MODEL, dtype=torch.bfloat16, local_files_only=True, cache_dir=C.CH).to(dev)
    if emission == "base":
        backbone.eval()
        return backbone, tok, False
    _apply_peft_gptq_patch()
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(backbone, lcfg)
    # FREEZE CHECK (load-bearing): only LoRA params may require grad.
    bad = [n for n, p in model.named_parameters() if p.requires_grad and "lora" not in n]
    assert not bad, f"backbone must be frozen; these non-LoRA params require grad: {bad[:4]}"
    return model, tok, True


def _train(model, tok, train_items, emission, ks, lids, dev, epochs, lr, bs, lam_kl, seed,
           test_items, log):
    """Train the adapter so each slot's emission matches its target (decoupled soft / coupled one-hot),
    + a fluency KL anchor. Per-epoch log line: mean loss, held-out oracle+marginal joint-EM, items/s."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_k):
            return it

    # flatten train into (item, slot) examples
    examples = [(it, t) for it in train_items for t in range(it["T"])]

    def emit_torch(item, t):
        return nodes.slot_logits(model, tok, item["prompt_slot"][t], item["k"], lids[item["k"]], dev)

    def heldout_oracle_marginal_em():
        """Quick in-memory eval used per-epoch: oracle edges + marginal decoder joint-EM on a held-out
        subset (capped for speed)."""
        model.eval()
        sub = test_items[: min(len(test_items), 64)]
        with torch.no_grad():
            def emit_np(item, t):
                return emit_torch(item, t).detach().float().cpu().numpy()
            res = evaluate_with(sub, emit_fn=emit_np, sources=("oracle",), seed=seed)
        return res["oracle"]["marginal"]["joint_em"]

    print(f"[{emission}] training: {len(train_items)} items / {len(examples)} (item,slot) examples, "
          f"epochs={epochs} bs={bs} lr={lr} lam_kl={lam_kl}", flush=True)
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        g = torch.Generator().manual_seed(1000 * seed + ep)
        perm = torch.randperm(len(examples), generator=g).tolist()
        opt.zero_grad()
        run, seen = 0.0, 0
        for step, ei in enumerate(tqdm(perm, desc=f"ep{ep}", leave=False)):
            item, t = examples[ei]
            lg = emit_torch(item, t)                         # tensor[k], fp32
            if emission == "decoupled":
                target = nodes.emission_target(np.array(item["local_unary"])[t], tau=1.0).to(dev)
                loss = nodes.loss_decoupled(lg, target)
            else:  # coupled: one-hot CE to the gold option
                onehot = torch.zeros(item["k"], device=dev)
                onehot[item["gold"][t]] = 1.0
                loss = nodes.loss_coupled(lg, onehot)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"[{emission}] non-finite training loss at ep{ep} step{step}")
            (loss / bs).backward()
            run += float(loss); seen += 1
            if (step + 1) % bs == 0 or step == len(perm) - 1:
                if lam_kl > 0:
                    kl = lam_kl * nodes.kl_to_base(model, tok, nodes.GENERAL[:4], dev)
                    if not torch.isfinite(kl):
                        raise FloatingPointError(f"[{emission}] non-finite KL anchor at ep{ep}")
                    kl.backward()
                opt.step(); opt.zero_grad()
        dt = time.time() - t0
        mean_loss = run / max(seen, 1)
        ho_em = heldout_oracle_marginal_em()
        ips = seen / dt if dt > 0 else float("nan")
        line = (f"  [{emission}] ep{ep}: mean_loss={mean_loss:.4f} "
                f"heldout_oracle_marginal_joint_em={ho_em:.3f} items/s={ips:.1f} ({seen} in {dt:.1f}s)")
        print(line, flush=True)
        if log is not None:
            log.write(line + "\n"); log.flush()
    return model


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--emission", choices=["base", "coupled", "decoupled"], required=True)
    ap.add_argument("--edges", default="self,cheap,oracle,shuffle",
                    help="comma subset of {self,cheap,oracle,shuffle}")
    ap.add_argument("--Ts", default="3,4")
    ap.add_argument("--ks", default="3,4")
    ap.add_argument("--Cs", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lam_kl", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cheap_model", default="meta-llama/Llama-3.2-1B")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    sources = tuple(s.strip() for s in a.edges.split(",") if s.strip())
    for s in sources:
        if s not in ALL_SOURCES:
            raise SystemExit(f"unknown edge source '{s}'; must be subset of {ALL_SOURCES}")
    # the decorrelation check needs the self builder as the phi reference; ensure it is always built if cheap is requested.
    if "cheap" in sources and "self" not in sources:
        sources = sources + ("self",)

    Ts = [int(x) for x in a.Ts.split(",")]
    ks = [int(x) for x in a.ks.split(",")]
    Cs = [int(x) for x in a.Cs.split(",")]

    items_path = HERE / "data" / "items.json"
    if not items_path.exists():
        raise SystemExit(f"missing {items_path}; run `python gen_data.py` first")
    all_items = json.loads(items_path.read_text())
    items = [it for it in all_items if it["T"] in Ts and it["k"] in ks and it["C"] in Cs]
    train_items = [it for it in items if it["split"] == "train"]
    test_items = [it for it in items if it["split"] == "test"]    # held-out; edges never see gold
    if not test_items:
        raise SystemExit("no held-out (test) items for this (T,k,C) grid")
    print(f"[{a.emission}] grid Ts={Ts} ks={ks} Cs={Cs}: "
          f"{len(train_items)} train / {len(test_items)} test items", flush=True)

    (HERE / "outputs").mkdir(exist_ok=True)
    log = open(HERE / "outputs" / f"log_{a.emission}_s{a.seed}.txt", "w")

    # ---- load model (frozen backbone; +LoRA unless base) ----
    model, tok, trainable = _load_model(a.emission, a.rank, dev)
    lids = {k: nodes.letter_ids(tok, k) for k in ks}

    adapter_dir = HERE / "outputs" / f"adapter_{a.emission}_s{a.seed}"
    if trainable:
        # ---- train ----
        _train(model, tok, train_items, a.emission, ks, lids, dev,
               a.epochs, a.lr, a.bs, a.lam_kl, a.seed, test_items, log)
        # ---- save + reload the adapter (checkpoint round-trip) ----
        model.save_pretrained(str(adapter_dir))

    # ---- build the real closures ----
    def emit_fn(item, t):
        with torch.no_grad():
            return nodes.slot_logits(model, tok, item["prompt_slot"][t], item["k"], lids[item["k"]], dev) \
                .detach().float().cpu().numpy()

    # self edges: Qwen3-4B answers the yes/no queries (cached by prompt string)
    self_raw = edges.make_score_yes_no(model, tok, dev)
    self_cache = {}

    def self_score(prompt):
        if prompt not in self_cache:
            self_cache[prompt] = float(self_raw(prompt))
        return self_cache[prompt]

    by_src = {"self": self_score}

    # cheap edges: a decorrelated ~1B model answers the queries (cached)
    cheap_model = cheap_tok = None
    if "cheap" in sources:
        try:
            frontier = _bp("frontier", DEPS / "frontier.py")
            cheap_model, cheap_tok = frontier.load(a.cheap_model, dev)
            cheap_raw = edges.make_score_yes_no(cheap_model, cheap_tok, dev)
            cheap_cache = {}

            def cheap_score(prompt):
                if prompt not in cheap_cache:
                    cheap_cache[prompt] = float(cheap_raw(prompt))
                return cheap_cache[prompt]

            by_src["cheap"] = cheap_score
        except Exception as e:
            raise SystemExit(f"cheap builder model '{a.cheap_model}' failed to load: {e!r}")

    # ---- final eval (in-memory) over the full held-out split, all sources x decoders ----
    model.eval()
    t0 = time.time()
    res_inmem = evaluate_with(test_items, emit_fn=emit_fn, score_yes_no_by_source=by_src,
                              sources=sources, seed=a.seed, ci_seed=a.seed)
    print(f"[{a.emission}] final in-memory eval done in {time.time()-t0:.1f}s "
          f"(elicitation={res_inmem['_elicitation']} queries; cache-deduped)", flush=True)

    # ---- reload-from-checkpoint check (entry mode 2): reload adapter, re-eval, assert identical ----
    if trainable:
        if not adapter_dir.exists():
            raise SystemExit(f"adapter checkpoint missing after save: {adapter_dir}")
        _apply_peft_gptq_patch()
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        backbone2 = AutoModelForCausalLM.from_pretrained(
            C.DEFAULT_MODEL, dtype=torch.bfloat16, local_files_only=True, cache_dir=C.CH).to(dev)
        try:
            model2 = PeftModel.from_pretrained(backbone2, str(adapter_dir)).eval()
        except Exception as e:
            raise SystemExit(f"adapter reload failed from {adapter_dir}: {e!r}")

        def emit_fn2(item, t):
            with torch.no_grad():
                return nodes.slot_logits(model2, tok, item["prompt_slot"][t], item["k"],
                                         lids[item["k"]], dev).detach().float().cpu().numpy()

        self_raw2 = edges.make_score_yes_no(model2, tok, dev)
        self_cache2 = {}

        def self_score2(prompt):
            if prompt not in self_cache2:
                self_cache2[prompt] = float(self_raw2(prompt))
            return self_cache2[prompt]

        by_src2 = {"self": self_score2}
        if "cheap" in by_src:
            by_src2["cheap"] = by_src["cheap"]          # same cheap model instance is fine
        res_reload = evaluate_with(test_items, emit_fn=emit_fn2, score_yes_no_by_source=by_src2,
                                   sources=sources, seed=a.seed, ci_seed=a.seed)
        # verify final eval == last in-memory eval within 1e-6 on joint-EM for every cell
        max_delta = 0.0
        for src in sources:
            for dec in DECODERS:
                d = abs(res_inmem[src][dec]["joint_em"] - res_reload[src][dec]["joint_em"])
                max_delta = max(max_delta, d)
        print(f"[{a.emission}] reload-identical check: max |Δ joint-EM| = {max_delta:.2e}", flush=True)
        if max_delta > 1e-6:
            raise SystemExit(f"reload mismatch: max |Δ joint-EM| = {max_delta:.2e} > 1e-6")
        res = res_reload
    else:
        res = res_inmem

    # ---- guard: never write a JSON with a NaN joint-EM silently ----
    for src in sources:
        for dec in DECODERS:
            v = res[src][dec]["joint_em"]
            if not np.isfinite(v):
                raise SystemExit(f"non-finite joint-EM for {src}/{dec}; aborting before write")

    # strip the per-item em_vec from the on-disk payload? keep it: analyze.py uses it for paired CIs.
    out = dict(
        emission=a.emission,
        config=dict(vars(a), device=str(dev), trainable=trainable, n_train=len(train_items),
                    n_test=len(test_items), sources=list(sources)),
        cells=res,
        elicitation=res["_elicitation"] if "_elicitation" in res else res_inmem["_elicitation"],
        phi=res.get("_phi", {}),
    )
    out_path = HERE / "outputs" / f"run_{a.emission}_s{a.seed}.json"
    out_path.write_text(json.dumps(out))
    # human-readable summary
    print(f"\n=== [{a.emission}] FINAL (held-out, {len(test_items)} items) ===", flush=True)
    for src in sources:
        cells = " ".join(f"{dec}={res[src][dec]['joint_em']:.3f}" for dec in DECODERS)
        print(f"  edges={src:8s} joint-EM: {cells}", flush=True)
    print(f"  edge phi (vs self): {res.get('_phi', {})}", flush=True)
    print(f"  elicitation queries: {out['elicitation']}", flush=True)
    print(f"  wrote {out_path}", flush=True)
    log.close()


if __name__ == "__main__":
    main()
