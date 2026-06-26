"""Frozen Qwen3-4B + LoRA(r16); train the option-letter distribution toward the known target
under one mode/regime; evaluate distance-to-truth per (k,j) every epoch + from the saved adapter."""
import argparse, importlib.util as ilu, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

def _bp(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m

core = _bp("core", HERE / "core.py")
metrics = _bp("metrics", HERE / "metrics.py")
C = core.C

def evaluate_with(score_fn, items, prompt_key, device="cuda"):
    """score_fn(prompt, k) -> logits[k]. Groups TV/mass/uniformity/det-acc by (k,j)."""
    by = defaultdict(list)
    for it in items:
        k = it["k"]
        lg = score_fn(it[prompt_key], k)
        p = F.softmax(torch.as_tensor(lg, dtype=torch.float32), 0).cpu().numpy()
        t = np.array([it["target"][L] for L in it["letters"]])
        surv = t > 0
        rec = dict(tv=metrics.tv(p, t), kl=metrics.kl(t, p),
                   mass=metrics.survivor_mass(p, surv),
                   nonunif=metrics.within_uniformity_tv(p, surv),
                   det_correct=int(np.argmax(p) == np.argmax(t)) if it["j"] == 1 else None)
        by[(k, it["j"])].append(rec)
    out = {}
    for (k, j), recs in by.items():
        tvs = [r["tv"] for r in recs]
        out[f"{k},{j}"] = dict(
            k=k, j=j, n=len(recs), tv_mean=float(np.mean(tvs)), tv_ci=metrics.boot_ci(tvs),
            kl_mean=float(np.mean([r["kl"] for r in recs])),
            mass_mean=float(np.mean([r["mass"] for r in recs])),
            nonunif_mean=float(np.mean([r["nonunif"] for r in recs])),
            det_acc=(float(np.mean([r["det_correct"] for r in recs])) if j == 1 else None))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "self"], required=True)
    ap.add_argument("--regime", choices=["decoupled", "coupled", "shuffle"], required=True)
    ap.add_argument("--ks", default="2,3,4,5")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lam_kl", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only_j", default="", help="comma list to restrict training j (cross-talk control)")
    a = ap.parse_args()
    dev = torch.device("cuda"); torch.manual_seed(a.seed)
    prompt_key = "prompt_stated" if a.mode == "oracle" else "prompt_clue"
    tag = f"{a.mode}_{a.regime}_s{a.seed}" + (f"_j{a.only_j.replace(',','')}" if a.only_j else "")

    items = json.loads((HERE / "data" / "items.json").read_text())
    ks = [int(x) for x in a.ks.split(",")]
    items = [it for it in items if it["k"] in ks]
    only_j = set(int(x) for x in a.only_j.split(",")) if a.only_j else None
    tr = [it for it in items if it["split"] == "train" and (only_j is None or it["j"] in only_j)]
    te = [it for it in items if it["split"] == "test"]

    # shuffle control: permute targets across train items (same letters set per k to stay valid)
    if a.regime == "shuffle":
        g = torch.Generator().manual_seed(a.seed)
        bykey = defaultdict(list)
        for it in tr:
            bykey[it["k"]].append(it)
        for k, group in bykey.items():
            perm = torch.randperm(len(group), generator=g).tolist()
            tgs = [group[i]["target"] for i in perm]          # permuted targets within this k
            for it, tgt in zip(group, tgs):
                it["target"] = tgt                            # train items are per-run copies; safe to mutate

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft.import_utils as _piu
    _piu.is_gptqmodel_available = lambda *x, **k: False
    try:
        import peft.tuners.lora.gptq as _pg; _pg.is_gptqmodel_available = lambda *x, **k: False
    except Exception:
        pass
    from peft import LoraConfig, get_peft_model, PeftModel
    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL, local_files_only=True, cache_dir=C.CH)
    base = AutoModelForCausalLM.from_pretrained(C.DEFAULT_MODEL, dtype=torch.bfloat16,
                                                local_files_only=True, cache_dir=C.CH).to(dev)
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lcfg)
    assert all((not p.requires_grad) for n, p in model.named_parameters() if "lora" not in n), "backbone must be frozen"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    lids = {k: core.letter_ids(tok, k) for k in ks}

    def score(prompt, k):
        return core.option_logits(model, tok, prompt, k, lids[k], dev)

    print(f"[{tag}] start: {len(tr)} train / {len(te)} test items, ks={ks}", flush=True)
    (HERE / "outputs").mkdir(exist_ok=True)
    log = open(HERE / "outputs" / f"log_{tag}.txt", "w")
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        g = torch.Generator().manual_seed(1000 * a.seed + ep)
        perm = torch.randperm(len(tr), generator=g).tolist()
        opt.zero_grad(); run, seen = 0.0, 0
        for step, j in enumerate(perm):
            it = tr[j]; lg = score(it[prompt_key], it["k"]); t = core.target_vec(it, dev)
            loss = core.loss_decoupled(lg, t) if a.regime != "coupled" else core.loss_coupled(lg, t)
            (loss / a.bs).backward(); run += float(loss); seen += 1
            if (step + 1) % a.bs == 0 or step == len(perm) - 1:
                if a.lam_kl > 0:
                    (a.lam_kl * core.kl_to_base(model, tok, core.GENERAL[:4], dev)).backward()
                opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            ev = evaluate_with(lambda p, k: score(p, k), te, prompt_key, dev)
        last_ev = ev                                          # keep final-epoch in-memory eval for reload check
        tv_all = float(np.mean([c["tv_mean"] for c in ev.values()]))
        ips = seen / (time.time() - t0)
        line = f"ep{ep}: loss={run/seen:.3f} meanTV={tv_all:.3f} items/s={ips:.1f}"
        print(" ", line, flush=True); log.write(line + "\n"); log.flush()
        if not np.isfinite(tv_all):
            raise FloatingPointError(f"non-finite TV at epoch {ep}")

    adir = HERE / "outputs" / f"adapter_{tag}"
    model.save_pretrained(str(adir))
    # final eval from the reloaded checkpoint (entry mode 2)
    del model; torch.cuda.empty_cache()
    base2 = AutoModelForCausalLM.from_pretrained(C.DEFAULT_MODEL, dtype=torch.bfloat16,
                                                 local_files_only=True, cache_dir=C.CH).to(dev)
    model = PeftModel.from_pretrained(base2, str(adir)).eval()
    def score2(prompt, k):
        return core.option_logits(model, tok, prompt, k, lids[k], dev)
    with torch.no_grad():
        final = evaluate_with(lambda p, k: score2(p, k), te, prompt_key, dev)
    # finite guard: never write a JSON with a non-finite cell TV
    if not all(np.isfinite(c["tv_mean"]) for c in final.values()):
        raise FloatingPointError("non-finite TV in final checkpoint eval")
    # reloaded-checkpoint eval should match the last in-memory eval (tiny bf16 nondeterminism is OK -> warn, don't raise)
    shared = set(final) & set(last_ev)
    reload_max_dtv = max((abs(final[c]["tv_mean"] - last_ev[c]["tv_mean"]) for c in shared), default=0.0)
    rline = f"reload_match: max|dtv|={reload_max_dtv:.4f}"
    print(" ", rline, flush=True); log.write(rline + "\n"); log.flush()
    if reload_max_dtv > 1e-2:
        print(f"WARNING: checkpoint reload diverges from in-memory eval (max|dtv|={reload_max_dtv:.4f})", flush=True)
    out = dict(tag=tag, mode=a.mode, regime=a.regime, cfg=vars(a), cells=final, reload_max_dtv=float(reload_max_dtv))
    (HERE / "outputs" / f"run_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"=== [{tag}] final meanTV={np.mean([c['tv_mean'] for c in final.values()]):.3f} ===", flush=True)
    log.close()

if __name__ == "__main__":
    main()
