"""One training run -> native determinate/ambiguous metrics + per-item arrays + saved adapter.

Frozen Qwen3-4B + LoRA(r16). Decoupled loss = margin discrimination (order, not magnitude) + (p-0.5)^2
calibration on an external-source ambiguous item + KL-to-base fluency leash. coupled = hard CE, no shape.
shuffle = shuffled determinate gold (control). Evaluates the NATIVE yes/no forward (no helper decoder).
Stores per-item gen_test det correctness and per-item held-out ambiguous |p-0.5| for paired bootstrap.

Inputs (all under core/data/): corpus.json (determinate items), ambiguous_qa.json (ambiguous items).
common.py (scoring/load helpers) sits alongside this file.

Run: CUDA_VISIBLE_DEVICES=4 python run.py --regime decoupled --lam_shape 40 --seed 0 --save_adapter
"""
import argparse, importlib.util as ilu, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ANS = "Answer with only 'yes' or 'no'."


def by_path(name, p):
    s = ilu.spec_from_file_location(name, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


C = by_path("ce_common", HERE / "common.py")

GENERAL = [
    "The sun rose slowly over the quiet harbor as the fishing boats returned.",
    "She opened the old notebook and began to write down everything she remembered.",
    "Economists disagree about the long-term effects of the new trade policy.",
    "After the rain stopped, the children ran outside to play in the puddles.",
    "The recipe calls for two cups of flour, a pinch of salt, and three eggs.",
    "He studied the map carefully before deciding which trail to take.",
    "The committee will meet next Thursday to review the annual budget.",
    "A gentle breeze carried the scent of pine through the open window.",
    "Researchers published their findings in a peer-reviewed journal last month.",
    "The train was delayed, so we waited on the platform and shared a coffee.",
    "Most of the documents were stored in a single folder on the shared drive.",
    "By the time the movie ended, almost everyone in the theater was asleep.",
]


def yn_logits(model, tok, user, dev):
    ids = C.chat_prefix_ids(tok, user, dev)
    logits = model(ids).logits[0, -1].float()
    yi = sorted({tok.encode(v, add_special_tokens=False)[0] for v in C.YES if tok.encode(v, add_special_tokens=False)})
    ni = sorted({tok.encode(v, add_special_tokens=False)[0] for v in C.NO if tok.encode(v, add_special_tokens=False)})
    ly = torch.logsumexp(logits[torch.tensor(yi, device=dev)], 0)
    ln = torch.logsumexp(logits[torch.tensor(ni, device=dev)], 0)
    return ly, ln


def kl_to_base(model, tok, sents, dev):
    tot = 0.0
    for s in sents:
        ids = tok(s, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad(), model.disable_adapter():
            base = F.softmax(model(ids).logits[0].float(), -1)
        adpt = F.log_softmax(model(ids).logits[0].float(), -1)
        tot = tot + F.kl_div(adpt, base, reduction="batchmean")
    return tot / len(sents)


@torch.no_grad()
def perplexity(model, tok, sents, dev):
    tot, ntok = 0.0, 0
    for s in sents:
        ids = tok(s, return_tensors="pt").input_ids.to(dev)
        lp = F.log_softmax(model(ids).logits[0].float(), -1)
        tgt = ids[0, 1:]; lps = lp[:-1][torch.arange(len(tgt)), tgt]
        tot += float(-lps.sum()); ntok += len(tgt)
    return float(np.exp(tot / ntok))


@torch.no_grad()
def native_pyes(model, tok, user, dev):
    ly, ln = yn_logits(model, tok, user, dev)
    return float(torch.sigmoid(ly - ln))


@torch.no_grad()
def evaluate(model, tok, det, amb, dev):
    det_corr = [int((native_pyes(model, tok, f"{it['scene']} {it['question']} {ANS}", dev) > 0.5) == (it["gold"] == "yes"))
                for it in det]
    amb_p = np.array([native_pyes(model, tok, f"{it['scene']} {it['question']} {ANS}", dev) for it in amb])
    dev_abs = np.abs(amb_p - 0.5)
    ent = -(amb_p * np.log(amb_p + 1e-9) + (1 - amb_p) * np.log(1 - amb_p + 1e-9))
    return dict(det_acc=float(np.mean(det_corr)), det_corr=[int(x) for x in det_corr],
                det_ids=[it["id"] for it in det],
                amb_abs_dev=float(np.mean(dev_abs)), amb_abs_dev_items=[float(x) for x in dev_abs],
                amb_ids=[it["id"] for it in amb],
                amb_entropy=float(np.mean(ent)),
                amb_overpeak_frac=float(np.mean(dev_abs > 0.4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["decoupled", "coupled", "shuffle"], required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--lam_shape", type=float, default=1.0)
    ap.add_argument("--lam_kl", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--save_adapter", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda"); torch.manual_seed(a.seed)
    tag = a.tag or (f"{a.regime}_ls{int(a.lam_shape)}_s{a.seed}" if a.regime == "decoupled"
                    else f"{a.regime}_s{a.seed}")

    corpus = json.loads((DATA / "corpus.json").read_text())
    by = lambda s: [it for it in corpus if it["split"] == s]
    det_tr, det_te = by("train"), by("gen_test")
    amb = json.loads((DATA / "ambiguous_qa.json").read_text())
    words = sorted({it["word"] for it in amb}); te_w = set(words[::3])
    amb_tr = [it for it in amb if it["word"] not in te_w]
    amb_te = [it for it in amb if it["word"] in te_w]

    gold = [it["gold"] == "yes" for it in det_tr]
    if a.regime == "shuffle":
        g = torch.Generator().manual_seed(a.seed)
        gold = [gold[i] for i in torch.randperm(len(gold), generator=g).tolist()]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import peft.import_utils as _piu
    _piu.is_gptqmodel_available = lambda *x, **k: False
    try:
        import peft.tuners.lora.gptq as _pg; _pg.is_gptqmodel_available = lambda *x, **k: False
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL, local_files_only=True, cache_dir=C.CH)
    model = AutoModelForCausalLM.from_pretrained(C.DEFAULT_MODEL, dtype=torch.bfloat16,
                                                 local_files_only=True, cache_dir=C.CH).to(dev)
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=a.targets.split(","))
    model = get_peft_model(model, lcfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    base = evaluate(model, tok, det_te, amb_te, dev)
    base_ppl = perplexity(model, tok, GENERAL, dev)
    print(f"[{tag}] init: det_acc={base['det_acc']:.3f} amb|p-.5|={base['amb_abs_dev']:.3f} ppl={base_ppl:.2f}",
          flush=True)

    for ep in range(a.epochs):
        model.train()
        g = torch.Generator().manual_seed(1000 * a.seed + ep)
        perm = torch.randperm(len(det_tr), generator=g).tolist()
        opt.zero_grad(); run = 0.0
        for step, j in enumerate(perm):
            it = det_tr[j]
            ly, ln = yn_logits(model, tok, f"{it['scene']} {it['question']} {ANS}", dev)
            if a.regime == "coupled":
                t = torch.tensor([0 if gold[j] else 1], device=dev)
                ldet = F.cross_entropy(torch.stack([ly, ln])[None], t)
            else:
                lg, lo = (ly, ln) if gold[j] else (ln, ly)
                ldet = F.relu(a.margin - (lg - lo))
            loss = ldet
            if a.regime != "coupled":
                ait = amb_tr[step % len(amb_tr)]
                aly, aln = yn_logits(model, tok, f"{ait['scene']} {ait['question']} {ANS}", dev)
                p_yes = torch.sigmoid(aly - aln)
                loss = loss + a.lam_shape * (p_yes - 0.5) ** 2
            (loss / a.bs).backward(); run += float(loss)
            if (step + 1) % a.bs == 0 or step == len(perm) - 1:
                if a.lam_kl > 0:
                    (a.lam_kl * kl_to_base(model, tok, GENERAL[:4], dev)).backward()
                opt.step(); opt.zero_grad()
        model.eval()
        ev = evaluate(model, tok, det_te, amb_te, dev)
        print(f"  ep{ep}: loss={run/len(perm):.3f} det_acc={ev['det_acc']:.3f} "
              f"amb|p-.5|={ev['amb_abs_dev']:.3f} amb_ent={ev['amb_entropy']:.3f}", flush=True)

    model.eval()
    final = evaluate(model, tok, det_te, amb_te, dev)
    final_ppl = perplexity(model, tok, GENERAL, dev)
    out = dict(tag=tag, regime=a.regime, cfg=vars(a),
               init=dict(det_acc=base["det_acc"], amb_abs_dev=base["amb_abs_dev"], ppl=base_ppl),
               final=dict({k: final[k] for k in final}, ppl=final_ppl, ppl_ratio=final_ppl / base_ppl))
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / f"run_{tag}.json").write_text(json.dumps(out, indent=2))
    if a.save_adapter:
        adir = HERE / "outputs" / f"adapter_{tag}"
        model.save_pretrained(str(adir))
        print(f"  saved adapter -> {adir}", flush=True)
    print(f"=== [{tag}] FINAL det={final['det_acc']:.3f} amb|p-.5|={final['amb_abs_dev']:.3f} "
          f"amb_ent={final['amb_entropy']:.3f} ppl x{final_ppl/base_ppl:.2f} ===", flush=True)


if __name__ == "__main__":
    main()
