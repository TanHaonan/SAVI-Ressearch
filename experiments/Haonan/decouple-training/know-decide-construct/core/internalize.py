"""In-the-forward vs at-the-probe test. Load base+adapter, then on the SAME forward pass the model answers
on, ask: does an external linear readout on the answer-position hidden state still recover det accuracy the
NATIVE forward leaves on the table? If readout >> native, the adapter leaves recoverable answer signal that
the native forward does not use, visible to the probe.
If readout ~= native, the forward already does the readout's job (the change is in the native forward).

Control (frozen model, --frozen): native ~0.53 while readout ~0.86 -> the +0.32 gap the adapter must close.
Input: core/data/corpus.json. common.py sits alongside this file.
Run: CUDA_VISIBLE_DEVICES=4 python internalize.py --adapter_dir outputs/adapter_decoupled_ls20_s0
"""
import argparse, importlib.util as ilu, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ANS = "Answer with only 'yes' or 'no'."


def by_path(name, p):
    s = ilu.spec_from_file_location(name, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


C = by_path("ce_common", HERE / "common.py")


@torch.no_grad()
def fwd_hidden(model, tok, item, dev):
    ids = C.chat_prefix_ids(tok, f"{item['scene']} {item['question']} {ANS}", dev)
    out = model(ids, output_hidden_states=True)
    return torch.stack(out.hidden_states, 0)[:, 0, -1, :].float().half().cpu()   # [L+1,H]


@torch.no_grad()
def native_pyes(model, tok, item, dev):
    ids = C.chat_prefix_ids(tok, f"{item['scene']} {item['question']} {ANS}", dev)
    logits = model(ids).logits[0, -1].float()
    yi = sorted({tok.encode(v, add_special_tokens=False)[0] for v in C.YES if tok.encode(v, add_special_tokens=False)})
    ni = sorted({tok.encode(v, add_special_tokens=False)[0] for v in C.NO if tok.encode(v, add_special_tokens=False)})
    ly = torch.logsumexp(logits[torch.tensor(yi, device=dev)], 0)
    ln = torch.logsumexp(logits[torch.tensor(ni, device=dev)], 0)
    return float(torch.sigmoid(ly - ln))


def _z(X, mu, sd):
    return (X - mu) / sd


def train_layer_readout(big, y, meta, layer, dev, epochs=300, wd=1e-2):
    """Linear readout on layer-L answer-position hidden state predicting gold yes. Select on val."""
    tr = [i for i, m in enumerate(meta) if m["split"] == "train"]
    va = [i for i, m in enumerate(meta) if m["split"] == "val"]
    feat = big[:, layer, :]
    H = feat.shape[-1]
    mu = feat[tr].mean(0); sd = feat[tr].std(0) + 1e-5
    head = nn.Linear(H, 1).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=wd)
    Xtr = _z(feat[tr], mu, sd).to(dev); ytr = y[tr].to(dev)
    Xva = _z(feat[va], mu, sd).to(dev); yva = y[va].to(dev)
    bv, st = -1.0, None
    for ep in range(epochs):
        head.train(); opt.zero_grad()
        F.binary_cross_entropy_with_logits(head(Xtr).squeeze(-1), ytr).backward(); opt.step()
        if ep % 5 == 0:
            head.eval()
            with torch.no_grad():
                acc = float(((head(Xva).squeeze(-1) > 0) == (yva > 0.5)).float().mean()) if len(va) else 0.0
            if acc >= bv:
                bv, st = acc, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(st)
    return head, mu.to(dev), sd.to(dev), bv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", default=None)
    ap.add_argument("--frozen", action="store_true", help="control: frozen base model, no adapter")
    a = ap.parse_args()
    dev = torch.device("cuda"); torch.manual_seed(0)
    corpus = json.loads((DATA / "corpus.json").read_text())

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft.import_utils as _piu
    _piu.is_gptqmodel_available = lambda *x, **k: False
    try:
        import peft.tuners.lora.gptq as _pg; _pg.is_gptqmodel_available = lambda *x, **k: False
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL, local_files_only=True, cache_dir=C.CH)
    model = AutoModelForCausalLM.from_pretrained(C.DEFAULT_MODEL, dtype=torch.bfloat16,
                                                 local_files_only=True, cache_dir=C.CH).to(dev)
    tag = "frozen"
    if not a.frozen:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter_dir)
        tag = Path(a.adapter_dir).name.replace("adapter_", "")
    model.eval()

    hs = torch.stack([fwd_hidden(model, tok, it, dev) for it in corpus]).float()   # [N,L+1,H]
    y = torch.tensor([1.0 if it["gold"] == "yes" else 0.0 for it in corpus])
    meta = [{"id": it["id"], "split": it["split"], "gold": it["gold"]} for it in corpus]
    gi = [i for i, m in enumerate(meta) if m["split"] == "gen_test"]

    # native det on gen_test (same items)
    nat = np.array([int((native_pyes(model, tok, corpus[i], dev) > 0.5) == (corpus[i]["gold"] == "yes")) for i in gi])

    # layer-swept linear readout, select layer on val, report on gen_test
    best = None
    for layer in range(1, hs.shape[1]):
        head, mu, sd, bv = train_layer_readout(hs, y, meta, layer, dev)
        with torch.no_grad():
            Xg = _z(hs[gi, layer, :].to(dev), mu, sd)
            pred = (head(Xg).squeeze(-1) > 0).cpu().numpy()
        corr = (pred == (y[gi].numpy() > 0.5)).astype(int)
        cand = dict(layer=layer, val_acc=bv, gen_test_acc=float(corr.mean()), readout_corr=[int(x) for x in corr])
        if best is None or cand["val_acc"] > best["val_acc"]:
            best = cand

    delta = best["gen_test_acc"] - float(nat.mean())
    out = dict(tag=tag, adapter_dir=a.adapter_dir, n_gen=len(gi),
               native_det=float(nat.mean()), native_corr=[int(x) for x in nat],
               readout_layer=best["layer"], readout_val=best["val_acc"],
               readout_det=best["gen_test_acc"], readout_corr=best["readout_corr"],
               readout_minus_native=delta)
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / f"internalize_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[internalize {tag}] native_det={nat.mean():.3f} readout_det={best['gen_test_acc']:.3f} "
          f"(L{best['layer']}) readout-native={delta:+.3f}", flush=True)


if __name__ == "__main__":
    main()
