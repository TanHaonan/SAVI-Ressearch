"""Generation-level lever: train `decoupled_genreg` = `decoupled` + ONE commit-slot calibration term.

Copies the training loop of controllable-posterior/core/run.py (mode oracle, seed 0, 6 epochs,
lr 1e-4, rank 16, bs 8, fluency lam_kl 0.5) UNCHANGED, except the per-item loss is:

    loss_decoupled(option_logits @ answer position, target)              # the control loss
  + LAM_COMMIT * loss_commit(commit_logits @ COMMIT slot over nouns, target)   # the ONE added term
  [+ lam_kl * kl_to_base(GENERAL) fluency anchor, exactly as run.py]

Saves to controllable-posterior/core/outputs/adapter_oracle_decoupled_genreg_s0 so the
state-emission eval harness's group->adapter map (cp_groups) finds it under the name
`decoupled_genreg`. Per-epoch log: total loss + the two components + answer-position meanTV
+ commit-slot meanTV on the held-out test slice -> gen-calibration/outputs/train_genreg.log.
"""
import argparse
import importlib.util as ilu
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CP_CORE = HERE.parent / "controllable-posterior" / "core"


def _bp(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


core = _bp("core", CP_CORE / "core.py")            # loss_decoupled, option_logits, letter_ids, kl_to_base, GENERAL, target_vec
metrics = _bp("metrics", CP_CORE / "metrics.py")   # tv
gl = _bp("genreg_loss", HERE / "core" / "genreg_loss.py")
C = core.C


def answer_meanTV(model, tok, items, lids, prompt_key, dev):
    """Answer-position mean-TV (the control readout) — must STAY low (guardrail)."""
    tvs = []
    for it in items:
        lg = core.option_logits(model, tok, it[prompt_key], it["k"], lids[it["k"]], dev)
        p = F.softmax(lg, 0).detach().cpu().numpy()
        t = np.array([it["target"][L] for L in it["letters"]])
        tvs.append(metrics.tv(p, t))
    return float(np.mean(tvs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="2,3,4,5")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lam_kl", type=float, default=0.5)        # fluency anchor (as run.py)
    ap.add_argument("--lam_commit", type=float, default=1.0)    # the ONE added term's weight
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template", default=gl.DEFAULT_TEMPLATE)
    ap.add_argument("--eval_slice", type=int, default=48, help="held-out test items for the per-epoch curve")
    a = ap.parse_args()

    dev = torch.device("cuda"); torch.manual_seed(a.seed)
    mode = "oracle"
    prompt_key = "prompt_stated"                                # oracle readout view (as run.py)
    tag = f"oracle_decoupled_genreg_s{a.seed}"

    items = json.loads((CP_CORE / "data" / "items.json").read_text())
    ks = [int(x) for x in a.ks.split(",")]
    items = [it for it in items if it["k"] in ks]
    tr = [it for it in items if it["split"] == "train"]
    te = [it for it in items if it["split"] == "test"]

    # held-out eval slice for the per-epoch curve: up to eval_slice items, balanced across (k,j) cells.
    by_cell = defaultdict(list)
    for it in sorted(te, key=lambda x: x["id"]):
        by_cell[(it["k"], it["j"])].append(it)
    eval_items = []
    cells = sorted(by_cell)
    i = 0
    while len(eval_items) < a.eval_slice and any(by_cell[c] for c in cells):
        c = cells[i % len(cells)]
        if by_cell[c]:
            eval_items.append(by_cell[c].pop(0))
        i += 1

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft.import_utils as _piu
    _piu.is_gptqmodel_available = lambda *x, **k: False
    try:
        import peft.tuners.lora.gptq as _pg; _pg.is_gptqmodel_available = lambda *x, **k: False
    except Exception:
        pass
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL, local_files_only=True, cache_dir=C.CH)
    base = AutoModelForCausalLM.from_pretrained(C.DEFAULT_MODEL, dtype=torch.bfloat16,
                                                local_files_only=True, cache_dir=C.CH).to(dev)
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lcfg)
    assert all((not p.requires_grad) for n, p in model.named_parameters() if "lora" not in n), "backbone must be frozen"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    lids = {k: core.letter_ids(tok, k) for k in ks}
    # noun_ids depends on the item's nouns (not just k), so resolve per item inside the loop; but the
    # per-epoch eval reuses a per-item cache keyed by id to avoid re-encoding.
    nid_cache = {}

    def nid_for(it):
        if it["id"] not in nid_cache:
            nid_cache[it["id"]] = gl.noun_ids(tok, it["nouns"])
        return nid_cache[it["id"]]

    outdir = HERE / "outputs"; outdir.mkdir(exist_ok=True)
    log = open(outdir / "train_genreg.log", "w")

    def emit(line):
        print(" ", line, flush=True); log.write(line + "\n"); log.flush()

    emit(f"[{tag}] start: {len(tr)} train / {len(te)} test, eval_slice={len(eval_items)}, "
         f"ks={ks}, lam_commit={a.lam_commit}, lam_kl={a.lam_kl}, template={a.template!r}")

    def eval_curve():
        """Answer-position and commit-slot mean-TV on the held-out slice (the per-epoch curve)."""
        a_tv = answer_meanTV(model, tok, eval_items, lids, prompt_key, dev)
        ctvs = []
        for it in eval_items:
            clg = gl.commit_logits(model, tok, it["prompt_b1"], a.template, it["nouns"], nid_for(it), dev)
            p = F.softmax(clg, 0).detach().cpu().numpy()
            t = np.array([it["target"][L] for L in it["letters"]])
            ctvs.append(metrics.tv(p, t))
        return a_tv, float(np.mean(ctvs))

    # epoch 0 (pre-training) baseline curve so the DROP is visible from the untrained adapter
    model.eval()
    with torch.no_grad():
        a_tv0, c_tv0 = eval_curve()
    emit(f"ep-: (pre-train) ans_meanTV={a_tv0:.3f} commit_meanTV={c_tv0:.3f}")

    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        g = torch.Generator().manual_seed(1000 * a.seed + ep)
        perm = torch.randperm(len(tr), generator=g).tolist()
        opt.zero_grad()
        run_tot = run_dec = run_com = 0.0
        seen = 0
        for step, j in enumerate(perm):
            it = tr[j]; k = it["k"]; t = core.target_vec(it, dev)
            lg = core.option_logits(model, tok, it[prompt_key], k, lids[k], dev)
            l_dec = core.loss_decoupled(lg, t)
            clg = gl.commit_logits(model, tok, it["prompt_b1"], a.template, it["nouns"], nid_for(it), dev)
            l_com = gl.loss_commit(clg, t)
            loss = l_dec + a.lam_commit * l_com
            (loss / a.bs).backward()
            run_tot += float(loss); run_dec += float(l_dec); run_com += float(l_com); seen += 1
            if (step + 1) % a.bs == 0 or step == len(perm) - 1:
                if a.lam_kl > 0:
                    (a.lam_kl * core.kl_to_base(model, tok, core.GENERAL[:4], dev)).backward()
                opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            a_tv, c_tv = eval_curve()
        ips = seen / (time.time() - t0)
        emit(f"ep{ep}: loss={run_tot/seen:.3f} (dec={run_dec/seen:.3f} commit={run_com/seen:.3f}) "
             f"ans_meanTV={a_tv:.3f} commit_meanTV={c_tv:.3f} items/s={ips:.1f}")
        if not np.isfinite(run_tot):
            raise FloatingPointError(f"non-finite loss at epoch {ep}")

    adir = CP_CORE / "outputs" / f"adapter_{tag}"
    model.save_pretrained(str(adir))
    emit(f"=== saved adapter -> {adir} ===")
    log.close()


if __name__ == "__main__":
    main()
