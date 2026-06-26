"""P1 (separability) + P3 (DECIDE-late beats greedy) on cached Qwen3-4B features. CPU.

P1: KNOW trained with pure RANK loss vs CE -> does discrimination need sharpness pressure? (separable in
    training?) Then sweep the DECIDE temperature -> discrimination (acc/AUC) flat while calibration (ECE) /
    commitment (entropy) move -> independent knobs.
P3: KNOW(best) -> DECIDE(late) answer accuracy vs the model's native forward answer (meta.local_correct).

Two-step run (see README.md):
  1. python extract_features.py            # builds data/feats_qwen3_4b.pt (needs a GPU once)
  2. python run_p1_p3.py                   # CPU-only analysis; writes outputs/p1_p3.json
"""
import json
from pathlib import Path
import numpy as np
import torch
import know_decide as K

HERE = Path(__file__).resolve().parent


def main():
    blob = K.load_blob()
    meta = blob["meta"]
    label = torch.tensor([0 if m["cued"] == "a" else 1 for m in meta])     # cued sense
    feat = blob["rev_mean"].float()                                        # [N,2,L,H]
    gi = K.idx_of(meta, "gen_test")
    q = np.array([0 if K.q_sense_of(meta[i]["id"], meta[i]["cued"], meta[i]["gold"]) == "a" else 1 for i in range(len(meta))])
    gold_yes = np.array([1 if meta[i]["gold"] == "yes" else 0 for i in range(len(meta))])
    local = np.array([1 if meta[i]["local_correct"] else 0 for i in range(len(meta))])

    out = {}

    # ---------- P1: rank-only vs CE discrimination ----------
    res = {}
    heads = {}
    for loss in ("ce", "rank"):
        accs, aucs = [], []
        for seed in (0, 1):
            head, mu, sd, layer, va = K.train_know(feat, label, meta, loss=loss, kind="linear", seed=seed)
            lg = K.know_logits(head, mu, sd, feat[:, :, layer, :])
            a, u = K.sense_metrics(lg[gi], label[gi])
            accs.append(a); aucs.append(u)
            if seed == 0:
                heads[loss] = (head, mu, sd, layer, lg)
        res[loss] = dict(layer=int(heads[loss][3]), sense_acc=float(np.mean(accs)), sense_auc=float(np.mean(aucs)))
    gap_disc = res["ce"]["sense_acc"] - res["rank"]["sense_acc"]
    out["P1_discrimination"] = dict(
        ce=res["ce"], rank_only=res["rank"], ce_minus_rank=float(gap_disc),
        verdict="rank-only matches CE (separable in training)" if abs(gap_disc) <= 0.03
                else "rank-only WORSE than CE (discrimination needs sharpness pressure -> not separable)")

    # ---------- P1: DECIDE temperature sweep (on rank-only KNOW) ----------
    _, _, _, _, lg_rank = heads["rank"]
    sweep = []
    for T in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        p_yes = K.decide(lg_rank[gi], q[gi], T=T).numpy()
        ans = (p_yes > 0.5).astype(int)
        acc = float((ans == gold_yes[gi]).mean())
        conf = float(np.mean(np.maximum(p_yes, 1 - p_yes)))
        ent = float(np.mean(-(p_yes * np.log(p_yes + 1e-9) + (1 - p_yes) * np.log(1 - p_yes + 1e-9))))
        e = K.ece(p_yes, gold_yes[gi])
        sweep.append(dict(T=T, answer_acc=acc, mean_conf=conf, mean_entropy=ent, ece=e))
    accs = [s["answer_acc"] for s in sweep]; eces = [s["ece"] for s in sweep]
    out["P1_decide_sweep"] = dict(
        sweep=sweep, acc_range=float(max(accs) - min(accs)), ece_range=float(max(eces) - min(eces)),
        verdict="independent knobs: acc flat, ECE moves" if (max(accs) - min(accs) < 0.01 and max(eces) - min(eces) >= 0.05)
                else "not cleanly independent")

    # ---------- P3: DECIDE-late vs native greedy ----------
    best_loss = "ce" if res["ce"]["sense_acc"] >= res["rank"]["sense_acc"] else "rank"
    _, _, _, _, lg_best = heads[best_loss]
    p_yes = K.decide(lg_best[gi], q[gi], T=1.0).numpy()
    decide_correct = ((p_yes > 0.5).astype(int) == gold_yes[gi]).astype(int)
    sense_acc = float((lg_best[gi].argmax(1).numpy() == label[gi].numpy()).mean())
    native = local[gi]
    diff = decide_correct - native
    rng = np.random.default_rng(0)
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)])
    out["P3_decide_vs_greedy"] = dict(
        best_know_loss=best_loss, know_sense_acc=sense_acc,
        decide_late_answer_acc=float(decide_correct.mean()), native_greedy_acc=float(native.mean()),
        gap=float(diff.mean()), gap_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        verdict="DECIDE-late >> greedy (recovers answers the native forward pass gets wrong)" if np.percentile(bs, 2.5) > 0
                else "no gap")

    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "p1_p3.json").write_text(json.dumps(out, indent=2))

    print("=== P1 discrimination (gen_test sense acc) ===")
    print(f"  CE        acc={res['ce']['sense_acc']:.3f} auc={res['ce']['sense_auc']:.3f} (L{res['ce']['layer']})")
    print(f"  rank-only acc={res['rank']['sense_acc']:.3f} auc={res['rank']['sense_auc']:.3f} (L{res['rank']['layer']})")
    print(f"  CE - rank = {gap_disc:+.3f}  -> {out['P1_discrimination']['verdict']}")
    print("\n=== P1 DECIDE temperature sweep (rank-only KNOW) ===")
    for s in sweep:
        print(f"  T={s['T']:<4} answer_acc={s['answer_acc']:.3f} conf={s['mean_conf']:.3f} ent={s['mean_entropy']:.3f} ECE={s['ece']:.3f}")
    print(f"  acc range={out['P1_decide_sweep']['acc_range']:.3f} ECE range={out['P1_decide_sweep']['ece_range']:.3f} -> {out['P1_decide_sweep']['verdict']}")
    print("\n=== P3 DECIDE-late vs native greedy (gen_test) ===")
    p3 = out["P3_decide_vs_greedy"]
    print(f"  KNOW sense acc={p3['know_sense_acc']:.3f}  DECIDE-late answer acc={p3['decide_late_answer_acc']:.3f}"
          f"  native greedy={p3['native_greedy_acc']:.3f}")
    print(f"  gap={p3['gap']:+.3f} CI{p3['gap_ci']} -> {p3['verdict']}")


if __name__ == "__main__":
    main()
