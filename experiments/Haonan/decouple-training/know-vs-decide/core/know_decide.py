"""Shared core for the belief-vs-commit minimal experiment.

belief = a 2-way distribution over the word's two senses, read by a per-candidate shared-weight scorer
        from the frozen backbone's per-sense scene-readout reps (the `rev_*` tensors saved by
        extract_features.py).
        (Per-candidate shared-weight, NOT a K-way head — a two-channel K-way head collapses.)
commit = answer "yes" iff the belief's argmax sense == the sense the QUESTION asks about (q_sense),
        applied LATE with a temperature T that is decoupled from the belief's training.

Features are precomputed once by extract_features.py into data/feats_qwen3_4b.pt; the separability
and late-commit analysis below then runs on CPU. See README.md for the two-step run order.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
FEATS = HERE / "data" / "feats_qwen3_4b.pt"
CORPUS = HERE / "data" / "corpus.json"


def load_blob(path=FEATS):
    return torch.load(path, weights_only=False)


def idx_of(meta, split):
    return [i for i, m in enumerate(meta) if m["split"] == split]


def q_sense_of(item_id, cued, gold):
    """Sense the QUESTION asks about (a property of the question). The id suffix is the CUED sense, NOT
    q_sense; q_sense is fixed by the design identity gold=='yes' iff cued==q_sense. No leak: DECIDE-correct
    (yes iff KNOW_sense==q_sense) reduces to (KNOW_sense==cued) regardless of q_sense's value."""
    return cued if gold == "yes" else ("b" if cued == "a" else "a")


class SenseScorer(nn.Module):
    """Per-candidate shared-weight scorer: same MLP/linear applied to each sense's rep -> a scalar score."""
    def __init__(self, H, kind="linear"):
        super().__init__()
        if kind == "linear":
            self.f = nn.Linear(H, 1)
        else:
            self.f = nn.Sequential(nn.Linear(H, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x2):                       # x2: [N,2,H] -> [N,2] logits over senses
        return self.f(x2).squeeze(-1)


def _z(X, mu, sd):
    return (X - mu) / sd


def rank_loss(logits, label, margin=1.0):
    """Pure ORDERING: hinge on (score_cued - score_other). No magnitude/sharpness target (saturates)."""
    s_cued = logits.gather(1, label[:, None]).squeeze(1)
    s_other = logits.gather(1, (1 - label)[:, None]).squeeze(1)
    return F.relu(margin - (s_cued - s_other)).mean()


def train_know(feat2, label, meta, loss="ce", kind="linear", layer=None, seed=0,
               epochs=300, lr=1e-3, wd=1e-2, device="cpu"):
    """feat2: [N,2,L,H] (all layers) or [N,2,H] (one layer). label: cued in {0,1}. Returns head,mu,sd,layer,val_acc."""
    torch.manual_seed(seed)
    tr, va = idx_of(meta, "train"), idx_of(meta, "val")
    if feat2.dim() == 4:                          # sweep layers, pick best val
        best = None
        for L in range(1, feat2.shape[2]):
            r = train_know(feat2[:, :, L, :], label, meta, loss, kind, L, seed, epochs, lr, wd, device)
            if best is None or r[4] > best[4]:
                best = r
        return best
    H = feat2.shape[-1]
    mu = feat2[tr].reshape(-1, H).mean(0); sd = feat2[tr].reshape(-1, H).std(0) + 1e-5
    Xtr = _z(feat2[tr], mu, sd).to(device); ytr = label[tr].to(device)
    Xva = _z(feat2[va], mu, sd).to(device); yva = label[va].to(device)
    head = SenseScorer(H, kind).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    best_acc, state = -1.0, None
    for ep in range(epochs):
        head.train(); opt.zero_grad()
        out = head(Xtr)
        loss_v = F.cross_entropy(out, ytr) if loss == "ce" else rank_loss(out, ytr)
        loss_v.backward(); opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            head.eval()
            with torch.no_grad():
                acc = float((head(Xva).argmax(1) == yva).float().mean()) if len(va) else 0.0
            if acc >= best_acc:
                best_acc, state = acc, {k: v.detach().clone() for k, v in head.state_dict().items()}
    if state:
        head.load_state_dict(state)
    return head, mu.to(device), sd.to(device), layer, best_acc


@torch.no_grad()
def know_logits(head, mu, sd, feat2_layer, device="cpu"):
    return head(_z(feat2_layer.to(device), mu, sd)).cpu()      # [N,2]


def sense_metrics(logits, label):
    """Discrimination of KNOW over senses: accuracy + AUC of (score_a - score_b) predicting cued==a."""
    pred = logits.argmax(1).numpy(); y = label.numpy()
    acc = float((pred == y).mean())
    diff = (logits[:, 0] - logits[:, 1]).numpy()              # higher -> sense a
    ya = (y == 0).astype(float)                               # cued==a
    auc = _auc(diff, ya)
    return acc, auc


def _auc(score, y):
    y = np.asarray(y); score = np.asarray(score)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def decide(logits, q_sense, T=1.0):
    """Late commitment. P(yes) = P(KNOW sense == q_sense) under softmax(logits/T). q_sense in {0:a,1:b}."""
    p = torch.softmax(logits / T, dim=1)                      # [N,2]
    p_yes = p.gather(1, torch.tensor(q_sense)[:, None]).squeeze(1)
    return p_yes


def ece(p_yes, gold_yes, bins=10):
    """Expected calibration error of the yes/no answer probabilities."""
    p = np.asarray(p_yes); y = np.asarray(gold_yes, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += abs(y[m].mean() - p[m].mean()) * m.sum() / len(p)
    return float(e)
