"""Build pairwise (k,k) log-potentials per constraint from one of four sources.

Each cross-slot constraint becomes one (k,k) matrix M where M[a,b] = how compatible option a of slot i is
with option b of slot j, in log-potential units. Four sources:
  - self / cheap : a language model answers a closed-form pairwise yes/no consistency query for every (a,b)
                   cell; the edge weight is logp(yes) - logp(no). Model-agnostic via an injected closure
                   score_yes_no(prompt) -> float, so tests need no model.
  - oracle       : the planted hard compatibility matrix (the ceiling); already HARD_PEN-encoded.
  - shuffle      : oracle matrices reassigned across items (control that breaks the signal).

API NOTES (confirmed against the deps):
  - the shared scoring helpers (_deps/common.py) expose p_yes(model, tok, user, device) which returns a
    PROBABILITY in [0,1] (P(yes) normalised against P(no)), NOT a logit. make_score_yes_no converts it
    back to a margin logp(yes)-logp(no) = log(p)-log(1-p) with clipping. (answer_logprobs is available as
    a fallback that returns raw per-class log-probs whose difference is the same margin.)
  - the cross-tokenizer model loader (_deps/frontier.py) exposes load(name, device) for the cheap ~1B model.
  - gen_data item schema (confirmed): each constraint dict has keys i, j, relation, text, compat[(k,k)];
    options[t] is a list of k nouns; slot_names[t] is the human slot label; letters/k present. The _query
    below names both slots and the two specific option nouns, as required.
"""
import importlib.util as ilu, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DEPS = HERE / "_deps"
# oracle.py is vendored in _deps/; used only for HARD_PEN here.
sys.path.insert(0, str(DEPS))
import oracle  # HARD_PEN
HARD = oracle.HARD_PEN


def _query(item, e, a, b):
    """Closed-form pairwise consistency prompt naming BOTH slots and the two specific option nouns."""
    ni = item["options"][e["i"]][a]
    nj = item["options"][e["j"]][b]
    return (f"{e['text']} If {item['slot_names'][e['i']]} hides the {ni} and "
            f"{item['slot_names'][e['j']]} hides the {nj}, is that consistent? Answer yes or no.")


def build_edges(item, score_yes_no):
    """For each constraint, fill the (k,k) margin matrix from pairwise queries. Returns [(i,j,M)].

    score_yes_no(prompt) -> float is the logp(yes)-logp(no) margin for that prompt. M[a,b] is the margin for
    'option a of slot i with option b of slot j'. A constant scorer therefore yields a flat (uninformative)
    edge; a truth-aligned scorer reproduces the oracle ranking."""
    k = item["k"]
    out = []
    for e in item["constraints"]:
        M = np.zeros((k, k), float)
        for a in range(k):
            for b in range(k):
                M[a, b] = score_yes_no(_query(item, e, a, b))
        out.append((e["i"], e["j"], M))
    return out


def build_edges_from_truth(item):
    """oracle source: the planted hard compat matrices (the ceiling), already HARD_PEN-encoded."""
    return [(e["i"], e["j"], np.array(e["compat"], float)) for e in item["constraints"]]


def shuffle_edges(items, seed=0):
    """control: reassign each item's oracle edge matrices to a DIFFERENT item (deterministic given seed).

    A fixed-point-free permutation (derangement when possible) so each item generally gets edges that are
    NOT its own -> the signal is genuinely broken. Edges are aligned by constraint position; on a shape
    mismatch (different C or k) that slot falls back to a flat zeros matrix (no signal). Identities (i,j)
    always stay the item's own, so the graph topology is unchanged and only the matrix content is scrambled."""
    n = len(items)
    rng = np.random.default_rng(seed)
    truths = [build_edges_from_truth(it) for it in items]
    perm = _derangement(n, rng)
    out = []
    for idx, it in enumerate(items):
        donor = truths[perm[idx]]
        mine = truths[idx]
        fixed = []
        for q, (i, j, M) in enumerate(mine):
            if q < len(donor) and np.array(donor[q][2]).shape == M.shape:
                fixed.append((i, j, np.array(donor[q][2], float)))
            else:
                fixed.append((i, j, np.zeros_like(M)))   # shape mismatch -> flat (broken signal)
        out.append(fixed)
    return out


def _derangement(n, rng):
    """A permutation with no fixed points when n > 1 (so no item keeps its own edges); identity if n <= 1."""
    if n <= 1:
        return np.arange(n)
    for _ in range(64):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    # fallback: a cyclic shift is fixed-point-free for n > 1
    return (np.arange(n) + 1) % n


def edges_equal(a, b):
    """Structural equality of two edge lists-of-lists: same shape, same (i,j), same matrices."""
    if len(a) != len(b):
        return False
    for ea, eb in zip(a, b):
        if len(ea) != len(eb):
            return False
        for (i1, j1, M1), (i2, j2, M2) in zip(ea, eb):
            if (i1, j1) != (i2, j2):
                return False
            if np.array(M1).shape != np.array(M2).shape or not np.allclose(M1, M2):
                return False
    return True


def edge_phi(err_a, err_b):
    """phi (mean-square-contingency) correlation of two boolean error vectors. Matches the shared phi
    helper semantics; returns 0.0 when a margin is degenerate (zero denominator)."""
    a = np.asarray(err_a, bool); b = np.asarray(err_b, bool)
    n11 = np.sum(a & b); n10 = np.sum(a & ~b); n01 = np.sum(~a & b); n00 = np.sum(~a & ~b)
    num = n11 * n00 - n10 * n01
    den = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return float(num / den) if den > 0 else 0.0


# --- real scorers (used by run.py, NOT exercised by tests; this is the only model-touching function) ---
def make_score_yes_no(model, tok, dev):
    """Closure scoring logp(yes)-logp(no) at the answer position, reusing the shared scoring helpers.

    p_yes(model, tok, user, device) returns P(yes) in [0,1] (probability, not a logit), so we invert
    it to the margin logp(yes)-logp(no) = log(p)-log(1-p) with clipping. This is a real margin (unbounded
    log-odds), not a probability. Works cross-tokenizer for the cheap model: the yes/no scoring is on that
    model's own tokens (load it via the cross-tokenizer loader and pass it here)."""
    _s = ilu.spec_from_file_location("dep_common", DEPS / "common.py")
    _C = ilu.module_from_spec(_s); _s.loader.exec_module(_C)

    def f(prompt):
        py = _C.p_yes(model, tok, prompt, dev)          # P(yes) in [0,1]
        py = min(max(py, 1e-4), 1 - 1e-4)               # clip away from 0/1 so the log-odds stays finite
        return float(np.log(py) - np.log(1.0 - py))     # margin = logp(yes) - logp(no)

    return f
