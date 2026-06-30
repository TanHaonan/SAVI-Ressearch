"""Pure metrics for the PRM boundary-step calibration check (no GPU / no models).

A "step observation" carries: config, sol_id, absolute idx, n_steps, signed
distance to the first-error boundary (None for all-correct solutions), truth
(1=good/on-path, 0=bad/off-path), and the PRM reward (positive-class prob).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class StepObs:
    config: str
    sol_id: str
    idx: int                 # absolute step index (0-based)
    n: int                   # number of steps in the solution
    dist: Optional[int]      # idx - label; None if solution is all-correct
    truth: int               # 1 good, 0 bad
    reward: float            # PRM positive-class probability in (0,1)


def build_step_table(records, rewards_by_id) -> list[StepObs]:
    """records: dicts with config,id,steps,label.  rewards_by_id: {id:[r0,...]}.

    Emits only *labeled* steps: all steps for all-correct solutions; for
    erroneous solutions the correct prefix + the first-error step (post-error
    steps are excluded — their true quality is unknown).
    """
    obs: list[StepObs] = []
    for r in records:
        sid = r["id"]
        rw = rewards_by_id.get(sid)
        if rw is None:
            continue
        steps = r["steps"]
        n = len(steps)
        k = r["label"]
        if k == -1:
            for i in range(n):
                obs.append(StepObs(r["config"], sid, i, n, None, 1, float(rw[i])))
        else:
            for i in range(k):
                obs.append(StepObs(r["config"], sid, i, n, i - k, 1, float(rw[i])))
            obs.append(StepObs(r["config"], sid, k, n, 0, 0, float(rw[k])))
    return obs


def is_error(o: StepObs, theta: float) -> int:
    """1 if the thresholded prediction mislabels this step."""
    pred_bad = o.reward < theta
    true_bad = o.truth == 0
    return int(pred_bad != true_bad)


def polarity(o: StepObs, theta: float) -> Optional[str]:
    """'FN' (bad scored good), 'FP' (good scored bad), or None if correct."""
    if not is_error(o, theta):
        return None
    return "FN" if o.truth == 0 else "FP"


def boot_ci(vals, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap of the mean -> (lo, mean, hi); matches cb.metrics."""
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (float(lo), float(a.mean()), float(hi))


def dist_bin(o: StepObs) -> Optional[str]:
    if o.dist is None:
        return None
    if o.dist == 0:
        return "0(bad)"
    if o.dist >= -4:
        return str(o.dist)        # -1, -2, -3, -4
    return "<=-5"


def pos_bin(o: StepObs) -> str:
    if o.n <= 1:
        return "mid"
    f = o.idx / (o.n - 1)
    if f < 1 / 3:
        return "early"
    if f > 2 / 3:
        return "late"
    return "mid"


def rate_by_bin(obs, key_fn: Callable[[StepObs], Optional[str]], theta: float,
                n_boot: int = 2000, seed: int = 0) -> dict:
    buckets: dict[str, list] = {}
    for o in obs:
        kb = key_fn(o)
        if kb is None:
            continue
        buckets.setdefault(kb, []).append(is_error(o, theta))
    out = {}
    for kb in sorted(buckets, key=str):
        lo, mean, hi = boot_ci(buckets[kb], n_boot, seed)
        out[kb] = {"n": len(buckets[kb]), "lo": lo, "mean": mean, "hi": hi}
    return out


def choose_threshold(obs, grid=None) -> tuple[float, float]:
    """θ maximizing per-step F1 (positive class = 'bad step'). Returns (θ, F1)."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best = (0.5, -1.0)
    for t in grid:
        tp = fp = fn = 0
        for o in obs:
            pred_bad = o.reward < t
            true_bad = o.truth == 0
            if pred_bad and true_bad:
                tp += 1
            elif pred_bad and not true_bad:
                fp += 1
            elif (not pred_bad) and true_bad:
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[1]:
            best = (float(t), f1)
    return best


def uniformity_report(obs, theta: float, ratio_thresh: float = 2.0,
                      min_n: int = 50, n_boot: int = 2000, seed: int = 0) -> dict:
    good = [o for o in obs if o.truth == 1]
    bad = [o for o in obs if o.truth == 0]
    fp_rate = boot_ci([is_error(o, theta) for o in good], n_boot, seed)
    fn_rate = boot_ci([is_error(o, theta) for o in bad], n_boot, seed)
    by_distance = rate_by_bin(good, dist_bin, theta, n_boot, seed)  # FP among good
    by_position = rate_by_bin(obs, pos_bin, theta, n_boot, seed)

    flags = []
    pooled_fp = fp_rate[1] if fp_rate[1] and fp_rate[1] > 0 else 1e-9
    fpm, fnm = fp_rate[1], fn_rate[1]
    if fpm == 0 and fnm == 0:
        polarity_ratio = 1.0                 # perfectly calibrated -> no asymmetry
    elif fpm == 0:
        polarity_ratio = float("inf")        # misses errors but never false-alarms
    else:
        polarity_ratio = fnm / fpm
    # polarity flag only when both classes have enough samples to be meaningful
    if len(good) >= min_n and len(bad) >= min_n and (
            polarity_ratio > ratio_thresh or polarity_ratio < 1.0 / ratio_thresh):
        flags.append(f"polarity (FN/FP={polarity_ratio:.2f})")

    dist_means = [b["mean"] for b in by_distance.values() if b["n"] >= min_n]
    dmaxr = (max(dist_means, default=0.0) / pooled_fp)
    if dist_means and dmaxr > ratio_thresh:
        flags.append(f"distance (max good-bin {dmaxr:.2f}x pooled FP)")

    if ("early" in by_position and "late" in by_position
            and by_position["early"]["n"] >= min_n and by_position["late"]["n"] >= min_n
            and by_position["early"]["mean"] > 0):
        pr = by_position["late"]["mean"] / by_position["early"]["mean"]
        if pr > ratio_thresh or pr < 1.0 / ratio_thresh:
            flags.append(f"position (late/early={pr:.2f})")

    return {
        "theta": theta,
        "n_good": len(good),
        "n_bad": len(bad),
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "polarity_ratio": polarity_ratio,
        "by_distance": by_distance,
        "by_position": by_position,
        "flags": flags,
        "verdict": "UNIFORM" if not flags else "CONCENTRATED",
    }
