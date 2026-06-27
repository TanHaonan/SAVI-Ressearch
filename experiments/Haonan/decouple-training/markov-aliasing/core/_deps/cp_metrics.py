"""Distance of the native distribution to the known true posterior. Exact truth -> no proxy metrics."""
import numpy as np

def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())

def kl(t, p):
    t = np.asarray(t, float); p = np.asarray(p, float); m = t > 0
    return float((t[m] * np.log(t[m] / np.clip(p[m], 1e-9, 1.0))).sum())

def survivor_mass(p, survivor_mask):
    return float(np.asarray(p, float)[survivor_mask].sum())

def within_uniformity_tv(p, survivor_mask):
    """TV between the survivor-renormalized native and uniform-over-survivors (0 = perfectly uniform)."""
    p = np.asarray(p, float); s = p[survivor_mask]
    s = s / (s.sum() + 1e-9); u = np.ones_like(s) / len(s)
    return 0.5 * float(np.abs(s - u).sum())

def boot_ci(vals, n_boot=2000, seed=0):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [float(np.percentile(means, q)) for q in (2.5, 50, 97.5)]
