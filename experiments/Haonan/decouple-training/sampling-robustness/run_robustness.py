"""Robustness probe on the DETERMINISTIC LETTER regime of the toy carrier.

A robustness check on the diversity-after-Phi result, NOT a new claim. Two outputs:

PART A -- N x temperature sweep (letter regime).
  For split=='test' items, per group, per temperature T, draw N_MAX letter samples ONCE (one seeded
  draw of N_MAX continuations per item), parse each with phi_parse, then read every prefix N in
  --n_prefixes off the SAME N_MAX draws (the N-curve = metrics on samples[:N], no re-sampling). This
  answers: at what sample-count does the calibrated spread stabilize (sampling-diversity flagged N=4
  as a misleading-collapse artifact), and does decoupled's advantage hold across temperature.
  Metrics per (T, N, k, j) cell: calibration_tv (to the known uniform-over-survivors target),
  coverage, k_eff_distinct, abstain_rate, with item-bootstrap CIs. Writes outputs/robustness_<group>.json.

PART B -- answer-position readout calibration (NO generation).
  Determinate items (j==1, single correct survivor): confidence = max option softmax prob,
  correct = (argmax option letter == the survivor). reliability_curve + ECE per group. The 'calibration'
  shared metric, finally produced as a proper number.
  Ambiguous items (j>=2): mean within_uniformity_tv -- the 'should-be-uniform' calibration (0 = the
  readout spreads mass uniformly over the survivors). Writes outputs/calibration_<group>.json.

Reuses (read-only) the ORIGINAL controllable-posterior items + cached oracle adapters; nothing here
trains. argparse-first: --help loads no model. Letter prompt = item['prompt_stated'] (ends with
"Answer with one letter."); generation max_new=8.
"""
import argparse
import importlib.util as ilu
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:                                # progress bar optional; degrade to identity
    def tqdm(x, **k):
        return x

HERE = Path(__file__).resolve().parent
CORE = HERE / "core"
OUT = HERE / "outputs"

# Reused trees (read-only). The state-emission dir vendors the shared infra under its core/_deps;
# we import those modules in place by file path (the spec_from_file_location convention) so we
# duplicate no code. The letter Phi lives in sampling-diversity.
SE = HERE.parent / "state-emission"
SE_CORE = SE / "core"
SE_DEPS = SE_CORE / "_deps"
SD_CORE = HERE.parent / "sampling-diversity" / "core"

# Original (frozen) carrier items + cached oracle adapters — independent of any running job.
# Resolved relative to the sibling controllable-posterior tree.
CP_ROOT = HERE.parent / "controllable-posterior"
ITEMS_PATH = CP_ROOT / "data" / "items.json"
CP_ADAPTER_ROOT = CP_ROOT / "outputs"


def by_path(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


# ---- reused infra (numpy/torch/re only at import; transformers/peft load lazily after argparse) ----
C = by_path("dep_common", SE_DEPS / "common.py")        # load, chat_prefix_ids, DEFAULT_MODEL, CH
cpcore = by_path("cp_core", SE_DEPS / "cp_core.py")     # option_logits, letter_ids
cpm = by_path("cp_metrics", SE_DEPS / "cp_metrics.py")  # tv, within_uniformity_tv, boot_ci
gen_mod = by_path("se_generator", SE_CORE / "generator.py")     # AdapterGenerator
sm = by_path("se_state_metrics", SE_CORE / "state_metrics.py")  # calibration_tv, coverage, k_eff, abstain
phi = by_path("sd_phi", SD_CORE / "phi.py")             # phi_parse (letter regime)
ece_mod = by_path("ece", CORE / "ece.py")               # reliability_curve, ece  (the new code)


def GROUPS():
    """group -> adapter path. 'base' = no adapter; oracle-mode seed-0 adapters otherwise."""
    return {
        "base": "base",
        "coupled": str(CP_ADAPTER_ROOT / "adapter_oracle_coupled_s0"),
        "decoupled": str(CP_ADAPTER_ROOT / "adapter_oracle_decoupled_s0"),
        "shuffle": str(CP_ADAPTER_ROOT / "adapter_oracle_shuffle_s0"),
    }


# ------------------------------------------------------------------ utils

def log_line(msg):
    """One human-readable line to stdout and appended to outputs/robustness.log."""
    OUT.mkdir(exist_ok=True)
    with open(OUT / "robustness.log", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def load_items(ks, per_cell, max_items=None):
    """Held-out subset of the ORIGINAL carrier: split=='test', k in ks, up to per_cell items per (k,j)."""
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(f"carrier items missing: {ITEMS_PATH}")
    items = json.loads(ITEMS_PATH.read_text())
    per_cell_items = defaultdict(list)
    for it in items:
        if it["split"] != "test" or it["k"] not in ks:
            continue
        per_cell_items[(it["k"], it["j"])].append(it)
    sub = []
    for cell in sorted(per_cell_items):
        sub.extend(sorted(per_cell_items[cell], key=lambda x: x["id"])[:per_cell])
    if max_items is not None:
        sub = sub[:max_items]
    if not sub:
        raise ValueError(f"empty item subset (ks={ks}, per_cell={per_cell})")
    return sub


def target_np(item):
    """True posterior as a numpy vector aligned to item['letters'] (uniform over survivors)."""
    return np.array([item["target"][L] for L in item["letters"]], dtype=float)


def load_backbone(adapter, dev):
    """Load Qwen3-4B (bf16, eval, frozen) + cached LoRA for non-base. Mirrors run_state_emission."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft.import_utils as _piu
    _piu.is_gptqmodel_available = lambda *x, **k: False
    try:
        import peft.tuners.lora.gptq as _pg
        _pg.is_gptqmodel_available = lambda *x, **k: False
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL, local_files_only=True, cache_dir=C.CH)
    model = AutoModelForCausalLM.from_pretrained(
        C.DEFAULT_MODEL, dtype=torch.bfloat16, local_files_only=True, cache_dir=C.CH).to(dev)
    if adapter != "base":
        if not Path(adapter).exists():
            raise FileNotFoundError(f"adapter dir missing: {adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


# -------------------------------------------------------- PART A: N x T sweep

def cell_key(T, N, k, j):
    """Parseable (T, N, k, j) key, e.g. 'T1.0_N16_k3_j2'."""
    return f"T{T:g}_N{N}_k{k}_j{j}"


def metrics_on_prefix(states_prefix, item):
    """The four sweep metrics from a list of letter-states (Phi pushforward) + the item's truth."""
    letters, survivors, target = item["letters"], item["survivors"], item["target"]
    return dict(
        calibration_tv=float(sm.calibration_tv(states_prefix, target, letters)),  # nan if all abstained
        coverage=float(sm.coverage(states_prefix, survivors)),
        k_eff_distinct=float(sm.k_eff_distinct(states_prefix, letters)),
        abstain_rate=float(sm.abstain_rate(states_prefix)),
    )


SWEEP_METRICS = ["calibration_tv", "coverage", "k_eff_distinct", "abstain_rate"]
# calibration_tv may legitimately be nan (a whole prefix abstained); it is dropped from its own
# aggregate then and the surviving n is reported.
_NAN_ALLOWED = {"calibration_tv"}


def aggregate(rows):
    """rows: per-item dicts carrying a metric value. -> {metric: {mean, ci, n}} (nan-aware)."""
    agg = {}
    for key in SWEEP_METRICS:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        if not vals:
            agg[key] = dict(mean=float("nan"), ci=[float("nan")] * 3, n=0)
        else:
            agg[key] = dict(mean=float(np.mean(vals)), ci=cpm.boot_ci(vals), n=len(vals))
    return agg


def assert_finite_cells(group, cells):
    """Guard: never write a non-finite metric mean except calibration_tv when its whole cell abstained."""
    for ck, agg in cells.items():
        for key, v in agg.items():
            if key in _NAN_ALLOWED and v["n"] == 0:
                continue
            if not np.isfinite(v["mean"]):
                raise FloatingPointError(f"non-finite cell: group={group} cell={ck} metric={key}")


def part_a_sweep(group, items, tok, model, dev, args):
    """Draw N_MAX letter samples ONCE per (item, T), read every prefix N off those draws, aggregate
    per (T, N, k, j). Returns (cells_dict, throughput_blob)."""
    render_state = lambda it: it["prompt_stated"]         # letter prompt (ends "Answer with one letter.")
    gen = gen_mod.AdapterGenerator(model, tok, dev, render_state, max_new=args.max_new)

    # per_item rows indexed by (T, N): list of metric dicts (one per item)
    rows = defaultdict(list)
    n_samp = n_tok = 0
    t0 = time.time()
    for it in tqdm(items, desc=f"A:{group}"):
        k, j = it["k"], it["j"]
        for T in args.temps:
            texts = gen.sample(it, args.n_max, T, args.seed)   # ONE draw of N_MAX; reused for every prefix
            states = [phi.phi_parse(t, it) for t in texts]
            n_samp += len(texts)
            n_tok += sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
            for N in args.n_prefixes:
                m = metrics_on_prefix(states[:N], it)
                m.update(k=k, j=j)
                rows[(T, N)].append(m)
    dt = max(time.time() - t0, 1e-9)

    cells = {}
    for (T, N), rlist in rows.items():
        by_cell = defaultdict(list)
        for r in rlist:
            by_cell[(r["k"], r["j"])].append(r)
        for (k, j), crows in by_cell.items():
            cells[cell_key(T, N, k, j)] = aggregate(crows)
    assert_finite_cells(group, cells)
    return cells, dict(samples_per_s=n_samp / dt, gen_tokens_per_s=n_tok / dt, n_items=len(items))


def write_part_a(group, cells, args):
    """outputs/robustness_<group>.json + a summary log line per (group, T, N) aggregated over j."""
    blob = dict(group=group, regime="letter", seed=args.seed, max_new=args.max_new,
                temps=args.temps, n_max=args.n_max, n_prefixes=args.n_prefixes, cells=cells)
    OUT.mkdir(exist_ok=True)
    (OUT / f"robustness_{group}.json").write_text(json.dumps(blob, indent=2))

    # log: per (T, N) aggregate over all j (mean of cell means, nan-aware)
    by_TN = defaultdict(list)
    for ck, agg in cells.items():
        # ck = 'T{T}_N{N}_k{k}_j{j}'
        T = ck.split("_")[0][1:]
        N = ck.split("_")[1][1:]
        by_TN[(T, N)].append(agg)
    for (T, N) in sorted(by_TN, key=lambda x: (float(x[0]), int(x[1]))):
        aggs = by_TN[(T, N)]
        def mean_of(metric):
            vs = [a[metric]["mean"] for a in aggs if np.isfinite(a[metric]["mean"])]
            return float(np.mean(vs)) if vs else float("nan")
        log_line(
            f"[A] {group:>9} T={T:>3} N={int(N):>2} | "
            f"cal_tv={mean_of('calibration_tv'):.3f} "
            f"cov={mean_of('coverage'):.3f} "
            f"k_eff={mean_of('k_eff_distinct'):.2f} "
            f"abst={mean_of('abstain_rate'):.2f} "
            f"(cells={len(aggs)})")


# -------------------------------------------------- PART B: readout calibration

def part_b_calibration(group, items, tok, model, dev, args):
    """Answer-position readout (no generation). Determinate (j==1): confidence = max option softmax,
    correct = (argmax letter == survivor) -> reliability_curve + ECE. Ambiguous (j>=2): mean
    within_uniformity_tv. Returns the calibration blob."""
    confidences, correct = [], []          # determinate readout calibration
    within_unifs = []                      # ambiguous 'should-be-uniform' readout
    n_det = n_amb = 0
    for it in tqdm(items, desc=f"B:{group}"):
        k = it["k"]
        lids = cpcore.letter_ids(tok, k)
        with torch.no_grad():
            lg = cpcore.option_logits(model, tok, it["prompt_stated"], k, lids, dev)
        p = torch.softmax(lg, 0).detach().cpu().numpy()
        if it["j"] == 1:
            conf = float(p.max())
            pred_letter = it["letters"][int(np.argmax(p))]
            survivor = it["survivors"][0]
            confidences.append(conf)
            correct.append(int(pred_letter == survivor))
            n_det += 1
        else:
            survivor_mask = np.array([L in set(it["survivors"]) for L in it["letters"]], dtype=bool)
            within_unifs.append(float(cpm.within_uniformity_tv(p, survivor_mask)))
            n_amb += 1

    curve = ece_mod.reliability_curve(confidences, correct, n_bins=args.n_bins)
    ece_det = ece_mod.ece(confidences, correct, n_bins=args.n_bins)
    mean_within = float(np.mean(within_unifs)) if within_unifs else float("nan")
    acc_det = float(np.mean(correct)) if correct else float("nan")
    mean_conf_det = float(np.mean(confidences)) if confidences else float("nan")

    return dict(
        group=group, n_bins=args.n_bins,
        determinate=dict(
            n=n_det, ece=(None if np.isnan(ece_det) else ece_det),
            accuracy=(None if np.isnan(acc_det) else acc_det),
            mean_confidence=(None if np.isnan(mean_conf_det) else mean_conf_det),
            reliability_curve=[dict(mean_conf=mc, acc=a, count=c) for (mc, a, c) in curve]),
        ambiguous=dict(
            n=n_amb,
            mean_within_uniformity_tv=(None if np.isnan(mean_within) else mean_within)),
    )


def write_part_b(blob, args):
    group = blob["group"]
    OUT.mkdir(exist_ok=True)
    (OUT / f"calibration_{group}.json").write_text(json.dumps(blob, indent=2))
    d, a = blob["determinate"], blob["ambiguous"]
    ece_s = "nan" if d["ece"] is None else f"{d['ece']:.3f}"
    unif_s = "nan" if a["mean_within_uniformity_tv"] is None else f"{a['mean_within_uniformity_tv']:.3f}"
    acc_s = "nan" if d["accuracy"] is None else f"{d['accuracy']:.3f}"
    log_line(
        f"[B] {group:>9} | ECE_determinate={ece_s} (n={d['n']}, acc={acc_s}) | "
        f"mean_within_unif_ambiguous={unif_s} (n={a['n']})")


# ---------------------------------------------------------------- driver

def run(args, dev):
    groups = GROUPS()
    items = load_items(args.ks, args.per_cell, args.max_items)
    log_line(f"[setup] groups={args.groups} ks={args.ks} per_cell={args.per_cell} "
             f"temps={args.temps} n_max={args.n_max} n_prefixes={args.n_prefixes} "
             f"n_items={len(items)} dev={dev} seed={args.seed}")
    for group in args.groups:
        if group not in groups:
            raise ValueError(f"unknown group '{group}' (choose from {list(groups)})")
        adapter = groups[group]
        log_line(f"=== group {group}: loading model (adapter={adapter}) ===")
        model, tok = load_backbone(adapter, dev)

        # PART A — N x T sweep (letter generation)
        cells, thru = part_a_sweep(group, items, tok, model, dev, args)
        write_part_a(group, cells, args)
        log_line(f"[A] {group} throughput: {thru['samples_per_s']:.2f} samples/s, "
                 f"{thru['gen_tokens_per_s']:.1f} gen-tokens/s (n_items={thru['n_items']})")

        # PART B — readout calibration / ECE (no generation)
        blob = part_b_calibration(group, items, tok, model, dev, args)
        write_part_b(blob, args)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="letter-regime N x T robustness sweep + readout ECE")
    ap.add_argument("--groups", default="base,coupled,decoupled,shuffle",
                    help="comma list from base,coupled,decoupled,shuffle")
    ap.add_argument("--ks", default="2,3,4,5", help="comma list of option-counts to include")
    ap.add_argument("--per_cell", type=int, default=10, help="items per (k,j) cell")
    ap.add_argument("--temps", default="0.7,1.0,1.3", help="temperature sweep")
    ap.add_argument("--n_max", type=int, default=48, help="letter samples drawn ONCE per (item,T)")
    ap.add_argument("--n_prefixes", default="4,8,16,24,48",
                    help="prefix sample-counts to read the N-curve off the single N_MAX draw")
    ap.add_argument("--max_new", type=int, default=8, help="max new tokens per letter sample")
    ap.add_argument("--n_bins", type=int, default=10, help="ECE / reliability-curve bins")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_items", type=int, default=None, help="optional global item cap (smoke)")
    args = ap.parse_args()

    args.groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    args.ks = [int(x) for x in args.ks.split(",") if x.strip()]
    args.temps = [float(x) for x in args.temps.split(",") if x.strip()]
    args.n_prefixes = sorted({int(x) for x in args.n_prefixes.split(",") if x.strip()})
    if max(args.n_prefixes) > args.n_max:
        raise ValueError(f"n_prefixes max {max(args.n_prefixes)} exceeds n_max {args.n_max}")

    # everything below touches the device / loads models — AFTER argparse so --help is import-light.
    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    log_line(f"==== robustness run: groups={args.groups} ks={args.ks} per_cell={args.per_cell} "
             f"temps={args.temps} n_max={args.n_max} n_prefixes={args.n_prefixes} "
             f"max_new={args.max_new} seed={args.seed} dev={dev} ====")
    t0 = time.time()
    run(args, dev)
    log_line(f"DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
