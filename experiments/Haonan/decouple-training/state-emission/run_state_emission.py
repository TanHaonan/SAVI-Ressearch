# state-emission/run_state_emission.py  [INTEGRATION]
"""State-emission measurement pipeline: does swapping one-hot CE -> decoupled make FREE GENERATION,
after a semantic-merge Phi, yield a non-degenerate AND calibrated distribution over semantic states?

For base/coupled/decoupled/shuffle, and per (k,j,temperature) cell, this reports -- AFTER the
semantic-merge function Phi maps each free-text continuation to a committed option-state:
  * coverage      -- fraction of the true survivors that show up in the samples,
  * calibration_tv -- distance between the empirical committed-state distribution and the KNOWN true
                      posterior (uniform over the j survivors). This is the HEADLINE scalar.
  * k_eff / eliminated_mass / abstain_rate / bar -- shape diagnostics,
placed BESIDE the answer-position readout_tv (read on the canonical oracle prompt via option_logits)
so the readout->generation gap (gap = calibration_tv - readout_tv) is explicit.

Two free-text regimes (replacing sampling-diversity's letter/freeform):
  b1 -- free reasoning + a deterministic `COMMIT: <thing>` slot; Phi = phi_b1 (deterministic, low-noise).
  b2 -- fully free answer (no slot); Phi must merge by meaning.
         Phase A uses the WEAK deterministic phi_b2_parse (a placeholder, never authoritative for b2);
         Phase B re-classifies with the decorrelated external Llama judge (authoritative for b2).

Two phases (one shared Phi/metrics core; never two big models loaded at once):
  Phase A  -- per group: load Qwen3-4B fresh (+ cached LoRA for non-base), build an AdapterGenerator
              with render_state = prompt_b1 / prompt_b2, temperature-sample continuations, map each to a
              state with the deterministic Phi, compute the L2 metrics, and read the answer-position
              posterior beside them. Raw samples are saved so metrics recompute WITHOUT re-sampling.
  Phase B  -- b2 only: load the Llama-3.1-8B judge ONCE, re-classify each saved b2 sample, recompute the
              metrics under LLM-Phi (authoritative for b2), and report deterministic<->LLM agreement +
              false-merge on the gold subset. If the judge is unavailable the run downgrades to
              deterministic-Phi only and continues (never crashes).

Two entry modes via one core:
  * fresh       -- load adapter, sample, score (default).
  * --from_dump -- recompute every metric from the saved samples_<group>_<regime>.json (no sampling,
                   no Qwen load); a re-run reproduces the numbers. Phase B also reuses Phase A's dumps.

Smoke invocation L1 runs (model loads happen inside main(), AFTER argparse, so --help is import-light):
  CUDA_VISIBLE_DEVICES=0 python run_state_emission.py \
      --groups base,decoupled --regimes b1,b2 --ks 3 --per_cell 2 --n 4 --temps 1.0 \
      --llm_subset 2 --tag smoke

Adapters: GROUPS[g] = <cp_root>/outputs/adapter_oracle_<g>_s0 (base = no adapter). Train them by
pointing controllable-posterior/core/run.py at THIS experiment's items.json; see TRAIN_NOTE.
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
except ImportError:  # progress bar is optional; degrade to identity
    def tqdm(x, **k):
        return x

HERE = Path(__file__).resolve().parent
CORE = HERE / "core"
DEPS = CORE / "_deps"
OUT = HERE / "outputs"

# How the adapters are produced (documented command, NOT run here). The decoupled-vs-coupled
# contrast needs four LoRA adapters trained on THIS experiment's carrier; training reuses the
# UNCHANGED controllable-posterior trainer:
#   1) python core/carrier.py                       # writes core/data/items.json (the carrier)
#   2) cp=<cp_root>/controllable-posterior          # the trainer dir (run.py)
#      cp data/items.json  ->  $cp/core/data/items.json   (point its trainer at this carrier's data)
#   3) for r in decoupled coupled shuffle:          # base = no adapter, nothing to train
#        python $cp/core/run.py --mode oracle --regime $r --seed 0
#      -> adapters land in $cp/core/outputs/adapter_oracle_<r>_s0/  (large, local, gitignored)
#   4) run THIS pipeline with --cp_root pointing at $cp so GROUPS resolves those adapter dirs.
TRAIN_NOTE = "see TRAIN_NOTE in run_state_emission.py header for the adapter training recipe"

JUDGE = "meta-llama/Llama-3.1-8B-Instruct"


def by_path(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


# ---- reused infra (repo importlib convention; no package install). These pull numpy/torch/re only;
# transformers/peft are imported lazily inside load_backbone / the judge path, AFTER argparse. ----
C = by_path("dep_common", DEPS / "common.py")        # load, chat_prefix_ids, answer_logprobs
cpcore = by_path("cp_core", DEPS / "cp_core.py")     # option_logits, letter_ids
cpm = by_path("cp_metrics", DEPS / "cp_metrics.py")  # tv, boot_ci
gen_mod = by_path("se_generator", CORE / "generator.py")     # AdapterGenerator, StubGenerator
phi = by_path("se_phi", CORE / "phi.py")                     # phi_b1, phi_b2_*, audit_false_merge
sm = by_path("se_state_metrics", CORE / "state_metrics.py")  # calibration_tv, coverage, k_eff, bar, ...

# Per-regime decode config: which prompt view render_state picks, and which deterministic Phi maps.
REGIMES = {
    "b1": dict(prompt_key="prompt_b1", phi=phi.phi_b1),
    "b2": dict(prompt_key="prompt_b2", phi=phi.phi_b2_parse),
}


# ------------------------------------------------------------------ utils

def cp_groups(cp_root):
    """group -> adapter path. 'base' = no adapter; oracle-mode seed-0 adapters otherwise.
    Resolved from a configurable root so the decode line can point at freshly trained LoRAs."""
    cp = Path(cp_root)
    return {
        "base": "base",
        "coupled": str(cp / "outputs/adapter_oracle_coupled_s0"),
        "decoupled": str(cp / "outputs/adapter_oracle_decoupled_s0"),
        "shuffle": str(cp / "outputs/adapter_oracle_shuffle_s0"),
        # gen-calibration lever: decoupled + a commit-slot calibration term (same dir convention).
        "decoupled_genreg": str(cp / "outputs/adapter_oracle_decoupled_genreg_s0"),
    }


def log_line(msg):
    """One human-readable line, both to stdout and appended to outputs/run.log."""
    OUT.mkdir(exist_ok=True)
    with open(OUT / "run.log", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def load_items(ks, per_cell, max_items=None):
    """Held-out subset of THIS experiment's carrier: split=='test', k in ks, up to per_cell items per
    (k,j) cell. Reads state-emission/core/data/items.json (NOT controllable-posterior's)."""
    path = CORE / "data" / "items.json"
    if not path.exists():
        raise FileNotFoundError(
            f"carrier items missing: {path}. Run `python core/carrier.py` first ({TRAIN_NOTE}).")
    items = json.loads(path.read_text())
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
        raise ValueError(f"empty item subset (ks={ks}, per_cell={per_cell}); nothing to measure")
    return sub


def target_np(item):
    """True posterior as a numpy vector aligned to item['letters']."""
    return np.array([item["target"][L] for L in item["letters"]], dtype=float)


# -------------------------------------------------------- per-item metrics core

# Metrics that enter the per-cell aggregate. calibration_tv & coverage may be nan (whole cell abstained
# / no survivors), and are dropped from their own aggregate then (reported via abstain_rate + n).
METRIC_KEYS = ["k_eff_distinct", "k_eff_entropy", "coverage", "calibration_tv",
               "eliminated_mass", "abstain_rate", "bar", "readout_tv", "gap"]


def item_metrics(states, texts, item, readout_tv, regime):
    """All per-item L2 numbers from a list of Phi-states + the raw texts, BESIDE the readout_tv.

    state_metrics is generic over the item's own alphabet: alphabet = item['letters'],
    survivors = item['survivors'], target = item['target']. calibration_tv is the headline scalar; it
    is nan when nothing committed (legitimate). gap = calibration_tv - readout_tv makes the
    readout->generation distance explicit per item (nan if calibration_tv is nan)."""
    letters, survivors, target = item["letters"], item["survivors"], item["target"]
    cal = float(sm.calibration_tv(states, target, letters))
    cov = float(sm.coverage(states, survivors))
    rt = float(readout_tv)
    return dict(
        k_eff_distinct=float(sm.k_eff_distinct(states, letters)),
        k_eff_entropy=float(sm.k_eff_entropy(states, letters)),
        coverage=cov,
        calibration_tv=cal,
        eliminated_mass=float(sm.eliminated_mass(states, survivors, letters)),
        abstain_rate=float(sm.abstain_rate(states)),
        bar=float(sm.bar(states)),
        readout_tv=rt,
        gap=(cal - rt) if np.isfinite(cal) else float("nan"),
        regime=regime,
    )


def cell_label(k, j, temp):
    """Unambiguous, parseable cell key, e.g. 'k3_j2_T1.0'. Temperature is part of the key so the
    sweep stays separate (one item -> one row per (k,j,temp) cell)."""
    return f"k{k}_j{j}_T{temp:g}"


def aggregate_cells(per_item):
    """per_item: list of dicts (one per item) carrying (k,j,temp) + metric values + readout_tv.
    Returns {'k{k}_j{j}_T{temp}': {metric: {mean, ci, n}}}, keyed by (k,j,temp) so the temperature
    sweep is recoverable and each item enters a cell exactly once (correct CI n).

    NaN-AWARE reduction: calibration_tv (and coverage, and the derived gap) are nan for an item that
    never committed; those items are dropped from THAT metric's aggregate only and the surviving n is
    reported. A metric with zero finite values yields mean=nan, ci=[nan]*3, n=0."""
    cells = defaultdict(list)
    for r in per_item:
        cells[(r["k"], r["j"], r["temp"])].append(r)
    out = {}
    for cell in sorted(cells):
        rows = cells[cell]
        agg = {}
        for key in METRIC_KEYS:
            vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
            if not vals:                       # e.g. every item abstained -> no finite calibration_tv
                agg[key] = dict(mean=float("nan"), ci=[float("nan")] * 3, n=0)
                continue
            agg[key] = dict(mean=float(np.mean(vals)), ci=cpm.boot_ci(vals), n=len(vals))
        out[cell_label(*cell)] = agg
    return out


# calibration_tv & coverage (and the derived gap) are the only means allowed to be non-finite, and
# only when the whole cell abstained (n==0). Every other metric must be finite on every written cell.
_NAN_ALLOWED = {"calibration_tv", "coverage", "gap"}


def assert_finite_cells(group, regime, cells):
    """Never write a JSON with a non-finite metric mean, EXCEPT calibration_tv/coverage/gap which are
    legitimately nan when a whole cell abstained (n==0 there)."""
    for cell, agg in cells.items():
        for key, v in agg.items():
            if key in _NAN_ALLOWED and v["n"] == 0:
                continue
            if not np.isfinite(v["mean"]):
                raise FloatingPointError(
                    f"non-finite metric cell: group={group} regime={regime} cell={cell} metric={key}")


# ----------------------------------------------------------- model loading

def load_backbone(adapter, dev):
    """Load Qwen3-4B (bf16, eval, frozen) and, for a non-'base' group, wrap with the cached LoRA.
    gptq monkeypatch is applied BEFORE importing peft (mirrors run_diversity.load_backbone)."""
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
        ap = Path(adapter)
        if not ap.exists():
            raise FileNotFoundError(
                f"adapter dir missing: {adapter} (expected a cached LoRA; {TRAIN_NOTE})")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()                                # eval-only; backbone frozen, no training
    return model, tok


# -------------------------------------------------------------- Phase A

def sample_group(group, regime, items, tok, model, dev, args):
    """Sample + deterministic-Phi + readout for one (group, regime). Returns (per_item rows, blob).
    The blob is saved so Phase B / a --from_dump re-run recompute metrics WITHOUT re-sampling."""
    cfg = REGIMES[regime]
    render_state = lambda it: it[cfg["prompt_key"]]
    phi_fn = cfg["phi"]
    gen = gen_mod.AdapterGenerator(model, tok, dev, render_state, max_new=args.max_new)

    per_item, sample_rows = [], []
    n_samp = n_tok = 0
    t0 = time.time()
    for it in tqdm(items, desc=f"{group}/{regime}"):
        k = it["k"]
        lids = cpcore.letter_ids(tok, k)
        # readout (answer-position posterior) BESIDE generation, on the canonical oracle prompt
        # (prompt_stated). option_logits is grad-friendly -> wrap in no_grad to keep the phase grad-free.
        readout_prompt = it["prompt_stated"] if args.mode == "oracle" else it["prompt_clue"]
        with torch.no_grad():
            lg = cpcore.option_logits(model, tok, readout_prompt, k, lids, dev)
        p = torch.softmax(lg, 0).detach().cpu().numpy()
        readout_tv = cpm.tv(p, target_np(it))

        for temp in args.temps:
            texts = gen.sample(it, args.n, temp, args.seed)
            states = [phi_fn(t, it) for t in texts]
            n_samp += len(texts)
            n_tok += sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
            m = item_metrics(states, texts, it, readout_tv, regime)
            m.update(k=k, j=it["j"], temp=temp, id=it["id"])
            per_item.append(m)
            sample_rows.append(dict(
                id=it["id"], k=k, j=it["j"], temp=temp, regime=regime,
                letters=it["letters"], nouns=it["nouns"], survivors=it["survivors"],
                target=it["target"], readout_tv=float(readout_tv),
                texts=texts, states=states))
    dt = max(time.time() - t0, 1e-9)
    return per_item, dict(
        group=group, regime=regime, mode=args.mode, n_items=len(items),
        samples_per_s=n_samp / dt, gen_tokens_per_s=n_tok / dt, rows=sample_rows)


def recompute_from_blob(blob):
    """Second entry mode: rebuild per_item metric rows from a saved samples blob WITHOUT re-sampling.
    Identical metric math to the fresh path (item_metrics), so --from_dump reproduces fresh numbers."""
    per_item = []
    for r in blob["rows"]:
        it = dict(letters=r["letters"], nouns=r["nouns"], survivors=r["survivors"], target=r["target"])
        m = item_metrics(r["states"], r["texts"], it, r["readout_tv"], r.get("regime", "b1"))
        m.update(k=r["k"], j=r["j"], temp=r["temp"], id=r["id"])
        per_item.append(m)
    return per_item


def write_group_outputs(group, regime_cells, args):
    """diversity_<group>.json (per-cell aggregates) + one run.log line per group x regime x cell."""
    blob = dict(group=group, mode=args.mode, tag=args.tag, cp_root=args.cp_root, regimes=regime_cells)
    (OUT / f"diversity_{group}.json").write_text(json.dumps(blob, indent=2))
    for regime, cells in regime_cells.items():
        for cell, agg in cells.items():
            log_line(
                f"[{args.tag}] {group:>9} {regime:>3} {cell:>10} | "
                f"k_eff={agg['k_eff_distinct']['mean']:.2f} "
                f"cov={agg['coverage']['mean']:.2f} "
                f"cal_tv={agg['calibration_tv']['mean']:.3f} "
                f"readout_tv={agg['readout_tv']['mean']:.3f} "
                f"gap={agg['gap']['mean']:.3f} "
                f"elim={agg['eliminated_mass']['mean']:.2f} "
                f"abst={agg['abstain_rate']['mean']:.2f} "
                f"bar={agg['bar']['mean']:.2f} "
                f"(n={agg['calibration_tv']['n']})")


# -------------------------------------------------------------- Phase B

def phase_b_faithfulness(args, dev):
    """b2 only: re-classify saved b2 samples with the external Llama judge, recompute the metrics
    under LLM-Phi (authoritative for b2), and report deterministic<->LLM agreement + false-merge on
    the gold subset. Downgrade to deterministic-Phi only (do NOT crash) if the judge is unavailable."""
    b2_blobs = []
    for group in args.groups:
        f = OUT / f"samples_{group}_b2.json"
        if f.exists():
            b2_blobs.append((group, json.loads(f.read_text())))
    if not b2_blobs:
        log_line(f"[{args.tag}] Phase B skipped: no b2 sample dumps present")
        return

    try:
        judge, jtok = C.load(JUDGE, dev)
    except Exception as e:                       # judge unavailable -> downgrade, do NOT crash
        log_line(f"[{args.tag}] WARN: judge unavailable -> deterministic-Phi only for b2 "
                 f"({type(e).__name__}: {e})")
        return

    report = {}
    for group, blob in b2_blobs:
        rows = blob["rows"][: args.llm_subset]   # cap judged rows at --llm_subset
        agree = total = 0
        false_commit = 0
        llm_per_item = []
        for r in tqdm(rows, desc=f"judge/{group}"):
            it = dict(letters=r["letters"], nouns=r["nouns"], survivors=r["survivors"], target=r["target"])
            llm_states = []
            for text, pstate in zip(r["texts"], r["states"]):
                lp = C.answer_logprobs(judge, jtok, phi.phi_b2_llm_prompt(text, it), dev,
                                       phi.phi_b2_llm_classes(it))
                cls = max(lp, key=lp.get)
                lstate = {"H": "hedge", "Z": "none"}.get(cls, cls)   # letter classes pass through
                llm_states.append(lstate)
                total += 1
                agree += int(lstate == pstate)
                false_commit += int(phi.audit_false_merge(it, lstate)["false_commit"])
            m = item_metrics(llm_states, r["texts"], it, r["readout_tv"], "b2")
            m.update(k=r["k"], j=r["j"], temp=r.get("temp", args.temp_free), id=r["id"])
            llm_per_item.append(m)
        cells = aggregate_cells(llm_per_item)
        assert_finite_cells(group, "b2_llm", cells)
        agreement = (agree / total) if total else float("nan")
        false_merge_rate = (false_commit / total) if total else float("nan")
        report[group] = dict(parse_llm_agreement=agreement, false_merge_rate=false_merge_rate,
                             n_judged=total, cells=cells)
        log_line(f"[{args.tag}] {group:>9} b2 LLM-Phi | parse<->LLM agreement={agreement:.3f} "
                 f"false_merge={false_merge_rate:.3f} (n={total})")
    (OUT / "faithfulness.json").write_text(
        json.dumps(dict(judge=JUDGE, tag=args.tag, report=report), indent=2))


# ---------------------------------------------------------------- driver

def run_phase_a(args, dev):
    groups = cp_groups(args.cp_root)
    for group in args.groups:
        if group not in groups:
            raise ValueError(f"unknown group '{group}' (choose from {list(groups)})")
        adapter = groups[group]

        if args.from_dump:
            log_line(f"[{args.tag}] === group {group} (recompute from dump, no sampling) ===")
            regime_cells = {}
            for regime in args.regimes:
                f = OUT / f"samples_{group}_{regime}.json"
                if not f.exists():
                    raise FileNotFoundError(f"--from_dump set but missing {f}; run a fresh pass first")
                blob = json.loads(f.read_text())
                per_item = recompute_from_blob(blob)
                if not per_item:
                    raise ValueError(f"empty sample list in {f}")
                cells = aggregate_cells(per_item)
                assert_finite_cells(group, regime, cells)
                regime_cells[regime] = cells
            write_group_outputs(group, regime_cells, args)
            continue

        log_line(f"[{args.tag}] === group {group}: loading model (adapter={adapter}) ===")
        model, tok = load_backbone(adapter, dev)
        items = load_items(args.ks, args.per_cell, args.max_items)
        log_line(f"[{args.tag}] group {group}: {len(items)} items, regimes={args.regimes}")

        regime_cells = {}
        for regime in args.regimes:
            if regime not in REGIMES:
                raise ValueError(f"unknown regime '{regime}' (choose from {list(REGIMES)})")
            per_item, sblob = sample_group(group, regime, items, tok, model, dev, args)
            if not per_item:
                raise ValueError(f"empty sample list: group={group} regime={regime}")
            (OUT / f"samples_{group}_{regime}.json").write_text(json.dumps(sblob, indent=2))
            cells = aggregate_cells(per_item)
            assert_finite_cells(group, regime, cells)
            regime_cells[regime] = cells
            log_line(f"[{args.tag}] {group}/{regime} throughput: "
                     f"{sblob['samples_per_s']:.2f} samples/s, "
                     f"{sblob['gen_tokens_per_s']:.1f} gen-tokens/s | "
                     f"MFU=n/a (generation-only, no training)")
        write_group_outputs(group, regime_cells, args)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()             # free before the next group / the judge


def write_recipe(args):
    """Hand the matched-compute recipe to the decode line so it can match generation compute."""
    OUT.mkdir(exist_ok=True)
    (OUT / "sampling_recipe.json").write_text(json.dumps(dict(
        N=args.n, max_new=args.max_new, temps=args.temps, temp_free=args.temp_free), indent=2))


def main():
    ap = argparse.ArgumentParser(description="state-emission measurement pipeline [INTEGRATION]")
    ap.add_argument("--groups", default="base,coupled,decoupled,shuffle",
                    help="comma list from base,coupled,decoupled,shuffle")
    ap.add_argument("--regimes", default="b1,b2", help="comma list from b1,b2")
    ap.add_argument("--ks", default="2,3,4,5", help="comma list of option-counts to include")
    ap.add_argument("--per_cell", type=int, default=20, help="items per (k,j) cell")
    ap.add_argument("--n", type=int, default=24, help="samples/item (free-text, both regimes)")
    ap.add_argument("--max_new", type=int, default=64, help="max new tokens per free-text sample")
    ap.add_argument("--temps", default="0.7,1.0,1.3", help="temperature sweep (both regimes)")
    ap.add_argument("--temp_free", type=float, default=1.0,
                    help="fallback temperature stamped on judged rows lacking a temp field")
    ap.add_argument("--llm_subset", type=int, default=60,
                    help="cap on b2 sample-rows judged by external LLM-Phi (rows[:llm_subset] per group)")
    ap.add_argument("--cp_root", default=str(HERE.parent / "controllable-posterior"),
                    help="root holding outputs/adapter_oracle_<group>_s0 LoRA dirs")
    ap.add_argument("--mode", default="oracle", choices=["oracle", "self"],
                    help="readout prompt view: oracle=prompt_stated, self=prompt_clue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from_dump", action="store_true",
                    help="recompute metrics from saved samples_*.json without re-sampling")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max_items", type=int, default=None, help="optional global item cap (smoke)")
    args = ap.parse_args()

    # parse comma lists
    args.groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    args.regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    args.ks = [int(x) for x in args.ks.split(",") if x.strip()]
    args.temps = [float(x) for x in args.temps.split(",") if x.strip()]

    # everything below this line touches torch device / loads models -- AFTER argparse on purpose,
    # so --help prints without importing transformers/peft or allocating a device.
    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    write_recipe(args)

    log_line(f"[{args.tag}] ==== state-emission run: groups={args.groups} regimes={args.regimes} "
             f"ks={args.ks} per_cell={args.per_cell} mode={args.mode} dev={dev} "
             f"cp_root={args.cp_root} from_dump={args.from_dump} seed={args.seed} ====")
    t0 = time.time()

    run_phase_a(args, dev)

    # Phase B: external-LLM-Phi faithfulness, b2 only (no-op if b2 not run / no dumps)
    if "b2" in args.regimes:
        phase_b_faithfulness(args, dev)
    else:
        log_line(f"[{args.tag}] Phase B skipped: b2 not in regimes")

    log_line(f"[{args.tag}] DONE in {time.time() - t0:.1f}s | MFU=n/a (generation-only, no training)")


if __name__ == "__main__":
    main()
