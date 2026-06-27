# sampling-diversity/core/run_diversity.py  [INTEGRATION]
"""Measure generation-side diversity-after-Phi for base/coupled/decoupled/shuffle on the
controllable-posterior carrier, beside the known readout-TV. No training; reuses cached adapters.

Per group (base/coupled/decoupled/shuffle) and per (k,j) cell this reports, after the semantic-merge
function Phi: the number of distinct option-states the temperature samples cover (k_eff), what fraction
of the true survivors show up (coverage), and the distance between the empirical committed-option
distribution and the true posterior (generation-state-TV) -- placed BESIDE the answer-position readout-TV
so the readout-vs-generation gap is explicit.

Two phases (one shared Phi/metrics core, never two big models loaded at once):
  Phase A  -- per group: load Qwen3-4B fresh (wrap with the LoRA adapter for non-base groups),
              temperature-sample continuations, map each to an option-state with the deterministic
              parser-Phi, compute the diversity metrics, and read the answer-position posterior beside
              them. Raw samples are saved so metrics can be recomputed WITHOUT re-sampling.
  Phase B  -- freeform only: load the Llama-3.1-8B judge ONCE, re-classify each saved freeform sample
              with an external LLM (the decorrelated cross-check), recompute the metrics under LLM-Phi,
              and report parser-vs-LLM agreement. If the judge is unavailable the run downgrades to
              parser-Phi only and continues.

Two entry modes via one core:
  * fresh   -- load adapter, sample, score (default).
  * --from_dump -- recompute every metric from the saved samples_<group>_<regime>.json (no sampling,
              no Qwen load) so a re-run reproduces the numbers; this is also how Phase B reuses Phase A.

Small-subset invocation that L1 runs (model loads happen inside main(), after argparse):
  CUDA_VISIBLE_DEVICES=0 python run_diversity.py --per_cell 2 --n_letter 4 --n_free 4 --temps 1.0 \
      --regimes letter,freeform --llm_subset 2 --groups base,decoupled --tag smoke
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
DEPS = HERE / "_deps"               # vendored deps -> importable without any repo-relative path
OUT = HERE / "outputs"

# Large local artifacts (NOT vendored here): the controllable-posterior carrier item bank
# (CP/data/items.json) and the cached LoRA adapters (CP/outputs/adapter_oracle_*_s0). These are
# big, machine-local files; only Phase A's `--from_dump=False` model run reads them. Override with
# --cp_root if your checkout differs. The module imports and `--help` without touching this path.
CP = HERE.parent.parent / "controllable-posterior"


def by_path(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


# ---- reused infra: vendored locally under core/_deps so this dir is self-contained ----
C = by_path("ce_common", DEPS / "common.py")
core = by_path("cp_core", DEPS / "cp_core.py")
cpm = by_path("cp_metrics", DEPS / "cp_metrics.py")
sample = by_path("sd_sample", HERE / "sample.py")
phi = by_path("sd_phi", HERE / "phi.py")
dm = by_path("sd_divmetrics", HERE / "divmetrics.py")

# group -> adapter path ("base" = no adapter); oracle-mode seed-0 adapters
GROUPS = {
    "base": "base",
    "coupled": str(CP / "outputs/adapter_oracle_coupled_s0"),
    "decoupled": str(CP / "outputs/adapter_oracle_decoupled_s0"),
    "shuffle": str(CP / "outputs/adapter_oracle_shuffle_s0"),
}
JUDGE = "meta-llama/Llama-3.1-8B-Instruct"


# ------------------------------------------------------------------ utils

def log_line(msg):
    """One human-readable line, both to stdout and appended to outputs/run.log."""
    OUT.mkdir(exist_ok=True)
    with open(OUT / "run.log", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def load_items(ks, per_cell, mode, max_items=None):
    """Held-out subset: split=='test', k in ks, up to per_cell items per (k,j) cell."""
    items = json.loads((CP / "data" / "items.json").read_text())
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
        raise ValueError(f"empty item subset (ks={ks}, per_cell={per_cell}, mode={mode}); nothing to measure")
    return sub


def target_np(item):
    """True posterior as a numpy vector aligned to item['letters']."""
    return np.array([item["target"][L] for L in item["letters"]], dtype=float)


def item_metrics(states, texts, item):
    """All per-item diversity numbers from a list of Phi-states + the raw texts."""
    letters, survivors = item["letters"], item["survivors"]
    return dict(
        k_eff_distinct=float(dm.k_eff_distinct(states, letters)),
        k_eff_entropy=float(dm.k_eff_entropy(states, letters)),
        coverage=float(dm.coverage(states, survivors)),
        gen_state_tv=float(dm.gen_state_tv(states, item["target"], letters)),
        eliminated_mass=float(dm.eliminated_mass(states, survivors, letters)),
        abstain_rate=float(dm.abstain_rate(states)),
        distinct_text=float(dm.distinct_text(texts)),
    )


METRIC_KEYS = ["k_eff_distinct", "k_eff_entropy", "coverage", "gen_state_tv",
               "eliminated_mass", "abstain_rate", "distinct_text", "readout_tv"]


def cell_label(k, j, temp):
    """Unambiguous, parseable cell key: e.g. 'k3_j2_T1.0'. Temperature is part of the key so the
    letter regime's temperature sweep stays separate (one item -> one row per (k,j,temp) cell)."""
    return f"k{k}_j{j}_T{temp:g}"


def aggregate_cells(per_item):
    """per_item: list of dicts (one per item) carrying (k,j,temp) + metric values + readout_tv.
    Returns {'k{k}_j{j}_T{temp}': {metric: {mean, ci}}}. Cells are keyed by (k,j,temp) so the
    temperature sweep is recoverable and each item enters a cell exactly once (correct CI n).
    gen_state_tv can be nan for an item that never committed; those items are dropped from the
    gen_state_tv aggregate only (reported via abstain_rate)."""
    cells = defaultdict(list)
    for r in per_item:
        cells[(r["k"], r["j"], r["temp"])].append(r)
    out = {}
    for cell in sorted(cells):
        rows = cells[cell]
        agg = {}
        for key in METRIC_KEYS:
            vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
            if not vals:                       # e.g. every item abstained -> no finite gen_state_tv
                agg[key] = dict(mean=float("nan"), ci=[float("nan")] * 3, n=0)
                continue
            agg[key] = dict(mean=float(np.mean(vals)), ci=cpm.boot_ci(vals), n=len(vals))
        out[cell_label(*cell)] = agg
    return out


def assert_finite_cells(group, regime, cells):
    """Mirror controllable-posterior/run.py: never write a JSON with a non-finite metric mean,
    EXCEPT gen_state_tv which is legitimately nan when a whole cell abstained (n==0 there)."""
    for cell, agg in cells.items():
        for key, v in agg.items():
            if key == "gen_state_tv" and v["n"] == 0:
                continue
            if not np.isfinite(v["mean"]):
                raise FloatingPointError(
                    f"non-finite metric cell: group={group} regime={regime} cell={cell} metric={key}")


# ----------------------------------------------------------- model loading

def load_backbone(adapter, dev):
    """Load Qwen3-4B (bf16, eval) and, for a non-'base' group, wrap with the cached LoRA adapter.
    gptq monkeypatch is applied BEFORE importing peft (copied from native-abstain/gen.py)."""
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
                f"adapter dir missing: {adapter} (expected a cached controllable-posterior LoRA)")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()                                # eval-only; backbone frozen, no training
    return model, tok


# -------------------------------------------------------------- Phase A

def sample_group(group, regime, items, tok, model, dev, args):
    """Sample + parser-Phi + readout for one (group, regime). Returns (per_item rows, samples_blob).
    samples_blob is saved so Phase B / a --from_dump re-run can recompute metrics WITHOUT re-sampling."""
    if regime == "letter":
        n, max_new, temps = args.n_letter, 4, args.temps
    else:
        n, max_new, temps = args.n_free, 48, [args.temp_free]

    per_item, sample_rows = [], []
    n_samp = n_tok = 0
    t0 = time.time()
    for it in tqdm(items, desc=f"{group}/{regime}"):
        k = it["k"]
        lids = core.letter_ids(tok, k)
        # readout (answer-position posterior) BESIDE generation, on the on-distribution letter prompt.
        # eval-only: option_logits is grad-friendly, so wrap it to keep the whole phase grad-free.
        lp_prompt = sample.letter_prompt(it, args.mode)
        with torch.no_grad():
            lg = core.option_logits(model, tok, lp_prompt, k, lids, dev)
        p = torch.softmax(lg, 0).detach().cpu().numpy()
        readout_tv = cpm.tv(p, target_np(it))

        # generation under the regime's prompt, swept over temperatures (letter) or T=1.0 (freeform)
        prompt = sample.letter_prompt(it, args.mode) if regime == "letter" else sample.freeform_prompt(it, args.mode)
        for temp in temps:
            texts = sample.sample_continuations(model, tok, prompt, dev, n, temp, max_new, args.seed)
            states = [phi.phi_parse(t, it) for t in texts]
            n_samp += len(texts)
            n_tok += sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
            m = item_metrics(states, texts, it)
            m.update(readout_tv=float(readout_tv), k=k, j=it["j"], temp=temp, id=it["id"])
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
    """Second entry mode: rebuild per_item metric rows from a saved samples blob WITHOUT re-sampling."""
    per_item = []
    for r in blob["rows"]:
        it = dict(letters=r["letters"], nouns=r["nouns"], survivors=r["survivors"], target=r["target"])
        m = item_metrics(r["states"], r["texts"], it)
        m.update(readout_tv=float(r["readout_tv"]), k=r["k"], j=r["j"], temp=r["temp"], id=r["id"])
        per_item.append(m)
    return per_item


def write_group_outputs(group, regime_cells, args):
    """diversity_<group>.json (per-cell aggregates) + one run.log line per group x regime x cell."""
    blob = dict(group=group, mode=args.mode, tag=args.tag, regimes=regime_cells)
    (OUT / f"diversity_{group}.json").write_text(json.dumps(blob, indent=2))
    for regime, cells in regime_cells.items():
        for cell, agg in cells.items():
            log_line(
                f"[{args.tag}] {group:>9} {regime:>8} {cell:>8} | "
                f"k_eff={agg['k_eff_distinct']['mean']:.2f} "
                f"k_eff_H={agg['k_eff_entropy']['mean']:.2f} "
                f"cov={agg['coverage']['mean']:.2f} "
                f"gen_tv={agg['gen_state_tv']['mean']:.3f} "
                f"readout_tv={agg['readout_tv']['mean']:.3f} "
                f"elim={agg['eliminated_mass']['mean']:.2f} "
                f"abst={agg['abstain_rate']['mean']:.2f} "
                f"distinct_text={agg['distinct_text']['mean']:.2f}")


# -------------------------------------------------------------- Phase B

def phase_b_faithfulness(args, dev):
    """Freeform only: re-classify saved freeform samples with the external Llama judge, recompute the
    metrics under LLM-Phi, and report parser<->LLM agreement. Downgrade to parser-only if unavailable."""
    free_blobs = []
    for group in args.groups:
        f = OUT / f"samples_{group}_freeform.json"
        if f.exists():
            free_blobs.append((group, json.loads(f.read_text())))
    if not free_blobs:
        log_line(f"[{args.tag}] Phase B skipped: no freeform sample dumps present")
        return

    try:
        judge, jtok = C.load(JUDGE, dev)
    except Exception as e:                       # judge unavailable -> downgrade, do NOT crash
        log_line(f"[{args.tag}] WARN: judge unavailable -> parser-Phi only ({type(e).__name__}: {e})")
        return

    report = {}
    for group, blob in free_blobs:
        # one judged sample per (item,temp); cap total judged rows at --llm_subset items
        rows = blob["rows"][: args.llm_subset]
        agree, total = 0, 0
        llm_per_item = []
        for r in tqdm(rows, desc=f"judge/{group}"):
            it = dict(letters=r["letters"], nouns=r["nouns"], survivors=r["survivors"], target=r["target"])
            llm_states = []
            for text, pstate in zip(r["texts"], r["states"]):
                lp = C.answer_logprobs(judge, jtok, phi.phi_llm_prompt(text, it), dev, phi.phi_llm_classes(it))
                cls = max(lp, key=lp.get)
                lstate = {"H": "hedge", "Z": "none"}.get(cls, cls)   # letter classes pass through
                llm_states.append(lstate)
                total += 1
                agree += int(lstate == pstate)
            m = item_metrics(llm_states, r["texts"], it)
            # carry temp so aggregate_cells can key by (k,j,temp); dumps written by Phase A carry it,
            # fall back to the freeform temperature if an older dump lacks the field.
            m.update(k=r["k"], j=r["j"], temp=r.get("temp", args.temp_free), id=r["id"])
            llm_per_item.append(m)
        cells = aggregate_cells(llm_per_item)
        agreement = (agree / total) if total else float("nan")
        report[group] = dict(parser_llm_agreement=agreement, n_judged=total, cells=cells)
        log_line(f"[{args.tag}] {group:>9} freeform LLM-Phi | parser<->LLM agreement={agreement:.3f} "
                 f"(n={total})")
    (OUT / "faithfulness.json").write_text(json.dumps(dict(judge=JUDGE, tag=args.tag, report=report), indent=2))


# ---------------------------------------------------------------- driver

def run_phase_a(args, dev):
    for group in args.groups:
        if group not in GROUPS:
            raise ValueError(f"unknown group '{group}' (choose from {list(GROUPS)})")
        adapter = GROUPS[group]

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
        items = load_items(args.ks, args.per_cell, args.mode, args.max_items)
        log_line(f"[{args.tag}] group {group}: {len(items)} items, regimes={args.regimes}")

        regime_cells = {}
        for regime in args.regimes:
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


def main():
    global CP, GROUPS                 # --cp_root rebinds the large-artifact root after argparse
    ap = argparse.ArgumentParser(description="sampling-diversity-after-Phi measurement pipeline")
    ap.add_argument("--groups", default="base,coupled,decoupled,shuffle",
                    help="comma list from base,coupled,decoupled,shuffle")
    ap.add_argument("--regimes", default="letter,freeform", help="comma list from letter,freeform")
    ap.add_argument("--ks", default="2,3,4,5", help="comma list of option-counts to include")
    ap.add_argument("--per_cell", type=int, default=20, help="items per (k,j) cell")
    ap.add_argument("--n_letter", type=int, default=24, help="samples/item, letter regime")
    ap.add_argument("--n_free", type=int, default=16, help="samples/item, freeform regime")
    ap.add_argument("--temps", default="0.7,1.0,1.3", help="letter-regime temperature sweep")
    ap.add_argument("--temp_free", type=float, default=1.0, help="freeform-regime temperature")
    ap.add_argument("--llm_subset", type=int, default=60,
                    help="cap on freeform sample-rows judged by external LLM-Phi (rows[:llm_subset] per group)")
    ap.add_argument("--mode", default="oracle", choices=["oracle", "self"], help="prompt mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from_dump", action="store_true",
                    help="recompute metrics from saved samples_*.json without re-sampling")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max_items", type=int, default=None, help="optional global item cap (smoke)")
    ap.add_argument("--cp_root", default=str(CP),
                    help="controllable-posterior artifact root (items.json + cached LoRA adapters); "
                         "large local files, only read by a fresh Phase-A model run")
    args = ap.parse_args()

    # parse comma lists
    args.groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    args.regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    args.ks = [int(x) for x in args.ks.split(",") if x.strip()]
    args.temps = [float(x) for x in args.temps.split(",") if x.strip()]

    # resolve the large-artifact root (items.json + adapters) AFTER argparse, before any run reads it
    CP = Path(args.cp_root)
    GROUPS = {
        "base": "base",
        "coupled": str(CP / "outputs/adapter_oracle_coupled_s0"),
        "decoupled": str(CP / "outputs/adapter_oracle_decoupled_s0"),
        "shuffle": str(CP / "outputs/adapter_oracle_shuffle_s0"),
    }

    # everything below this line touches torch device / loads models -- AFTER argparse on purpose,
    # so --help prints without importing transformers/peft or allocating a device.
    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    log_line(f"[{args.tag}] ==== sampling-diversity run: groups={args.groups} regimes={args.regimes} "
             f"ks={args.ks} per_cell={args.per_cell} mode={args.mode} dev={dev} "
             f"from_dump={args.from_dump} seed={args.seed} ====")
    t0 = time.time()

    run_phase_a(args, dev)

    # Phase B: external-LLM-Phi faithfulness, freeform only (no-op if freeform not run / no dumps)
    if "freeform" in args.regimes:
        phase_b_faithfulness(args, dev)
    else:
        log_line(f"[{args.tag}] Phase B skipped: freeform not in regimes")

    log_line(f"[{args.tag}] DONE in {time.time() - t0:.1f}s | MFU=n/a (generation-only, no training)")


if __name__ == "__main__":
    main()
