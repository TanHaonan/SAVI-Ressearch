"""Fluency guardrail: did decoupled_genreg's added commit term collapse the general LM?

Two cheap held-out checks on generic text the training never optimized against:
  1) KL(base ‖ adapter) next-token divergence (core.kl_to_base) — the SAME anchor used in training,
     but on the HELD-OUT generic sentences (GENERAL[4:], training uses GENERAL[:4]).
  2) generic-text perplexity under each adapter (lower/equal-ish = fluency intact).

Reports both for `decoupled` and `decoupled_genreg`; a collapse shows as a large KL or a blown-up
perplexity for genreg relative to decoupled.
"""
import importlib.util as ilu
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CP_CORE = HERE.parent / "controllable-posterior" / "core"


def _bp(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


core = _bp("core", CP_CORE / "core.py")
C = core.C

# Held-out generic text: GENERAL[4:] is NOT used by the training fluency anchor (GENERAL[:4]).
HELDOUT = core.GENERAL[4:]


def perplexity(model, tok, sents, dev):
    """Mean per-token perplexity over the held-out sentences under the (adapter-on) model."""
    tot_nll = tot_tok = 0.0
    for s in sents:
        ids = tok(s, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        logp = F.log_softmax(logits[:-1], -1)
        tgt = ids[0, 1:]
        nll = -logp[range(len(tgt)), tgt]
        tot_nll += float(nll.sum()); tot_tok += len(tgt)
    return float(torch.exp(torch.tensor(tot_nll / tot_tok)))


def main():
    dev = torch.device("cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft.import_utils as _piu
    _piu.is_gptqmodel_available = lambda *x, **k: False
    try:
        import peft.tuners.lora.gptq as _pg; _pg.is_gptqmodel_available = lambda *x, **k: False
    except Exception:
        pass
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL, local_files_only=True, cache_dir=C.CH)
    base = AutoModelForCausalLM.from_pretrained(C.DEFAULT_MODEL, dtype=torch.bfloat16,
                                                local_files_only=True, cache_dir=C.CH).to(dev)

    # base (no adapter) perplexity reference
    base.eval()
    base_ppl = perplexity(base, tok, HELDOUT, dev)
    print(f"[fluency] base (no adapter): heldout_ppl={base_ppl:.2f}")

    for g in ["decoupled", "decoupled_genreg"]:
        adir = CP_CORE / "outputs" / f"adapter_oracle_{g}_s0"
        if not adir.exists():
            print(f"[fluency] {g}: adapter missing ({adir}) -- skip")
            continue
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(C.DEFAULT_MODEL, dtype=torch.bfloat16,
                                                 local_files_only=True, cache_dir=C.CH).to(dev),
            str(adir)).eval()
        kl = float(core.kl_to_base(model, tok, HELDOUT, dev))
        ppl = perplexity(model, tok, HELDOUT, dev)
        print(f"[fluency] {g:>16}: heldout_ppl={ppl:.2f}  KL(base‖adapter)={kl:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
