"""Tests for the Tier-A sweep harness (run_corr.py).

The harness is the arm-matrix + sweep driver (PREREG §5/§6/§7/§8/§9). These tests run
the TINY one-cell smoke (``run_corr.run_smoke``, a single competence p over a reduced
β/seed/budget grid, finishes in a few seconds on CPU) and assert:

1. SCHEMA — the written ``tierA_smoke.json`` has the documented shape: a config, one
   cell per p, each cell carrying the four-arm read (iid_curve + per-β corr_curves +
   oracle_ceiling + temp-matched H4), the four validity gates V1–V4, and the H1/H2/H3/H4
   instruments; the doc carries the cross-p H5 + Holm + summary.

2. V2 β=0 EQUIVALENCE PASSES — the well-posed gate (corr(β=0) ≡ iid_bok byte-for-byte)
   must hold (it is by construction; MECHANISM §2). This is the load-bearing correctness
   check for the whole experiment.

3. ISO-TOKEN LEDGER — the iso-token budget read is sound: the iid arm's realized tokens
   at B* are within the budget, and corr's step-mode N× tax shows up (corr realizes fewer
   chains than iid at the same B; V3.tax_present).

4. DETERMINISM — two smoke runs produce byte-identical results (modulo timestamps).

5. RESUMABILITY — extending the p-grid reuses already-computed cells (the resume key
   excludes the p-grid iteration axis), and changing a cell-determining knob recomputes.

The smoke deliberately uses a tiny grid, so its H1/H3/H4 NUMBERS are not prereg-grade
(they may be negative / non-significant); these tests check the PIPELINE and SCHEMA, not
a positive scientific outcome.
"""

import json

import pytest

import core_boot as cb  # noqa: F401  (ensures the substrate is on sys.path)
import run_corr as rc


# A module-scoped smoke run reused across the schema/V2/iso-token/determinism tests, so
# the (few-second) sweep is paid once. Written to a test-local path, not outputs/.
@pytest.fixture(scope="module")
def smoke_doc(tmp_path_factory):
    out = tmp_path_factory.mktemp("cc") / "tierA_smoke.json"
    doc = rc.run_smoke(out_path=out, verbose=False)
    return doc, out


# ===========================================================================
# 1. SCHEMA
# ===========================================================================

def test_smoke_schema_top_level(smoke_doc):
    doc, out = smoke_doc
    assert out.is_file()
    assert doc["schema"] == "correlated-coverage/tierA/v2"
    assert doc["tier"] == "A-mock"
    assert doc["frozen_decode_core"] is True
    assert doc["partial"] is False
    for key in ("config", "cells", "H5", "holm", "summary"):
        assert key in doc, key
    # The written file equals the returned doc (atomic write of the final state).
    on_disk = json.loads(out.read_text())
    assert on_disk["cells"][0]["p"] == doc["cells"][0]["p"]
    assert on_disk["cells"][0]["collapse"] == doc["cells"][0]["collapse"]


def test_smoke_cell_has_all_arms_and_gates_and_instruments(smoke_doc):
    doc, _ = smoke_doc
    assert len(doc["cells"]) == 1  # one (p, collapse) cell in the smoke
    cell = doc["cells"][0]
    # The cell is keyed by (p, collapse) — collapse is now a sweep axis (RETUNE).
    assert "collapse" in cell
    # The arms are represented: iid baseline curve, per-β corr_step curves, per-β corr_cheap
    # curves, oracle ceiling, and the temp-matched arm inside H4.
    assert "iid_curve" in cell and "coverage" in cell["iid_curve"]
    assert "corr_curves" in cell and len(cell["corr_curves"]) >= 2
    assert "cheap_curves" in cell and len(cell["cheap_curves"]) >= 2
    assert "oracle_ceiling" in cell
    assert "tau_matched" in cell["H4"]
    # The instance filter is recorded (oracle-solvable + iid-in-band + corr-K>=4).
    assert "n_instances_kept" in cell and "instance_filter" in cell
    assert "keep_mask" in cell["instance_filter"]
    # The four validity gates.
    for g in ("V1", "V2", "V3", "V4"):
        assert g in cell["gates"]
        assert "pass" in cell["gates"][g]
    # V2 now checks BOTH repulsion arms' β=0 equivalence.
    assert "mismatches_step" in cell["gates"]["V2"]
    assert "mismatches_cheap" in cell["gates"]["V2"]
    # The instruments: H1 (corr_step), H1_cheap (corr_cheap), H2/H3/H4.
    assert "delta" in cell["H1"] and "pass" in cell["H1"]
    assert "delta" in cell["H1_cheap"] and "pass" in cell["H1_cheap"]
    assert "mean_prompt_overhead_tokens" in cell["H1_cheap"]
    assert "corr_slope" in cell["H2"] and "iid_slope" in cell["H2"]
    assert "coverageK_over_beta" in cell["H3"] and "beta_star_chain_matched" in cell["H3"]
    assert "cheap_coverageK_over_beta" in cell["H3"]
    assert "jd_corr_leaf" in cell["H4"] and "jd_temp_leaf" in cell["H4"]
    # H4 fix: the τ-match target entropy is measured at K>=4 (not the degenerate K=1).
    assert cell["H4"]["tau_match_target_K"] >= 4
    assert cell["H4"]["K_star"] >= 4


def test_smoke_curves_aligned_to_budget_grid(smoke_doc):
    doc, _ = smoke_doc
    cell = doc["cells"][0]
    budgets = cell["iid_curve"]["budgets"]
    assert len(cell["iid_curve"]["coverage"]) == len(budgets)
    assert len(cell["iid_curve"]["mean_K"]) == len(budgets)
    for b, curve in cell["corr_curves"].items():
        assert curve["budgets"] == budgets
        assert len(curve["coverage"]) == len(budgets)
    for b, curve in cell["cheap_curves"].items():
        assert curve["budgets"] == budgets
        assert len(curve["coverage"]) == len(budgets)
    # B_idx points at B_star within the budget grid.
    assert budgets[cell["B_idx"]] == cell["B_star"]


def test_smoke_h5_holm_present(smoke_doc):
    doc, _ = smoke_doc
    # H5 is a cross-cell regression; with one cell its slope is nan (honest "cannot fit"),
    # but the structure must be present.
    assert "slope" in doc["H5"] and "points" in doc["H5"]
    # Holm now covers the H1 / H1_cheap / H2 / H4 family (the cheap arm is adjudicated too).
    for h in ("H1", "H1_cheap", "H2", "H4"):
        assert h in doc["holm"]
        assert "p_adj" in doc["holm"][h] and "reject" in doc["holm"][h]


# ===========================================================================
# 2. V2 β=0 EQUIVALENCE (the load-bearing correctness gate) — BOTH repulsion arms
# ===========================================================================

def test_smoke_v2_beta0_equivalence_passes_for_both_arms(smoke_doc):
    """corr_step(β=0) AND corr_cheap(β=0) ≡ iid_bok byte-for-byte (MECHANISM §2).

    Must pass for BOTH arms — it is by construction. If either arm's β=0 broke, this gate
    (and the whole correction) would fail.
    """
    doc, _ = smoke_doc
    v2 = doc["cells"][0]["gates"]["V2"]
    assert v2["pass"] is True, v2
    assert v2["mismatches"] == 0
    assert v2["mismatches_step"] == 0
    assert v2["mismatches_cheap"] == 0
    assert v2["checked_cells"] > 0


# ===========================================================================
# 3. ISO-TOKEN LEDGER  (V3 — the binding fairness axis)
# ===========================================================================

def test_smoke_v3_iso_token_ledger_sound(smoke_doc):
    """iid tokens ≤ B*; corr's step-mode N× tax present (corr realizes fewer chains)."""
    doc, _ = smoke_doc
    v3 = doc["cells"][0]["gates"]["V3"]
    B = v3["budget_B"]
    # iid arm respects the budget ceiling.
    assert v3["iid_tokens_at_B"] <= B + 1e-6
    # corr pays more per chain -> fewer chains realized at the same B (the N× tax).
    assert v3["tax_present"] is True
    assert v3["corr_mean_K"] <= v3["iid_mean_K"]
    assert v3["pass"] is True, v3


def test_smoke_cheap_overhead_charged_into_budget(smoke_doc):
    """The cheap arm's prompt-overhead is charged into Budget (the headline cheap H1 read).

    The cheap arm's per-β curves carry ``mean_prompt_overhead_tokens``; at least one β>0
    must charge a strictly-positive conditioning overhead (β=0 charges none), proving the
    conditioning context is NOT free in the sweep's ledger.
    """
    doc, _ = smoke_doc
    cell = doc["cells"][0]
    overheads = {float(b): c["mean_prompt_overhead_tokens"]
                 for b, c in cell["cheap_curves"].items()}
    # β=0 charges zero overhead; some β>0 charges a positive overhead.
    assert overheads.get(0.0, 0.0) == 0.0
    assert any(v > 0.0 for b, v in overheads.items() if b > 0.0), overheads
    # The cheap H1 read records the overhead it actually paid at its β*.
    assert "mean_prompt_overhead_tokens" in cell["H1_cheap"]


def test_smoke_v4_oracle_ceiling_above_iid(smoke_doc):
    """V4: the solvability ceiling exceeds the realized i.i.d. coverage (head-cover-room)."""
    doc, _ = smoke_doc
    v4 = doc["cells"][0]["gates"]["V4"]
    assert v4["oracle_ceiling"] >= v4["iid_cov_at_B"]
    assert v4["shared_eval_path"] is True


def test_smoke_v1_band_recorded(smoke_doc):
    """V1: head-cover-room — iid coverage@B* is recorded and the band is the prereg band."""
    doc, _ = smoke_doc
    v1 = doc["cells"][0]["gates"]["V1"]
    assert v1["band"] == [0.2, 0.9]
    assert 0.0 <= v1["iid_cov_at_B"] <= 1.0


# ===========================================================================
# 4. DETERMINISM
# ===========================================================================

def test_smoke_deterministic(tmp_path):
    """Two smoke runs are byte-identical modulo the wall-clock timestamps."""
    d1 = rc.run_smoke(out_path=tmp_path / "a.json", verbose=False)
    d2 = rc.run_smoke(out_path=tmp_path / "b.json", verbose=False)

    def _strip(d):
        d = json.loads(json.dumps(d))  # deep copy
        d.pop("generated_unix", None)
        d.pop("elapsed_sec", None)
        for c in d["cells"]:
            c.pop("elapsed_sec", None)
        return d

    assert json.dumps(_strip(d1), sort_keys=True) == json.dumps(_strip(d2), sort_keys=True)


# ===========================================================================
# 5. RESUMABILITY
# ===========================================================================

def test_resume_reuses_cells_and_extends_grid(tmp_path):
    """Extending the p-grid reuses already-computed (p, collapse) cells (resume key excludes
    the iteration axes)."""
    out = tmp_path / "resume.json"
    insts = rc.make_instances()
    common = dict(collapse_grid=[0.8], beta_grid=[0.0, 1.0], seeds=[1, 2],
                  budgets=[96, 192], instances=insts, boot_n=300, n_corr=3, verbose=False)
    rc.run_sweep(p_grid=[0.5], out_path=out, resume=False, **common)
    first = json.loads(out.read_text())
    assert [c["p"] for c in first["cells"]] == [0.5]

    rc.run_sweep(p_grid=[0.5, 0.6], out_path=out, resume=True, **common)
    second = json.loads(out.read_text())
    assert sorted(c["p"] for c in second["cells"]) == [0.5, 0.6]
    # The cached (p=0.5, collapse=0.8) cell's numbers are unchanged by the resume.
    c05_first = next(c for c in first["cells"] if c["p"] == 0.5)
    c05_second = next(c for c in second["cells"] if c["p"] == 0.5)
    assert c05_first["B_star"] == c05_second["B_star"]
    assert c05_first["iid_cov_at_B"] == c05_second["iid_cov_at_B"]
    assert c05_first["n_instances_kept"] == c05_second["n_instances_kept"]


def test_resume_extends_collapse_grid(tmp_path):
    """Extending the COLLAPSE grid reuses prior cells (collapse is an iteration axis too)."""
    out = tmp_path / "resume_collapse.json"
    insts = rc.make_instances()
    common = dict(p_grid=[0.5], beta_grid=[0.0, 1.0], seeds=[1, 2], budgets=[96, 192],
                  instances=insts, boot_n=300, n_corr=3, verbose=False)
    rc.run_sweep(collapse_grid=[0.8], out_path=out, resume=False, **common)
    first = json.loads(out.read_text())
    assert sorted(c["collapse"] for c in first["cells"]) == [0.8]
    rc.run_sweep(collapse_grid=[0.8, 0.95], out_path=out, resume=True, **common)
    second = json.loads(out.read_text())
    assert sorted(c["collapse"] for c in second["cells"]) == [0.8, 0.95]
    c08_first = next(c for c in first["cells"] if c["collapse"] == 0.8)
    c08_second = next(c for c in second["cells"] if c["collapse"] == 0.8)
    assert c08_first["B_star"] == c08_second["B_star"]


def test_resume_recomputes_on_knob_change(tmp_path):
    """Changing a cell-determining knob (β grid) invalidates the cache (fresh recompute)."""
    out = tmp_path / "resume2.json"
    insts = rc.make_instances()
    rc.run_sweep(p_grid=[0.5], collapse_grid=[0.8], beta_grid=[0.0, 1.0], seeds=[1, 2],
                 budgets=[96, 192], instances=insts, boot_n=300, n_corr=3, out_path=out,
                 resume=False, verbose=False)
    # Different β grid -> resume key mismatch -> cell recomputed (different #betas).
    rc.run_sweep(p_grid=[0.5], collapse_grid=[0.8], beta_grid=[0.0, 1.0, 4.0], seeds=[1, 2],
                 budgets=[96, 192], instances=insts, boot_n=300, n_corr=3, out_path=out,
                 resume=True, verbose=False)
    doc = json.loads(out.read_text())
    cell = next(c for c in doc["cells"] if c["p"] == 0.5)
    assert len(cell["corr_curves"]) == 3  # recomputed with the new 3-β grid
    assert len(cell["cheap_curves"]) == 3


# ===========================================================================
# Direct unit checks on the fast iso-token curve (prefix equivalence)
# ===========================================================================

def test_arm_curve_matches_slow_instrument():
    """The fast single-run-at-K_max curve equals the slow per-K instrument exactly."""
    import mock_backend as mb
    import corr_sampler as cs
    import instruments as ins

    DOM = rc.DOMAIN
    sample = mb.make_competence_backend(p=0.6, collapse=0.5, depth_cap=8, domain=DOM)
    insts = rc.make_instances()[:3]
    seeds = [1, 2]
    budgets = [12, 24, 48]
    run = lambda inst, K, seed: cs.corr_step(DOM, sample, inst, K=K, N=4, beta=1.0,
                                             tau=1.0, seed=seed)
    fast = rc.arm_curve(run, insts, seeds, budgets)
    slow = ins.coverage_vs_tokens(run, insts, seeds, budgets, K_max=64)
    assert fast["coverage"] == slow["coverage"]
    assert fast["mean_K"] == slow["mean_K"]
    for B in budgets:
        assert fast["per_cell"][B] == slow["per_cell"][B]


def test_arm_curve_matches_slow_instrument_cheap_arm():
    """The fast iso-token curve also matches the slow instrument for the CHEAP arm.

    The cheap arm's per-chain cost (1× generation + a growing prompt-overhead) is recorded
    in ``cum_tokens`` just like the step arm, so the prefix-scan curve must equal the slow
    per-K instrument exactly for it too.
    """
    import mock_backend as mb
    import corr_sampler as cs
    import instruments as ins

    DOM = rc.DOMAIN
    sample = mb.make_competence_backend(p=0.6, collapse=0.6, depth_cap=8, domain=DOM)
    insts = rc.make_instances()[:3]
    seeds = [1, 2]
    budgets = [24, 48, 96]
    run = lambda inst, K, seed: cs.corr_cheap(DOM, sample, inst, K=K, N=4, beta=1.0,
                                              tau=1.0, seed=seed,
                                              prompt_overhead_per_state=1)
    fast = rc.arm_curve(run, insts, seeds, budgets)
    slow = ins.coverage_vs_tokens(run, insts, seeds, budgets, K_max=64)
    assert fast["coverage"] == slow["coverage"]
    assert fast["mean_K"] == slow["mean_K"]
    for B in budgets:
        assert fast["per_cell"][B] == slow["per_cell"][B]
