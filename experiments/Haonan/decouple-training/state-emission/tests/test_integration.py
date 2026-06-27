# tests/test_integration.py
"""MODEL-FREE end-to-end integration test for the state-emission measurement pipeline.

No model is loaded. We drive the pipeline's pure functions on a tiny, hand-checkable shape:

  scripted structured-commit texts  ->  phi_b1 (Subtask 3)  ->  item_metrics + aggregate_cells (run_state_emission)

and assert the cell numbers are exactly the ones a human can compute by hand for a constructed
2-survivor item:
  * a BALANCED committed set {surv0, surv1, surv0, surv1} -> coverage == 1.0, calibration_tv == 0.0;
  * a COLLAPSED set (all surv0)                            -> coverage == 0.5, calibration_tv == 0.5.

We also assert that the second entry mode (`--from_dump`) reproduces the fresh metric numbers
EXACTLY: write a tiny samples blob in the on-disk schema, run recompute_from_blob over it, and
compare the resulting per-item rows and the aggregated cells byte-for-byte against the fresh path.

The StubGenerator (Subtask 1) is exercised to confirm the sample() seam returns scripted texts that
then flow through the real phi_b1 and the real metrics — i.e. the wiring, not a mock of the metrics.
This file imports ONLY core modules + run_state_emission's pure functions; it never imports a model
and run_state_emission never imports from tests/.
"""
import importlib.util as ilu
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def bp(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


G = bp("generator", ROOT / "core/generator.py")
PHI = bp("phi", ROOT / "core/phi.py")
M = bp("state_metrics", ROOT / "core/state_metrics.py")
# The pipeline under test. Importing it must NOT load transformers/torch-models (argparse-first design);
# importing the module top-level is allowed (it pulls torch, which is a hard dep of generator already).
RUN = bp("run_state_emission", ROOT / "run_state_emission.py")


# --------------------------------------------------------------------- fixtures
def _two_survivor_item():
    """A constructed k=3, j=2 item whose true posterior is uniform over survivors {A, B}.
    Letters/nouns/survivors/target hand-fixed so the test arithmetic is checkable by eye:
      survivors = [A, B] (mass 0.5 each); C is eliminated (mass 0)."""
    return dict(
        id="k3_j2_0", k=3, j=2,
        letters=["A", "B", "C"],
        nouns=["apple", "train", "cloud"],
        survivors=["A", "B"],
        target={"A": 0.5, "B": 0.5, "C": 0.0},
        prompt_b1="setup ... COMMIT: <thing>",
        prompt_b2="setup ... say which and why",
        prompt_stated="setup, equally likely behind the apple, the train. Answer with one letter.",
        prompt_clue="setup, it is not the cloud. Answer with one letter.",
        split="test",
    )


def _commit(noun):
    """A structured-commit continuation that commits to `noun` via the deterministic COMMIT slot."""
    return f"Some reasoning that may mention things.\nCOMMIT: the {noun}"


def _balanced_texts(it):
    """Scripted structured-commit texts whose phi_b1 states are exactly {A, B, A, B} (balanced over survivors)."""
    n0 = it["nouns"][it["letters"].index("A")]   # apple
    n1 = it["nouns"][it["letters"].index("B")]   # train
    return [_commit(n0), _commit(n1), _commit(n0), _commit(n1)]


def _collapsed_texts(it):
    """Scripted structured-commit texts whose phi_b1 states are all A (collapsed onto one survivor)."""
    n0 = it["nouns"][it["letters"].index("A")]
    return [_commit(n0)] * 4


# ----------------------------------------------------------- phi wiring sanity
def test_phi_b1_maps_scripted_commits_to_expected_states():
    it = _two_survivor_item()
    bal = [PHI.phi_b1(t, it) for t in _balanced_texts(it)]
    col = [PHI.phi_b1(t, it) for t in _collapsed_texts(it)]
    assert bal == ["A", "B", "A", "B"]
    assert col == ["A", "A", "A", "A"]


def test_stub_generator_feeds_phi():
    """The sample() seam returns the scripted texts unchanged, which then map through real phi_b1."""
    it = _two_survivor_item()
    scripted = _balanced_texts(it)
    gen = G.StubGenerator(render_state=lambda s: s["prompt_b1"], scripted=scripted)
    texts = gen.sample(it, N=4, temperature=1.0, seed=0)
    states = [PHI.phi_b1(t, it) for t in texts]
    assert texts == scripted
    assert states == ["A", "B", "A", "B"]


# ------------------------------------------------------------ item_metrics math
def test_item_metrics_balanced_hand_checked():
    it = _two_survivor_item()
    states = ["A", "B", "A", "B"]
    m = RUN.item_metrics(states, _balanced_texts(it), it, readout_tv=0.0, regime="b1")
    assert m["coverage"] == pytest.approx(1.0)
    assert m["calibration_tv"] == pytest.approx(0.0)
    assert m["k_eff_distinct"] == pytest.approx(2.0)
    assert m["eliminated_mass"] == pytest.approx(0.0)
    assert m["abstain_rate"] == pytest.approx(0.0)
    # gap = calibration_tv - readout_tv
    assert m["gap"] == pytest.approx(0.0)


def test_item_metrics_collapsed_hand_checked():
    it = _two_survivor_item()
    states = ["A", "A", "A", "A"]
    m = RUN.item_metrics(states, _collapsed_texts(it), it, readout_tv=0.0, regime="b1")
    # only survivor A appears -> coverage 1 of 2 survivors = 0.5
    assert m["coverage"] == pytest.approx(0.5)
    # empirical committed dist = {A:1, B:0}; target {A:0.5, B:0.5} -> TV = 0.5
    assert m["calibration_tv"] == pytest.approx(0.5)
    assert m["k_eff_distinct"] == pytest.approx(1.0)


# ----------------------------------------------------------- aggregation cells
def _rows_for(it, states_per_item, temp=1.0, readout_tv=0.0):
    """Build per_item rows (the aggregate_cells input) for a list of state-lists, one per item."""
    rows = []
    for states in states_per_item:
        m = RUN.item_metrics(states, ["x"] * len(states), it, readout_tv=readout_tv, regime="b1")
        m.update(k=it["k"], j=it["j"], temp=temp, id=it["id"])
        rows.append(m)
    return rows


def test_aggregate_cells_balanced_cell():
    it = _two_survivor_item()
    rows = _rows_for(it, [["A", "B", "A", "B"], ["A", "B", "A", "B"]])
    cells = RUN.aggregate_cells(rows)
    key = RUN.cell_label(it["k"], it["j"], 1.0)
    assert key in cells
    agg = cells[key]
    assert agg["coverage"]["mean"] == pytest.approx(1.0)
    assert agg["calibration_tv"]["mean"] == pytest.approx(0.0)
    assert agg["calibration_tv"]["n"] == 2
    # CI is a 3-tuple of finite numbers
    assert len(agg["calibration_tv"]["ci"]) == 3
    assert all(v == v for v in agg["calibration_tv"]["ci"])  # not nan


def test_aggregate_cells_collapsed_cell():
    it = _two_survivor_item()
    rows = _rows_for(it, [["A", "A", "A", "A"], ["A", "A", "A", "A"]])
    cells = RUN.aggregate_cells(rows)
    agg = cells[RUN.cell_label(it["k"], it["j"], 1.0)]
    assert agg["coverage"]["mean"] == pytest.approx(0.5)
    assert agg["calibration_tv"]["mean"] == pytest.approx(0.5)


# --------------------------------------------- nan-aware reduction (full abstain)
def test_aggregate_cells_nan_aware_calibration_when_cell_abstains():
    """If every item in a cell fully abstains, calibration_tv (and the derived gap) is nan per item,
    so the cell mean is nan with n==0 (legitimate, not a crash). coverage stays FINITE here -- with
    survivors present but none committed, coverage is 0.0 (0 of 2 survivors seen), not nan, so it
    keeps the full n. abstain_rate is 1.0. assert_finite_cells must tolerate the calibration/gap
    nan-with-n==0 while leaving coverage's finite 0.0 untouched."""
    it = _two_survivor_item()
    rows = _rows_for(it, [["hedge", "hedge", "none", "none"], ["none", "none", "hedge", "hedge"]])
    cells = RUN.aggregate_cells(rows)
    agg = cells[RUN.cell_label(it["k"], it["j"], 1.0)]
    assert agg["calibration_tv"]["n"] == 0
    assert agg["calibration_tv"]["mean"] != agg["calibration_tv"]["mean"]  # nan
    assert agg["gap"]["n"] == 0
    assert agg["gap"]["mean"] != agg["gap"]["mean"]                        # nan (derived from cal_tv)
    # coverage is finite (0 survivors seen out of 2) -> mean 0.0, full n, not dropped
    assert agg["coverage"]["n"] == 2
    assert agg["coverage"]["mean"] == pytest.approx(0.0)
    assert agg["abstain_rate"]["n"] == 2
    assert agg["abstain_rate"]["mean"] == pytest.approx(1.0)
    # must not raise: calibration_tv & gap nan-with-n==0 are allowed non-finite means
    RUN.assert_finite_cells("base", "b1", cells)


def test_assert_finite_cells_rejects_unexpected_nan():
    """A non-finite mean on a metric OTHER than calibration_tv/coverage (or with n>0) must raise."""
    bad = {"k3_j2_T1": {"abstain_rate": {"mean": float("nan"), "ci": [0, 0, 0], "n": 2}}}
    with pytest.raises(FloatingPointError):
        RUN.assert_finite_cells("base", "b1", bad)


# --------------------------------------------------- --from_dump exact parity
def test_from_dump_reproduces_fresh_numbers_exactly(tmp_path):
    """Write a tiny samples blob in the on-disk schema, recompute metrics from it, and assert the
    per-item rows AND the aggregated cells are byte-identical to the fresh path. This is the
    `--from_dump` contract: numbers reproduce without re-sampling or loading a model."""
    it = _two_survivor_item()
    balanced = _balanced_texts(it)
    collapsed = _collapsed_texts(it)
    states_bal = [PHI.phi_b1(t, it) for t in balanced]
    states_col = [PHI.phi_b1(t, it) for t in collapsed]

    # ---- fresh path: per_item rows computed directly ----
    fresh_rows = []
    for texts, states in [(balanced, states_bal), (collapsed, states_col)]:
        m = RUN.item_metrics(states, texts, it, readout_tv=0.25, regime="b1")
        m.update(k=it["k"], j=it["j"], temp=1.0, id=it["id"])
        fresh_rows.append(m)
    fresh_cells = RUN.aggregate_cells(fresh_rows)

    # ---- dump path: same data serialized to the blob schema, recomputed ----
    blob = dict(
        group="stub", regime="b1", mode="oracle", n_items=1,
        samples_per_s=0.0, gen_tokens_per_s=0.0,
        rows=[
            dict(id=it["id"], k=it["k"], j=it["j"], temp=1.0, regime="b1",
                 letters=it["letters"], nouns=it["nouns"], survivors=it["survivors"],
                 target=it["target"], readout_tv=0.25, texts=balanced, states=states_bal),
            dict(id=it["id"], k=it["k"], j=it["j"], temp=1.0, regime="b1",
                 letters=it["letters"], nouns=it["nouns"], survivors=it["survivors"],
                 target=it["target"], readout_tv=0.25, texts=collapsed, states=states_col),
        ],
    )
    blob_path = tmp_path / "samples_stub_b1.json"
    blob_path.write_text(json.dumps(blob))
    reread = json.loads(blob_path.read_text())
    dump_rows = RUN.recompute_from_blob(reread)
    dump_cells = RUN.aggregate_cells(dump_rows)

    # per-item rows identical (same keys, same values)
    assert [sorted(r) for r in dump_rows] == [sorted(r) for r in fresh_rows]
    for fr, dr in zip(fresh_rows, dump_rows):
        for key in fr:
            assert dr[key] == fr[key], f"row mismatch on {key}: {dr[key]} != {fr[key]}"
    # aggregated cells identical byte-for-byte (JSON round-trip to normalize tuple/list)
    assert json.dumps(dump_cells, sort_keys=True) == json.dumps(fresh_cells, sort_keys=True)
