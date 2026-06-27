# sampling-diversity/core/tests/test_aggregate.py
"""Guards for the cell-aggregation contract: (Fix 1) cells keyed by (k,j,temp) so the temperature
sweep does not collapse into one averaged cell; (Fix 2) distinct_text reaches the aggregated output.
Loads run_diversity (the core) -- tests are allowed to import core. No model is loaded."""
import importlib.util as ilu
from pathlib import Path
HERE = Path(__file__).resolve().parent
def _load(name, fn):
    s = ilu.spec_from_file_location(name, HERE.parent / fn); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m
rd = _load("run_diversity", "run_diversity.py")


def _row(temp, k=3, j=2):
    """One per-item aggregate row carrying every METRIC_KEYS field + (k,j,temp)."""
    return dict(k_eff_distinct=2.0, k_eff_entropy=2.0, coverage=1.0, gen_state_tv=0.1,
                eliminated_mass=0.0, abstain_rate=0.0, distinct_text=5.0, readout_tv=0.02,
                k=k, j=j, temp=temp, id=f"k{k}_j{j}_T{temp}")


def test_temp_splits_into_separate_cells():
    # same (k,j), two temperatures -> must produce TWO cells, not one averaged cell
    cells = rd.aggregate_cells([_row(0.7), _row(1.3)])
    assert len(cells) == 2
    assert "k3_j2_T0.7" in cells and "k3_j2_T1.3" in cells
    # each cell sees the item once -> CI n == 1 (not inflated by the other temp)
    assert cells["k3_j2_T0.7"]["readout_tv"]["n"] == 1
    assert cells["k3_j2_T1.3"]["readout_tv"]["n"] == 1


def test_distinct_text_in_aggregate():
    cells = rd.aggregate_cells([_row(1.0), _row(1.0)])
    assert "distinct_text" in rd.METRIC_KEYS
    agg = cells["k3_j2_T1"]
    assert "distinct_text" in agg
    assert abs(agg["distinct_text"]["mean"] - 5.0) < 1e-9
    assert agg["distinct_text"]["n"] == 2


def test_phase_b_row_shape_is_aggregatable():
    """Regression for the L1-caught KeyError: a Phase-B (LLM-Phi) per-item row is built by
    item_metrics(...) + {k, j, temp, id}; it must aggregate without KeyError and the cell must
    carry the right temp. Mirrors exactly what phase_b_faithfulness produces."""
    it = dict(letters=["A", "B", "C"], nouns=["apple", "river", "cloud"],
              survivors=["A", "B", "C"], target={"A": 1/3, "B": 1/3, "C": 1/3})
    llm_states = ["A", "B", "C", "hedge"]
    texts = ["the apple", "the river", "the cloud", "it is ambiguous"]
    m = rd.item_metrics(llm_states, texts, it)
    m.update(k=3, j=3, temp=1.0, id="k3_j3_0")     # same update phase_b_faithfulness does
    cells = rd.aggregate_cells([m])                # must NOT KeyError on 'temp'
    assert list(cells.keys()) == ["k3_j3_T1"]
    assert cells["k3_j3_T1"]["coverage"]["n"] == 1
