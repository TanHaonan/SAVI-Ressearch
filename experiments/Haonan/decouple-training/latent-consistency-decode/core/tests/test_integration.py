"""Integration smoke test for the latent-consistency-decode pipeline.

NO model load. Exercises run.evaluate_with on a tiny shape with a FAKE emission
(emit_fn returns the local-bias unary as logits) and a FAKE yes/no scorer
(score_yes_no returns a constant), then asserts every (edge source x decoder)
cell yields a finite joint exact-match. This pins the eval core so the real run
(which passes closures over nodes.slot_logits + edges.make_score_yes_no) cannot
silently produce a NaN/inf cell or crash on any edge source.

The test imports run.py; run.py must NEVER import from tests/.
"""
import importlib.util as ilu, random, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_deps"))

spec = ilu.spec_from_file_location("run", HERE.parent / "run.py")
R = ilu.module_from_spec(spec); spec.loader.exec_module(R)
gspec = ilu.spec_from_file_location("gen", HERE.parent / "gen_data.py")
gen = ilu.module_from_spec(gspec); gspec.loader.exec_module(gen)


def _items(n=4):
    rng = random.Random(0)
    return [gen.make_item(rng, 3, 3, 2, i) for i in range(n)]


def test_evaluate_finite_on_tiny():
    items = _items(4)

    # fake emission: the local-bias unary, recentred to logits (no model).
    def emit(item, t):
        u = np.array(item["local_unary"])[t]
        return u - u.max()

    # fake yes/no scorer: a constant margin (flat edge) -> still finite everywhere.
    def score(prompt):
        return 0.0

    res = R.evaluate_with(items, emit_fn=emit, score_yes_no=score)
    for src in ("oracle", "self", "shuffle"):
        for dec in ("greedy", "marginal", "viterbi"):
            cell = res[src][dec]
            assert np.isfinite(cell["joint_em"]), f"{src}/{dec} joint_em not finite"
            assert np.isfinite(cell["acc"]), f"{src}/{dec} acc not finite"


def test_every_source_decoder_cell_present():
    items = _items(4)

    def emit(item, t):
        u = np.array(item["local_unary"])[t]
        return u - u.max()

    def score(prompt):
        # a non-trivial-but-finite scorer (depends on prompt length) to exercise build_edges
        return float(len(prompt) % 3) - 1.0

    res = R.evaluate_with(items, emit_fn=emit, score_yes_no=score)
    for src in ("oracle", "self", "shuffle"):
        assert src in res, f"missing edge source {src}"
        for dec in ("greedy", "marginal", "viterbi"):
            assert dec in res[src], f"missing decoder {dec} under {src}"
            cell = res[src][dec]
            # every cell carries joint-EM + per-slot acc + a feasibility split + an item-bootstrap CI
            assert "joint_em" in cell and "acc" in cell
            assert "joint_em_ci" in cell and len(cell["joint_em_ci"]) == 3
            assert "feas_split" in cell
            assert np.isfinite(cell["joint_em"])


def test_oracle_beats_or_matches_greedy_with_perfect_emission():
    """Sanity: with the TRUE unary as emission and ORACLE edges, the wiring must carry the
    planted signal end-to-end. The MAP decoder (viterbi) recovers gold exactly (joint-EM == 1
    by construction: gold IS the unique MAP of unary+edges). The per-position marginal recovers
    the large majority but may lag viterbi on a small fraction of items -- the sum-product MPM
    is not the joint MAP, so a per-slot-argmax/joint-MAP gap is EXPECTED (carried instruction #4:
    P3 is non-trivial on this flat-unary carrier). Greedy is bitten (reverse-bite by construction)."""
    items = _items(16)

    def emit(item, t):
        # perfect emission == the planted local-bias unary (what a calibrated readout would give)
        u = np.array(item["local_unary"])[t]
        return u - u.max()

    def score(prompt):
        return 0.0

    res = R.evaluate_with(items, emit_fn=emit, score_yes_no=score)
    # viterbi (exact MAP) recovers gold on every item; marginal recovers most; greedy is bitten.
    assert res["oracle"]["viterbi"]["joint_em"] == 1.0
    assert res["oracle"]["marginal"]["joint_em"] >= 0.9
    assert res["oracle"]["marginal"]["joint_em"] >= res["oracle"]["greedy"]["joint_em"]
    assert res["oracle"]["greedy"]["joint_em"] < 1.0
