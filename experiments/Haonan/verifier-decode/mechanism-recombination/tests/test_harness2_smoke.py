"""Tiny PLAN2 harness smoke: finite metrics, all arms present, deterministic + resumable."""

import math
import os

import run_mechanism2 as rm2


def _finite_tree(obj):
    rm2._check_finite_tree(obj, "smoke")


def test_harness2_smoke_finite_and_complete(tmp_path):
    out = str(tmp_path / "smoke")
    result = rm2.run(
        domain_name="lattice",
        depths=[8],
        K_grid=[8],
        p_grid=[0.7],
        seeds=[1],
        N=8,
        inst_per_depth=4,
        outdir=out,
    )
    assert os.path.exists(os.path.join(out, "results2.json"))
    assert len(result["per_cell"]) == 1
    cell = result["per_cell"][0]
    # every arm has a finite pass@1.
    for arm in rm2.ARMS:
        v = cell["pass_at_1"][arm]
        assert isinstance(v, float) and math.isfinite(v)
    # ceiling == 1.0 (solvable by construction).
    assert cell["pass_at_1"]["ceiling"] == 1.0
    # headline deltas finite where computable.
    for key in ("D1_exec", "D_verif", "D_merge"):
        g = cell[key]
        if g.get("status") == "ok":
            for k in ("delta", "lo", "hi"):
                assert math.isfinite(g[k])
    # exec parity holds: iso exec >= savi exec.
    assert cell["iso_exec_geq_savi"] is True
    # whole tree finite.
    _finite_tree(result)


def test_harness2_axis_artifact_cell(tmp_path):
    """The single (depth, K) artifact cell is populated with both axes' pass@1."""
    out = str(tmp_path / "art")
    result = rm2.run(
        domain_name="lattice",
        depths=[8],          # single depth -> it is the mid -> artifact cell
        K_grid=[8],
        p_grid=[0.7],
        seeds=[1, 2],
        N=8,
        inst_per_depth=4,
        outdir=out,
    )
    art = result["axis_artifact_cell"]
    assert art is not None
    assert math.isfinite(art["iso_candidates_pass"])
    assert math.isfinite(art["iso_exec_pass"])
    # Candidate parity is at least as strong as exec parity (more attempts).
    assert art["iso_candidates_pass"] >= art["iso_exec_pass"] - 1e-9


def test_harness2_deterministic(tmp_path):
    a = rm2.run("lattice", [8], [8], [0.7], [1], 8, 4, str(tmp_path / "a"))
    b = rm2.run("lattice", [8], [8], [0.7], [1], 8, 4, str(tmp_path / "b"))
    ca, cb = a["per_cell"][0], b["per_cell"][0]
    assert ca["pass_at_1"] == cb["pass_at_1"]
    assert ca["D1_exec"] == cb["D1_exec"]


def test_harness2_resumable_per_depth(tmp_path):
    """--depth run separately into a shared outdir resumes and unions depths."""
    out = str(tmp_path / "resume")
    # Run depth 4 only.
    r4 = rm2.run("lattice", [4, 8], [8], [0.7], [1], 8, 4, out, only_depth=4)
    depths_present = {c["depth"] for c in r4["per_cell"]}
    assert depths_present == {4}
    # Now run depth 8 into the SAME outdir; both depths should appear.
    r8 = rm2.run("lattice", [4, 8], [8], [0.7], [1], 8, 4, out, only_depth=8)
    depths_present2 = {c["depth"] for c in r8["per_cell"]}
    assert depths_present2 == {4, 8}
    # Cache exists.
    assert os.path.exists(os.path.join(out, "cache.json"))
