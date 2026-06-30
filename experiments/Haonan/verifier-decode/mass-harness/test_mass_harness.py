"""Unit tests for the generalized novelty/stitch metrics (mock domain, no GPU)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mass_harness as MH  # noqa: E402


class MockDom:
    """State = running sum (int); canon = the sum, so different move-orders reaching
    the same sum MERGE (mirrors a real canonical mergeable state)."""
    def canon(self, s):
        return s

    def apply(self, s, m):
        return s + m


D = MockDom()


def test_canon_seq():
    assert MH.canon_seq(D, 0, (1, 2, 3)) == (0, 1, 3, 6)


def test_stitch_genuine_recombination():
    # node "3" reachable by 1+2 (chain A prefix) and 2+1 (chain B); decode (1,2,3)
    # = A's prefix to 3 + B's final edge 3->6. No single chain is (1,2,3).
    chains = [(1, 2, 1), (2, 1, 3)]   # canon seqs (0,1,3,4) and (0,2,3,6)
    info = MH.analyze(D, 0, (1, 2, 3), chains)
    assert info["novel"] and info["stitch"] and info["n_covered"] == info["n_edges"]


def test_not_stitch_when_edge_uncovered():
    chains = [(1, 2, 1)]              # covers (0->1),(1->3) but not (3->6)
    info = MH.analyze(D, 0, (1, 2, 3), chains)
    assert info["novel"] and not info["stitch"] and info["n_covered"] == 2


def test_not_novel_when_path_present():
    chains = [(1, 2, 3), (2, 1, 3)]
    info = MH.analyze(D, 0, (1, 2, 3), chains)
    assert not info["novel"] and not info["stitch"]
