# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Tests for FragmentationAnalyzer.detect_splits (split/merge/birth/death)."""
from __future__ import annotations

import itertools
import os
import warnings

import numpy as np
import pandas as pd
import pytest


def _nodes(ids, xyz, eul):
    return pd.DataFrame(
        {
            "grain_id": ids,
            "X": xyz[:, 0],
            "Y": xyz[:, 1],
            "Z": xyz[:, 2],
            "GrainRadius": np.ones(len(ids)),
            "Eul0": eul[:, 0],
            "Eul1": eul[:, 1],
            "Eul2": eul[:, 2],
        }
    )


class TestDetectGrainSplits:
    def test_one_to_two_split(self):
        pytest.importorskip("neml2")
        pytest.importorskip("scipy")
        from graintrace.fragmentation import FragmentationAnalyzer

        # Step A: two well-separated grains. Step B: grain 1 splits into 1a/1b at
        # nearby centroids with a small orientation divergence; grain 2 persists.
        a = _nodes(
            [1, 2],
            np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
            np.array([[10.0, 20.0, 30.0], [80.0, 40.0, 10.0]]),
        )
        b = _nodes(
            [11, 12, 13],
            np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
            np.array([[11.0, 20.0, 30.0], [9.0, 21.0, 30.0], [80.0, 40.0, 10.0]]),
        )
        res = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees"
        )
        assert res["n_splits"] == 1
        assert res["n_matches"] == 1
        split = res["split_correspondences"]
        assert len(split) == 1
        assert int(split.iloc[0]["parent_a"]) == 1
        assert int(split.iloc[0]["n_children"]) == 2
        assert set(map(int, split.iloc[0]["children_b"].split(";"))) == {11, 12}

    def test_merge_detected(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a = _nodes(
            [1, 2],
            np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[11.0, 20.0, 30.0], [9.0, 21.0, 30.0]]),
        )
        b = _nodes(
            [11],
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[10.0, 20.0, 30.0]]),
        )
        res = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees"
        )
        assert res["n_merges"] == 1

    def test_birth_and_death(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        # A grain far away (death) and a new isolated B grain (birth).
        a = _nodes(
            [1],
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[10.0, 20.0, 30.0]]),
        )
        b = _nodes(
            [11],
            np.array([[500.0, 500.0, 500.0]]),
            np.array([[70.0, 10.0, 5.0]]),
        )
        res = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees"
        )
        assert res["n_deaths"] == 1
        assert res["n_births"] == 1

    def test_adjacency_rejects_spurious_child(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        # Grain 1 -> children 11, 12 (adjacent). Child 13 is orientation/centroid
        # compatible but not adjacent to 11/12; adjacency consistency drops it.
        a = _nodes(
            [1],
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[10.0, 20.0, 30.0]]),
        )
        b = _nodes(
            [11, 12, 13],
            np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
            np.array([[11.0, 20.0, 30.0], [9.0, 21.0, 30.0], [10.0, 19.0, 31.0]]),
        )
        res = FragmentationAnalyzer().detect_splits(
            a,
            b,
            d_tol=10.0,
            theta_tol_deg=5.0,
            angle_type="degrees",
            adjacency_b=[(11, 12)],
        )
        assert res["n_splits"] == 1
        children = set(
            map(int, res["split_correspondences"].iloc[0]["children_b"].split(";"))
        )
        assert children == {11, 12}

    def test_adjacency_keeps_best_candidate_when_no_pair_is_adjacent(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        # Grain 1 has two compatible candidates (11, 12) that are NOT adjacent to
        # each other. There is no split to reject, so the parent must keep its best
        # candidate and stay matched -- not become a death with two spurious births.
        a = _nodes(
            [1],
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[10.0, 20.0, 30.0]]),
        )
        b = _nodes(
            [11, 12, 13],
            np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [300.0, 300.0, 300.0]]),
            np.array([[10.0, 20.0, 30.0], [9.0, 21.0, 30.0], [70.0, 10.0, 5.0]]),
        )
        res = FragmentationAnalyzer().detect_splits(
            a,
            b,
            d_tol=10.0,
            theta_tol_deg=5.0,
            angle_type="degrees",
            adjacency_b=[(11, 13), (12, 13)],
        )
        assert res["n_deaths"] == 0
        assert res["n_splits"] == 0
        assert res["n_matches"] == 1
        # 11 is the exact orientation match, so it is the kept candidate.
        matched = [c for c in res["components"] if c["kind"] == "match"][0]
        assert matched["a_ids"] == [1]
        assert matched["b_ids"] == [11]
        # The rejected sibling and the unrelated grain are births, not children.
        assert res["n_births"] == 2


class TestSyntheticFFEndToEnd:
    def test_generate_and_detect_mixed_multiplicity(self, tmp_path):
        pytest.importorskip("neml2")

        from graintrace.fragmentation import FragmentationAnalyzer
        from synthetic_fragmentation import generate_ff_split

        # default multiplicities [1,2,3,1,2]: two whole, two binary, one ternary.
        gt = generate_ff_split(
            str(tmp_path / "ff"),
            n_steps=5,
            theta_max_deg=12.0,
            misorientation_tol_deg=5.0,
            seed=3,
        )
        assert (tmp_path / "ff" / "ground_truth.json").exists()
        assert gt["split_step"] is not None
        assert gt["n_splitting_grains"] == 3

        # compare the pre-split step with the fully-split step.
        s = gt["split_step"]
        res = FragmentationAnalyzer().detect_splits(
            gt["step_csvs"][s - 1],
            gt["step_csvs"][s],
            d_tol=25.0,
            theta_tol_deg=15.0,  # generous cross-step link (children diverge by theta/2)
            angle_type="degrees",
        )
        # all three splitting grains recovered, with the ternary among them.
        assert res["n_splits"] == 3
        n_children = sorted(res["split_correspondences"]["n_children"].tolist())
        assert n_children == [2, 2, 3]
        # the ternary parent must map to exactly 3 children.
        assert int(res["split_correspondences"]["n_children"].max()) == 3


class TestSyntheticGenerators:
    def test_ebsd_and_nf_write_files(self, tmp_path):
        pytest.importorskip("neml2")
        import glob

        from synthetic_fragmentation import (
            generate_ebsd_split,
            generate_nf_split,
        )

        eb = generate_ebsd_split(str(tmp_path / "ebsd"), nx=8, ny=8, nz=2, n_steps=4)
        assert len(eb["step_csvs"]) == 4
        assert all(os.path.exists(p) for p in eb["step_csvs"])

        nf = generate_nf_split(
            str(tmp_path / "nf"), nx=8, ny=8, nz=2, n_steps=3, exp_file_token="layer"
        )
        assert len(nf["step_dirs"]) == 3
        mics = glob.glob(os.path.join(nf["step_dirs"][0], "*.mic"))
        assert len(mics) == 2  # nz layers


class TestTopKAndDTolCompose:
    """``top_k`` caps the candidate set; ``top_k=None`` removes the cap (#49)."""

    def test_capping_before_or_after_d_tol_gives_the_same_set(self):
        """The reorder proposed in #49 is a no-op, so the cap is the only lever.

        Every B grain inside the radius is nearer than every B grain outside
        it, so "k nearest, then filter by d_tol" and "filter by d_tol, then k
        nearest" select identical candidates. Pinned here so nobody reorders
        these two filters expecting the search to widen.
        """
        pytest.importorskip("scipy")
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(0)
        k = 8
        for _ in range(200):
            coords_a = rng.random((4, 3)) * 10.0
            coords_b = rng.random((40, 3)) * 10.0
            tree = cKDTree(coords_b)
            d_tol = float(rng.uniform(0.5, 6.0))
            for i in range(len(coords_a)):
                dist, jdx = tree.query(coords_a[i], k=k)
                cap_then_filter = {int(j) for dd, j in zip(dist, jdx) if dd <= d_tol}
                within = tree.query_ball_point(coords_a[i], r=d_tol)
                filter_then_cap = {
                    int(j)
                    for _, j in sorted(
                        (float(np.linalg.norm(coords_b[j] - coords_a[i])), j)
                        for j in within
                    )[:k]
                }
                assert cap_then_filter == filter_then_cap

    def _dense_split(self):
        """Parent at the origin fragmenting into 5 near-identical children."""
        a = _nodes(
            [1],
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[10.0, 20.0, 30.0]]),
        )
        offsets = np.array(
            [[-2.0, 0, 0], [-1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]
        )
        b = _nodes(
            [11, 12, 13, 14, 15],
            offsets,
            np.tile(np.array([[10.0, 20.0, 30.0]]), (5, 1)),
        )
        return a, b

    def test_cap_limits_the_number_of_children(self):
        pytest.importorskip("neml2")
        pytest.importorskip("scipy")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._dense_split()
        res = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees", top_k=2
        )
        assert int(res["split_correspondences"].iloc[0]["n_children"]) == 2

    def test_raising_d_tol_alone_does_not_widen_a_capped_search(self):
        pytest.importorskip("neml2")
        pytest.importorskip("scipy")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._dense_split()
        narrow = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees", top_k=2
        )
        wide = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=1000.0, theta_tol_deg=5.0, angle_type="degrees", top_k=2
        )
        assert (
            int(narrow["split_correspondences"].iloc[0]["n_children"])
            == int(wide["split_correspondences"].iloc[0]["n_children"])
            == 2
        )

    def test_top_k_none_takes_every_candidate_within_d_tol(self):
        pytest.importorskip("neml2")
        pytest.importorskip("scipy")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._dense_split()
        res = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees", top_k=None
        )
        assert int(res["split_correspondences"].iloc[0]["n_children"]) == 5

    def test_top_k_none_matches_a_cap_wider_than_the_candidate_set(self):
        """With no grain excluded by the cap, both paths must agree exactly."""
        pytest.importorskip("neml2")
        pytest.importorskip("scipy")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._dense_split()
        uncapped = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees", top_k=None
        )
        wide_cap = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees", top_k=50
        )
        assert uncapped["n_splits"] == wide_cap["n_splits"]
        assert (
            uncapped["split_correspondences"].iloc[0]["children_b"]
            == wide_cap["split_correspondences"].iloc[0]["children_b"]
        )


def _sized_nodes(ids, xyz, eul, radii, volumes=None):
    """Grain table with explicit per-grain radii (and optionally volumes)."""
    df = _nodes(ids, xyz, eul)
    df["GrainRadius"] = np.asarray(radii, dtype=float)
    if volumes is not None:
        df["volume"] = np.asarray(volumes, dtype=float)
    return df


class TestCandidatePairGate:
    """The distance gate on its own: pure NumPy/SciPy, so it runs without neml2."""

    @staticmethod
    def _parent_and_subgrains(radius=50.0, frac=0.8):
        """One A grain of ``radius`` with three B sub-grains at ``frac * radius``."""
        coords_a = np.array([[0.0, 0.0, 0.0]])
        off = frac * radius
        coords_b = np.array([[off, 0.0, 0.0], [-off, 0.0, 0.0], [0.0, off, 0.0]])
        return coords_a, coords_b, np.array([radius]), np.full(3, 5.0)

    def test_absolute_gate_misses_subgrains_inside_a_large_parent(self):
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _candidate_pairs

        ca, cb, ra, rb = self._parent_and_subgrains()
        ai, _, _, gate = _candidate_pairs(
            ca, cb, ra, rb, 10.0, "absolute", 1.0, 1.0, 1.0, 8
        )
        assert len(ai) == 0
        assert gate.shape == (0,)

    def test_radius_gate_reaches_subgrains_inside_a_large_parent(self):
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _candidate_pairs

        ca, cb, ra, rb = self._parent_and_subgrains()
        ai, bj, dist, gate = _candidate_pairs(
            ca, cb, ra, rb, 10.0, "radius", 1.0, 1.0, 1.0, 8
        )
        assert sorted(bj.tolist()) == [0, 1, 2]
        assert np.all(ai == 0)
        # Gate is the parent radius (the d_tol floor and the small B radii lose).
        assert np.allclose(gate, 50.0)
        assert np.allclose(dist, 40.0)

    def test_radius_sum_gate(self):
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _candidate_pairs

        # Centres 30 apart, radii 10 and 12: inside alpha=1.5, outside alpha=1.0.
        ca, cb = np.array([[0.0, 0.0, 0.0]]), np.array([[30.0, 0.0, 0.0]])
        ra, rb = np.array([10.0]), np.array([12.0])
        tight, _, _, _ = _candidate_pairs(
            ca, cb, ra, rb, 0.0, "radius_sum", 1.0, 1.0, 1.0, 8
        )
        loose, _, _, gate = _candidate_pairs(
            ca, cb, ra, rb, 0.0, "radius_sum", 1.0, 1.0, 1.5, 8
        )
        assert len(tight) == 0
        assert len(loose) == 1
        assert np.allclose(gate, 1.5 * 22.0)

    def test_radius_mode_search_is_not_capped_by_top_k(self):
        """`top_k` is an 'absolute'-mode knob; the radius search must cover the gate."""
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _candidate_pairs

        n = 12
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        cb = np.column_stack(
            [40.0 * np.cos(angles), 40.0 * np.sin(angles), np.zeros(n)]
        )
        ca, ra, rb = np.array([[0.0, 0.0, 0.0]]), np.array([50.0]), np.full(n, 4.0)

        capped, _, _, _ = _candidate_pairs(
            ca, cb, ra, rb, 10.0, "absolute", 1.0, 1.0, 1.0, 3
        )
        uncapped, _, _, _ = _candidate_pairs(
            ca, cb, ra, rb, 10.0, "radius", 1.0, 1.0, 1.0, 3
        )
        assert len(capped) == 0  # every B centroid is beyond the absolute d_tol
        assert len(uncapped) == n

    def test_alpha_b_lets_a_large_b_grain_reach_a_small_a_grain(self):
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _candidate_pairs

        ca, cb = np.array([[0.0, 0.0, 0.0]]), np.array([[30.0, 0.0, 0.0]])
        ra, rb = np.array([1.0]), np.array([40.0])
        ai, _, _, gate = _candidate_pairs(
            ca, cb, ra, rb, 5.0, "radius", 1.0, 1.0, 1.0, 8
        )
        assert len(ai) == 1
        assert np.allclose(gate, 40.0)

    def test_empty_side_returns_empty_arrays(self):
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _candidate_pairs

        empty3 = np.empty((0, 3))
        for ca, cb, ra, rb in (
            (empty3, np.zeros((2, 3)), np.empty(0), np.ones(2)),
            (np.zeros((2, 3)), empty3, np.ones(2), np.empty(0)),
        ):
            for mode in ("absolute", "radius", "radius_sum"):
                ai, bj, dist, gate = _candidate_pairs(
                    ca, cb, ra, rb, 10.0, mode, 1.0, 1.0, 1.0, 8
                )
                assert len(ai) == len(bj) == len(dist) == len(gate) == 0

    def test_candidate_order_is_independent_of_the_ball_query_order(self):
        """Pairs are emitted sorted per A grain, so downstream tie-breaks are stable."""
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _candidate_pairs

        rng = np.random.default_rng(3)
        cb = rng.uniform(-20.0, 20.0, (15, 3))
        ca, ra, rb = np.zeros((1, 3)), np.array([40.0]), np.full(15, 1.0)
        ai, bj, _, _ = _candidate_pairs(
            ca, cb, ra, rb, 0.0, "radius", 1.0, 1.0, 1.0, None
        )
        assert np.all(ai == 0)
        assert np.all(np.diff(bj) > 0)


class TestOneToOneBlocks:
    """The Hungarian assignment helper on its own (no neml2 needed)."""

    def test_assignment_is_strict_and_optimal(self):
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _one_to_one_blocks

        # One component, 3 A x 3 B, fully connected with a known optimum.
        comp = np.zeros(6, dtype=np.int64)
        ai = np.repeat([0, 1, 2], 3)
        bj = np.tile([0, 1, 2], 3)
        cost_matrix = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
        cost = cost_matrix[ai, bj]

        blocks = _one_to_one_blocks(comp, ai, bj, cost)
        pairs = {int(a): int(b) for blk in blocks for a, b in zip(blk[0], blk[1])}

        # Brute force over all 3! permutations.
        best = min(
            itertools.permutations(range(3)),
            key=lambda p: sum(cost_matrix[i, p[i]] for i in range(3)),
        )
        assert pairs == {i: best[i] for i in range(3)}
        assert len(set(pairs.values())) == len(pairs)  # no B used twice

    def test_each_component_is_solved_independently(self):
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _one_to_one_blocks

        # Two disjoint components; a split parent may keep only one child.
        comp = np.array([0, 1, 0, 0, 1], dtype=np.int64)  # A: 0,1  B: 0,1,2
        ai = np.array([0, 0, 1])
        bj = np.array([0, 1, 2])
        cost = np.array([0.9, 0.2, 0.5])

        blocks = _one_to_one_blocks(comp, ai, bj, cost)
        pairs = sorted(
            (int(a), int(b)) for blk in blocks for a, b in zip(blk[0], blk[1])
        )
        assert pairs == [(0, 1), (1, 2)]

    def test_no_candidates_yields_no_blocks(self):
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _one_to_one_blocks

        assert (
            _one_to_one_blocks(
                np.zeros(2, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=float),
            )
            == []
        )


class TestRecallHelpers:
    """Per-side recall accounting (no neml2 needed)."""

    def test_counts_and_volume_weighting(self):
        from graintrace.fragmentation import _side_recall

        linked = np.array([True, False, True])
        mask = np.ones(3, dtype=bool)
        out = _side_recall(linked, mask, np.array([10.0, 1.0, 10.0]), "a")
        assert out["n_a_linked"] == 2
        assert out["n_a_total"] == 3
        assert out["a_fraction"] == pytest.approx(2.0 / 3.0)
        assert out["a_fraction_volume"] == pytest.approx(20.0 / 21.0)

    def test_overlap_mask_restricts_the_denominator(self):
        from graintrace.fragmentation import _side_recall

        linked = np.array([True, False, False])
        mask = np.array([True, False, False])
        out = _side_recall(linked, mask, None, "b")
        assert out["b_fraction"] == 1.0
        assert out["n_b_total"] == 1
        assert "b_fraction_volume" not in out

    def test_empty_mask_gives_nan_not_a_zero_division(self):
        from graintrace.fragmentation import _side_recall

        out = _side_recall(np.zeros(2, bool), np.zeros(2, bool), None, "a")
        assert np.isnan(out["a_fraction"])

    def test_overlap_mask_accepts_a_column_name_or_an_array(self):
        from graintrace.fragmentation import _overlap_mask

        df = pd.DataFrame({"in_ov": [True, False, True]})
        assert _overlap_mask("in_ov", df, 3, "in_overlap_a").tolist() == [
            True,
            False,
            True,
        ]
        assert _overlap_mask(None, df, 3, "in_overlap_a").all()
        assert _overlap_mask([1, 0, 1], df, 3, "in_overlap_a").tolist() == [
            True,
            False,
            True,
        ]

    def test_wrong_length_overlap_mask_raises(self):
        from graintrace.fragmentation import _overlap_mask

        with pytest.raises(ValueError, match="in_overlap_a must have length 3"):
            _overlap_mask([True, False], pd.DataFrame(), 3, "in_overlap_a")

    def test_unknown_overlap_column_names_the_parameter_and_the_columns(self):
        """A bare pandas KeyError does not say which argument was wrong."""
        from graintrace.fragmentation import _overlap_mask

        df = pd.DataFrame({"in_ov": [True, False, True]})
        with pytest.raises(ValueError, match="in_overlap_a='in_ovv'"):
            _overlap_mask("in_ovv", df, 3, "in_overlap_a")
        with pytest.raises(ValueError, match=r"available columns are \['in_ov'\]"):
            _overlap_mask("in_ovv", df, 3, "in_overlap_a")


class TestSizeAwareSplitDetection:
    """``d_tol_mode`` end to end through :meth:`detect_splits`."""

    @staticmethod
    def _parent_and_subgrains():
        """Issue #59's case: one R=50 parent, three sub-grains at 0.8 R."""
        a = _sized_nodes(
            [1], np.array([[0.0, 0.0, 0.0]]), np.array([[10.0, 20.0, 30.0]]), [50.0]
        )
        off = 40.0
        b = _sized_nodes(
            [11, 12, 13],
            np.array([[off, 0.0, 0.0], [-off, 0.0, 0.0], [0.0, off, 0.0]]),
            np.array([[11.0, 20.0, 30.0], [9.0, 21.0, 30.0], [10.0, 19.5, 30.5]]),
            [5.0, 5.0, 5.0],
        )
        return a, b

    def test_absolute_gate_reports_births_and_a_death(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._parent_and_subgrains()
        res = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees"
        )
        assert res["n_births"] == 3
        assert res["n_deaths"] == 1
        assert res["n_splits"] == 0

    def test_radius_gate_reports_one_split_with_three_children(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._parent_and_subgrains()
        res = FragmentationAnalyzer().detect_splits(
            a,
            b,
            d_tol=10.0,
            theta_tol_deg=5.0,
            angle_type="degrees",
            d_tol_mode="radius",
        )
        assert res["n_splits"] == 1
        assert res["n_births"] == res["n_deaths"] == 0
        row = res["split_correspondences"].iloc[0]
        assert int(row["parent_a"]) == 1
        assert set(map(int, row["children_b"].split(";"))) == {11, 12, 13}

    def test_radius_sum_gate_reports_one_split(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._parent_and_subgrains()
        res = FragmentationAnalyzer().detect_splits(
            a,
            b,
            d_tol=0.0,
            theta_tol_deg=5.0,
            angle_type="degrees",
            d_tol_mode="radius_sum",
            alpha=1.0,
        )
        assert res["n_splits"] == 1

    def test_unknown_mode_raises(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._parent_and_subgrains()
        with pytest.raises(ValueError, match="Unknown d_tol_mode"):
            FragmentationAnalyzer().detect_splits(
                a, b, d_tol=1.0, theta_tol_deg=1.0, d_tol_mode="nope"
            )

    def test_radius_mode_without_a_radius_column_raises(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._parent_and_subgrains()
        with pytest.raises(ValueError, match="needs the 'GrainRadius' column"):
            FragmentationAnalyzer().detect_splits(
                a.drop(columns=["GrainRadius"]),
                b,
                d_tol=1.0,
                theta_tol_deg=1.0,
                d_tol_mode="radius",
            )

    def test_non_default_top_k_warns_in_a_radius_mode(self):
        """A cap the radius search cannot honour must not be swallowed silently."""
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._parent_and_subgrains()
        kw = dict(d_tol=1.0, theta_tol_deg=5.0, angle_type="degrees")
        with pytest.warns(UserWarning, match="is ignored when d_tol_mode"):
            FragmentationAnalyzer().detect_splits(
                a, b, d_tol_mode="radius", top_k=3, **kw
            )
        # The default is not a choice the caller made, so it stays quiet; so does
        # the absolute mode, which actually uses the cap.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FragmentationAnalyzer().detect_splits(a, b, d_tol_mode="radius", **kw)
            FragmentationAnalyzer().detect_splits(
                a, b, d_tol_mode="absolute", top_k=3, **kw
            )

    def test_zero_theta_tol_does_not_produce_a_nan_cost(self):
        """``theta_tol_deg=0`` is degenerate but legal; the cost must stay finite."""
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        # Two exactly-equal orientation pairs, so the 0-degree filter keeps them
        # and the miso/theta_tol term would be 0/0 without the guard.
        a = _nodes(
            [1, 2],
            np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
            np.array([[10.0, 20.0, 30.0], [40.0, 20.0, 30.0]]),
        )
        b = _nodes(
            [11, 12],
            np.array([[1.0, 0.0, 0.0], [51.0, 0.0, 0.0]]),
            np.array([[10.0, 20.0, 30.0], [40.0, 20.0, 30.0]]),
        )
        out = FragmentationAnalyzer().detect_splits(
            a,
            b,
            d_tol=10.0,
            theta_tol_deg=0.0,
            angle_type="degrees",
            assignment="one_to_one",
        )
        assert out["n_matches"] == 2
        assert np.isfinite(out["one_to_one"]["cost"].to_numpy()).all()

    def test_defaults_are_unchanged_by_the_new_parameters(self):
        """Backward compatibility: the default call is the 'absolute' path."""
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._parent_and_subgrains()
        kw = dict(d_tol=45.0, theta_tol_deg=5.0, angle_type="degrees")
        plain = FragmentationAnalyzer().detect_splits(a, b, **kw)
        explicit = FragmentationAnalyzer().detect_splits(
            a, b, d_tol_mode="absolute", top_k=8, **kw
        )
        keys = ("n_matches", "n_splits", "n_merges", "n_births", "n_deaths")
        assert {k: plain[k] for k in keys} == {k: explicit[k] for k in keys}
        assert plain["components"] == explicit["components"]

    @pytest.mark.parametrize(
        "case",
        ["parent_and_subgrains", "two_steps", "dense_split", "adjacency"],
    )
    def test_defaults_reproduce_the_pre_change_implementation(self, case):
        """Issue #59: "default arguments reproduce today's output exactly".

        The test above compares the new code against itself with the defaults
        spelled out, so both sides move together and it cannot catch a
        regression. This one compares against ``tests/_detect_splits_original``,
        a frozen copy of the implementation as it stood before this change --
        the pattern ``tests/_prune_original.py`` already establishes.
        """
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        from _detect_splits_original import detect_splits_original

        if case == "parent_and_subgrains":
            a, b = self._parent_and_subgrains()
            kw = dict(d_tol=45.0, theta_tol_deg=5.0, angle_type="degrees")
        elif case == "two_steps":
            a, b = TestOneToOneAssignment._two_steps()
            kw = dict(d_tol=20.0, theta_tol_deg=10.0, angle_type="degrees")
        elif case == "dense_split":
            a, b = TestTopKAndDTolCompose()._dense_split()
            kw = dict(d_tol=30.0, theta_tol_deg=8.0, angle_type="degrees")
        else:
            a = _nodes(
                [1],
                np.array([[0.0, 0.0, 0.0]]),
                np.array([[10.0, 20.0, 30.0]]),
            )
            b = _nodes(
                [11, 12, 13],
                np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
                np.array([[11.0, 20.0, 30.0], [9.0, 21.0, 30.0], [10.0, 19.0, 31.0]]),
            )
            kw = dict(
                d_tol=10.0,
                theta_tol_deg=5.0,
                angle_type="degrees",
                adjacency_b=[(11, 12)],
            )

        new = FragmentationAnalyzer().detect_splits(a, b, **kw)
        old = detect_splits_original("432", a, b, **kw)

        for key in ("n_matches", "n_splits", "n_merges", "n_births", "n_deaths"):
            assert new[key] == old[key], key
        assert new["components"] == old["components"]
        # "recall" and "null" are new keys added by this change; backward
        # compatibility is about the ones that already existed.
        pd.testing.assert_frame_equal(
            new["split_correspondences"], old["split_correspondences"]
        )


class TestOneToOneAssignment:
    """``assignment="one_to_one"`` through :meth:`detect_splits`."""

    @staticmethod
    def _two_steps():
        rng = np.random.default_rng(11)
        xyz = rng.uniform(-100.0, 100.0, (12, 3))
        eul = rng.uniform(0.0, 60.0, (12, 3))
        a = _sized_nodes(list(range(1, 13)), xyz, eul, np.full(12, 8.0))
        b = _sized_nodes(
            list(range(101, 113)),
            xyz + rng.normal(0.0, 1.5, (12, 3)),
            eul + rng.normal(0.0, 0.5, (12, 3)),
            np.full(12, 8.0),
        )
        return a, b

    def test_no_grain_is_assigned_twice(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._two_steps()
        res = FragmentationAnalyzer().detect_splits(
            a,
            b,
            d_tol=20.0,
            theta_tol_deg=10.0,
            angle_type="degrees",
            assignment="one_to_one",
        )
        pairs = res["one_to_one"]
        assert list(pairs.columns) == ["a_id", "b_id", "miso_deg", "dist", "cost"]
        assert pairs["a_id"].is_unique
        assert pairs["b_id"].is_unique
        assert (pairs["miso_deg"] <= 10.0).all()

    def test_a_split_parent_keeps_exactly_one_child(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = TestSizeAwareSplitDetection._parent_and_subgrains()
        res = FragmentationAnalyzer().detect_splits(
            a,
            b,
            d_tol=10.0,
            theta_tol_deg=5.0,
            angle_type="degrees",
            d_tol_mode="radius",
            assignment="one_to_one",
        )
        # The lineage view still reports the split; the assignment is strict.
        assert res["n_splits"] == 1
        assert len(res["one_to_one"]) == 1

    def test_lineage_output_is_unaffected_by_requesting_an_assignment(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._two_steps()
        kw = dict(d_tol=20.0, theta_tol_deg=10.0, angle_type="degrees")
        without = FragmentationAnalyzer().detect_splits(a, b, **kw)
        with_asg = FragmentationAnalyzer().detect_splits(
            a, b, assignment="one_to_one", **kw
        )
        assert without["components"] == with_asg["components"]
        assert "one_to_one" not in without

    def test_unknown_assignment_raises(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._two_steps()
        with pytest.raises(ValueError, match="Unknown assignment"):
            FragmentationAnalyzer().detect_splits(
                a, b, d_tol=1.0, theta_tol_deg=1.0, assignment="greedy"
            )


class TestRecallAndNull:
    """Per-side recall and the chance null through :meth:`detect_splits`."""

    @staticmethod
    def _one_linked_one_missed():
        a = _sized_nodes(
            [1, 2],
            np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]]),
            np.array([[10.0, 20.0, 30.0], [40.0, 30.0, 20.0]]),
            [10.0, 10.0],
            volumes=[1000.0, 1.0],
        )
        b = _sized_nodes(
            [11],
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[10.5, 20.0, 30.0]]),
            [10.0],
        )
        return a, b

    def test_recall_is_reported_per_side(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._one_linked_one_missed()
        res = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees"
        )
        rec = res["recall"]
        assert rec["n_a_linked"] == 1 and rec["n_a_total"] == 2
        assert rec["a_fraction"] == pytest.approx(0.5)
        assert rec["b_fraction"] == pytest.approx(1.0)
        # Volume weighting: the linked A grain holds nearly all the volume.
        assert rec["a_fraction_volume"] == pytest.approx(1000.0 / 1001.0)
        # nodes_b has no volume column, so no volume entry for that side.
        assert "b_fraction_volume" not in rec

    def test_overlap_mask_excludes_material_the_other_step_cannot_see(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._one_linked_one_missed()
        res = FragmentationAnalyzer().detect_splits(
            a,
            b,
            d_tol=10.0,
            theta_tol_deg=5.0,
            angle_type="degrees",
            in_overlap_a=[True, False],
        )
        assert res["recall"]["a_fraction"] == pytest.approx(1.0)
        assert res["recall"]["n_a_total"] == 1

    def test_null_links_are_about_zero_on_well_separated_grains(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = TestOneToOneAssignment._two_steps()
        res = FragmentationAnalyzer().detect_splits(
            a,
            b,
            d_tol=20.0,
            theta_tol_deg=10.0,
            angle_type="degrees",
            n_null=20,
            seed=42,
        )
        null = res["null"]
        assert null["n_null"] == 20
        assert null["observed_links"] == 12
        # A 10 deg gate under cubic symmetry is a tiny slice of orientation space.
        assert null["mean_links"] < 1.0

    def test_observed_links_is_measured_where_the_null_measures_it(self):
        """The adjacency prune must not deflate the count ``mean_links`` faces.

        ``_chance_links`` reruns the misorientation filter over the
        distance-gated candidates and never sees the adjacency prune, so an
        ``observed_links`` measured after that prune would read falsely weak
        against ``mean_links``. The post-prune count gets its own key.
        """
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        # Parent 1 has three compatible candidates; adjacency (11, 12) drops 13,
        # so the prune provably bites: three links before it, two after.
        a = _nodes(
            [1],
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[10.0, 20.0, 30.0]]),
        )
        b = _nodes(
            [11, 12, 13],
            np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
            np.array([[11.0, 20.0, 30.0], [9.0, 21.0, 30.0], [10.0, 19.0, 31.0]]),
        )
        kwargs = dict(
            d_tol=10.0,
            theta_tol_deg=5.0,
            angle_type="degrees",
            n_null=5,
            seed=42,
        )
        free = FragmentationAnalyzer().detect_splits(a, b, **kwargs)["null"]
        pruned = FragmentationAnalyzer().detect_splits(
            a, b, adjacency_b=[(11, 12)], **kwargs
        )["null"]

        # The prune really did drop a link, so the comparison below is not vacuous.
        assert pruned["observed_links_after_adjacency"] == 2
        assert pruned["observed_links"] == 3
        # ... and the count the null is compared against is untouched by it.
        assert pruned["observed_links"] == free["observed_links"]
        # With no adjacency supplied the two keys describe the same set.
        assert free["observed_links_after_adjacency"] == free["observed_links"] == 3

    def test_null_is_reproducible_under_a_seed(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = TestOneToOneAssignment._two_steps()
        kw = dict(
            d_tol=20.0, theta_tol_deg=10.0, angle_type="degrees", n_null=8, seed=7
        )
        first = FragmentationAnalyzer().detect_splits(a, b, **kw)
        second = FragmentationAnalyzer().detect_splits(a, b, **kw)
        assert first["null"] == second["null"]

    def test_null_is_absent_by_default(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._one_linked_one_missed()
        res = FragmentationAnalyzer().detect_splits(
            a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees"
        )
        assert "null" not in res


class TestRadiusSanitizingAndValidation:
    """NaN/negative radii fall back to ``d_tol``; negative tolerances are rejected."""

    @staticmethod
    def _parent_subgrains_and_a_nan_outlier():
        """A parent of R=50, three sub-grains at 0.8R, one far B grain with NaN R."""
        a, b = TestSizeAwareSplitDetection._parent_and_subgrains()
        far = _sized_nodes(
            [99],
            np.array([[1000.0, 0.0, 0.0]]),
            np.array([[60.0, 40.0, 10.0]]),
            [np.nan],
        )
        return a, pd.concat([b, far], ignore_index=True)

    def test_candidate_search_survives_a_nan_b_radius(self):
        pytest.importorskip("scipy")
        from graintrace.fragmentation import _candidate_pairs

        ca = np.zeros((1, 3))
        cb = np.array([[40.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
        ai, bj, _, _ = _candidate_pairs(
            ca,
            cb,
            np.array([50.0]),
            np.array([5.0, np.nan]),
            10.0,
            "radius",
            1.0,
            1.0,
            1.0,
            8,
        )
        assert ai.tolist() == [0] and bj.tolist() == [0]

    def test_one_nan_b_radius_does_not_drop_every_link(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._parent_subgrains_and_a_nan_outlier()
        with pytest.warns(UserWarning, match="1 grain.*nodes_b.*NaN or negative"):
            res = FragmentationAnalyzer().detect_splits(
                a,
                b,
                d_tol=10.0,
                theta_tol_deg=5.0,
                angle_type="degrees",
                d_tol_mode="radius",
            )
        assert res["n_splits"] == 1
        assert res["n_births"] == 1  # only the unrelated far grain
        assert res["n_deaths"] == 0

    @pytest.mark.parametrize("bad", [np.nan, -5.0])
    def test_bad_a_radius_falls_back_to_the_d_tol_floor(self, bad):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        # One A grain with a bad radius and one B grain 8 away: d_tol=10 must link it.
        a = _sized_nodes([1], np.zeros((1, 3)), np.array([[10.0, 20.0, 30.0]]), [bad])
        b = _sized_nodes(
            [11],
            np.array([[8.0, 0.0, 0.0]]),
            np.array([[10.5, 20.0, 30.0]]),
            [1.0],
        )
        with pytest.warns(UserWarning, match="nodes_a"):
            res = FragmentationAnalyzer().detect_splits(
                a,
                b,
                d_tol=10.0,
                theta_tol_deg=5.0,
                angle_type="degrees",
                d_tol_mode="radius",
            )
        assert res["n_matches"] == 1

    def test_absolute_mode_ignores_radii_and_stays_quiet(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = self._parent_subgrains_and_a_nan_outlier()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FragmentationAnalyzer().detect_splits(
                a, b, d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees"
            )

    @pytest.mark.parametrize(
        "name", ["d_tol", "theta_tol_deg", "alpha", "alpha_a", "alpha_b"]
    )
    def test_negative_tolerance_or_multiplier_raises(self, name):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        a, b = TestSizeAwareSplitDetection._parent_and_subgrains()
        kw = dict(d_tol=10.0, theta_tol_deg=5.0, angle_type="degrees")
        kw[name] = -1.0
        with pytest.raises(ValueError, match=f"{name} must be non-negative"):
            FragmentationAnalyzer().detect_splits(a, b, **kw)

