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

import os

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
