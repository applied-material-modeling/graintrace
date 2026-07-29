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

"""Tests for rei_comparison.REIComparison and the rare-points CSV export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _cube_csv(path, lo, hi, spacing=1.0, cid=1):
    """Solid axis-aligned cube of voxels on a regular grid."""
    xs = np.arange(lo, hi, spacing)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    df = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "z": Z.ravel()})
    df["rare_cluster_id"] = cid
    df.to_csv(path, index=False)
    return str(path)


def _blobs_csv(path, blobs, spacing=1.0):
    """blobs = list of (cid, (cx,cy,cz), radius); union of spheres on a grid."""
    parts = []
    for cid, center, radius in blobs:
        lo = np.floor(np.array(center) - radius).astype(int)
        hi = np.ceil(np.array(center) + radius).astype(int) + 1
        xs = np.arange(lo[0], hi[0])
        ys = np.arange(lo[1], hi[1])
        zs = np.arange(lo[2], hi[2])
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        m = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (
            Z - center[2]
        ) ** 2 <= radius**2
        sub = pd.DataFrame(
            {"x": X[m] * spacing, "y": Y[m] * spacing, "z": Z[m] * spacing}
        )
        sub["rare_cluster_id"] = cid
        parts.append(sub)
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    return str(path)


class TestREIComparisonMetrics:
    def test_identical_clouds_iou_one(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        a = _cube_csv(tmp_path / "a.csv", 0, 10)
        b = _cube_csv(tmp_path / "b.csv", 0, 10)
        out = REIComparison(
            a, b, str(tmp_path / "o"), spacing_1=1.0, spacing_2=1.0
        ).run_comparison()
        m = out["metrics"]
        assert m["iou"] == pytest.approx(1.0)
        assert m["dice"] == pytest.approx(1.0)
        assert m["containment_1"] == pytest.approx(1.0)
        assert m["containment_2"] == pytest.approx(1.0)

    def test_disjoint_clouds_iou_zero(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        a = _cube_csv(tmp_path / "a.csv", 0, 10)
        b = _cube_csv(tmp_path / "b.csv", 100, 110)
        out = REIComparison(
            a, b, str(tmp_path / "o"), spacing_1=1.0, spacing_2=1.0
        ).run_comparison()
        assert out["metrics"]["iou"] == 0.0
        assert out["metrics"]["n_voxels_intersection"] == 0

    def test_known_partial_overlap(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        # A = [0,10)^3, B = [5,15)^3 -> intersection 5^3, union 2*1000-125
        a = _cube_csv(tmp_path / "a.csv", 0, 10)
        b = _cube_csv(tmp_path / "b.csv", 5, 15)
        out = REIComparison(
            a, b, str(tmp_path / "o"), spacing_1=1.0, spacing_2=1.0
        ).run_comparison()
        expected_iou = 125.0 / (1000.0 + 1000.0 - 125.0)
        assert out["metrics"]["iou"] == pytest.approx(expected_iou)
        assert out["metrics"]["n_voxels_intersection"] == 125

    def test_coarse_vs_fine_high_containment(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        # same region, fine (1.0) vs coarse (2.0): coarse cells nearly contained
        a = _cube_csv(tmp_path / "a.csv", 0, 20, spacing=1.0)
        b = _cube_csv(tmp_path / "b.csv", 0, 20, spacing=2.0)
        out = REIComparison(
            a, b, str(tmp_path / "o"), spacing_1=1.0, spacing_2=2.0
        ).run_comparison()
        assert out["metrics"]["iou"] > 0.5
        assert out["metrics"]["containment_1"] > 0.8

    def test_auto_spacing_detection(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        a = _cube_csv(tmp_path / "a.csv", 0, 10, spacing=2.0)
        b = _cube_csv(tmp_path / "b.csv", 0, 10, spacing=2.0)
        comp = REIComparison(a, b, str(tmp_path / "o"))  # spacings=None
        out = comp.run_comparison()
        assert np.allclose(comp.spacing_1, 2.0)
        assert out["metrics"]["iou"] == pytest.approx(1.0)


class TestREIComparisonClusters:
    def test_cluster_matching_by_overlap(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        # two well-separated blobs; REI 2 has DIFFERENT labels + a small shift
        a = _blobs_csv(
            tmp_path / "a.csv", [(1, (10, 10, 10), 4), (2, (40, 40, 40), 4)]
        )
        b = _blobs_csv(
            tmp_path / "b.csv", [(77, (11, 10, 10), 4), (88, (41, 40, 40), 4)]
        )
        out = REIComparison(
            a, b, str(tmp_path / "o"), spacing_1=1.0, spacing_2=1.0
        ).run_comparison()
        match = pd.read_csv(out["cluster_match_path"])
        pairs = match[(match.cluster_1 > 0) & (match.cluster_2 > 0)][
            ["cluster_1", "cluster_2"]
        ].values.tolist()
        assert [1, 77] in pairs
        assert [2, 88] in pairs
        assert out["metrics"]["n_matched_clusters"] == 2

    def test_cluster_col_none_skips_matching(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        a = _cube_csv(tmp_path / "a.csv", 0, 8)
        b = _cube_csv(tmp_path / "b.csv", 0, 8)
        out = REIComparison(
            a, b, str(tmp_path / "o"), spacing_1=1.0, spacing_2=1.0, cluster_col=None
        ).run_comparison()
        assert out["cluster_match_path"] is None


class TestREIComparisonOutputs:
    def test_output_files_written(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        a = _cube_csv(tmp_path / "a.csv", 0, 8)
        b = _cube_csv(tmp_path / "b.csv", 4, 12)
        out_dir = tmp_path / "cmp"
        out = REIComparison(
            a, b, str(out_dir), spacing_1=1.0, spacing_2=1.0
        ).run_comparison()
        assert Path(out["metrics_path"]).exists()
        assert Path(out["overlap_vtk_path"]).exists()
        assert Path(out["cluster_match_path"]).exists()
        # VTK is a legacy ASCII polydata point cloud
        head = Path(out["overlap_vtk_path"]).read_text()[:200]
        assert "DATASET POLYDATA" in head


class TestREIComparisonValidation:
    def test_missing_file_raises(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        a = _cube_csv(tmp_path / "a.csv", 0, 5)
        with pytest.raises(FileNotFoundError):
            REIComparison(a, str(tmp_path / "nope.csv"), str(tmp_path / "o"))

    def test_missing_columns_raises(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        bad = tmp_path / "bad.csv"
        pd.DataFrame({"x": [0, 1], "y": [0, 1]}).to_csv(bad, index=False)
        a = _cube_csv(tmp_path / "a.csv", 0, 5)
        with pytest.raises(ValueError, match="missing columns"):
            REIComparison(str(bad), a, str(tmp_path / "o"))

    def test_bad_spacing_raises(self, tmp_path):
        from graintrace.rei_comparison import REIComparison

        a = _cube_csv(tmp_path / "a.csv", 0, 5)
        b = _cube_csv(tmp_path / "b.csv", 0, 5)
        comp = REIComparison(a, b, str(tmp_path / "o"), spacing_1=-1.0, spacing_2=1.0)
        with pytest.raises(ValueError, match="positive"):
            comp.run_comparison()


class TestRarePointsCSVExport:
    """The new rare_points_csv_path output of IdentifyRareClusters.run_get_rare_cluster."""

    def _minimal_bundle(self):
        # 6 grid points; stage-1 labels 0/1/2 -> super labels 10/11/12
        input_df = pd.DataFrame(
            {
                "id": np.arange(6),
                "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "y": np.zeros(6),
                "z": np.zeros(6),
                "feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        gsc_labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        indicator_points_df = pd.DataFrame(
            {"cluster_id": [0, 1, 2], "cluster_label": [10, 11, 12]}
        )
        indicator_clusters_df = pd.DataFrame(
            {"cluster_label": [10, 11, 12], "n": [2, 2, 2]}
        )
        return {
            "input_df": input_df,
            "gsc_labels": gsc_labels,
            "indicator_points_df": indicator_points_df,
            "indicator_clusters_df": indicator_clusters_df,
        }

    def test_rare_points_csv_written(self, tmp_path):
        from graintrace.rare_cluster_indicator import IdentifyRareClusters
        from graintrace.user_data_class import RareCriteria

        csv_in = tmp_path / "field.csv"
        self._minimal_bundle()["input_df"].to_csv(csv_in, index=False)
        irc = IdentifyRareClusters(
            input_csv_path=str(csv_in), id_col="id", coord_cols=("x", "y", "z")
        )

        # select super-labels 11 and 12 as rare
        criteria = RareCriteria(selector=lambda df: np.array([11, 12]))
        rare_csv = tmp_path / "rare_points.csv"
        out = irc.run_get_rare_cluster(
            bundle=self._minimal_bundle(),
            criteria=criteria,
            output_vtk_path=str(tmp_path / "rare.vtk"),
            first_rare_block_id=2,
            rare_points_csv_path=str(rare_csv),
        )
        assert out["rare_points_csv_path"] == str(rare_csv)
        assert rare_csv.exists()
        df = pd.read_csv(rare_csv)
        # rare = the 4 points in stage-1 clusters 1 and 2 (x = 2,3,4,5)
        assert set(df.columns) == {"x", "y", "z", "rare_cluster_id"}
        assert len(df) == 4
        assert sorted(df["x"].tolist()) == [2.0, 3.0, 4.0, 5.0]
        assert (df["rare_cluster_id"] >= 2).all()
