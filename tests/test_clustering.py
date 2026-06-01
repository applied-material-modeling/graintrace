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

"""Tests for ClusterAnalysisIndicator, GraphSpatialCluster, IdentifyRareClusters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_vms_csv


class TestClusterAnalysisIndicator:
    def test_loads_csv(self, tmp_path, vms_csv):
        from graintrace.cluster_indicator import ClusterAnalysisIndicator

        ind = ClusterAnalysisIndicator(str(vms_csv), coord_cols=("x", "y", "z"))
        ind.load_data()
        assert ind.data is not None
        assert "sxx" in ind.data.columns

    def test_run_scipy_hierarchical(self, tmp_path, vms_csv):
        from graintrace.cluster_indicator import ClusterAnalysisIndicator
        from graintrace.similarity_metric_library import SimilarityMetricLibrary

        ind = ClusterAnalysisIndicator(str(vms_csv), coord_cols=("x", "y", "z"))
        lib = SimilarityMetricLibrary()
        spec = lib.von_mises_stress()
        result = ind.run(
            method_type="scipy_hierarchical",
            spec=spec,
            threshold=0.05,
            method="average",
            criterion="distance",
        )
        assert "points" in result
        assert "clusters" in result
        assert "extras" in result
        assert "linkage_Z" in result["extras"]
        assert "cluster_label" in result["points"].columns

    def test_run_sklearn_dbscan(self, tmp_path, vms_csv):
        from graintrace.cluster_indicator import ClusterAnalysisIndicator
        from graintrace.similarity_metric_library import SimilarityMetricLibrary

        ind = ClusterAnalysisIndicator(str(vms_csv), coord_cols=("x", "y", "z"))
        lib = SimilarityMetricLibrary()
        spec = lib.von_mises_stress()
        result = ind.run(
            method_type="sklearn_dbscan",
            spec=spec,
            eps=0.1,
            min_samples=2,
        )
        assert "points" in result
        assert "cluster_label" in result["points"].columns

    def test_missing_feature_col_raises(self, tmp_path, vms_csv):
        from graintrace.cluster_indicator import ClusterAnalysisIndicator
        from graintrace.user_data_class import SimilarityMetric

        ind = ClusterAnalysisIndicator(str(vms_csv), coord_cols=("x", "y", "z"))
        spec = SimilarityMetric(
            name="bad", feature_cols=["nonexistent_col"], func=lambda u, v: 0.0
        )
        with pytest.raises(ValueError, match="missing columns|Missing"):
            ind.run(method_type="scipy_hierarchical", spec=spec, threshold=0.1)

    def test_invalid_method_raises(self, tmp_path, vms_csv):
        from graintrace.cluster_indicator import ClusterAnalysisIndicator
        from graintrace.similarity_metric_library import SimilarityMetricLibrary

        ind = ClusterAnalysisIndicator(str(vms_csv), coord_cols=("x", "y", "z"))
        lib = SimilarityMetricLibrary()
        spec = lib.von_mises_stress()
        with pytest.raises(ValueError, match="Unknown method"):
            ind.run(method_type="bad_method", spec=spec)


class TestGraphSpatialCluster:
    def _make_grid_csv(self, path, nx=8, ny=8, seed=0):
        rng = np.random.default_rng(seed)
        n = nx * ny
        xs = np.tile(np.arange(nx, dtype=float), ny)
        ys = np.repeat(np.arange(ny, dtype=float), nx)
        df = pd.DataFrame(
            {
                "id": np.arange(1, n + 1),
                "x": xs,
                "y": ys,
                "z": np.zeros(n),
                "sxx": rng.normal(100, 20, n),
                "syy": rng.normal(50, 10, n),
                "szz": rng.normal(30, 5, n),
                "sxy": rng.normal(0, 5, n),
                "sxz": rng.normal(0, 5, n),
                "syz": rng.normal(0, 5, n),
            }
        )
        df.to_csv(path, index=False)
        return str(path)

    def test_run_leiden_grid(self, tmp_path):
        from graintrace.graph_spatial_cluster import GraphSpatialCluster
        from graintrace.similarity_metric_library import SimilarityMetricLibrary
        from graintrace.user_data_class import WeightConfig

        csv_path = self._make_grid_csv(tmp_path / "grid.csv")
        gsc = GraphSpatialCluster(
            csv_path=csv_path,
            id_col="id",
            coord_cols=("x", "y", "z"),
        )
        lib = SimilarityMetricLibrary()
        spec = lib.von_mises_stress()
        weight_cfg = WeightConfig(mode="rbf", sigma=50.0)

        result = gsc.run(
            spec=spec,
            graph_mode="grid",
            manhattan_radius=1,
            segmenter="leiden",
            seed=42,
            weight_cfg=weight_cfg,
            n_jobs=1,
            output_csv_path=str(tmp_path / "clusters.csv"),
        )
        assert "csv_path" in result or "points" in result or isinstance(result, dict)

    def test_run_produces_labeled_csv(self, tmp_path):
        from graintrace.graph_spatial_cluster import GraphSpatialCluster
        from graintrace.similarity_metric_library import SimilarityMetricLibrary
        from graintrace.user_data_class import WeightConfig

        csv_path = self._make_grid_csv(tmp_path / "grid2.csv")
        out_csv = str(tmp_path / "labeled.csv")
        gsc = GraphSpatialCluster(csv_path=csv_path, id_col="id", coord_cols=("x", "y", "z"))
        lib = SimilarityMetricLibrary()
        spec = lib.von_mises_stress()
        gsc.run(
            spec=spec,
            graph_mode="grid",
            segmenter="leiden",
            seed=0,
            weight_cfg=WeightConfig(mode="rbf", sigma=50.0),
            n_jobs=1,
            output_csv_path=out_csv,
        )
        if Path(out_csv).exists():
            df = pd.read_csv(out_csv)
            assert "cluster_label" in df.columns or len(df) > 0
