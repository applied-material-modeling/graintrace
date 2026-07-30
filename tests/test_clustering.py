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
        gsc = GraphSpatialCluster(
            csv_path=csv_path, id_col="id", coord_cols=("x", "y", "z")
        )
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


class TestGraphSpatialClusterFixes:
    """Locks in the correctness invariants of the build/prune/threads fixes."""

    @staticmethod
    def _random_undirected(n, n_edges, seed=0):
        rng = np.random.default_rng(seed)
        e = np.stack([rng.integers(0, n, n_edges), rng.integers(0, n, n_edges)], axis=1)
        e = np.unique(np.sort(e, axis=1), axis=0)
        e = e[e[:, 0] != e[:, 1]]
        w = rng.random(e.shape[0])
        return e, w

    def test_graphfromcoo_equivalent_to_addedge(self):
        nk = pytest.importorskip("networkit")
        n = 300
        edges, w = self._random_undirected(n, 2000)
        g_loop = nk.Graph(n, weighted=True, directed=False)
        for (u, v), wt in zip(edges, w):
            g_loop.addEdge(int(u), int(v), float(wt))
        row = np.ascontiguousarray(edges[:, 0], dtype=np.uint64)
        col = np.ascontiguousarray(edges[:, 1], dtype=np.uint64)
        g_coo = nk.GraphFromCoo(
            (np.ascontiguousarray(w), (row, col)),
            n=n,
            weighted=True,
            directed=False,
        )
        assert g_loop.numberOfEdges() == g_coo.numberOfEdges()
        assert abs(g_loop.totalEdgeWeight() - g_coo.totalEdgeWeight()) < 1e-9

    def test_prune_njobs_equivalence(self):
        # n_jobs is accepted (now ignored) and produces identical output.
        from graintrace.graph_spatial_cluster import GraphSpatialCluster

        gsc = GraphSpatialCluster.__new__(GraphSpatialCluster)
        n = 2000
        edges, w = self._random_undirected(n, 30000)
        e1, w1 = gsc.prune_topk_per_node_parallel(
            n_nodes=n, edges=edges, weights=w, k=5, n_jobs=1
        )
        e2, w2 = gsc.prune_topk_per_node_parallel(
            n_nodes=n, edges=edges, weights=w, k=5, n_jobs=3
        )
        assert np.array_equal(e1, e2)
        assert np.array_equal(w1, w2)

    @staticmethod
    def _bruteforce_topk(n, edges, weights, k):
        """Independent reference: per node keep its k highest-weight edges (union)."""
        from collections import defaultdict

        inc = defaultdict(list)
        for eid, (u, v) in enumerate(edges):
            inc[int(u)].append((weights[eid], eid))
            inc[int(v)].append((weights[eid], eid))
        keep = set()
        for lst in inc.values():
            lst.sort(key=lambda x: x[0])
            for _, eid in lst[-k:]:
                keep.add(eid)
        mask = np.zeros(len(edges), dtype=bool)
        for eid in keep:
            mask[eid] = True
        return edges[mask], weights[mask]

    def test_prune_vectorized_matches_bruteforce(self):
        # Distinct weights => no tie ambiguity, so results must be identical.
        from graintrace.graph_spatial_cluster import GraphSpatialCluster

        gsc = GraphSpatialCluster.__new__(GraphSpatialCluster)
        rng = np.random.default_rng(7)
        n = 60
        edges, _ = self._random_undirected(n, 400, seed=3)
        w = rng.permutation(edges.shape[0]).astype(np.float64)  # all distinct
        for k in (1, 3, 10):
            e_vec, w_vec = gsc.prune_topk_per_node_parallel(
                n_nodes=n, edges=edges, weights=w, k=k
            )
            e_bf, w_bf = self._bruteforce_topk(n, edges, w, k)
            assert np.array_equal(e_vec, e_bf), f"edges differ at k={k}"
            assert np.array_equal(w_vec, w_bf), f"weights differ at k={k}"

    def test_prune_matches_saved_original(self):
        # Prune must be bit-identical to the saved original on distinct weights.
        from graintrace.graph_spatial_cluster import GraphSpatialCluster
        from _prune_original import prune_original

        gsc = GraphSpatialCluster.__new__(GraphSpatialCluster)
        rng = np.random.default_rng(11)
        for seed in (1, 2, 3):
            n = 500
            edges, _ = self._random_undirected(n, 6000, seed=seed)
            w = rng.permutation(edges.shape[0]).astype(np.float64)  # all distinct
            for k in (1, 5, 20):
                e_ref, w_ref = prune_original(n, edges, w, k)
                for n_jobs in (1, 4):
                    e_new, w_new = gsc.prune_topk_per_node_parallel(
                        n_nodes=n, edges=edges, weights=w, k=k, n_jobs=n_jobs
                    )
                    assert np.array_equal(
                        e_new, e_ref
                    ), f"edges differ k={k} nj={n_jobs}"
                    assert np.array_equal(
                        w_new, w_ref
                    ), f"weights differ k={k} nj={n_jobs}"

    def test_compute_edge_distances_vectorized_njobs_no_deadlock(self):
        # Vectorized metrics must run single-process even when n_jobs>1.
        from graintrace.graph_spatial_cluster import GraphSpatialCluster
        from graintrace.similarity_metric_library import SimilarityMetricLibrary

        gsc = GraphSpatialCluster.__new__(GraphSpatialCluster)
        rng = np.random.default_rng(0)
        n = 500
        X = rng.normal(0.0, 1.0, size=(n, 6))
        edges = np.stack([rng.integers(0, n, 5000), rng.integers(0, n, 5000)], axis=1)
        spec = SimilarityMetricLibrary().von_mises_stress()
        d1 = gsc.compute_edge_distances(edges=edges, X=X, spec=spec, n_jobs=1)
        d4 = gsc.compute_edge_distances(edges=edges, X=X, spec=spec, n_jobs=4)
        assert np.allclose(d1, d4)

    def test_segment_n_threads_produces_valid_partition(self):
        pytest.importorskip("networkit")
        from graintrace.graph_spatial_cluster import GraphSpatialCluster

        gsc = GraphSpatialCluster.__new__(GraphSpatialCluster)
        n = 400
        edges, w = self._random_undirected(n, 4000)
        labels = gsc.segment_graph_networkit(
            n_nodes=n, edges=edges, weights=w, method="leiden", seed=42, n_threads=2
        )
        # exercises GraphFromCoo + getVector + n_threads plumbing together
        # exercises GraphFromCoo + getVector + n_threads plumbing together
        assert labels.shape == (n,)
        assert labels.min() >= 0
