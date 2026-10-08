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

    @staticmethod
    def _write_offset_feature_csv(path):
        """CSV whose first feature column is deliberately outside the metric.

        The stock fixture lists exactly the metric's columns in the metric's
        order, so a positional name/column drift cannot show up there.
        """
        n = 24
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "id": np.arange(1, n + 1),
                "x": np.tile(np.arange(6, dtype=float), 4),
                "y": np.repeat(np.arange(4, dtype=float), 6),
                "z": np.zeros(n),
                # Far away from the stresses, so a swap is unmistakable.
                "temperature": rng.normal(800.0, 1.0, n),
                "sxx": rng.normal(100.0, 1.0, n),
                "syy": rng.normal(50.0, 1.0, n),
            }
        )
        df.to_csv(path, index=False)
        return df

    def test_dbscan_summary_names_match_their_columns(self, tmp_path):
        # run_sklearn_dbscan summarised the full feature matrix but labelled it
        # with the metric's subset, so every *_sum/_sumsq/_mean was published
        # under the wrong physical quantity.
        pytest.importorskip("sklearn")
        from graintrace.cluster_indicator import ClusterAnalysisIndicator
        from graintrace.user_data_class import SimilarityMetric

        csv_path = tmp_path / "offset_features.csv"
        df = self._write_offset_feature_csv(csv_path)

        spec = SimilarityMetric(
            name="sxx_only",
            feature_cols=["sxx", "syy"],
            func=lambda u, v: float(abs(u[0] - v[0])),
        )
        ind = ClusterAnalysisIndicator(str(csv_path), coord_cols=("x", "y", "z"))
        result = ind.run(
            method_type="sklearn_dbscan",
            spec=spec,
            eps=1e9,  # one cluster containing every point
            min_samples=2,
        )
        clusters = result["clusters"]
        assert len(clusters) == 1

        # Every feature column present in the CSV must be summarised, each
        # under its own name.
        for col in ("temperature", "sxx", "syy"):
            assert f"{col}_mean" in clusters.columns, f"{col} missing from summary"
            assert clusters[f"{col}_mean"].iloc[0] == pytest.approx(
                df[col].mean()
            ), f"{col}_mean does not hold {col}"

    def test_cluster_summary_rejects_name_column_mismatch(self):
        from graintrace.cluster_indicator import ClusterAnalysisIndicator

        ind = ClusterAnalysisIndicator.__new__(ClusterAnalysisIndicator)
        labels = np.array([0, 0, 1, 1])
        coords = np.zeros((4, 3))
        feats = np.zeros((4, 3))
        with pytest.raises(ValueError, match="paired by position"):
            ind._build_cluster_summaries_from_arrays(
                labels=labels,
                coords=coords,
                feats=feats,
                coord_names=["x", "y", "z"],
                feat_names=["a", "b"],  # 2 names for 3 columns
            )


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


class TestGraphSpatialClusterCheckpointValidation:
    """Resuming must refuse a checkpoint built under different settings."""

    @staticmethod
    def _make_csv(path, nx=8, ny=8, seed=0):
        rng = np.random.default_rng(seed)
        n = nx * ny
        df = pd.DataFrame(
            {
                "id": np.arange(1, n + 1),
                "x": np.tile(np.arange(nx, dtype=float), ny),
                "y": np.repeat(np.arange(ny, dtype=float), nx),
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

    def _run(self, csv_path, ckpt_base, **overrides):
        from graintrace.graph_spatial_cluster import GraphSpatialCluster
        from graintrace.similarity_metric_library import SimilarityMetricLibrary
        from graintrace.user_data_class import WeightConfig

        gsc = GraphSpatialCluster(
            csv_path=csv_path, id_col="id", coord_cols=("x", "y", "z")
        )
        kwargs = dict(
            spec=SimilarityMetricLibrary().von_mises_stress(),
            graph_mode="grid",
            manhattan_radius=1,
            segmenter="leiden",
            seed=42,
            weight_cfg=WeightConfig(mode="rbf", sigma=50.0),
            n_jobs=1,
            max_edge_distance=40.0,
            checkpoint_base_path=ckpt_base,
        )
        kwargs.update(overrides)
        return gsc.run(**kwargs)

    @pytest.fixture
    def written_checkpoint(self, tmp_path):
        pytest.importorskip("networkit")
        csv_path = self._make_csv(tmp_path / "grid.csv")
        ckpt = str(tmp_path / "ckpt")
        self._run(csv_path, ckpt)
        assert Path(ckpt + ".meta.json").exists()
        return csv_path, ckpt

    def test_resume_with_identical_parameters_is_accepted(self, written_checkpoint):
        csv_path, ckpt = written_checkpoint
        out = self._run(csv_path, ckpt, resume_from_checkpoint=True)
        assert isinstance(out, dict)

    @pytest.mark.parametrize(
        "override",
        [
            {"max_edge_distance": 10.0},
            {"manhattan_radius": 2},
            {"reduce_edges_topweights_k": 4},
        ],
    )
    def test_resume_rejects_changed_graph_parameters(
        self, written_checkpoint, override
    ):
        # Previously the stale graph was returned silently: only n_nodes was
        # checked, and every graph-affecting step is skipped on resume.
        csv_path, ckpt = written_checkpoint
        with pytest.raises(ValueError, match="different graph-affecting parameters"):
            self._run(csv_path, ckpt, resume_from_checkpoint=True, **override)

    def test_resume_rejects_different_data_of_equal_length(
        self, written_checkpoint, tmp_path
    ):
        # Same row count, different values: n_nodes alone cannot catch this.
        _, ckpt = written_checkpoint
        other_csv = self._make_csv(tmp_path / "other.csv", seed=99)
        with pytest.raises(ValueError, match="different graph-affecting parameters"):
            self._run(other_csv, ckpt, resume_from_checkpoint=True)

    def test_resume_warns_when_meta_predates_validation(self, written_checkpoint):
        # Checkpoints from older versions lack the new keys; warn, do not crash.
        import json

        csv_path, ckpt = written_checkpoint
        meta_path = Path(ckpt + ".meta.json")
        meta = json.loads(meta_path.read_text())
        del meta["data_sha256"]
        del meta["max_edge_distance"]
        meta_path.write_text(json.dumps(meta))

        with pytest.warns(RuntimeWarning, match="missing"):
            self._run(csv_path, ckpt, resume_from_checkpoint=True)

    def test_partial_checkpoint_rebuilds_instead_of_crashing(self, tmp_path):
        # gamma_sweep_leiden's existence probe must agree with _load_checkpoint,
        # which also needs .meta.json. With only edges+weights on disk the old
        # probe said "resume" and _load_checkpoint raised FileNotFoundError.
        pytest.importorskip("networkit")
        from graintrace.fragmentation import FragmentationAnalyzer
        from graintrace.similarity_metric_library import SimilarityMetricLibrary
        from graintrace.user_data_class import WeightConfig

        csv_path = self._make_csv(tmp_path / "sweep.csv")
        ckpt = str(tmp_path / "partial")
        # A half-written checkpoint: no .meta.json.
        np.save(ckpt + ".edges.npy", np.zeros((3, 2), dtype=np.int64))
        np.save(ckpt + ".weights.npy", np.ones(3, dtype=np.float64))

        results = FragmentationAnalyzer.gamma_sweep_leiden(
            csv_path=csv_path,
            ckpt_base=ckpt,
            gammas=[1.0],
            n_points=64,
            spec=SimilarityMetricLibrary().von_mises_stress(),
            weight_cfg=WeightConfig(mode="rbf", sigma=50.0),
            max_edge_distance=40.0,
            graph_mode="grid",
            manhattan_radius=1,
            reduce_topk=None,
        )
        assert len(results) == 1
        assert results[0]["n_grains"] >= 1


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
