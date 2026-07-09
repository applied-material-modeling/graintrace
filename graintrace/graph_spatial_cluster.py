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

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Callable, Dict, Any
from .user_data_class import SimilarityMetric, WeightConfig
from tqdm import tqdm
import os
import json
import re


# numba-accelerated top-k pruning (thread-parallel, no multiprocessing); falls back
# to a single-threaded numpy path (GraphSpatialCluster._topk_nodes_worker) if absent.
try:
    from numba import njit, prange, set_num_threads as _nb_set_threads

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is a declared dependency
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(parallel=True, nogil=True, cache=True)
    def _prune_topk_numba(indptr, order, w_half, k, keep_half):
        """Mark the top-k highest-weight half-edges per node into keep_half.

        order groups half-edges by node into contiguous segments; keep_half is
        bool[2E] in sorted-position space, so per-node segments never race.
        """
        n = indptr.shape[0] - 1
        for nidx in prange(n):
            s = indptr[nidx]
            e = indptr[nidx + 1]
            m = e - s
            if m <= 0:
                continue
            if m <= k:
                for j in range(s, e):
                    keep_half[j] = True
            else:
                seg = np.empty(m, dtype=np.float64)
                for j in range(m):
                    seg[j] = w_half[order[s + j]]
                idx = np.argsort(seg)  # ascending; top-k = last k
                for t in range(m - k, m):
                    keep_half[s + idx[t]] = True


class GraphSpatialCluster:

    def __init__(
        self,
        csv_path: str,
        id_col: str = "id",
        coord_cols: Tuple[str, str, str] = ("x", "y", "z"),
    ) -> None:

        self.csv_path: str = csv_path
        self.id_col: str = id_col
        self.coord_cols: Tuple[str, str, str] = coord_cols
        self.data: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Load CSV and populate data, features, coords."""
        if self.data is not None:
            return

        df = pd.read_csv(self.csv_path)

        required = [self.id_col, *self.coord_cols]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(df.columns) == len(required):
            raise ValueError("No feature columns found in the data.")

        self.data = df

    def check_feature_matrix(self, spec: SimilarityMetric) -> None:

        if self.data is None:
            self.load_data()

        df = self.data

        missing = [c for c in spec.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Metric '{spec.name}' requires missing columns: {missing}"
            )

    def run(
        self,
        spec: SimilarityMetric,
        graph_mode: str = "auto",  # "auto" | "knn" | "grid"
        k: int = 16,  # knn
        manhattan_radius: int = 1,  # manhattan radius for grid connectivity if graph_mode="grid"
        grid_tol: float = 1e-6,
        n_jobs: int = 1,
        weight_chunk_size: int = 1_000_000,
        nodes_chunk: int = 250_000,
        segmenter: str = "auto",  # "auto" | "leiden" | "plm"
        seed: int = 42,
        n_threads: int = 1,  # NetworKit/Leiden thread count; ~physical cores is optimal
        feature_names: Optional[List[str]] = None,
        output_csv_path: Optional[str] = None,
        return_labels: bool = False,
        reduce_edges_topweights_k: Optional[
            int
        ] = None,  # if not None, keep only top k edges per node by weight before clustering
        max_edge_distance: Optional[
            float
        ] = None,  # if not None, remove edges with distances below threshold
        weight_cfg: WeightConfig = WeightConfig(mode="inverse", eps=1e-8),
        networkit_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_base_path: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        mp_start_method: Optional[str] = None,
    ) -> Dict[str, Any]:

        print("\n=== Running GraphSpatialCluster ===\n")

        self.load_data()
        self.check_feature_matrix(spec)

        print("Data loaded.\n")

        df = self.data
        coords = df[list(self.coord_cols)].to_numpy(dtype=np.float64)
        X = df[spec.feature_cols].to_numpy(dtype=np.float64)

        if resume_from_checkpoint:
            if checkpoint_base_path is None:
                raise ValueError(
                    "resume_from_checkpoint=True requires checkpoint_base_path"
                )

            print(f"Resuming from checkpoint: {checkpoint_base_path}")
            edges_ck, weights_ck, meta = self._load_checkpoint(checkpoint_base_path)

            edges = np.asarray(edges_ck)  # still mmapped underneath
            weights = np.asarray(weights_ck)

            if edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError(f"Bad checkpoint edges shape: {edges.shape}")
            if weights.ndim != 1 or weights.shape[0] != edges.shape[0]:
                raise ValueError("Checkpoint weights/edges length mismatch")
            if int(meta.get("n_nodes", -1)) != int(coords.shape[0]):
                raise ValueError(
                    "Checkpoint n_nodes does not match current CSV row count"
                )

            mode = "checkpoint"
        else:
            mode = graph_mode.lower()
            if mode == "auto":
                mode = "grid" if self._detect_grid(coords, tol=grid_tol) else "knn"
                print("Graph mode auto-detected as:", mode)
            elif mode not in ("knn", "grid"):
                raise ValueError("graph_mode must be one of {'auto','knn','grid'}")

            print("Building graph with mode:", mode)

            if mode == "grid":
                edges = self._build_grid_edges(
                    coords, manhattan_radius=manhattan_radius, tol=grid_tol
                )
            else:
                edges = self._build_mutual_knn_edges(coords, k=k)

            print(
                f"Graph is built. Number of edges: {edges.shape[0]}, "
                f"Number of nodes: {coords.shape[0]},"
                f" Number of features: {X.shape[1]}\n"
            )

            print("Computing edge distances with metric:", spec.name)
            distances = self.compute_edge_distances(
                edges=edges,
                X=X,
                spec=spec,
                n_jobs=n_jobs,
                chunk_size=weight_chunk_size,
                mp_start_method=mp_start_method,
            )

            if max_edge_distance is not None:
                max_edge_distance = float(max_edge_distance)
                if max_edge_distance <= 0:
                    raise ValueError("max_edge_distance must be > 0")

                keep = distances <= max_edge_distance
                edges = edges[keep]
                distances = distances[keep]

                print(f"\nKeeping edges with distance <= {max_edge_distance}")
                print(f"Updated number of edges: {edges.shape[0]}\n")

            if weight_cfg.mode.lower() in ("rbf", "exp") and weight_cfg.sigma is None:
                if distances.shape[0] == 0:
                    raise ValueError(
                        "No edges remain after max_edge_distance filtering; cannot estimate sigma."
                    )
                sigma = self.estimate_sigma_from_distances(
                    distances=distances,
                    quantile=weight_cfg.sigma_auto["quantile"],
                )
                weight_cfg = WeightConfig(**{**weight_cfg.__dict__, "sigma": sigma})
                print(f"Estimated sigma for weight function: {sigma}\n")

            print("Converting edge distances to weights")
            weights = self.distances_to_weights(distances, weight_cfg)

            if reduce_edges_topweights_k is not None:
                print(
                    f"\nRemoving edges to keep only top {reduce_edges_topweights_k} weights per node"
                )
                edges, weights = self.prune_topk_per_node_parallel(
                    n_nodes=coords.shape[0],
                    edges=edges,
                    weights=weights,
                    k=reduce_edges_topweights_k,
                    n_jobs=n_jobs,
                    nodes_chunk=nodes_chunk,
                    mp_start_method=mp_start_method,
                )
                print(f"Updated number of edges: {edges.shape[0]}\n")

        if (checkpoint_base_path is not None) and (not resume_from_checkpoint):
            meta = {
                "n_nodes": int(coords.shape[0]),
                "n_edges": int(edges.shape[0]),
                "metric": spec.name,
                "weight_mode": weight_cfg.mode,
                "segmenter": segmenter,
                "reduced_topk": (
                    int(reduce_edges_topweights_k)
                    if reduce_edges_topweights_k is not None
                    else None
                ),
                "graph_mode": mode,
                "k": int(k),
                "manhattan_radius": int(manhattan_radius),
                "grid_tol": float(grid_tol),
                "max_edge_distance": (
                    float(max_edge_distance) if max_edge_distance is not None else None
                ),
            }
            print(f"Saving checkpoint: {checkpoint_base_path} (edges/weights/meta)")
            self._save_checkpoint(
                checkpoint_base_path, edges=edges, weights=weights, meta=meta
            )

        print("\nSegmenting graph with method:", segmenter)
        print(
            "Segmenter parameters:", networkit_kwargs if networkit_kwargs else "default"
        )

        if networkit_kwargs is None:
            networkit_kwargs = {}

        labels = self.segment_graph_networkit(
            n_nodes=coords.shape[0],
            edges=edges,
            weights=weights,
            method=segmenter,
            seed=seed,
            n_threads=n_threads,
            **networkit_kwargs,
        )

        if feature_names is None:
            feature_names = [
                c
                for c in df.columns
                if c not in ([self.id_col] + list(self.coord_cols))
            ]
        else:
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                raise ValueError(f"feature_names contains missing columns: {missing}")

        X = df[feature_names].to_numpy(dtype=float)

        clusters = self.get_cluster_properties(
            labels=labels,
            coords=coords,
            X=X,
            feature_names=feature_names,
        )

        csv_path = None
        if output_csv_path is not None:
            clusters.to_csv(output_csv_path, index=False)
            csv_path = output_csv_path

        extras = {
            "n_nodes": int(coords.shape[0]),
            "n_edges": int(edges.shape[0]),
            "n_clusters": int(clusters.shape[0]),
            "metric": spec.name,
            "segmenter": segmenter,
        }

        if return_labels:
            extras["labels"] = labels

        return {
            "clusters": clusters,
            "csv_path": csv_path,
            "extras": extras,
        }

    def segment_graph_networkit(
        self,
        n_nodes: int,
        edges: np.ndarray,
        weights: np.ndarray,
        method: str = "leiden",
        seed: int = 42,
        n_threads: int = 1,
        **networkit_kwargs: Any,
    ) -> np.ndarray:

        import networkit as nk

        if edges.shape[0] != weights.shape[0]:
            raise ValueError("edges and weights length mismatch")

        # Set NetworKit thread count explicitly; cap at machine max (oversubscribing
        # past physical cores regresses).
        if n_threads is not None and int(n_threads) >= 1:
            set_threads = min(int(n_threads), nk.getMaxNumberOfThreads())
            nk.setNumberOfThreads(set_threads)
            print(
                f"  NetworKit Leiden threads set to {set_threads} "
                f"(machine max {nk.getMaxNumberOfThreads()})"
            )

        # Build the graph from COO arrays in one vectorized C++ call; fall back to a
        # per-edge addEdge loop on older NetworKit without GraphFromCoo.
        if hasattr(nk, "GraphFromCoo"):
            row = np.ascontiguousarray(edges[:, 0], dtype=np.uint64)
            col = np.ascontiguousarray(edges[:, 1], dtype=np.uint64)
            w = np.ascontiguousarray(weights, dtype=np.float64)
            G = nk.GraphFromCoo(
                (w, (row, col)), n=n_nodes, weighted=True, directed=False
            )
        else:
            G = nk.Graph(n_nodes, weighted=True, directed=False)
            for (u, v), wt in zip(edges, weights):
                G.addEdge(int(u), int(v), float(wt))

        nk.setSeed(seed, True)

        chosen = method.lower()

        if chosen == "leiden":
            if not hasattr(nk.community, "ParallelLeiden"):
                raise ValueError(
                    "NetworKit ParallelLeiden not available in this install."
                )
            algo = nk.community.ParallelLeiden(G, **networkit_kwargs)
        elif chosen == "plm":
            algo = nk.community.PLM(G, **networkit_kwargs)
        elif chosen == "plp":
            algo = nk.community.PLP(G, **networkit_kwargs)
        else:
            raise ValueError("method must be one of {'leiden','plm','plp'}")

        algo.run()
        part = algo.getPartition()

        # getVector() returns the full subset-id array in one C++ call.
        try:
            labels = np.asarray(part.getVector(), dtype=np.int64)
        except AttributeError:
            labels = np.fromiter(
                (part.subsetOf(i) for i in range(n_nodes)),
                dtype=np.int64,
                count=n_nodes,
            )
        return labels

    @staticmethod
    def _is_regular_1d_grid(vals: np.ndarray, tol: float) -> bool:
        u = np.unique(vals)
        if u.size < 3:
            return False
        du = np.diff(u)
        step = np.median(du)
        if step <= 0:
            return False
        return np.max(np.abs(du - step)) <= tol * max(1.0, abs(step))

    def _detect_grid(self, coords: np.ndarray, tol: float = 1e-6) -> bool:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        if not (
            self._is_regular_1d_grid(x, tol)
            and self._is_regular_1d_grid(y, tol)
            and self._is_regular_1d_grid(z, tol)
        ):
            return False
        ux, uy, uz = np.unique(x), np.unique(y), np.unique(z)
        return (ux.size * uy.size * uz.size) == coords.shape[0]

    def _build_grid_edges(
        self,
        coords: np.ndarray,
        manhattan_radius: int = 1,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Build grid edges (undirected, (E,2) with i<j) for a full 3D grid.

        manhattan_radius r gives 6/24/62/124/... neighbors for r=1/2/3/4.
        """
        if manhattan_radius < 1:
            raise ValueError("manhattan_radius must be >= 1")

        def unique_with_tolerance(arr, tol):
            sorted_arr = np.sort(arr)
            diffs = np.diff(sorted_arr)
            mask = np.append(True, diffs > tol)
            return sorted_arr[mask]

        xs = unique_with_tolerance(coords[:, 0], tol)
        ys = unique_with_tolerance(coords[:, 1], tol)
        zs = unique_with_tolerance(coords[:, 2], tol)

        def map_to_bins(vals, uniques):
            pos = np.searchsorted(uniques, vals)
            pos = np.clip(pos, 0, uniques.size - 1)
            left = np.maximum(pos - 1, 0)
            choose_left = np.abs(vals - uniques[left]) <= np.abs(vals - uniques[pos])
            return np.where(choose_left, left, pos).astype(np.int64)

        ix = map_to_bins(coords[:, 0], xs)
        iy = map_to_bins(coords[:, 1], ys)
        iz = map_to_bins(coords[:, 2], zs)

        nx, ny, nz = xs.size, ys.size, zs.size
        lin = ix + nx * (iy + ny * iz)

        inv = np.empty(nx * ny * nz, dtype=np.int64)
        inv[lin] = np.arange(coords.shape[0], dtype=np.int64)

        r = manhattan_radius
        offsets = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx == dy == dz == 0:
                        continue
                    if abs(dx) + abs(dy) + abs(dz) <= r:
                        offsets.append((dx, dy, dz))

        edges_chunks = []

        for dx, dy, dz in tqdm(offsets, desc="Building grid edges per node"):
            ix2 = ix + dx
            iy2 = iy + dy
            iz2 = iz + dz

            mask = (
                (ix2 >= 0)
                & (ix2 < nx)
                & (iy2 >= 0)
                & (iy2 < ny)
                & (iz2 >= 0)
                & (iz2 < nz)
            )
            if not np.any(mask):
                continue

            lin2 = ix2[mask] + nx * (iy2[mask] + ny * iz2[mask])
            j = inv[lin2]
            i = np.nonzero(mask)[0].astype(np.int64)

            keep = i < j  # undirected: keep i<j only
            if np.any(keep):
                edges_chunks.append(np.stack([i[keep], j[keep]], axis=1))

        if not edges_chunks:
            return np.empty((0, 2), dtype=np.int64)

        edges = np.concatenate(edges_chunks, axis=0).astype(np.int64)
        return edges

    # build edges via mutual kNN: edge (i,j) exists if i in kNN(j) and j in kNN(i)
    def _build_mutual_knn_edges(self, coords: np.ndarray, k: int) -> np.ndarray:

        if k < 1:
            raise ValueError("k must be >= 1")

        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=k + 1, workers=-1)
        nbrs = idx[:, 1:]

        N = coords.shape[0]
        src = np.repeat(np.arange(N, dtype=np.int64), k)
        dst = nbrs.reshape(-1).astype(np.int64)

        nbrs_sorted = np.sort(nbrs, axis=1)

        keep = np.zeros(src.shape[0], dtype=bool)
        order = np.argsort(dst, kind="mergesort")
        src_s = src[order]
        dst_s = dst[order]

        starts = np.flatnonzero(np.r_[True, dst_s[1:] != dst_s[:-1]])
        ends = np.r_[starts[1:], dst_s.size]

        for a, b in tqdm(
            zip(starts, ends), total=starts.size, desc="Building edges from nodes"
        ):
            j = int(dst_s[a])
            block = src_s[a:b]
            row = nbrs_sorted[j]
            pos = np.searchsorted(row, block)
            ok = np.zeros(block.shape[0], dtype=bool)
            m = pos < row.size
            ok[m] = row[pos[m]] == block[m]
            keep[order[a:b]] = ok

        keep = keep & (src < dst)

        u = src[keep]
        v = dst[keep]
        edges = np.stack([u, v], axis=1)

        return edges

    @staticmethod
    def _dist_to_weight(d: float, cfg: WeightConfig) -> float:
        m = cfg.mode.lower()
        if m == "inverse":
            return 1.0 / (d + cfg.eps)
        if m == "rbf":
            if cfg.sigma <= 0:
                raise ValueError("sigma must be > 0 for rbf")
            x = d / cfg.sigma
            return float(np.exp(-(x**cfg.power)))
        if m == "exp":
            if cfg.sigma <= 0:
                raise ValueError("sigma must be > 0 for exp")
            return float(np.exp(-(d / cfg.sigma)))
        if m == "log_inv":
            return float(-np.log(d + cfg.eps))
        raise ValueError(f"Unknown weight mode: {cfg.mode}")

    @staticmethod
    def _dist_to_weight_vec(d: np.ndarray, cfg: WeightConfig) -> np.ndarray:
        m = cfg.mode.lower()
        if m == "inverse":
            return 1.0 / (d + cfg.eps)
        if m == "rbf":
            if cfg.sigma <= 0:
                raise ValueError("sigma must be > 0 for rbf")
            x = d / cfg.sigma
            return np.exp(-(x**cfg.power))
        if m == "exp":
            if cfg.sigma <= 0:
                raise ValueError("sigma must be > 0 for exp")
            return np.exp(-(d / cfg.sigma))
        if m == "log_inv":
            return -np.log(d + cfg.eps)
        raise ValueError(f"Unknown weight mode: {cfg.mode}")

    @staticmethod
    def _distances_worker(
        args: Tuple[np.ndarray, np.ndarray, SimilarityMetric],
    ) -> np.ndarray:
        edges_chunk, X, spec = args

        dist_edges = getattr(spec, "dist_edges", None)
        if dist_edges is not None:
            d = np.asarray(dist_edges(X, edges_chunk), dtype=np.float64)
            if d.shape != (edges_chunk.shape[0],):
                d = d.reshape(-1)
            if d.shape != (edges_chunk.shape[0],):
                raise ValueError(
                    f"dist_edges must return shape ({edges_chunk.shape[0]},), got {d.shape}"
                )
            return d.astype(np.float64, copy=False)

        d = np.empty(edges_chunk.shape[0], dtype=np.float64)
        for t, (i, j) in enumerate(edges_chunk):
            d[t] = float(spec.func(X[i], X[j]))
        return d

    def compute_edge_distances(
        self,
        edges: np.ndarray,
        X: np.ndarray,
        spec: SimilarityMetric,
        n_jobs: int = 1,
        chunk_size: int = 1_000_000,
        mp_start_method: Optional[str] = None,
    ) -> np.ndarray:

        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must be shape (E,2)")
        if edges.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)

        # Single-process, chunked (no multiprocessing); built-in metrics are
        # vectorized. n_jobs / mp_start_method kept for backward compat only.
        distances = np.empty(edges.shape[0], dtype=np.float64)
        total_edges = edges.shape[0]
        with tqdm(total=total_edges, desc="Edge distances", unit="edge") as pbar:
            for s in range(0, total_edges, chunk_size):
                e = min(total_edges, s + chunk_size)
                distances[s:e] = self._distances_worker((edges[s:e], X, spec))
                pbar.update(e - s)
        return distances

    @staticmethod
    def distances_to_weights(
        distances: np.ndarray,
        weight_cfg: WeightConfig,
    ) -> np.ndarray:
        return GraphSpatialCluster._dist_to_weight_vec(
            np.asarray(distances, dtype=np.float64),
            weight_cfg,
        ).astype(np.float64, copy=False)

    @staticmethod
    def estimate_sigma_from_distances(
        distances: np.ndarray,
        quantile: float = 0.5,
    ) -> float:
        if distances.shape[0] == 0:
            raise ValueError("No distances available to estimate sigma from.")
        sigma = float(np.quantile(distances, quantile))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Bad sigma estimate: {sigma}")
        return sigma

    def compute_edge_weights(
        self,
        edges: np.ndarray,
        X: np.ndarray,
        spec: SimilarityMetric,
        weight_cfg: WeightConfig,
        n_jobs: int = 1,
        chunk_size: int = 1_000_000,
    ) -> np.ndarray:
        distances = self.compute_edge_distances(
            edges=edges,
            X=X,
            spec=spec,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )
        return self.distances_to_weights(distances, weight_cfg)

    # drop weak edges via top-k per node
    @staticmethod
    def _topk_nodes_worker(args) -> np.ndarray:
        indptr, adj_eid, adj_w, a, b, k = args
        kept_chunks = []
        for nidx in range(a, b):
            s = int(indptr[nidx])
            e = int(indptr[nidx + 1])
            m = e - s
            if m <= 0:
                continue
            if m <= k:
                kept_chunks.append(adj_eid[s:e])
            else:
                w_slice = adj_w[s:e]
                idx = np.argpartition(w_slice, -k)[-k:]
                kept_chunks.append(adj_eid[s:e][idx])
        if kept_chunks:
            return np.concatenate(kept_chunks)
        return np.empty((0,), dtype=np.int64)

    @staticmethod
    def prune_topk_per_node_parallel(
        n_nodes: int,
        edges: np.ndarray,
        weights: np.ndarray,
        k: Optional[int],
        n_jobs: int = 1,
        nodes_chunk: int = 250_000,  # accepted for backward compat; ignored
        mp_start_method: Optional[str] = None,  # accepted for backward compat; ignored
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Keep each node's k highest-weight incident edges (union over endpoints).

        Uses a numba thread-parallel kernel when available (n_jobs = thread count),
        else a single-threaded numpy fallback. nodes_chunk / mp_start_method are
        accepted for backward compat only and have no effect.
        """
        if k is None:
            return edges, weights
        k = int(k)
        if k < 1 or edges.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

        # Build incidence (one half-edge per endpoint). Build half-edge weights
        # after the sort and use quicksort (no merge buffer) to keep peak memory low.
        E = edges.shape[0]
        node = np.concatenate(
            [
                edges[:, 0].astype(np.int64, copy=False),
                edges[:, 1].astype(np.int64, copy=False),
            ]
        )  # (2E,)

        deg = np.bincount(node, minlength=n_nodes).astype(np.int64)
        indptr = np.empty(n_nodes + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(deg, out=indptr[1:])

        # group half-edges by node so each node's incidence is a contiguous slice
        order = np.argsort(node, kind="quicksort")
        del node  # free before the top-k pass

        adj_w_half = np.concatenate(
            [
                weights.astype(np.float64, copy=False),
                weights.astype(np.float64, copy=False),
            ]
        )  # (2E,)

        keep_edge = np.zeros(E, dtype=bool)

        if _HAS_NUMBA:
            if n_jobs is not None and int(n_jobs) >= 1:
                _nb_set_threads(int(n_jobs))
            keep_half = np.zeros(2 * E, dtype=np.bool_)
            _prune_topk_numba(indptr, order, adj_w_half, k, keep_half)
            kept_global = order[keep_half]
            # half-edge global index -> edge id: first half [0,E), second [E,2E)
            kept_eids = np.where(kept_global >= E, kept_global - E, kept_global)
            keep_edge[kept_eids] = True
        else:
            # single-threaded numpy fallback
            adj_eid = np.concatenate(
                [np.arange(E, dtype=np.int64), np.arange(E, dtype=np.int64)]
            )[order]
            adj_w = adj_w_half[order]
            kept = GraphSpatialCluster._topk_nodes_worker(
                (indptr, adj_eid, adj_w, 0, n_nodes, k)
            )
            if kept.size:
                keep_edge[kept] = True

        return edges[keep_edge], weights[keep_edge]

    def get_cluster_properties(
        self,
        labels: np.ndarray,
        coords: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
    ) -> pd.DataFrame:

        if labels.ndim != 1:
            labels = labels.ravel()

        uniq, inv = np.unique(labels, return_inverse=True)
        k = uniq.size
        n = np.bincount(inv, minlength=k).astype(np.int64)

        # coordinate means
        cx = np.bincount(inv, weights=coords[:, 0], minlength=k) / n
        cy = np.bincount(inv, weights=coords[:, 1], minlength=k) / n
        cz = np.bincount(inv, weights=coords[:, 2], minlength=k) / n

        out = {
            "cluster_id": uniq.astype(np.int64),
            "n": n,
            "x": cx,
            "y": cy,
            "z": cz,
        }

        # average every feature
        for j, fname in enumerate(feature_names):
            sums = np.bincount(inv, weights=X[:, j], minlength=k)
            out[f"{fname}_mean"] = sums / n

        # Frobenius norm of any full 3x3 tensor among the features
        ij_set = {"11", "12", "13", "21", "22", "23", "31", "32", "33"}
        pat = re.compile(r"^(?P<prefix>.+)_(?P<ij>[123]{2})$")

        # prefix -> {ij: column_index}
        tensors: Dict[str, Dict[str, int]] = {}

        for j, name in enumerate(feature_names):
            m = pat.match(name)
            if not m:
                continue
            ij = m.group("ij")
            if ij not in ij_set:
                continue
            prefix = m.group("prefix")
            tensors.setdefault(prefix, {})[ij] = j

        full_prefixes = [p for p, mp in tensors.items() if len(mp) == 9]

        for prefix in full_prefixes:
            idx = [
                tensors[prefix][ij]
                for ij in ("11", "12", "13", "21", "22", "23", "31", "32", "33")
            ]

            norm_per_point = np.linalg.norm(X[:, idx], axis=1)

            sums = np.bincount(inv, weights=norm_per_point, minlength=k)
            out[f"{prefix}_norm_mean"] = sums / n

        return pd.DataFrame(out)

    def estimate_sigma_from_sampled_edges(
        self,
        edges: np.ndarray,
        X: np.ndarray,
        spec: SimilarityMetric,
        sample_size: int = 200_000,
        quantile: float = 0.5,  # 0.5 = median
        seed: int = 0,
    ) -> float:
        if edges.shape[0] == 0:
            raise ValueError("No edges to estimate sigma from.")

        rng = np.random.default_rng(seed)
        m = min(sample_size, edges.shape[0])
        idx = rng.choice(edges.shape[0], size=m, replace=False)

        ds = np.empty(m, dtype=np.float64)
        for t, (i, j) in enumerate(edges[idx]):
            ii = int(i)
            jj = int(j)
            ds[t] = float(spec.func(X[ii], X[jj]))

        sigma = float(np.quantile(ds, quantile))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Bad sigma estimate: {sigma}")
        return sigma

    @staticmethod
    def _ckpt_paths(base_path: str) -> Dict[str, str]:
        return {
            "edges": base_path + ".edges.npy",
            "weights": base_path + ".weights.npy",
            "meta": base_path + ".meta.json",
        }

    @staticmethod
    def _save_checkpoint(
        base_path: str, *, edges: np.ndarray, weights: np.ndarray, meta: Dict[str, Any]
    ) -> None:
        os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
        p = GraphSpatialCluster._ckpt_paths(base_path)
        np.save(p["edges"], edges.astype(np.int64, copy=False), allow_pickle=False)
        np.save(
            p["weights"], weights.astype(np.float64, copy=False), allow_pickle=False
        )
        with open(p["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @staticmethod
    def _load_checkpoint(
        base_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        p = GraphSpatialCluster._ckpt_paths(base_path)
        if not (
            os.path.exists(p["edges"])
            and os.path.exists(p["weights"])
            and os.path.exists(p["meta"])
        ):
            raise FileNotFoundError(
                f"Checkpoint files not found for base_path={base_path}"
            )
        edges = np.load(p["edges"], mmap_mode="r")  # mmap avoids full load
        weights = np.load(p["weights"], mmap_mode="r")
        with open(p["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        return edges, weights, meta
