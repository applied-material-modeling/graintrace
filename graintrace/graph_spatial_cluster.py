from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import List, Optional, Tuple, Callable, Dict, Any
from .user_data_class import SimilarityMetric, WeightConfig
from tqdm import tqdm
import os
import json
import re


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

        # check essential columns
        required = [self.id_col, *self.coord_cols]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # if there are no other columns other than required, raise error
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

    ## RUN
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

            # Hard sanity checks (cheap, prevents silent corruption)
            if edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError(f"Bad checkpoint edges shape: {edges.shape}")
            if weights.ndim != 1 or weights.shape[0] != edges.shape[0]:
                raise ValueError("Checkpoint weights/edges length mismatch")
            if int(meta.get("n_nodes", -1)) != int(coords.shape[0]):
                raise ValueError(
                    "Checkpoint n_nodes does not match current CSV row count"
                )

            # Skip graph construction + weight computation below
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

    ##

    # graph clustering
    def segment_graph_networkit(
        self,
        n_nodes: int,
        edges: np.ndarray,
        weights: np.ndarray,
        method: str = "leiden",
        seed: int = 42,
        **networkit_kwargs: Any,
    ) -> np.ndarray:

        import networkit as nk

        if edges.shape[0] != weights.shape[0]:
            raise ValueError("edges and weights length mismatch")

        G = nk.Graph(n_nodes, weighted=True, directed=False)
        for (u, v), w in zip(edges, weights):
            G.addEdge(int(u), int(v), float(w))

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

        labels = np.fromiter(
            (part.subsetOf(i) for i in range(n_nodes)), dtype=np.int64, count=n_nodes
        )
        return labels

    # detect grid and build edges from that
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
        """
        Assumes full 3D grid population if grid detection passed.
        manhattan_radius r:
        r=1 -> 6 neighbors
        r=2 -> 24 neighbors
        r=3 -> 62 neighbors
        r=4 -> 124 neighbors
        ...
        Returns undirected edges as (E,2) with i<j.
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

            # Undirected: keep i<j only
            keep = i < j
            if np.any(keep):
                edges_chunks.append(np.stack([i[keep], j[keep]], axis=1))

        if not edges_chunks:
            return np.empty((0, 2), dtype=np.int64)

        edges = np.concatenate(edges_chunks, axis=0).astype(np.int64)
        return edges

    # if not grid, build edges via mutual kNN (edge (i,j) exists if i in kNN(j) and j in kNN(i))
    def _build_mutual_knn_edges(self, coords: np.ndarray, k: int) -> np.ndarray:

        if k < 1:
            raise ValueError("k must be >= 1")

        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=k + 1, workers=-1)
        nbrs = idx[:, 1:]  # (N,k)

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

    # parallel operations for edge weight computation
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

        if n_jobs is None or n_jobs < 1:
            n_jobs = 1

        if n_jobs == 1:
            return self._distances_worker((edges, X, spec))

        if mp_start_method is None:
            ctx = mp.get_context()
        else:
            ctx = mp.get_context(mp_start_method)
        tasks = (
            (edges[s:e], X, spec)
            for s in range(0, edges.shape[0], chunk_size)
            for e in [min(edges.shape[0], s + chunk_size)]
        )

        distances = np.empty(edges.shape[0], dtype=np.float64)
        off = 0
        total_edges = edges.shape[0]

        with ctx.Pool(processes=n_jobs) as pool:
            with tqdm(total=total_edges, desc="Edge distances", unit="edge") as pbar:
                for d_chunk in pool.imap(self._distances_worker, tasks, chunksize=1):
                    chunk_size_actual = d_chunk.size
                    distances[off : off + chunk_size_actual] = d_chunk
                    off += chunk_size_actual
                    pbar.update(chunk_size_actual)

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

    # in case too much edges, this allow to remove of weak edges via topk
    @staticmethod
    def _topk_nodes_worker(args) -> np.ndarray:
        indptr, adj_eid, adj_w, a, b, k = args
        kept_chunks = []

        for n in range(a, b):
            s = int(indptr[n])
            e = int(indptr[n + 1])
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
        nodes_chunk: int = 250_000,
        mp_start_method: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if k is None:
            return edges, weights
        k = int(k)
        if k < 1 or edges.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

        # ---- Build incidence locally ----
        E = edges.shape[0]
        u = edges[:, 0].astype(np.int64, copy=False)
        v = edges[:, 1].astype(np.int64, copy=False)
        w = weights.astype(np.float64, copy=False)

        node = np.concatenate([u, v], axis=0)  # (2E,)
        eid = np.concatenate(
            [np.arange(E, dtype=np.int64), np.arange(E, dtype=np.int64)],
            axis=0,
        )
        adj_w_half = np.concatenate([w, w], axis=0)  # half-edge weights (2E,)

        deg = np.bincount(node, minlength=n_nodes).astype(np.int64)
        indptr = np.empty(n_nodes + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(deg, out=indptr[1:])

        # Sort half-edges by node so each node's incidence is a contiguous slice
        order = np.argsort(node, kind="mergesort")  # stable; C-level
        adj_eid = eid[order]
        adj_w = adj_w_half[order]

        keep_edge = np.zeros(E, dtype=bool)

        if n_jobs is None or n_jobs < 2:
            kept = GraphSpatialCluster._topk_nodes_worker(
                (indptr, adj_eid, adj_w, 0, n_nodes, k)
            )
            if kept.size:
                keep_edge[kept] = True
        else:
            if mp_start_method is None:
                ctx = mp.get_context()
            else:
                ctx = mp.get_context(mp_start_method)
            tasks = []
            for a in range(0, n_nodes, nodes_chunk):
                b = min(n_nodes, a + nodes_chunk)
                tasks.append((indptr, adj_eid, adj_w, a, b, k))

            print("n_jobs:", n_jobs, "n_nodes:", n_nodes, "nodes_chunk:", nodes_chunk)

            with ctx.Pool(processes=n_jobs) as pool:
                with tqdm(
                    total=n_nodes, desc="Reducing edges from weights for nodes"
                ) as pbar:
                    for task, kept in zip(
                        tasks,
                        pool.imap(
                            GraphSpatialCluster._topk_nodes_worker, tasks, chunksize=1
                        ),
                    ):
                        a, b = task[3], task[4]
                        if kept.size:
                            keep_edge[kept] = True
                        pbar.update(b - a)

        return edges[keep_edge], weights[keep_edge]

    ## DEFINE HOW TO GET CLUSTER PROPERTIES / FEATURES
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

        # average every feature names
        for j, fname in enumerate(feature_names):
            sums = np.bincount(inv, weights=X[:, j], minlength=k)
            out[f"{fname}_mean"] = sums / n

        # other cluster properties can be added here as needed

        # norm of 3x3
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

            # per-point Frobenius norm of the 9-vector
            norm_per_point = np.linalg.norm(X[:, idx], axis=1)

            sums = np.bincount(inv, weights=norm_per_point, minlength=k)
            out[f"{prefix}_norm_mean"] = sums / n

        return pd.DataFrame(out)

    # other utility methods
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
        # base_path like "/path/to/run1_ckpt"
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
        # Fast, no compression. Best for very large arrays.
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
        edges = np.load(
            p["edges"], mmap_mode="r"
        )  # mmap avoids loading full file immediately
        weights = np.load(p["weights"], mmap_mode="r")
        with open(p["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        return edges, weights, meta
