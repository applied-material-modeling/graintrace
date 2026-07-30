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

"""Compare two rare-event-identification point clouds for spatial overlap."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .rare_cluster_indicator import IdentifyRareClusters

SpacingLike = Optional[Union[float, int, Sequence[float]]]


class REIComparison:
    """Compare two rare-event-identification (REI) point clouds.

    Each input CSV is a voxelized point cloud of a rare region on a regular grid
    (columns ``coord_cols`` plus an optional integer ``cluster_col``). The two
    grids may have *different* spacings but are assumed to share an origin (no
    rotation / translation is applied here).

    A rare point is treated as the center of its voxel cube, so each REI is a
    union of axis-aligned cubes and the overlap is a boolean volume
    intersection. Both regions are resampled onto a common finer lattice
    (``s_ref = min(spacing_1, spacing_2)`` per axis); membership is then an O(1)
    integer-index hash lookup. Carrying ``cluster_col`` through the same pass
    yields a per-cluster overlap matrix used for a 1-to-1 (Hungarian) cluster
    correspondence.

    Outputs written to ``output_dir``:
      * ``overlap_metrics.json`` -- IoU/Dice/containment + counts/volumes
      * ``overlap_cloud.vtk``    -- classified point cloud
                                    (membership 1=only-1, 2=only-2, 3=both)
      * ``cluster_match.csv``    -- 1-to-1 cluster pairing (if ``cluster_col``)
    """

    def __init__(
        self,
        rei_csv_1: str,
        rei_csv_2: str,
        output_dir: str,
        spacing_1: SpacingLike = None,
        spacing_2: SpacingLike = None,
        coord_cols: Tuple[str, str, str] = ("x", "y", "z"),
        cluster_col: Optional[str] = "rare_cluster_id",
        origin: Optional[Sequence[float]] = None,
        s_ref: SpacingLike = None,
        supersample: int = 1,
        split_merge_fraction: float = 0.2,
    ) -> None:
        if not os.path.exists(rei_csv_1):
            raise FileNotFoundError(f"REI CSV 1 not found: {rei_csv_1}")
        if not os.path.exists(rei_csv_2):
            raise FileNotFoundError(f"REI CSV 2 not found: {rei_csv_2}")

        if len(coord_cols) != 3:
            raise ValueError("coord_cols must be a 3-tuple (x, y, z column names).")
        if int(supersample) < 1:
            raise ValueError("supersample must be a positive integer (>= 1).")

        required = set(coord_cols)
        if cluster_col is not None:
            required.add(cluster_col)
        for path, label in [(rei_csv_1, "rei_csv_1"), (rei_csv_2, "rei_csv_2")]:
            try:
                cols = set(pd.read_csv(path, nrows=0).columns)
            except Exception as exc:  # pragma: no cover - passthrough of pandas error
                raise ValueError(f"Failed to read {label}: {exc}") from exc
            missing = required - cols
            if missing:
                raise ValueError(
                    f"{label} missing columns: {', '.join(sorted(missing))}"
                )

        self.rei_csv_1 = rei_csv_1
        self.rei_csv_2 = rei_csv_2
        self.output_dir = os.path.abspath(output_dir)
        self.coord_cols = tuple(coord_cols)
        self.cluster_col = cluster_col
        self.supersample = int(supersample)
        self.split_merge_fraction = float(split_merge_fraction)

        self._spacing_1_in = spacing_1
        self._spacing_2_in = spacing_2
        self._s_ref_in = s_ref
        self._origin_in = None if origin is None else np.asarray(origin, dtype=float)

        os.makedirs(self.output_dir, exist_ok=True)

        # populated by run_comparison()
        self.df_1: Optional[pd.DataFrame] = None
        self.df_2: Optional[pd.DataFrame] = None
        self.spacing_1: Optional[np.ndarray] = None
        self.spacing_2: Optional[np.ndarray] = None
        self.s_ref: Optional[np.ndarray] = None
        self.origin: Optional[np.ndarray] = None
        self.metrics: Dict[str, Any] = {}
        self.cluster_match: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ public

    def run_comparison(self) -> Dict[str, Any]:
        """Run the full comparison, write outputs, and return the metrics dict."""
        self._load_data()
        self._resolve_geometry()

        coords_1 = self.df_1[list(self.coord_cols)].to_numpy(dtype=np.float64)
        coords_2 = self.df_2[list(self.coord_cols)].to_numpy(dtype=np.float64)
        cid_1 = self._cluster_ids(self.df_1, coords_1.shape[0])
        cid_2 = self._cluster_ids(self.df_2, coords_2.shape[0])

        # occupied cells in each grid's own integer index space (nearest center)
        cells_1, cid_cells_1 = self._occupied_cells(
            coords_1, cid_1, self.origin, self.spacing_1
        )
        cells_2, cid_cells_2 = self._occupied_cells(
            coords_2, cid_2, self.origin, self.spacing_2
        )

        # rasterize both regions onto the common fine lattice
        fine_1, fcid_1 = self._rasterize_to_fine(
            cells_1, cid_cells_1, self.spacing_1, self.s_ref
        )
        fine_2, fcid_2 = self._rasterize_to_fine(
            cells_2, cid_cells_2, self.spacing_2, self.s_ref
        )

        result = self._compare_fine(fine_1, fcid_1, fine_2, fcid_2)
        return result

    # ------------------------------------------------------------------ loading

    def _load_data(self) -> None:
        self.df_1 = pd.read_csv(self.rei_csv_1)
        self.df_2 = pd.read_csv(self.rei_csv_2)
        print(
            f"Loaded {len(self.df_1)} points (REI 1) and "
            f"{len(self.df_2)} points (REI 2)."
        )

    def _cluster_ids(self, df: pd.DataFrame, n: int) -> np.ndarray:
        if self.cluster_col is None:
            return np.ones(n, dtype=np.int64)
        return (
            pd.to_numeric(df[self.cluster_col], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

    def _resolve_geometry(self) -> None:
        coords_1 = self.df_1[list(self.coord_cols)].to_numpy(dtype=np.float64)
        coords_2 = self.df_2[list(self.coord_cols)].to_numpy(dtype=np.float64)

        self.spacing_1 = self._resolve_spacing(self._spacing_1_in, coords_1, "REI 1")
        self.spacing_2 = self._resolve_spacing(self._spacing_2_in, coords_2, "REI 2")

        if self._s_ref_in is not None:
            s_ref = self._as_vec3(self._s_ref_in)
        else:
            s_ref = np.minimum(self.spacing_1, self.spacing_2)
        self.s_ref = s_ref / float(self.supersample)

        if self._origin_in is not None:
            self.origin = self._origin_in.astype(float)
        else:
            self.origin = np.minimum(coords_1.min(axis=0), coords_2.min(axis=0))

        print(
            "Grid geometry:\n"
            f"  spacing_1 = {self.spacing_1}\n"
            f"  spacing_2 = {self.spacing_2}\n"
            f"  s_ref     = {self.s_ref} (supersample={self.supersample})\n"
            f"  origin    = {self.origin}"
        )

    def _resolve_spacing(
        self, spacing_in: SpacingLike, coords: np.ndarray, label: str
    ) -> np.ndarray:
        if spacing_in is not None:
            return self._as_vec3(spacing_in)
        spacing = np.empty(3, dtype=float)
        for ax in range(3):
            s = self._infer_spacing_axis(coords[:, ax])
            spacing[ax] = 1.0 if s is None else s
        print(f"  auto-detected spacing for {label}: {spacing}")
        return spacing

    # ------------------------------------------------------------- voxelization

    @staticmethod
    def _as_vec3(val: Union[float, int, Sequence[float]]) -> np.ndarray:
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            arr = np.full(3, float(arr))
        if arr.shape != (3,):
            raise ValueError("spacing must be a scalar or a length-3 sequence.")
        if np.any(arr <= 0):
            raise ValueError("spacing values must be positive.")
        return arr

    @staticmethod
    def _infer_spacing_axis(vals: np.ndarray, tol: float = 1e-4) -> Optional[float]:
        u = np.unique(vals)
        if u.size < 2:
            return None  # degenerate axis (single layer)
        du = np.diff(u)
        step = float(du.min())
        if step <= 0:
            return None
        ratios = du / step
        if np.max(np.abs(ratios - np.round(ratios))) > tol * max(1.0, ratios.max()):
            print(
                "  [warning] coordinate spacing is not perfectly regular; "
                f"using min positive step = {step:.6g}. "
                "Pass an explicit spacing if this is wrong."
            )
        return step

    @staticmethod
    def _unique_rows(idx: np.ndarray) -> np.ndarray:
        """First-occurrence indices of unique integer rows (via 1-D key encoding).

        Faster than ``np.unique(axis=0)`` on large arrays: encode each (i,j,k)
        into a single int64 key over the array's own bounding box, then unique.
        """
        if idx.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        mn = idx.min(axis=0)
        dims = (idx.max(axis=0) - mn + 1).astype(np.int64)
        shifted = (idx - mn).astype(np.int64)
        keys = np.ravel_multi_index(
            (shifted[:, 0], shifted[:, 1], shifted[:, 2]),
            (int(dims[0]), int(dims[1]), int(dims[2])),
        )
        _, keep = np.unique(keys, return_index=True)
        keep.sort()
        return keep

    @classmethod
    def _occupied_cells(
        cls,
        coords: np.ndarray,
        cluster_ids: np.ndarray,
        origin: np.ndarray,
        spacing: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map centers to integer cell indices (nearest lattice cell) and dedup."""
        idx = np.round((coords - origin) / spacing).astype(np.int64)
        keep = cls._unique_rows(idx)  # keep first cluster id seen per cell
        return idx[keep], cluster_ids[keep]

    def _rasterize_to_fine(
        self,
        cells: np.ndarray,
        cluster_ids: np.ndarray,
        spacing: np.ndarray,
        s_ref: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Express occupied cells of one grid on the common fine lattice.

        Returns (fine_indices (K,3) int64, cluster_ids (K,)). Each fine cell is a
        lattice cell of spacing ``s_ref`` whose center falls inside the source
        cell (nearest-neighbour occupancy upsampling).
        """
        if cells.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)

        # identity fast-path: this grid already matches the fine lattice
        if np.allclose(spacing, s_ref, rtol=1e-9, atol=0.0):
            base = np.round((cells.astype(float) * spacing) / s_ref).astype(np.int64)
            return base, cluster_ids

        ratio = spacing / s_ref  # >= 1 per axis
        n_off = np.ceil(ratio).astype(int) + 1  # fine cells spanning a coarse cell

        # per-axis fine-index start = first fine center inside the coarse cell
        # coarse-cell lower face (relative to origin) = spacing*cell - spacing/2
        lower = spacing[None, :] * cells - spacing[None, :] / 2.0
        start = np.ceil(lower / s_ref[None, :] - 0.5).astype(np.int64)  # (M,3)

        # stencil of candidate offsets (product of per-axis ranges)
        offs = [np.arange(int(n_off[ax])) for ax in range(3)]
        ox, oy, oz = np.meshgrid(offs[0], offs[1], offs[2], indexing="ij")
        stencil = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)  # (S,3)

        cand = start[:, None, :] + stencil[None, :, :]  # (M,S,3)
        src_cid = np.broadcast_to(cluster_ids[:, None], cand.shape[:2]).reshape(-1)
        src_cell = np.broadcast_to(cells[:, None, :], cand.shape).reshape(-1, 3)
        cand = cand.reshape(-1, 3)

        # keep candidates whose nearest source-grid cell is the originating cell
        fine_center = cand.astype(float) * s_ref[None, :]  # relative to origin
        nearest = np.round(fine_center / spacing[None, :]).astype(np.int64)
        keep = np.all(nearest == src_cell, axis=1)

        fine = cand[keep]
        fcid = src_cid[keep]
        # dedup fine cells (a fine cell can only belong to one source cell here)
        uidx = self._unique_rows(fine)
        return fine[uidx], fcid[uidx]

    # ----------------------------------------------------------------- compare

    def _compare_fine(
        self,
        fine_1: np.ndarray,
        fcid_1: np.ndarray,
        fine_2: np.ndarray,
        fcid_2: np.ndarray,
    ) -> Dict[str, Any]:
        # shared integer key space over both regions
        if fine_1.shape[0] == 0 and fine_2.shape[0] == 0:
            raise ValueError("Both REI point clouds are empty after voxelization.")

        all_idx = np.vstack([fine_1, fine_2]) if fine_2.shape[0] else fine_1
        if fine_1.shape[0] == 0:
            all_idx = fine_2
        mn = all_idx.min(axis=0)
        dims = (all_idx.max(axis=0) - mn + 1).astype(np.int64)
        dims_t = (int(dims[0]), int(dims[1]), int(dims[2]))

        keys_1 = self._ravel(fine_1, mn, dims_t)
        keys_2 = self._ravel(fine_2, mn, dims_t)

        # sort for searchsorted-based lookups (keys already unique per region)
        o1 = np.argsort(keys_1, kind="mergesort")
        keys_1, fcid_1 = keys_1[o1], fcid_1[o1]
        o2 = np.argsort(keys_2, kind="mergesort")
        keys_2, fcid_2 = keys_2[o2], fcid_2[o2]

        union_keys = np.union1d(keys_1, keys_2)
        in1 = np.isin(union_keys, keys_1, assume_unique=True)
        in2 = np.isin(union_keys, keys_2, assume_unique=True)
        both = in1 & in2

        cid1_u = self._lookup(union_keys, keys_1, fcid_1)
        cid2_u = self._lookup(union_keys, keys_2, fcid_2)

        membership = np.where(both, 3, np.where(in1, 1, 2)).astype(np.int32)

        n1 = int(keys_1.size)
        n2 = int(keys_2.size)
        n_inter = int(np.count_nonzero(both))
        n_union = int(union_keys.size)
        vcell = float(np.prod(self.s_ref))

        iou = n_inter / n_union if n_union else 0.0
        dice = (2.0 * n_inter) / (n1 + n2) if (n1 + n2) else 0.0
        cont_1 = n_inter / n1 if n1 else 0.0
        cont_2 = n_inter / n2 if n2 else 0.0

        self.metrics = {
            "s_ref": self.s_ref.tolist(),
            "voxel_volume": vcell,
            "n_voxels_1": n1,
            "n_voxels_2": n2,
            "n_voxels_intersection": n_inter,
            "n_voxels_union": n_union,
            "volume_1": n1 * vcell,
            "volume_2": n2 * vcell,
            "volume_intersection": n_inter * vcell,
            "volume_union": n_union * vcell,
            "iou": iou,
            "dice": dice,
            "containment_1": cont_1,
            "containment_2": cont_2,
        }

        metrics_path = os.path.join(self.output_dir, "overlap_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        print("\n=== REI Comparison Metrics ===")
        for k, v in self.metrics.items():
            print(f"  {k:24s}: {v}")
        print(f"Saved metrics: {metrics_path}")

        # cluster-level 1-to-1 matching
        if self.cluster_col is not None:
            self._match_clusters(
                keys_1, fcid_1, keys_2, fcid_2, both, cid1_u, cid2_u, vcell
            )

        # export classified point cloud
        vtk_path = self._export_vtk(union_keys, mn, dims_t, membership, cid1_u, cid2_u)

        return {
            "output_dir": self.output_dir,
            "metrics": self.metrics,
            "metrics_path": metrics_path,
            "overlap_vtk_path": vtk_path,
            "cluster_match_path": (
                os.path.join(self.output_dir, "cluster_match.csv")
                if self.cluster_col is not None
                else None
            ),
        }

    def _match_clusters(
        self,
        _keys_1: np.ndarray,
        fcid_1: np.ndarray,
        _keys_2: np.ndarray,
        fcid_2: np.ndarray,
        both: np.ndarray,
        cid1_u: np.ndarray,
        cid2_u: np.ndarray,
        vcell: float,
    ) -> None:
        labels_1, cnt_1 = np.unique(fcid_1, return_counts=True)  # sorted labels
        labels_2, cnt_2 = np.unique(fcid_2, return_counts=True)
        if labels_1.size == 0 or labels_2.size == 0:
            return

        # per-label voxel counts (whole region, not just overlap)
        size_1 = {int(l): int(c) for l, c in zip(labels_1, cnt_1)}
        size_2 = {int(l): int(c) for l, c in zip(labels_2, cnt_2)}

        # overlap matrix over the intersection voxels (vectorized label -> index)
        c1 = cid1_u[both]
        c2 = cid2_u[both]
        O = np.zeros((labels_1.size, labels_2.size), dtype=np.int64)
        if c1.size:
            i1 = np.searchsorted(labels_1, c1)
            i2 = np.searchsorted(labels_2, c2)
            np.add.at(O, (i1, i2), 1)

        row_ind, col_ind = linear_sum_assignment(-O)

        rows: List[Dict[str, Any]] = []
        matched_1, matched_2 = set(), set()
        for r, c in zip(row_ind, col_ind):
            overlap = int(O[r, c])
            if overlap <= 0:
                continue
            lab1, lab2 = int(labels_1[r]), int(labels_2[c])
            union = size_1[lab1] + size_2[lab2] - overlap
            rows.append(
                {
                    "cluster_1": lab1,
                    "cluster_2": lab2,
                    "overlap_voxels": overlap,
                    "overlap_volume": overlap * vcell,
                    "size_voxels_1": size_1[lab1],
                    "size_voxels_2": size_2[lab2],
                    "jaccard": overlap / union if union else 0.0,
                    "containment_1": overlap / size_1[lab1] if size_1[lab1] else 0.0,
                    "containment_2": overlap / size_2[lab2] if size_2[lab2] else 0.0,
                }
            )
            matched_1.add(lab1)
            matched_2.add(lab2)

        # unmatched clusters (no overlap partner)
        for lab1 in labels_1:
            if int(lab1) not in matched_1:
                rows.append(
                    {
                        "cluster_1": int(lab1),
                        "cluster_2": -1,
                        "overlap_voxels": 0,
                        "overlap_volume": 0.0,
                        "size_voxels_1": size_1[int(lab1)],
                        "size_voxels_2": 0,
                        "jaccard": 0.0,
                        "containment_1": 0.0,
                        "containment_2": 0.0,
                    }
                )
        for lab2 in labels_2:
            if int(lab2) not in matched_2:
                rows.append(
                    {
                        "cluster_1": -1,
                        "cluster_2": int(lab2),
                        "overlap_voxels": 0,
                        "overlap_volume": 0.0,
                        "size_voxels_1": 0,
                        "size_voxels_2": size_2[int(lab2)],
                        "jaccard": 0.0,
                        "containment_1": 0.0,
                        "containment_2": 0.0,
                    }
                )

        # split / merge detection from the full overlap matrix
        frac = self.split_merge_fraction
        n_split = 0  # one cluster_1 overlaps multiple cluster_2 significantly
        for r in range(labels_1.size):
            sig = np.count_nonzero(O[r, :] >= frac * max(1, size_1[int(labels_1[r])]))
            if sig > 1:
                n_split += 1
        n_merge = 0  # one cluster_2 covered by multiple cluster_1 significantly
        for c in range(labels_2.size):
            sig = np.count_nonzero(O[:, c] >= frac * max(1, size_2[int(labels_2[c])]))
            if sig > 1:
                n_merge += 1

        match_df = pd.DataFrame(rows)
        match_path = os.path.join(self.output_dir, "cluster_match.csv")
        match_df.to_csv(match_path, index=False)
        self.cluster_match = match_df

        self.metrics["n_clusters_1"] = int(labels_1.size)
        self.metrics["n_clusters_2"] = int(labels_2.size)
        self.metrics["n_matched_clusters"] = len(matched_1)
        self.metrics["n_splits"] = int(n_split)
        self.metrics["n_merges"] = int(n_merge)

        print(
            f"\nCluster matching: {len(matched_1)} matched "
            f"({labels_1.size} in REI 1, {labels_2.size} in REI 2); "
            f"splits={n_split}, merges={n_merge}."
        )
        print(f"Saved cluster match table: {match_path}")

    def _export_vtk(
        self,
        union_keys: np.ndarray,
        mn: np.ndarray,
        dims: Tuple[int, int, int],
        membership: np.ndarray,
        cid1_u: np.ndarray,
        cid2_u: np.ndarray,
    ) -> str:
        # np.unravel_index returns one array per dim; pylint can't infer the count
        # pylint: disable-next=unbalanced-tuple-unpacking
        ix, iy, iz = np.unravel_index(union_keys, dims)
        idx = np.stack([ix, iy, iz], axis=1) + mn
        coords = self.origin[None, :] + idx.astype(float) * self.s_ref[None, :]

        point_data = {
            "membership": membership.astype(np.int32),
            "cluster_id_1": cid1_u.astype(np.int64),
            "cluster_id_2": cid2_u.astype(np.int64),
        }
        vtk_path = os.path.join(self.output_dir, "overlap_cloud.vtk")
        # Reuse the REI VTK point-cloud writer (shared internal helper).
        IdentifyRareClusters._write_polydata_points_vtk(  # pylint: disable=protected-access
            vtk_path, coords, point_data
        )
        print(f"Saved classified overlap cloud: {vtk_path}")
        return vtk_path

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _ravel(
        idx: np.ndarray, mn: np.ndarray, dims: Tuple[int, int, int]
    ) -> np.ndarray:
        if idx.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        shifted = (idx - mn).astype(np.int64)
        return np.ravel_multi_index(
            (shifted[:, 0], shifted[:, 1], shifted[:, 2]), dims
        ).astype(np.int64)

    @staticmethod
    def _lookup(
        query_keys: np.ndarray, sorted_keys: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """Return values for query_keys found in sorted_keys, else -1."""
        out = np.full(query_keys.shape[0], -1, dtype=np.int64)
        if sorted_keys.size == 0:
            return out
        pos = np.searchsorted(sorted_keys, query_keys)
        pos = np.clip(pos, 0, sorted_keys.size - 1)
        hit = sorted_keys[pos] == query_keys
        out[hit] = values[pos[hit]]
        return out
