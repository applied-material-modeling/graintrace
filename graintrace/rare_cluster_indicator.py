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

"""Identify and export spatially coherent rare clusters from CPFE field data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .user_data_class import RareCriteria
from .graph_spatial_cluster import GraphSpatialCluster
from .cluster_indicator import ClusterAnalysisIndicator


class IdentifyRareClusters:
    """Graph cluster -> merge via indicator -> select rare clusters and export VTK."""

    def __init__(
        self,
        input_csv_path: str,
        id_col: str = "id",
        coord_cols: Tuple[str, str, str] = ("x", "y", "z"),
    ) -> None:
        self.input_csv_path = input_csv_path
        self.id_col = id_col
        self.coord_cols = coord_cols

        self._input_df: Optional[pd.DataFrame] = None

    def run_clustering(
        self,
        gsc,
        indicator,
        *,
        gsc_run_kwargs: Dict[str, Any],
        indicator_run_kwargs: Dict[str, Any],
        reduced_csv_path: str,
    ) -> Dict[str, Any]:
        """Run graph spatial clustering then the merge indicator; return a result bundle."""
        print("Running clustering analysis...")

        input_df = self._load_input_df()

        gsc_out = gsc.run(
            output_csv_path=reduced_csv_path,
            return_labels=True,
            **gsc_run_kwargs,
        )

        if "extras" not in gsc_out or "labels" not in gsc_out["extras"]:
            raise ValueError(
                "GraphSpatialCluster must be run with return_labels=True to obtain per-point labels."
            )

        gsc_labels = np.asarray(gsc_out["extras"]["labels"], dtype=np.int64)

        labels_path = reduced_csv_path.rsplit(".", 1)[0] + "_gsc_labels.npy"
        np.save(labels_path, gsc_labels, allow_pickle=False)
        print("Saved GSC labels:", labels_path)

        if len(gsc_labels) != len(input_df):
            raise ValueError(
                "Length mismatch: per-point labels must align with input rows."
            )

        print("Reduced CSV saved:", gsc_out["csv_path"])

        print("\nRunning cluster indicator\n")
        ind_out = indicator.run(minimal_return=False, **indicator_run_kwargs)

        if "points" not in ind_out or ind_out["points"] is None:
            raise ValueError(
                "ClusterAnalysisIndicator must return 'points' (minimal_return=False) to build mapping."
            )

        indicator_points_df = ind_out["points"]
        indicator_clusters_df = ind_out["clusters"]

        if (
            "cluster_id" not in indicator_points_df.columns
            or "cluster_label" not in indicator_points_df.columns
        ):
            raise ValueError(
                "indicator 'points' must include columns: 'cluster_id' and 'cluster_label'."
            )

        return {
            "input_df": input_df,
            "gsc_labels": gsc_labels,
            "reduced_csv_path": gsc_out.get("csv_path", reduced_csv_path),
            "gsc_extras": gsc_out.get("extras", {}),
            "indicator_points_df": indicator_points_df,
            "indicator_clusters_df": indicator_clusters_df,
            "indicator_extras": ind_out.get("extras", {}),
        }

    def run_get_rare_cluster(
        self,
        bundle: Dict[str, Any],
        *,
        criteria: RareCriteria,
        output_vtk_path: str,
        export_control: str = "auto",  # "auto" | "grid" | "points"
        background_block_id: int = 1,
        first_rare_block_id: int = 2,
        also_write_final_label: bool = True,
        rare_reduced_stats_csv_path: Optional[str] = None,
        rare_points_csv_path: Optional[str] = None,
        use_sample_std: bool = False,  # pylint: disable=unused-argument  # public API kwarg
    ) -> Dict[str, Any]:
        """Select rare clusters from the bundle and export them to VTK (and optional CSVs)."""
        input_df: pd.DataFrame = bundle["input_df"]
        gsc_labels: np.ndarray = bundle["gsc_labels"]
        indicator_points_df: pd.DataFrame = bundle["indicator_points_df"]
        indicator_clusters_df: pd.DataFrame = bundle["indicator_clusters_df"]

        super_label_map = self._build_super_label_map(indicator_points_df)

        final_label = np.vectorize(super_label_map.get, otypes=[np.int64])(gsc_labels)

        if np.any(pd.isna(final_label)):
            raise ValueError(
                "Some stage-1 cluster_ids were not found in indicator mapping (cluster_id -> cluster_label)."
            )

        rare_super_labels = self._select_rare_super_labels(
            indicator_clusters_df, criteria
        )

        block_id = np.full(len(input_df), background_block_id, dtype=np.int32)
        rare_super_labels_sorted = list(rare_super_labels)

        if "n" in indicator_clusters_df.columns:
            n_map = dict(
                zip(indicator_clusters_df["cluster_label"], indicator_clusters_df["n"])
            )
            rare_super_labels_sorted.sort(key=lambda lab: n_map.get(lab, np.inf))

        label_to_block: Dict[int, int] = {}
        next_block = first_rare_block_id
        for lab in rare_super_labels_sorted:
            label_to_block[int(lab)] = next_block
            next_block += 1

        if rare_reduced_stats_csv_path is not None:
            cdf = indicator_clusters_df.copy()

            if "cluster_label" not in cdf.columns or "n" not in cdf.columns:
                raise ValueError(
                    "indicator_clusters_df must contain 'cluster_label' and 'n'."
                )

            cdf = cdf[cdf["cluster_label"].isin(rare_super_labels_sorted)].copy()
            cdf["rare_cluster_id"] = (
                cdf["cluster_label"].map(label_to_block).astype("Int64")
            )

            sum_cols = [c for c in cdf.columns if c.endswith("_sum")]
            bases = [c[:-4] for c in sum_cols if c[:-4] + "_sumsq" in cdf.columns]

            new_cols = {}
            n = pd.to_numeric(cdf["n"], errors="coerce").astype(float)

            for b in bases:
                s = pd.to_numeric(cdf[f"{b}_sum"], errors="coerce").astype(float)
                ss = pd.to_numeric(cdf[f"{b}_sumsq"], errors="coerce").astype(float)

                mean = s / n
                var_pop = (ss / n) - (mean * mean)
                var_pop = var_pop.clip(lower=0.0)

                new_cols[f"{b}_mean"] = mean
                new_cols[f"{b}_var"] = var_pop
                new_cols[f"{b}_std"] = np.sqrt(var_pop)

            cdf = pd.concat([cdf, pd.DataFrame(new_cols, index=cdf.index)], axis=1)

            keep = ["cluster_label", "rare_cluster_id", "n"]
            for b in bases:
                keep += [f"{b}_mean", f"{b}_var", f"{b}_std"]
                for extra in (f"{b}_min", f"{b}_max"):
                    if extra in cdf.columns:
                        keep.append(extra)

            out_df = cdf[keep].sort_values(
                ["n", "cluster_label"], ascending=[True, True]
            )
            out_df.to_csv(rare_reduced_stats_csv_path, index=False)

            print("\nSaved rare cluster stats CSV:", rare_reduced_stats_csv_path)

        for lab, bid in label_to_block.items():
            block_id[final_label == lab] = bid

        coords = input_df[list(self.coord_cols)].to_numpy(dtype=np.float64)

        if rare_points_csv_path is not None:
            rare_mask = block_id >= first_rare_block_id
            rare_df = pd.DataFrame(coords[rare_mask], columns=list(self.coord_cols))
            rare_df["rare_cluster_id"] = block_id[rare_mask].astype(np.int64)
            rare_df.to_csv(rare_points_csv_path, index=False)
            print(
                f"\nSaved rare point cloud CSV ({int(rare_mask.sum())} points): "
                f"{rare_points_csv_path}"
            )

        mode = export_control.lower()
        if mode == "auto":
            mode = "grid" if self._detect_full_grid(coords, tol=1e-6) else "points"
        elif mode not in ("grid", "points"):
            raise ValueError("export_control must be one of {'auto','grid','points'}")

        point_data: Dict[str, np.ndarray] = {
            "rare_cluster_id": block_id.astype(np.int32),
        }
        if also_write_final_label:
            point_data["final_label"] = final_label.astype(np.int64)

        feature_cols = [
            c
            for c in input_df.columns
            if c not in ([self.id_col] + list(self.coord_cols))
        ]
        for c in feature_cols:
            s = pd.to_numeric(input_df[c], errors="coerce")
            point_data[c] = s.to_numpy(dtype=np.float64)

        if mode == "grid":
            self._write_structured_grid_vtk(output_vtk_path, coords, point_data)
        else:
            self._write_polydata_points_vtk(output_vtk_path, coords, point_data)

        return {
            "output_vtk_path": output_vtk_path,
            "export_mode": mode,
            "n_points": int(len(coords)),
            "n_rare_clusters": int(len(rare_super_labels_sorted)),
            "rare_super_labels": rare_super_labels_sorted,
            "label_to_block": label_to_block,
            "rare_reduced_stats_csv_path": rare_reduced_stats_csv_path,
            "rare_points_csv_path": rare_points_csv_path,
        }

    def _build_super_label_map(
        self, indicator_points_df: pd.DataFrame
    ) -> Dict[int, int]:
        cluster_id = indicator_points_df["cluster_id"].to_numpy()
        cluster_label = indicator_points_df["cluster_label"].to_numpy()
        return {int(cid): int(clab) for cid, clab in zip(cluster_id, cluster_label)}

    def _select_rare_super_labels(
        self, indicator_clusters_df: pd.DataFrame, criteria: RareCriteria
    ) -> List[int]:
        if criteria.selector is not None:
            out = criteria.selector(indicator_clusters_df)
            labs = list(np.asarray(out).tolist())
            return [int(x) for x in labs]

        if "cluster_label" not in indicator_clusters_df.columns:
            raise ValueError(
                "indicator_clusters_df must contain 'cluster_label' to select rare clusters."
            )
        if "n" not in indicator_clusters_df.columns:
            raise ValueError(
                "Default rarity selection requires 'n' column in indicator_clusters_df. Provide criteria.selector."
            )

        df = indicator_clusters_df.copy()
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["n"])
        df = df[df["n"] >= criteria.min_size]

        if df.empty:
            return []

        q = float(criteria.size_quantile)
        q = min(max(q, 0.0), 1.0)
        thresh = float(df["n"].quantile(q))
        rare_df = df[df["n"] <= thresh].sort_values("n", ascending=True)

        if criteria.max_rare is not None:
            rare_df = rare_df.head(int(criteria.max_rare))

        return [int(x) for x in rare_df["cluster_label"].to_list()]

    def make_stage_objects(
        self,
        *,
        graph_cluster_out: str,
    ) -> Tuple[GraphSpatialCluster, ClusterAnalysisIndicator]:
        """Construct the GraphSpatialCluster and ClusterAnalysisIndicator stage objects."""
        gsc = GraphSpatialCluster(
            csv_path=self.input_csv_path,
            id_col=self.id_col,
            coord_cols=self.coord_cols,
        )

        indicator = ClusterAnalysisIndicator(
            csv_path=graph_cluster_out,
            id_col="cluster_id",
            coord_cols=self.coord_cols,
        )

        return gsc, indicator

    def _load_input_df(self) -> pd.DataFrame:
        if self._input_df is not None:
            return self._input_df
        df = pd.read_csv(self.input_csv_path)
        required = [self.id_col, *self.coord_cols]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")
        self._input_df = df
        return df

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

    def _detect_full_grid(self, coords: np.ndarray, tol: float = 1e-6) -> bool:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        if not (
            self._is_regular_1d_grid(x, tol)
            and self._is_regular_1d_grid(y, tol)
            and self._is_regular_1d_grid(z, tol)
        ):
            return False
        ux, uy, uz = np.unique(x), np.unique(y), np.unique(z)
        return (ux.size * uy.size * uz.size) == coords.shape[0]

    @staticmethod
    def _write_polydata_points_vtk(
        path: str,
        coords: np.ndarray,
        point_data: Dict[str, np.ndarray],
    ) -> None:
        n = coords.shape[0]
        with open(path, "w", encoding="utf-8") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Rare clusters (points)\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {n} float\n")
            for x, y, z in coords:
                f.write(f"{x:.9g} {y:.9g} {z:.9g}\n")

            f.write(f"VERTICES {n} {2*n}\n")
            for i in range(n):
                f.write(f"1 {i}\n")

            f.write(f"POINT_DATA {n}\n")
            for name, arr in point_data.items():
                IdentifyRareClusters._write_vtk_scalar_array(f, name, arr)

    @staticmethod
    def _write_structured_grid_vtk(
        path: str,
        coords: np.ndarray,
        point_data: Dict[str, np.ndarray],
        tol: float = 1e-6,
    ) -> None:
        xs = np.unique(coords[:, 0])
        ys = np.unique(coords[:, 1])
        zs = np.unique(coords[:, 2])
        nx, ny, nz = xs.size, ys.size, zs.size
        n = coords.shape[0]
        if nx * ny * nz != n:
            raise ValueError("Not a full grid; cannot export STRUCTURED_GRID.")

        # map each point to (ix,iy,iz) by nearest bin
        def map_to_bins(vals, uniques):
            pos = np.searchsorted(uniques, vals)
            pos = np.clip(pos, 0, uniques.size - 1)
            left = np.maximum(pos - 1, 0)
            choose_left = (
                np.abs(vals - uniques[left]) <= np.abs(vals - uniques[pos]) + tol
            )
            return np.where(choose_left, left, pos).astype(np.int64)

        ix = map_to_bins(coords[:, 0], xs)
        iy = map_to_bins(coords[:, 1], ys)
        iz = map_to_bins(coords[:, 2], zs)

        lin = ix + nx * (iy + ny * iz)  # x-fastest
        order = np.argsort(lin, kind="mergesort")

        coords_ord = coords[order]
        point_data_ord = {k: np.asarray(v)[order] for k, v in point_data.items()}

        with open(path, "w", encoding="utf-8") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Rare clusters (structured grid)\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
            f.write(f"POINTS {n} float\n")
            for x, y, z in coords_ord:
                f.write(f"{x:.9g} {y:.9g} {z:.9g}\n")

            f.write(f"POINT_DATA {n}\n")
            for name, arr in point_data_ord.items():
                IdentifyRareClusters._write_vtk_scalar_array(f, name, arr)

    @staticmethod
    def _write_vtk_scalar_array(f, name: str, arr: np.ndarray) -> None:
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError(
                f"VTK writer supports 1D scalar arrays only; got {name} with shape {arr.shape}"
            )

        if np.issubdtype(arr.dtype, np.integer):
            vtk_type = "int"
        else:
            vtk_type = "float"

        f.write(f"SCALARS {name} {vtk_type} 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in arr:
            f.write(f"{v}\n")
