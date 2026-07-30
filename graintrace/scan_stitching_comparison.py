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

"""Compare stitched HEDM grain data against ground truth (ScanStitchingComparison)."""

from __future__ import annotations

import os
import json
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from .orientation_helper import misorientation


class ScanStitchingComparison:
    """Match stitched vs. true grains and report position/orientation error metrics."""

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        output_dir: str,
        true_csv: str,
        stitch_csv: str,
        position_tolerance: float = 1,
        orientation_tolerance: float = 1,
        radius_tolerance: float = 1,
        orientation_units: str = "degrees",
        orientation_convention: str = "bunge",
        symmetry: str = "432",
        # read-only default, only stored to self.weights
        weights: dict = {"pos": 1.0, "ori": 0.0, "rad": 0.0},
        min_neighbors: int = 5,
    ):
        """Compare true vs stitched grain data (both CSVs need columns
        X, Y, Z, GrainRadius, Eul0, Eul1, Eul2)."""

        if not os.path.exists(true_csv):
            raise FileNotFoundError(f"True CSV not found: {true_csv}")
        if not os.path.exists(stitch_csv):
            raise FileNotFoundError(f"Stitched CSV not found: {stitch_csv}")

        if min_neighbors < 2:
            raise ValueError("min_neighbors must be at least 2.")

        required_cols = {"X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"}
        for file_path, label in [(true_csv, "true_csv"), (stitch_csv, "stitch_csv")]:
            try:
                cols = set(pd.read_csv(file_path, nrows=0).columns)
            except Exception as e:
                raise ValueError(f"Failed to read {label}: {e}") from e
            if not required_cols.issubset(cols):
                missing = required_cols - cols
                raise ValueError(f"{label} missing columns: {', '.join(missing)}")

        if orientation_units not in {"radians", "degrees"}:
            raise ValueError("orientation_units must be 'radians' or 'degrees'.")

        self.true_csv = true_csv
        self.stitch_csv = stitch_csv
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.radius_tolerance = radius_tolerance
        self.orientation_units = orientation_units
        self.weights = weights
        self.min_neighbors = min_neighbors
        self.output_dir = os.path.abspath(output_dir)

        self.orientation_convention = orientation_convention
        self.symmetry = symmetry

        os.makedirs(self.output_dir, exist_ok=True)

        self.df_true = None
        self.df_stitch = None
        self.kdtree_true = None
        self.kdtree_stitch = None
        self.matches = None
        self.inverse_matches = None
        self.metrics = {}

    def run_comparison(self) -> Dict[str, Any]:
        """Run the full compare pipeline and return the metrics dict."""
        self.load_data()

        self._build_kdtree()
        self._match_grains()

        self._compute_statistics()
        self._plot_histograms()

    def load_data(self) -> None:
        """Load and preprocess true and stitched grain data."""
        self.df_true = pd.read_csv(self.true_csv)
        self.df_stitch = pd.read_csv(self.stitch_csv)

        base_cols = ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]

        # keep any extra cols that exist in the CSVs (debug columns, ScanID, etc.)
        extra_true = [c for c in self.df_true.columns if c not in base_cols]
        extra_stitch = [c for c in self.df_stitch.columns if c not in base_cols]

        self.df_true = self.df_true[base_cols + extra_true].copy()
        self.df_stitch = self.df_stitch[base_cols + extra_stitch].copy()

        self.df_true[base_cols] = self.df_true[base_cols].astype(float)
        self.df_stitch[base_cols] = self.df_stitch[base_cols].astype(float)

        print(f"Loaded {len(self.df_true)} grains from true dataset.")
        print(f"Loaded {len(self.df_stitch)} grains from stitched dataset.\n")

    def _build_kdtree(self) -> None:
        """Construct KD-trees for true and stitched grain centroids."""
        if self.df_true is None or self.df_stitch is None:
            raise RuntimeError("Datasets not loaded. Run load_data() first.")

        coords_true = self.df_true[["X", "Y", "Z"]].to_numpy()
        self.kdtree_true = cKDTree(coords_true)
        print(f"KD-tree built with {len(coords_true)} true grains.")

        coords_stitch = self.df_stitch[["X", "Y", "Z"]].to_numpy()
        self.kdtree_stitch = cKDTree(coords_stitch)
        print(f"KD-tree built with {len(coords_stitch)} stitched grains.\n")

    def _match_grains(self) -> None:
        """Match stitched to true grains via KD-tree search and optimal assignment."""

        if self.kdtree_true is None or self.kdtree_stitch is None:
            raise RuntimeError("KD-trees not built. Run _build_kdtree() first.")
        if self.df_true is None or self.df_stitch is None:
            raise RuntimeError("Data not loaded. Run load_data() first.")

        self.matches = self._build_mapping(
            self.df_stitch, self.df_true, self.kdtree_true
        )
        mask = self._valid_match_mask(self.matches)
        print(
            f"\nMatched {mask.sum()} stitched grains to true grains (unmatched: {len(self.matches)-mask.sum()}).\n"
        )

        self.inverse_matches = self._build_mapping(
            self.df_true, self.df_stitch, self.kdtree_stitch
        )
        mask = self._valid_match_mask(self.inverse_matches)
        print(
            f"Matched {mask.sum()} true grains to stitched grains (unmatched: {len(self.inverse_matches)-mask.sum()}).\n"
        )

    def _build_mapping(self, source_df, target_df, target_tree):
        """Build a 1-1 source->target mapping via Hungarian assignment, allowing
        unmatched entries via dummy nodes. One row per source; idx_target = -1 if
        unmatched."""
        coords_source = source_df[["X", "Y", "Z"]].to_numpy()
        n_source, n_target = len(source_df), len(target_df)

        if n_source == 0 or n_target == 0:
            return pd.DataFrame(
                [
                    (int(s), -1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
                    for s in range(n_source)
                ],
                columns=[
                    "idx_source",
                    "idx_target",
                    "diff_pos_norm2",
                    "diff_rad_percentage",
                    "diff_ori",
                    "diff_pos_x",
                    "diff_pos_y",
                    "diff_pos_z",
                ],
            )

        # keep full coordinates for component-wise diffs
        coords_target = target_df[["X", "Y", "Z"]].to_numpy()

        # KD-tree query for candidate edges
        k = min(self.min_neighbors, n_target)
        dist, idx = target_tree.query(coords_source, k=k, workers=-1)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        s_idx, _ = np.indices(dist.shape)
        s_idx = s_idx.ravel().astype(int)
        t_idx = idx.ravel().astype(int)
        diff_pos = dist.ravel().astype(float)

        dx_all = coords_source[s_idx, 0] - coords_target[t_idx, 0]
        dy_all = coords_source[s_idx, 1] - coords_target[t_idx, 1]
        dz_all = coords_source[s_idx, 2] - coords_target[t_idx, 2]

        ori_source = source_df[["Eul0", "Eul1", "Eul2"]].to_numpy(float)
        ori_target = target_df[["Eul0", "Eul1", "Eul2"]].to_numpy(float)
        rad_source = source_df["GrainRadius"].to_numpy(float)
        rad_target = target_df["GrainRadius"].to_numpy(float)

        diff_ori_t = misorientation(
            ori_source[s_idx],
            ori_target[t_idx],
            angle_convention=self.orientation_convention,
            angle_type=self.orientation_units,
            symmetry=self.symmetry,
        )
        diff_ori = diff_ori_t.detach().cpu().numpy().astype(float)
        diff_rad = np.abs(rad_source[s_idx] - rad_target[t_idx]) / np.maximum(
            rad_target[t_idx], 1e-14
        )

        w_pos = float(self.weights.get("pos", 1.0))
        w_ori = float(self.weights.get("ori", 0.0))
        w_rad = float(self.weights.get("rad", 0.0))

        eps = 1e-14
        ptol = self.position_tolerance if self.position_tolerance != 0.0 else eps
        otol = self.orientation_tolerance if self.orientation_tolerance != 0.0 else eps
        rtol = self.radius_tolerance if self.radius_tolerance != 0.0 else eps

        # feasibility mask
        ok = np.ones_like(diff_pos, dtype=bool)
        if self.position_tolerance != -1:
            ok &= diff_pos <= self.position_tolerance
        if self.orientation_tolerance != -1:
            ok &= diff_ori <= self.orientation_tolerance
        if self.radius_tolerance != -1 and w_rad > 0.0:
            ok &= diff_rad <= self.radius_tolerance

        s_idx = s_idx[ok]
        t_idx = t_idx[ok]
        diff_pos = diff_pos[ok]
        diff_ori = diff_ori[ok]
        diff_rad = diff_rad[ok]

        dx_all = dx_all[ok]
        dy_all = dy_all[ok]
        dz_all = dz_all[ok]

        # nothing feasible => everything unmatched
        if s_idx.size == 0:
            return pd.DataFrame(
                [
                    (int(s), -1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
                    for s in range(n_source)
                ],
                columns=[
                    "idx_source",
                    "idx_target",
                    "diff_pos_norm2",
                    "diff_rad_percentage",
                    "diff_ori",
                    "diff_pos_x",
                    "diff_pos_y",
                    "diff_pos_z",
                ],
            )

        cost = (
            w_pos * (diff_pos / ptol)
            + w_ori * (diff_ori / otol)
            + w_rad * (diff_rad / rtol)
        )

        # unmatch cost slightly above max feasible cost
        max_real = 0.0
        if w_pos > 0 and self.position_tolerance != -1:
            max_real += w_pos * 1.0
        if w_ori > 0 and self.orientation_tolerance != -1:
            max_real += w_ori * 1.0
        if w_rad > 0 and self.radius_tolerance != -1:
            max_real += w_rad * 1.0
        UNMATCH_COST = max_real + 1e-6
        BIG = 1e9

        # best edge per (s,t) if duplicates show up
        pair = {}
        for s, t, dp, dr, do, cc, dx, dy, dz in zip(
            s_idx, t_idx, diff_pos, diff_rad, diff_ori, cost, dx_all, dy_all, dz_all
        ):
            key = (int(s), int(t))
            if key not in pair or cc < pair[key][7]:
                pair[key] = (
                    float(dp),  # 0
                    float(dr),  # 1
                    float(do),  # 2
                    float(dx),  # 3
                    float(dy),  # 4
                    float(dz),  # 5
                    float(cc),  # 6
                    float(cc),  # 7 (for comparison)
                )

        # augmented square cost matrix
        N = n_source + n_target
        C = np.full((N, N), BIG, dtype=float)

        for (s, t), vals in pair.items():
            cc = vals[6]
            C[s, t] = cc

        # real source -> dummy cols (unmatch each source)
        C[0:n_source, n_target : n_target + n_source] = UNMATCH_COST
        # dummy rows (target unmatched) -> real target cols
        C[n_source : n_source + n_target, 0:n_target] = UNMATCH_COST
        # dummy rows -> dummy cols
        C[n_source : n_source + n_target, n_target : n_target + n_source] = 0.0

        row_ind, col_ind = linear_sum_assignment(C)

        # per-source result, default unmatched
        idx_target_out = np.full(n_source, -1, dtype=int)
        dp_out = np.full(n_source, np.inf, dtype=float)
        dr_out = np.full(n_source, np.inf, dtype=float)
        do_out = np.full(n_source, np.inf, dtype=float)

        dx_out = np.full(n_source, np.inf, dtype=float)
        dy_out = np.full(n_source, np.inf, dtype=float)
        dz_out = np.full(n_source, np.inf, dtype=float)

        # interpret assignments for real source rows only
        for r, c in zip(row_ind, col_ind):
            if r >= n_source:
                continue  # dummy row
            s = int(r)
            if c < n_target:
                # match to real target only if cheaper than unmatch
                if C[r, c] < UNMATCH_COST and C[r, c] < BIG * 0.5:
                    t = int(c)
                    idx_target_out[s] = t
                    dp, dr, do, dx, dy, dz, _, _ = pair.get(
                        (s, t),
                        (
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ),
                    )
                    dp_out[s] = dp
                    dr_out[s] = dr
                    do_out[s] = do
                    dx_out[s] = dx
                    dy_out[s] = dy
                    dz_out[s] = dz
                else:
                    pass  # matched to dummy => unmatched source

        rows = [
            (
                int(s),
                int(idx_target_out[s]),
                dp_out[s],
                dr_out[s],
                do_out[s],
                dx_out[s],
                dy_out[s],
                dz_out[s],
            )
            for s in range(n_source)
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "idx_source",
                "idx_target",
                "diff_pos_norm2",
                "diff_rad_percentage",
                "diff_ori",
                "diff_pos_x",
                "diff_pos_y",
                "diff_pos_z",
            ],
        )

    def _valid_match_mask(self, df_matches, pos_tol=None, ori_tol=None):
        if df_matches is None:
            raise RuntimeError("Matches not computed.")

        pos_tol = pos_tol if pos_tol is not None else self.position_tolerance
        ori_tol = ori_tol if ori_tol is not None else self.orientation_tolerance

        if len(df_matches) == 0:
            return np.zeros(0, dtype=bool)

        return (
            (df_matches["idx_target"] >= 0)
            & (df_matches["diff_pos_norm2"] <= pos_tol)
            & (df_matches["diff_ori"] <= ori_tol)
        )

    def _compute_statistics(self):
        """Compute matching statistics on matches within the default tolerances."""
        if self.matches is None or self.inverse_matches is None:
            raise RuntimeError("Matches not computed. Run _match_grains() first.")

        # forward: stitched -> true
        mask_fwd = self._valid_match_mask(self.matches)
        matches_valid = self.matches[mask_fwd]

        # inverse: true -> stitched
        mask_inv = self._valid_match_mask(self.inverse_matches)
        inverse_valid = self.inverse_matches[mask_inv]

        # splits: one true grain matched by multiple stitched grains
        split_detail = (
            matches_valid.groupby("idx_target")["idx_source"]
            .apply(list)
            .reset_index()
            .rename(
                columns={
                    "idx_target": "TrueGrainID",
                    "idx_source": "MappedStitchedGrains",
                }
            )
        )
        split_detail["NumMappedStitched"] = split_detail["MappedStitchedGrains"].apply(
            len
        )
        split_detail = split_detail[split_detail["NumMappedStitched"] > 1]
        split_detail.to_csv(
            os.path.join(self.output_dir, "split_grains.csv"), index=False
        )

        # merges: one stitched grain matched by multiple true grains
        merge_detail = (
            inverse_valid.groupby("idx_target")["idx_source"]
            .apply(list)
            .reset_index()
            .rename(
                columns={
                    "idx_target": "StitchedGrainID",
                    "idx_source": "MappedTrueGrains",
                }
            )
        )
        merge_detail["NumMappedTrue"] = merge_detail["MappedTrueGrains"].apply(len)
        merge_detail = merge_detail[merge_detail["NumMappedTrue"] > 1]
        merge_detail.to_csv(
            os.path.join(self.output_dir, "merge_grains.csv"), index=False
        )

        def safe_stat(series, fn):
            return fn(series) if len(series) > 0 else None

        self.metrics = {
            "n_true": len(self.df_true),
            "n_stitch": len(self.df_stitch),
            "n_matched": len(matches_valid),
            "n_inverse_matched": len(inverse_valid),
            "n_splits": len(split_detail),
            "n_merges": len(merge_detail),
            "mean_pos_abs_error": safe_stat(matches_valid["diff_pos_norm2"], np.mean),
            "mean_pos_error_x": safe_stat(matches_valid["diff_pos_x"], np.mean),
            "mean_pos_error_y": safe_stat(matches_valid["diff_pos_y"], np.mean),
            "mean_pos_error_z": safe_stat(matches_valid["diff_pos_z"], np.mean),
            "mean_ori_error": safe_stat(matches_valid["diff_ori"], np.mean),
            "mean_rad_error": safe_stat(matches_valid["diff_rad_percentage"], np.mean),
            "median_pos_error": safe_stat(matches_valid["diff_pos_norm2"], np.median),
            "median_ori_error": safe_stat(matches_valid["diff_ori"], np.median),
            "median_rad_error": safe_stat(
                matches_valid["diff_rad_percentage"], np.median
            ),
        }

        with open(
            os.path.join(self.output_dir, "statistics_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.metrics, f, indent=2)

        self.matches.to_csv(
            os.path.join(self.output_dir, "stitch_to_true_matches.csv"), index=False
        )
        self.inverse_matches.to_csv(
            os.path.join(self.output_dir, "true_to_stitch_matches.csv"), index=False
        )

        print("\n=== Stitching Comparison Statistics (within tolerances) ===")
        for k, v in self.metrics.items():
            print(f"{k:20s}: {v}")

    def _plot_histograms(self):
        """Plot histograms of match differences."""
        os.makedirs(os.path.join(self.output_dir, "figures"), exist_ok=True)

        valid_mask = self._valid_match_mask(self.matches)
        matches_valid = self.matches[valid_mask]

        fig, axes = plt.subplots(3, 1, figsize=(6, 9), tight_layout=True)
        plot_info = [
            ("diff_pos_norm2", "Position Error (length units)"),
            ("diff_ori", f"Orientation Error ({self.orientation_units})"),
            ("diff_rad_percentage", "Radius Error (relative)"),
        ]

        for ax, (col, label) in zip(axes, plot_info):
            if col not in matches_valid.columns:
                continue
            ax.hist(matches_valid[col], bins=40, alpha=0.75, edgecolor="black")
            ax.set_xlabel(label)
            ax.set_ylabel("Number of Grains")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        fig.savefig(
            os.path.join(self.output_dir, "figures", "error_histograms.png"), dpi=300
        )
        plt.close(fig)

        print(
            f"Error histograms saved to: {os.path.join(self.output_dir, 'figures', 'error_histograms.png')}\n"
        )

    def _get_unmatched_grains(
        self,
        bounding_box: list | None = None,
        plot: bool = True,
        pos_tol: float | None = None,
        ori_tol: float | None = None,
        view3D: bool = False,
    ):
        """Identify unmatched grains beyond given tolerances and optionally plot them.

        bounding_box defaults to the true dataset extent; pos_tol/ori_tol default to
        the initialized class tolerances."""
        if self.matches is None:
            raise RuntimeError("Matches not computed. Run _match_grains() first.")
        if self.inverse_matches is None:
            raise RuntimeError(
                "inverse_matches not computed. Run _match_grains() first."
            )

        pos_tol = pos_tol or self.position_tolerance
        ori_tol = ori_tol or self.orientation_tolerance

        # valid forward links: stitched -> true
        valid_fwd = self._valid_match_mask(self.matches, pos_tol, ori_tol)
        claimed_true_by_fwd = set(self.matches.loc[valid_fwd, "idx_target"])

        # valid inverse links: true -> stitched
        valid_inv = self._valid_match_mask(self.inverse_matches, pos_tol, ori_tol)
        claimed_stitch_by_inv = set(self.inverse_matches.loc[valid_inv, "idx_target"])

        all_true = set(range(len(self.df_true)))
        all_stitch = set(range(len(self.df_stitch)))

        unmatched_true_fwd = list(all_true - claimed_true_by_fwd)  # plot RED
        unmatched_stitch_inv = list(all_stitch - claimed_stitch_by_inv)  # plot BLUE

        df_unmatched_true = self.df_true.iloc[unmatched_true_fwd]
        df_unmatched_stitch = self.df_stitch.iloc[unmatched_stitch_inv]

        df_unmatched_true.to_csv(
            os.path.join(self.output_dir, "unmatched_true.csv"), index=False
        )
        df_unmatched_stitch.to_csv(
            os.path.join(self.output_dir, "unmatched_stitch.csv"), index=False
        )

        print(
            f"Unmatched true grains: {len(df_unmatched_true)}. \nDetails saved to '{self.output_dir}/unmatched_true.csv'\n"
        )
        print(
            f"Unmatched stitched grains: {len(df_unmatched_stitch)}. \nDetails saved to '{self.output_dir}/unmatched_stitch.csv'"
        )

        if bounding_box is None:
            xyz = self.df_true[["X", "Y", "Z"]].to_numpy()
            mins = xyz.min(axis=0)
            maxs = xyz.max(axis=0)
            bounding_box = [*mins, *maxs]
            print("Bounding box auto-inferred from true dataset:")
            print(f"  X: [{mins[0]:.3f}, {maxs[0]:.3f}]")
            print(f"  Y: [{mins[1]:.3f}, {maxs[1]:.3f}]")
            print(f"  Z: [{mins[2]:.3f}, {maxs[2]:.3f}]")

        if plot and (len(df_unmatched_true) > 0 or len(df_unmatched_stitch) > 0):

            fig = plt.figure(figsize=(6, 12))
            xmin, xmax, ymin, ymax, zmin, zmax = bounding_box

            def draw_box_2d(ax, corners, color="black", lw=0.8):
                for (x0, y0), (x1, y1) in [
                    (corners[0], corners[1]),
                    (corners[1], corners[2]),
                    (corners[2], corners[3]),
                    (corners[3], corners[0]),
                ]:
                    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw)

            # XY projection
            if view3D:
                ax1 = fig.add_subplot(3, 1, 1)
            else:
                ax1 = fig.add_subplot(2, 1, 1)
            if len(df_unmatched_true) > 0:
                ax1.scatter(
                    df_unmatched_true["X"],
                    df_unmatched_true["Y"],
                    c="red",
                    s=8,
                    label="forward",
                )
            if len(df_unmatched_stitch) > 0:
                ax1.scatter(
                    df_unmatched_stitch["X"],
                    df_unmatched_stitch["Y"],
                    c="blue",
                    s=8,
                    label="backward",
                )
            draw_box_2d(ax1, [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_title("XY Projection")
            ax1.set_aspect("equal")
            ax1.legend(loc="upper right", fontsize=6)

            # XZ projection
            if view3D:
                ax2 = fig.add_subplot(3, 1, 2)
            else:
                ax2 = fig.add_subplot(2, 1, 2)
            if len(df_unmatched_true) > 0:
                ax2.scatter(
                    df_unmatched_true["X"],
                    df_unmatched_true["Z"],
                    c="red",
                    s=8,
                    label="forward",
                )
            if len(df_unmatched_stitch) > 0:
                ax2.scatter(
                    df_unmatched_stitch["X"],
                    df_unmatched_stitch["Z"],
                    c="blue",
                    s=8,
                    label="backward",
                )
            draw_box_2d(ax2, [(xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)])
            ax2.set_xlabel("X")
            ax2.set_ylabel("Z")
            ax2.set_title("XZ Projection")
            ax2.set_aspect("equal")
            ax2.legend(loc="upper right", fontsize=6)

            # 3D view
            if view3D:
                ax3 = fig.add_subplot(3, 1, 3, projection="3d")
                if len(df_unmatched_true) > 0:
                    ax3.scatter(
                        df_unmatched_true["X"],
                        df_unmatched_true["Y"],
                        df_unmatched_true["Z"],
                        c="red",
                        s=8,
                        label="forward",
                    )
                if len(df_unmatched_stitch) > 0:
                    ax3.scatter(
                        df_unmatched_stitch["X"],
                        df_unmatched_stitch["Y"],
                        df_unmatched_stitch["Z"],
                        c="blue",
                        s=8,
                        label="backward",
                    )

                # bounding box edges
                for s, e in [
                    ((xmin, ymin, zmin), (xmax, ymin, zmin)),
                    ((xmin, ymax, zmin), (xmax, ymax, zmin)),
                    ((xmin, ymin, zmax), (xmax, ymin, zmax)),
                    ((xmin, ymax, zmax), (xmax, ymax, zmax)),
                    ((xmin, ymin, zmin), (xmin, ymax, zmin)),
                    ((xmax, ymin, zmin), (xmax, ymax, zmin)),
                    ((xmin, ymin, zmax), (xmin, ymax, zmax)),
                    ((xmax, ymin, zmax), (xmax, ymax, zmax)),
                    ((xmin, ymin, zmin), (xmin, ymin, zmax)),
                    ((xmax, ymin, zmin), (xmax, ymin, zmax)),
                    ((xmin, ymax, zmin), (xmin, ymax, zmax)),
                    ((xmax, ymax, zmin), (xmax, ymax, zmax)),
                ]:
                    ax3.plot3D(*zip(s, e), color="black", linewidth=0.5)

                ax3.set_xlabel("X")
                ax3.set_ylabel("Y")
                ax3.set_zlabel("Z")
                ax3.set_title("3D View")
                ax3.legend(loc="upper right", fontsize=6)

            plt.tight_layout()
            out_path = os.path.join(self.output_dir, "figures", "unmatched_grains.png")
            plt.savefig(out_path, dpi=300)
            plt.close(fig)

            print(f"\nUnmatched grain plot saved to {out_path}")

        return df_unmatched_true, df_unmatched_stitch
