import os
import json
import pandas as pd
import sys
import matplotlib.pyplot as plt
import shutil
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from orientation_helper import misorientation 

class ScanStitchingComparison:
    def __init__(
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
        weights: dict = {"pos": 1.0, "ori": 0.0, "rad": 0.0},
        min_neighbors: int = 5):
    
        """
        Initialize comparison between true and stitched grain data.

        Parameters
        ----------
        true_csv : str
            Path to ground-truth CSV file.
            (must contain columns: X, Y, Z, GrainRadius, Eul0, Eul1, Eul2)
        stitch_csv : str
            Path to stitched or reconstructed CSV file.
            (must contain columns: X, Y, Z, GrainRadius, Eul0, Eul1, Eul2)
        position_tolerance : float, default=1e-3
            Max centroid distance for candidate matching.
        orientation_tolerance : float, default=1e-3
            Max Euler misorientation difference.
        radius_tolerance : float, default=1e-3
            Relative tolerance for grain radius.
        orientation_units : str, default="radians"
            Units of Euler angles ("radians" or "degrees").
        weights : dict, optional
            Weights for position/orientation/radius cost terms.
        min_neighbors : int, default=3
            Minimum KD-tree neighbors for adaptive search.
        """

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
                raise ValueError(f"Failed to read {label}: {e}")
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

        # placeholders
        self.df_true = None
        self.df_stitch = None
        self.kdtree_true = None
        self.kdtree_stitch = None
        self.matches = None
        self.inverse_matches = None
        self.metrics = {}

    def run_comparison(self):
        
        self.load_data()

        # identify matching pairs
        self._build_kdtree()
        self._match_grains()

        # plotting and printout statistics (these called are independent)
        self._compute_statistics()
        self._plot_histograms()

    def load_data(self):
        """Load and preprocess true and stitched grain data."""
        # --- Read CSVs ---
        self.df_true = pd.read_csv(self.true_csv)
        self.df_stitch = pd.read_csv(self.stitch_csv)

        # --- Enforce column order and numeric types ---
        base_cols = ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]

        # keep any extra cols that exist in the CSVs (debug columns, ScanID, etc.)
        extra_true = [c for c in self.df_true.columns if c not in base_cols]
        extra_stitch = [c for c in self.df_stitch.columns if c not in base_cols]

        self.df_true = self.df_true[base_cols + extra_true].copy()
        self.df_stitch = self.df_stitch[base_cols + extra_stitch].copy()

        self.df_true[base_cols] = self.df_true[base_cols].astype(float)
        self.df_stitch[base_cols] = self.df_stitch[base_cols].astype(float)

        # --- Convert Euler angles to radians if specified in degrees ---
        if self.orientation_units == "degrees":
            for col in ["Eul0", "Eul1", "Eul2"]:
                self.df_true[col] = np.deg2rad(self.df_true[col])
                self.df_stitch[col] = np.deg2rad(self.df_stitch[col])

        # --- Basic console summary ---
        print(f"Loaded {len(self.df_true)} grains from true dataset.")
        print(f"Loaded {len(self.df_stitch)} grains from stitched dataset.\n")
    
    def _build_kdtree(self):
        """Construct KD-trees for true and stitched grain centroids."""
        if self.df_true is None or self.df_stitch is None:
            raise RuntimeError("Datasets not loaded. Run load_data() first.")

        # True reference KD-tree
        coords_true = self.df_true[["X", "Y", "Z"]].to_numpy()
        self.kdtree_true = cKDTree(coords_true)
        print(f"KD-tree built with {len(coords_true)} true grains.")

        # Stitched dataset KD-tree
        coords_stitch = self.df_stitch[["X", "Y", "Z"]].to_numpy()
        self.kdtree_stitch = cKDTree(coords_stitch)
        print(f"KD-tree built with {len(coords_stitch)} stitched grains.\n")
    
    def _match_grains(self):
        """
        Match stitched grains to true grains using KD-tree neighbor search and optimal assignment.
        """

        if self.kdtree_true is None or self.kdtree_stitch is None:
            raise RuntimeError("KD-trees not built. Run _build_kdtree() first.")
        if self.df_true is None or self.df_stitch is None:
            raise RuntimeError("Data not loaded. Run load_data() first.")

        self.matches = self._build_mapping(
            self.df_stitch, self.df_true, self.kdtree_true
        )
        print(f"\nMatched {len(self.matches)} stitched grains to true grains.\n")

        self.inverse_matches = self._build_mapping(
            self.df_true, self.df_stitch, self.kdtree_stitch
        )
        print(f"Matched {len(self.inverse_matches)} true grains to stitched grains.\n")

    def _build_mapping(self, source_df, target_df, target_tree):
        """
        Build nearest-neighbor mapping between two datasets.

        No hard cutoff in the cost. All k-NN candidates are allowed,
        and tolerances are enforced later via masks.
        """

        coords_source = source_df[["X", "Y", "Z"]].to_numpy()
        n_source, n_target = len(source_df), len(target_df)

        if n_source == 0 or n_target == 0:
            return pd.DataFrame(
                columns=[
                    "idx_source", "idx_target",
                    "diff_pos_norm2", "diff_rad_percentage", "diff_ori",
                ]
            )

        # KD-tree query
        k = min(self.min_neighbors, n_target)
        dist, idx = target_tree.query(coords_source, k=k, workers=-1)

        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        s_idx, _ = np.indices(dist.shape)
        s_idx = s_idx.ravel()
        t_idx = idx.ravel()
        diff_pos = dist.ravel()

        ori_source = source_df[["Eul0", "Eul1", "Eul2"]].to_numpy()
        ori_target = target_df[["Eul0", "Eul1", "Eul2"]].to_numpy()
        rad_source = source_df["GrainRadius"].to_numpy()
        rad_target = target_df["GrainRadius"].to_numpy()

        
        # diff_ori = np.linalg.norm(ori_source[s_idx] - ori_target[t_idx], axis=1)
        diff_ori_t = misorientation(
            ori_source[s_idx], ori_target[t_idx],
            angle_convention=self.orientation_convention,   # e.g. "kocks"
            angle_type=self.orientation_units,        # "degrees" or "radians"
            symmetry=self.symmetry,                   # e.g. "432"
        )
        diff_ori = diff_ori_t.detach().cpu().numpy().astype(float)
        
        diff_rad = np.abs(rad_source[s_idx] - rad_target[t_idx]) / rad_target[t_idx]

        # tolerances and weights
        w_pos = self.weights.get("pos", 1.0)
        w_ori = self.weights.get("ori", 0.0)
        w_rad = self.weights.get("rad", 0.0)

        eps = 1e-14
        ptol = self.position_tolerance if self.position_tolerance != 0.0 else eps
        otol = self.orientation_tolerance if self.orientation_tolerance != 0.0 else eps
        rtol = self.radius_tolerance if self.radius_tolerance != 0.0 else eps

        cost = (
            w_pos * (diff_pos / ptol) +
            w_ori * (diff_ori / otol) +
            w_rad * (diff_rad / rtol)
        )

        order = np.lexsort((cost, s_idx))  # group by source, then cost
        s_s = s_idx[order]
        t_s = t_idx[order]
        dp_s = diff_pos[order]
        dr_s = diff_rad[order]
        do_s = diff_ori[order]

        best_t = np.full(n_source, -1, dtype=int)
        best_dp = np.full(n_source, np.inf)
        best_dr = np.full(n_source, np.inf)
        best_do = np.full(n_source, np.inf)

        seen = np.zeros(n_source, dtype=bool)
        for s, t, dp, dr, do in zip(s_s, t_s, dp_s, dr_s, do_s):
            if not seen[s]:
                best_t[s] = int(t)
                best_dp[s] = float(dp)
                best_dr[s] = float(dr)
                best_do[s] = float(do)
                seen[s] = True

                rows = []

        rows = [
            (int(s), int(best_t[s]), best_dp[s], best_dr[s], best_do[s])
            for s in range(n_source) if best_t[s] >= 0
        ]

        return pd.DataFrame(
            rows,
            columns=[
                "idx_source",
                "idx_target",
                "diff_pos_norm2",
                "diff_rad_percentage",
                "diff_ori",
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
            (df_matches["diff_pos_norm2"] <= pos_tol ) &
            (df_matches["diff_ori"] <= ori_tol )
        )

    def _compute_statistics(self):
        """
        Compute matching statistics on matches that satisfy the default tolerances.
        """
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
            .rename(columns={"idx_target": "TrueGrainID", "idx_source": "MappedStitchedGrains"})
        )
        split_detail["NumMappedStitched"] = split_detail["MappedStitchedGrains"].apply(len)
        split_detail = split_detail[split_detail["NumMappedStitched"] > 1]
        split_detail.to_csv(os.path.join(self.output_dir, "split_grains.csv"), index=False)

        # merges: one stitched grain matched by multiple true grains
        merge_detail = (
            inverse_valid.groupby("idx_target")["idx_source"]
            .apply(list)
            .reset_index()
            .rename(columns={"idx_target": "StitchedGrainID", "idx_source": "MappedTrueGrains"})
        )
        merge_detail["NumMappedTrue"] = merge_detail["MappedTrueGrains"].apply(len)
        merge_detail = merge_detail[merge_detail["NumMappedTrue"] > 1]
        merge_detail.to_csv(os.path.join(self.output_dir, "merge_grains.csv"), index=False)

        def safe_stat(series, fn):
            return fn(series) if len(series) > 0 else None

        self.metrics = {
            "n_true": len(self.df_true),
            "n_stitch": len(self.df_stitch),
            "n_matched": len(matches_valid),
            "n_inverse_matched": len(inverse_valid),
            "n_splits": len(split_detail),
            "n_merges": len(merge_detail),
            "mean_pos_error": safe_stat(matches_valid["diff_pos_norm2"], np.mean),
            "mean_ori_error": safe_stat(matches_valid["diff_ori"], np.mean),
            "mean_rad_error": safe_stat(matches_valid["diff_rad_percentage"], np.mean),
            "max_pos_error": safe_stat(matches_valid["diff_pos_norm2"], np.max),
            "max_ori_error": safe_stat(matches_valid["diff_ori"], np.max),
            "max_rad_error": safe_stat(matches_valid["diff_rad_percentage"], np.max),
        }

        with open(os.path.join(self.output_dir, "statistics_summary.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)

        self.matches.to_csv(os.path.join(self.output_dir, "stitch_to_true_matches.csv"), index=False)
        self.inverse_matches.to_csv(os.path.join(self.output_dir, "true_to_stitch_matches.csv"), index=False)

        print("\n=== Stitching Comparison Statistics (within tolerances) ===")
        for k, v in self.metrics.items():
            print(f"{k:20s}: {v}")


    def _plot_histograms(self):
        """Plot histograms of differences."""

        os.makedirs(os.path.join(self.output_dir, "figures"), exist_ok=True)

        valid_mask = self._valid_match_mask(self.matches)
        matches_valid = self.matches[valid_mask]

        fig, axes = plt.subplots(3, 1, figsize=(6, 9), tight_layout=True)
        plot_info = [
            ("diff_pos_norm2", "Position Error (length units)"),
            ("diff_ori", "Orientation Error (degrees)"),
            ("diff_rad_percentage", "Radius Error (relative)"),
        ]

        for ax, (col, label) in zip(axes, plot_info):
            if col not in matches_valid.columns:
                continue
            ax.hist(matches_valid[col], bins=40, alpha=0.75, edgecolor="black")
            ax.set_xlabel(label)
            ax.set_ylabel("Number of Grains")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        fig.savefig(os.path.join(self.output_dir, "figures", "error_histograms.png"), dpi=300)
        plt.close(fig)

        print(f"Error histograms saved to: {os.path.join(self.output_dir, 'figures', 'error_histograms.png')}\n")
    
    def _get_unmatched_grains(
            self,
            bounding_box: list | None = None,
            plot: bool = True,
            pos_tol: float | None = None,
            ori_tol: float | None = None,
            view3D: bool = False
    ):
        """
        Identify unmatched grains beyond given tolerances and optionally plot them.

        Parameters
        ----------
        bounding_box : list or None
            Domain bounding box [xmin, xmax, ymin, ymax, zmin, zmax].
            If None, inferred from true dataset.
        plot : bool, default=True
            Whether to generate a 3D scatter plot of unmatched grains.
        pos_tol : float or None
            Tolerances for position differences.
            If None, defaults to initialized class tolerances.
        """
        if self.matches is None:
            raise RuntimeError("Matches not computed. Run _match_grains() first.")

        pos_tol = pos_tol or self.position_tolerance
        ori_tol = ori_tol or self.orientation_tolerance

        # --- Identify unmatched grains ---
        valid_mask = self._valid_match_mask(self.matches, pos_tol, ori_tol)

        matched_true = set(self.matches.loc[valid_mask, "idx_target"])
        matched_stitch = set(self.matches.loc[valid_mask, "idx_source"])

        all_true = set(range(len(self.df_true)))
        all_stitch = set(range(len(self.df_stitch)))

        unmatched_true = list(all_true - matched_true)
        unmatched_stitch = list(all_stitch - matched_stitch)

        df_unmatched_true = self.df_true.iloc[unmatched_true]
        df_unmatched_stitch = self.df_stitch.iloc[unmatched_stitch]

        df_unmatched_true.to_csv(
            os.path.join(self.output_dir, "unmatched_true.csv"), index=False
        )
        df_unmatched_stitch.to_csv(
            os.path.join(self.output_dir, "unmatched_stitch.csv"), index=False
        )

        print(f"Unmatched true grains: {len(df_unmatched_true)}. \nDetails saved to '{self.output_dir}/unmatched_true.csv'\n")
        print(f"Unmatched stitched grains: {len(df_unmatched_stitch)}. \nDetails saved to '{self.output_dir}/unmatched_stitch.csv'")

        # --- Determine bounding box ---
        if bounding_box is None:
            xyz = self.df_true[["X", "Y", "Z"]].to_numpy()
            mins = xyz.min(axis=0)
            maxs = xyz.max(axis=0)
            bounding_box = [*mins, *maxs]
            print("Bounding box auto-inferred from true dataset:")
            print(f"  X: [{mins[0]:.3f}, {maxs[0]:.3f}]")
            print(f"  Y: [{mins[1]:.3f}, {maxs[1]:.3f}]")
            print(f"  Z: [{mins[2]:.3f}, {maxs[2]:.3f}]")

        # --- Plot unmatched grains ---
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

            # --- XY projection ---
            if view3D:
                ax1 = fig.add_subplot(3, 1, 1)
            else:
                ax1 = fig.add_subplot(2, 1, 1)
            if len(df_unmatched_true) > 0:
                ax1.scatter(df_unmatched_true["X"], df_unmatched_true["Y"], c="red", s=8, label="True")
            if len(df_unmatched_stitch) > 0:
                ax1.scatter(df_unmatched_stitch["X"], df_unmatched_stitch["Y"], c="blue", s=8, label="Stitch")
            draw_box_2d(ax1, [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_title("XY Projection")
            ax1.set_aspect('equal')
            ax1.legend(loc="upper right", fontsize=6)

            # --- XZ projection ---
            if view3D:
                ax2 = fig.add_subplot(3, 1, 2)
            else:
                ax2 = fig.add_subplot(2, 1, 2)
            if len(df_unmatched_true) > 0:
                ax2.scatter(df_unmatched_true["X"], df_unmatched_true["Z"], c="red", s=8, label="True")
            if len(df_unmatched_stitch) > 0:
                ax2.scatter(df_unmatched_stitch["X"], df_unmatched_stitch["Z"], c="blue", s=8, label="Stitch")
            draw_box_2d(ax2, [(xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)])
            ax2.set_xlabel("X")
            ax2.set_ylabel("Z")
            ax2.set_title("XZ Projection")
            ax2.set_aspect('equal')
            ax2.legend(loc="upper right", fontsize=6)

            # --- 3D view ---
            if view3D:
                ax3 = fig.add_subplot(3, 1, 3, projection="3d")
                if len(df_unmatched_true) > 0:
                    ax3.scatter(df_unmatched_true["X"], df_unmatched_true["Y"], df_unmatched_true["Z"], c="red", s=8, label="True")
                if len(df_unmatched_stitch) > 0:
                   ax3.scatter(df_unmatched_stitch["X"], df_unmatched_stitch["Y"], df_unmatched_stitch["Z"], c="blue", s=8, label="Stitch")

                # Bounding box (3D)
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