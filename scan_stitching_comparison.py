import os
import json
import pandas as pd
import sys
import matplotlib.pyplot as plt
import shutil
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

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
        cols = ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]
        self.df_true = self.df_true[cols].astype(float)
        self.df_stitch = self.df_stitch[cols].astype(float)

        # --- Convert Euler angles to radians if specified in degrees ---
        if self.orientation_units == "degrees":
            for col in ["Eul0", "Eul1", "Eul2"]:
                self.df_true[col] = np.deg2rad(self.df_true[col])
                self.df_stitch[col] = np.deg2rad(self.df_stitch[col])

        # --- Check for NaN values ---
        if self.df_true.isnull().any().any() or self.df_stitch.isnull().any().any():
            raise ValueError("NaN values detected in one or both CSV files.")

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
        """

        coords_source = source_df[["X", "Y", "Z"]].to_numpy()

        # Query nearest neighbors
        dist, idx = target_tree.query(coords_source, k=self.min_neighbors, workers=-1)

        # get the distances of potential matches
        s_idx, n_idx = np.indices(dist.shape)
        s_idx = s_idx.ravel()
        t_idx = idx.ravel()
        diff_pos = dist.ravel()

        # get orientation and radius differences
        ori_source = source_df[["Eul0", "Eul1", "Eul2"]].to_numpy()
        ori_target = target_df[["Eul0", "Eul1", "Eul2"]].to_numpy()
        rad_source = source_df["GrainRadius"].to_numpy()
        rad_target = target_df["GrainRadius"].to_numpy()

        diff_ori = np.linalg.norm(ori_source[s_idx] - ori_target[t_idx], axis=1)
        diff_rad = np.abs(rad_source[s_idx] - rad_target[t_idx]) /rad_target[t_idx]

        # cost matrix construction for scipy linear sum assignment
        cost = (self.weights["pos"] * diff_pos / self.position_tolerance
            + self.weights["ori"] * diff_ori / self.orientation_tolerance
            + self.weights["rad"] * diff_rad / self.radius_tolerance)

        n_source, n_target = len(source_df), len(target_df)
        cost_matrix = np.full((n_source, n_target), np.inf)
        cost_matrix[s_idx, t_idx] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        valid = cost_matrix[row_ind, col_ind] < np.inf

        # Construct mapping table
        match_rows = []
        for s, t, ok in zip(row_ind, col_ind, valid):
            if ok:
                mask = (s_idx == s) & (t_idx == t)
                if np.any(mask):
                    dp, dr, do = diff_pos[mask][0], diff_rad[mask][0], diff_ori[mask][0]
                    match_rows.append((int(s), int(t), dp, dr, do))

        return pd.DataFrame(
            match_rows,
            columns=["idx_source", "idx_target", "diff_pos_norm2", "diff_rad_percentage", "diff_ori_norm2"]
        )

    def _compute_statistics(self):
        """
        Compute matching statistics.
        """
        if self.matches is None or self.inverse_matches is None:
            raise RuntimeError("Matches not computed. Run _match_grains() first.")

        # --- Identify splits (trueID to multiple stitched IDs) ---
        split_detail = (
            self.matches.groupby("idx_target")["idx_source"]
            .apply(list)
            .reset_index()
            .rename(columns={"idx_target": "TrueGrainID", "idx_source": "MappedStitchedGrains"})
        )
        split_detail["NumMappedStitched"] = split_detail["MappedStitchedGrains"].apply(len)
        split_detail = split_detail[split_detail["NumMappedStitched"] > 1]
        split_detail.to_csv(os.path.join(self.output_dir, "split_grains.csv"), index=False)

        # --- Identify merges (stitchID to multiple trueIDs) ---
        merge_detail = (
            self.inverse_matches.groupby("idx_target")["idx_source"]
            .apply(list)
            .reset_index()
            .rename(columns={"idx_target": "StitchedGrainID", "idx_source": "MappedTrueGrains"})
        )
        merge_detail["NumMappedTrue"] = merge_detail["MappedTrueGrains"].apply(len)
        merge_detail = merge_detail[merge_detail["NumMappedTrue"] > 1]
        merge_detail.to_csv(os.path.join(self.output_dir, "merge_grains.csv"), index=False)

        # --- Summary dictionary ---
        self.metrics = {
            "n_true": len(self.df_true),
            "n_stitch": len(self.df_stitch),
            "n_matched": len(self.matches),
            "n_inverse_matched": len(self.inverse_matches),
            "n_splits": len(split_detail),
            "n_merges": len(merge_detail),
            "mean_pos_error": self.matches["diff_pos_norm2"].mean(),
            "mean_ori_error": self.matches["diff_ori_norm2"].mean(),
            "mean_rad_error": self.matches["diff_rad_percentage"].mean(),
            "max_pos_error": self.matches["diff_pos_norm2"].max(),
            "max_ori_error": self.matches["diff_ori_norm2"].max(),
            "max_rad_error": self.matches["diff_rad_percentage"].max(),
        }

        # Save summary
        with open(os.path.join(self.output_dir, "statistics_summary.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)

        # --- Save match tables ---
        self.matches.to_csv(os.path.join(self.output_dir, "stitch_to_true_matches.csv"), index=False)
        self.inverse_matches.to_csv(os.path.join(self.output_dir, "true_to_stitch_matches.csv"), index=False)

        print("\n=== Stitching Comparison Statistics ===")
        for k, v in self.metrics.items():
            print(f"{k:20s}: {v}")

        print(f"\nResults saved to: {self.output_dir}\n")
        print("Summary file:\n- statistics_summary.json\n\nDetailed match data:")
        print("- stitch_matches_to_true.csv")
        print("- true_matches_to_stitch.csv")
        print("\nSplit / merge analysis:")
        print("- split_grains.csv")
        print("- merge_grains.csv\n")

    def _plot_histograms(self):
        """Plot histograms of differences."""

        os.makedirs(os.path.join(self.output_dir, "figures"), exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(6, 9), tight_layout=True)
        plot_info = [
            ("diff_pos_norm2", "Position Error (length units)"),
            ("diff_ori_norm2", "Orientation Error (degrees)"),
            ("diff_rad_percentage", "Radius Error (relative)"),
        ]

        for ax, (col, label) in zip(axes, plot_info):
            if col not in self.matches.columns:
                continue
            ax.hist(self.matches[col], bins=40, alpha=0.75, edgecolor="black")
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
            rad_tol: float | None = None,
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
        pos_tol, ori_tol, rad_tol : float or None
            Tolerances for position/orientation/radius differences.
            If None, defaults to initialized class tolerances.
        """
        if self.matches is None:
            raise RuntimeError("Matches not computed. Run _match_grains() first.")

        pos_tol = pos_tol or self.position_tolerance
        ori_tol = ori_tol or self.orientation_tolerance
        rad_tol = rad_tol or self.radius_tolerance

        # --- Identify unmatched grains ---
        valid_mask = (
            (self.matches["diff_pos_norm2"] <= pos_tol)
            & (self.matches["diff_ori_norm2"] <= ori_tol)
            & (self.matches["diff_rad_percentage"] <= rad_tol)
        )

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
            ax1 = fig.add_subplot(3, 1, 1)
            if len(df_unmatched_true) > 0:
                ax1.scatter(df_unmatched_true["X"], df_unmatched_true["Y"], c="red", s=8, label="True")
            if len(df_unmatched_stitch) > 0:
                ax1.scatter(df_unmatched_stitch["X"], df_unmatched_stitch["Y"], c="blue", s=8, label="Stitch")
            draw_box_2d(ax1, [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_title("XY Projection")
            ax1.legend(loc="upper right", fontsize=6)

            # --- XZ projection ---
            ax2 = fig.add_subplot(3, 1, 2)
            if len(df_unmatched_true) > 0:
                ax2.scatter(df_unmatched_true["X"], df_unmatched_true["Z"], c="red", s=8, label="True")
            if len(df_unmatched_stitch) > 0:
                ax2.scatter(df_unmatched_stitch["X"], df_unmatched_stitch["Z"], c="blue", s=8, label="Stitch")
            draw_box_2d(ax2, [(xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)])
            ax2.set_xlabel("X")
            ax2.set_ylabel("Z")
            ax2.set_title("XZ Projection")
            ax2.legend(loc="upper right", fontsize=6)

            # --- 3D view ---
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