#!/usr/bin/env python3
# construct_voronoi_mesh.py

import os
import subprocess
import pandas as pd
import sys
import subprocess
import shutil
import numpy as np

class VoronoiMeshBuilder:
    def __init__(
        self,
        input_csv,
        output_dir,
        bounding_box,
        dim=3,
        weighted=False,
        gmsh_version="4.12.2",
        neper_version="4.10.1",
        auto_fix_bbox=False,
        bbox_fix_mode=None,
        bbox_tolerance=0.0,  # percentage tolerance (e.g., 5.0 = 5%)
        auto_rotate=False,
        rotate_angles=(0, 0, 0),    # rotation angles (rotate around X -> Y -> Z)
        rotate_convention= "xyz", 
        angle_identifier=None,
        orientation_descriptor = "euler-bunge",
        orientation_active_convention=False,
        unit="deg",          # rotation unit ('deg' or 'rad')
        elastic_strain_identifier=None,
        strain_unit="microstrain",
        env=None
    ):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.bounding_box = bounding_box
        self.dim = dim
        self.weighted = weighted
        self.gmsh_version = gmsh_version
        self.neper_version = neper_version
        self.auto_fix_bbox = auto_fix_bbox
        self.bbox_fix_mode = bbox_fix_mode
        self.bbox_tolerance = bbox_tolerance
        self.rotate_convention = rotate_convention
        self.ori_descriptor = orientation_descriptor
        self.angle_id = angle_identifier
        self.rotate_matrix = np.eye(3)
        self.auto_rotate = auto_rotate
        self.orientation_active_convention = orientation_active_convention
        self.data = None
        self.elastic_strain_id = elastic_strain_identifier
        self.strain_unit = strain_unit

        self.env = self.check_dependencies() if env is None else env

        # --- Rotation parameters ---
        if unit not in ("deg", "rad"):
            raise ValueError("Rotation unit must be 'deg' or 'rad'.")
        self.rotate_angles = rotate_angles
        self.unit = unit

        if self.rotate_convention != "xyz":
            raise ValueError("currently only rotate_convention = 'xyz' is supported.")

        valid_modes = ["remove_points", "extend_bounding_box"]

        if self.auto_fix_bbox:
            if self.bbox_fix_mode not in valid_modes:
                raise ValueError(
                    f"Invalid bbox_fix_mode='{self.bbox_fix_mode}'. "
                    f"When auto_fix_bbox=True, bbox_fix_mode must be one of {valid_modes}."
                )
        else:
            if self.bbox_fix_mode is not None:
                print(
                    f"Warning: bbox_fix_mode='{self.bbox_fix_mode}' "
                    f"is ignored since auto_fix_bbox=False."
                )
        
        # -- check strain validity ---
        if self.elastic_strain_id is not None:
            if len(self.elastic_strain_id) != 9:
                raise ValueError("elastic_strain_identifier must contain exactly 9 components.")
            if self.strain_unit not in ("microstrain", "strain"):
                raise ValueError(f"Invalid strain_unit '{self.strain_unit}'. Must be 'microstrain' or 'strain'.")

        if self.angle_id is not None:
            if not isinstance(self.angle_id, (list, tuple)) or len(self.angle_id) != 3:
                raise ValueError("angle_identifier must contain 3 elements if provided.")

        # -- check dim validity ---
        if self.dim not in [2, 3]:
            raise ValueError("Dimension 'dim' must be either 2 or 3.")

    def check_dependencies(self):
        """
        Check and install Neper & Gmsh locally under ~/.local if not present.
        Note: this include installing the GSL and OpenBLAS libraries as well (for Neper).
        """

        home = os.path.expanduser("~")
        prefix = os.path.join(home, ".local")
        env = os.environ.copy()
        env["PATH"] = f"{prefix}/bin:" + env["PATH"]
        env["LD_LIBRARY_PATH"] = f"{prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")

        def is_installed(cmd):
            return shutil.which(cmd, path=env["PATH"]) is not None

        def run(cmd, cwd=None):
            print(">", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=cwd, env=env)

        os.makedirs(prefix, exist_ok=True)

        # --- Gmsh via pip ---
        gmsh_installed = False
        try:
            import gmsh
            gmsh_installed = True
            print(f"Gmsh already available (v{gmsh.__version__})")
        except ImportError:
            print(f"Gmsh not found — installing via pip...")
            run([sys.executable, "-m", "pip", "install", f"gmsh=={self.gmsh_version}"])
            gmsh_installed = True
            print("Gmsh installed successfully via pip.")

        # --- Install GSL locally if missing ---
        gsl_lib = os.path.join(prefix, "lib", "libgsl.so")
        if not os.path.exists(gsl_lib):
            print("Installing GSL locally...")
            run(["wget", "https://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz", "-O", os.path.join(home, "gsl.tar.gz")])
            run(["tar", "-xzf", "gsl.tar.gz"], cwd=home)
            gsl_src = next(
                (os.path.join(home, d) for d in os.listdir(home) if d.startswith("gsl-")),
                None
            )
            if gsl_src:
                run(["./configure", f"--prefix={prefix}"], cwd=gsl_src)
                run(["make", "-j", str(os.cpu_count())], cwd=gsl_src)
                run(["make", "install"], cwd=gsl_src)
            else:
                raise RuntimeError("GSL extraction failed.")

        # --- Install OpenBLAS locally if missing ---
        openblas_lib = os.path.join(prefix, "lib", "libopenblas.so.0")
        if not os.path.exists(openblas_lib):
            print("Installing OpenBLAS locally...")
            progs_dir = os.path.expanduser("~/Progs")
            os.makedirs(progs_dir, exist_ok=True)
            run(["git", "clone", "https://github.com/xianyi/OpenBLAS.git"], cwd=progs_dir)
            openblas_src = os.path.join(progs_dir, "OpenBLAS")
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            run([
                "make",
                f"PREFIX={prefix}",
                f"FC={conda_prefix}/bin/x86_64-conda-linux-gnu-gfortran",
                "-j", str(os.cpu_count())
            ], cwd=openblas_src)
            run(["make", "install", f"PREFIX={prefix}"], cwd=openblas_src)

        # --- Neper installation ---
        neper_installed = is_installed("neper")
        

        if not neper_installed:
            print("Installing Neper locally...")
            stable_version = "4.10.1"
            stable_url = f"https://neper.info/download/neper-{stable_version}.tar.gz"
            progs_dir = os.path.expanduser("~/Progs")
            os.makedirs(progs_dir, exist_ok=True)

            try:
                # --- TR2 stable release first ---
                print(f"Attempting official stable release v{stable_version}...")
                run(["wget", stable_url, "-O", os.path.join(progs_dir, f"neper-{stable_version}.tar.gz")])
                run(["tar", "-zxf", f"neper-{stable_version}.tar.gz"], cwd=progs_dir)
                neper_src_dir = os.path.join(progs_dir, f"neper-{stable_version}", "src")
            except subprocess.CalledProcessError:
                # --- Fallback to GitHub repositoR2 ---
                print("Stable release unavailable, cloning GitHub master instead...")
                repo_url = "https://github.com/rquey/neper.git"
                neper_src_dir = os.path.join(progs_dir, "neper", "src")
                if not os.path.exists(os.path.join(progs_dir, "neper")):
                    run(["git", "clone", repo_url, os.path.join(progs_dir, "neper")])
                else:
                    run(["git", "-C", os.path.join(progs_dir, "neper"), "pull"])

            # --- Build & install ---
            build_dir = os.path.join(neper_src_dir, "build")
            os.makedirs(build_dir, exist_ok=True)
            run([
                "cmake",
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
                "-DNEPER_INSTALL_BASH_COMPLETION=OFF",  # disable sudo path
                ".."
            ], cwd=build_dir)
            run(["make", "-j", str(os.cpu_count())], cwd=build_dir)
            run(["make", "install"], cwd=build_dir)

            if is_installed("neper"):
                neper_installed = True
                print("Neper installed successfully (stable or GitHub build).")
            else:
                raise RuntimeError("Neper installation failed.")

        else:
            print("Neper already available in PATH.")
        
        home = os.path.expanduser("~")
        prefix = os.path.join(home, ".local")

        final_env = os.environ.copy()
        final_env["PATH"] = f"{prefix}/bin:" + final_env.get("PATH", "")
        final_env["LD_LIBRARY_PATH"] = (
            f"{prefix}/lib:{prefix}/lib64:/usr/local/lib:/usr/lib:/usr/lib64:/lib/x86_64-linux-gnu"
        )

        self.env = final_env

        return final_env

    def read_input(self):
        """
        Read and validate input CSV file.
        Keeps required spatial columns (X, Y[, Z]) and optionally GrainRadius if present.
        """

        print(f"Reading input file: {self.input_csv}")
        df = pd.read_csv(self.input_csv)

        # --- Required coordinates ---
        if self.dim == 3:
            required_cols = ["X", "Y", "Z"]
        elif self.dim == 2:
            required_cols = ["X", "Y"]
        else:
            raise ValueError("dim must be 2 or 3.")

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for {self.dim}D input: {missing}")

        # --- Include GrainRadius if available ---
        optional_cols = ["GrainRadius"] if "GrainRadius" in df.columns else []
        
        # --- Include angle if available ---
        angle_cols = []

        if self.angle_id is not None:
            if isinstance(self.angle_id, (list, tuple)) and len(self.angle_id) > 0:
                present = [c for c in self.angle_id if c in df.columns]

                if len(present) not in (0, len(self.angle_id)):
                    missing_angles = [c for c in self.angle_id if c not in df.columns]
                    raise ValueError(
                        f"Partial orientation columns found. Present: {present}; "
                        f"Missing: {missing_angles}. Either include all of {self.angle_id} or none."
                    )

                if len(present) == len(self.angle_id):
                    angle_cols = present
                    print(f"Orientation columns found: {angle_cols}")
                else:
                    print("No orientation columns detected (angle_identifier not present in file).")
            else:
                raise TypeError("angle_identifier must be a list or tuple of 3 names if provided.")
        else:
            print("angle_identifier=None -> skipping orientation column parsing.")
        
        # --- Include elastic strain if available ---
        strain_cols = []

        if self.elastic_strain_id is not None:
            if isinstance(self.elastic_strain_id, (list, tuple)) and len(self.elastic_strain_id) > 0:
                present = [c for c in self.elastic_strain_id if c in df.columns]

                if len(present) not in (0, len(self.elastic_strain_id)):
                    missing_strain = [c for c in self.elastic_strain_id if c not in df.columns]
                    raise ValueError(
                        f"Partial strain columns found. Present: {present}; "
                        f"Missing: {missing_strain}. Either include all of {self.elastic_strain_id} or none."
                    )

                if len(present) == len(self.elastic_strain_id):
                    strain_cols = present
                    print(f"Elastic strain columns found: {strain_cols}")
                else:
                    print("No elastic strain columns detected (elastic_strain_identifier not present in file).")
            else:
                raise TypeError("elastic_strain_identifier must be a list or tuple of 9 names if provided.")
        else:
            print("elastic_strain_identifier=None → skipping strain column parsing.")

        used_cols = required_cols + optional_cols + angle_cols + strain_cols
        df = df[used_cols]

        # --- Convert radians to degrees if necessary ---
        if angle_cols and self.unit == "rad":
            print("Converting orientation angles from radians to degrees.")
            df[angle_cols] = np.degrees(df[angle_cols])
        
        # --- Convert strain units if necessary ---
        if strain_cols:
            if self.strain_unit == "microstrain":
                print("Converting microstrain to strain (x1e-6).")
                df[strain_cols] = df[strain_cols] * 1e-6

        print(f"Loaded {len(df)} points, using columns: {used_cols}\n")
        self.data = df

        # Apply rotation
        self.rotate()
    
    def rotate(self):
        """
        Rotate or align data based on user settings.

        Parameters
        ----------
        auto : bool, default=False
            If True, align dataset to its principal axes using PCA.
            If False, apply explicit rotation based on rotate_angles and rotate_convention ('xyz').

        Notes
        -----
        - When auto=True, rotation is determined automatically via PCA.
        - When auto=False, and rotate_convention == 'xyz', rotation uses scipy.spatial.transform.Rotation.
        - Weighted PCA is supported if 'GrainRadius' exists.
        """

        if self.data is None:
            raise RuntimeError("No data loaded. Call read_input() first.")

        df = self.data.copy()

        # --- Automatic alignment using PCA ---
        if self.auto_rotate:
            print("Auto-rotation enabled: aligning data to principal axes.")
            eigenvectors, _ = self.compute_principal_axes()

            coords = df[["X", "Y", "Z"]].to_numpy() if self.dim == 3 else df[["X", "Y"]].to_numpy()
            mean = np.mean(coords, axis=0)
            centered = coords - mean

            # Apply PCA rotation actively
            rotated = np.dot(centered, eigenvectors)

            # Ensure right-handed orientation
            if np.linalg.det(eigenvectors) < 0:
                eigenvectors[:, 2] *= -1
                rotated = np.dot(centered, eigenvectors)

            # Align PCA axes’ signs to global XYZ
            ref_axis = np.eye(3)
            signs = np.sign(np.diag(np.dot(eigenvectors.T, ref_axis)))
            eigenvectors *= signs
            rotated = np.dot(centered, eigenvectors)

            if self.dim == 3:
                df[["X", "Y", "Z"]] = rotated
            else:
                df[["X", "Y"]] = rotated[:, :2]

            self.data = df
            self.rotate_matrix = eigenvectors
            print("Data rotated to align with principal axes.\n")
            return
        else:
            # --- Manual rotation using scipy ---
            print("Manual rotation:")
            from scipy.spatial.transform import Rotation as R

            ang = np.array(self.rotate_angles, dtype=float)
            if self.unit == "deg":
                ang = np.radians(ang)

            if self.rotate_convention == "xyz":
                rot = R.from_euler("xyz", ang, degrees=False)
                self.rotate_matrix = rot.as_matrix()
                if self.dim == 3:
                    df[["X", "Y", "Z"]] = np.dot(df[["X", "Y", "Z"]].to_numpy(), self.rotate_matrix.T)
                else:
                    df[["X", "Y"]] = np.dot(df[["X", "Y"]].to_numpy(), self.rotate_matrix[:2, :2].T)

            self.data = df
            print(f"Rotated dataset by rotate_angles {self.rotate_angles} ({self.unit}).\n")

    def compute_principal_axes(self):
        """
        Compute principal axes of the dataset via PCA (equal weighting).
        Returns (eigenvectors, eigenvalues), where eigenvectors form columns of a rotation matrix.

        Notes
        -----
        - Supports both 2D and 3D.
        - Ensures right-handed orientation of eigenvectors.
        - Ignores any weighting or GrainRadius column.
        """

        from sklearn.decomposition import PCA

        if self.data is None or self.data.empty:
            raise RuntimeError("Data not loaded. Call read_input() first.")

        df = self.data.copy()
        coords = df[["X", "Y"]].to_numpy() if self.dim == 2 else df[["X", "Y", "Z"]].to_numpy()

        # --- Center coordinates ---
        mean = np.mean(coords, axis=0)
        centered = coords - mean

        # --- PCA (equal weights) ---
        pca = PCA(n_components=self.dim)
        pca.fit(centered)

        eigenvectors = pca.components_.T  # columns = principal axes
        eigenvalues = pca.explained_variance_

        ratios = pca.explained_variance_ratio_
        print(ratios, " -> ", np.round(ratios * 100, 1), "%")

        # --- Ensure right-handed orientation ---
        if self.dim == 3 and np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1
        
        # --- Renormalize ---
        u, _, vT = np.linalg.svd(eigenvectors)
        eigenvectors = np.dot(u, vT)

        print(f"Principal axes (columns):\n{eigenvectors}")

        return eigenvectors, eigenvalues

    def plot_centroids(self, plot_box=True,save_path="centroid_plot.png"):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        """
        Plot centroid data in three orthogonal views:
        - Z view (X vs Y)
        - X view (Y vs Z)
        - Y view (X vs Z)
        """

        if self.data is None or self.data.empty:
            raise RuntimeError("No centroid data loaded. Call read_input() first.")

        df = self.data.copy()
        bb = self.bounding_box

        if self.dim == 2:

            xmin, xmax, ymin, ymax = map(float, bb)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(df["X"], df["Y"], s=10, color="black", alpha=0.6)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")

            # bbox rectangle
            if plot_box:
                ax.add_patch(Rectangle((bb[0], bb[2]), bb[1] - bb[0], bb[3] - bb[2],
                                   fill=False, lw=1, ls="-", ec="r"))

            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"2D centroid plot saved: {save_path}")
            return

        # 3D case: three projections
        xmin, xmax, ymin, ymax, zmin, zmax = map(float, bb)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Z view (X-Y)
        if plot_box:
            axes[0].add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, lw=1, ls="-", ec="r"))

        axes[0].scatter(df["X"], df["Y"], s=10, color="black", alpha=0.6)
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].set_title("Z-view (X-Y)")

        # X view (Y-Z)
        if plot_box:
            axes[1].add_patch(Rectangle((ymin, zmin), ymax - ymin, zmax - zmin,
                                    fill=False, lw=1, ls="-", ec="r"))
        axes[1].scatter(df["Y"], df["Z"], s=10, color="black", alpha=0.6)
        axes[1].set_xlabel("Y")
        axes[1].set_ylabel("Z")
        axes[1].set_title("X-view (Y-Z)")

        # Y view (X-Z)
        if plot_box:
            axes[2].add_patch(Rectangle((xmin, zmin), xmax - xmin, zmax - zmin,
                                    fill=False, lw=1, ls="-", ec="r"))
        axes[2].scatter(df["X"], df["Z"], s=10, color="black", alpha=0.6)
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Z")
        axes[2].set_title("Y-view (X-Z)")

        for ax in axes:
            ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    def remove_outside_points(self, bbox=None, tolerance=0.0):
        """
        Remove out-of-bounds points and all associated per-grain data
        (coordinates, GrainRadius, Euler angles, strain, etc.)
        stored in self.data.
        """
        if self.data is None or self.data.empty:
            raise RuntimeError("No data loaded. Call read_input() first.")

        bbox = np.array(bbox if bbox is not None else self.bounding_box, dtype=float)
        tol = tolerance / 100.0

        # shrink inward
        for i in range(0, len(bbox), 2):
            low, high = bbox[i], bbox[i + 1]
            size = high - low
            bbox[i] = low + tol * size
            bbox[i + 1] = high - tol * size

        df = self.data

        # print before bounds
        if self.dim == 3:
            print(
                f"\nBounding box: "
                f"X=({bbox[0]:.4f},{bbox[1]:.4f}), "
                f"Y=({bbox[2]:.4f},{bbox[3]:.4f}), "
                f"Z=({bbox[4]:.4f},{bbox[5]:.4f})"
            )
            print(
                f" Data extents before: "
                f"X=({df['X'].min():.4f},{df['X'].max():.4f}), "
                f"Y=({df['Y'].min():.4f},{df['Y'].max():.4f}), "
                f"Z=({df['Z'].min():.4f},{df['Z'].max():.4f})"
            )
        else:
            print(
                f"\nBounding box: "
                f"X=({bbox[0]:.4f},{bbox[1]:.4f}), "
                f"Y=({bbox[2]:.4f},{bbox[3]:.4f})"
            )
            print(
                f"Data extents before: "
                f"X=({df['X'].min():.4f},{df['X'].max():.4f}), "
                f"Y=({df['Y'].min():.4f},{df['Y'].max():.4f})"
            )

        # mask for in-box points
        if self.dim == 2:
            mask = (
                (df["X"] >= bbox[0]) & (df["X"] <= bbox[1]) &
                (df["Y"] >= bbox[2]) & (df["Y"] <= bbox[3])
            )
        else:
            mask = (
                (df["X"] >= bbox[0]) & (df["X"] <= bbox[1]) &
                (df["Y"] >= bbox[2]) & (df["Y"] <= bbox[3]) &
                (df["Z"] >= bbox[4]) & (df["Z"] <= bbox[5])
            )

        before = len(df)
        self.data = df.loc[mask].reset_index(drop=True)
        removed = before - len(self.data)

        print(
            f"\n\nRemoved {removed} out-of-bounds points; "
            f"{len(self.data)} remain.\n\n"
        )

        # after filtering, recompute
        dfa = self.data
        if self.dim == 3:
            print(
                f"Data extents after:  "
                f"X=({dfa['X'].min():.4f},{dfa['X'].max():.4f}), "
                f"Y=({dfa['Y'].min():.4f},{dfa['Y'].max():.4f}), "
                f"Z=({dfa['Z'].min():.4f},{dfa['Z'].max():.4f})"
            )
        else:
            print(
                f"Data extents after:  "
                f"X=({dfa['X'].min():.4f},{dfa['X'].max():.4f}), "
                f"Y=({dfa['Y'].min():.4f},{dfa['Y'].max():.4f})"
            )

    def validate_bounding_box(self):
        """
        Validate or correct bounding box.
        Supports:
        - remove_points: remove out-of-bound points (with tolerance)
        - extend_bounding_box: expand domain to enclose all points (+ margin)
        """
        print("Validating bounding box...")

        if self.data is None:
            raise RuntimeError("validate_bounding_box() called before reading input data.")

        if not self.auto_fix_bbox:
            print("Auto-fix disabled — using provided bounding box as-is.")
        else:
        

            df = self.data.copy()

            if self.bbox_fix_mode == "remove_points":
                print(
                    f"Auto-fix enabled: removing points outside bounding box "
                    f"with {self.bbox_tolerance}% tolerance."
                )
                self.remove_outside_points(self.bounding_box, self.bbox_tolerance)

            elif self.bbox_fix_mode == "extend_bounding_box":
                print(
                    f"Auto-fix enabled: extending bounding box to enclose all points "
                    f"with +{self.bbox_tolerance}% margin."
                )

                bbox = np.array(self.bounding_box, dtype=float)
                tol = self.bbox_tolerance / 100.0

                # Determine data extents
                if self.dim == 2:
                    data_extents = [
                        df["X"].min(), df["X"].max(),
                        df["Y"].min(), df["Y"].max()
                    ]
                else:
                    data_extents = [
                        df["X"].min(), df["X"].max(),
                        df["Y"].min(), df["Y"].max(),
                        df["Z"].min(), df["Z"].max()
                    ]

                # Expand bounding box to include data + margin
                new_bbox = bbox.copy()
                for i in range(0, len(bbox), 2):
                    low, high = bbox[i], bbox[i + 1]
                    data_min, data_max = data_extents[i], data_extents[i + 1]
                    # ensure box covers all data, then extend by tolerance
                    low_new = min(low, data_min)
                    high_new = max(high, data_max)
                    size = high_new - low_new
                    new_bbox[i] = low_new - tol * size
                    new_bbox[i + 1] = high_new + tol * size

                print(f"-> Original bounding box: {bbox.tolist()}")
                print(f"-> Data extents:          {data_extents}")
                print(f"-> Extended bounding box: {new_bbox.tolist()}")

                # Update bounding box for tessellation
                self.bounding_box = new_bbox.tolist()

            else:
                raise ValueError(
                    f"Unexpected bbox_fix_mode='{self.bbox_fix_mode}'. "
                    "Valid options are: ['remove_points', 'extend_bounding_box']."
                )
        
        df = self.data
        bx = self.bounding_box

        if self.dim == 2:
            min_x, max_x = df["X"].min(), df["X"].max()
            min_y, max_y = df["Y"].min(), df["Y"].max()
            if bx[0] > min_x or bx[1] < max_x or bx[2] > min_y or bx[3] < max_y:
                raise ValueError(
                    f"Bounding box invalid for 2D data.\n"
                    f"Data range X=({min_x}, {max_x}), Y=({min_y}, {max_y})\n"
                    f"Bounding box X=({bx[0]}, {bx[1]}), Y=({bx[2]}, {bx[3]})"
                )
        else:
            min_x, max_x = df["X"].min(), df["X"].max()
            min_y, max_y = df["Y"].min(), df["Y"].max()
            min_z, max_z = df["Z"].min(), df["Z"].max()
            if (
                bx[0] > min_x or bx[1] < max_x or
                bx[2] > min_y or bx[3] < max_y or
                bx[4] > min_z or bx[5] < max_z
            ):
                raise ValueError(
                    f"Bounding box invalid for 3D data.\n"
                    f"Data range X=({min_x}, {max_x}), Y=({min_y}, {max_y}), Z=({min_z}, {max_z})\n"
                    f"Bounding box X=({bx[0]}, {bx[1]}), Y=({bx[2]}, {bx[3]}), Z=({bx[4]}, {bx[5]})"
                )

    def build_voronoi(self, 
                      option: str = "voronoi", 
                      generate_mesh: bool = False, 
                      relative_el_size: float = None,
                      morphoalgo: str = "praxis",
                      mesh_quality_min: float = 0.9,
                      CVT_iter: int = 1000):
        """
        Build Voronoi (or Laguerre) tessellation using Neper.
        """

        # check options validity
        valid_options = ["voronoi", "centroid", "centroidal", "centroidsize"]
        if option not in valid_options:
            raise ValueError(f"Invalid option='{option}'. Must be one of {valid_options}.")

        if self.data is None or self.data.empty:
            self.read_input()

        self.validate_bounding_box()

        df = self.data.copy()
        
        os.makedirs(self.output_dir, exist_ok=True)

        input_path = os.path.join(self.output_dir, "points.dat")
        weight_path = os.path.join(self.output_dir, "weights.dat")
        orientation_path = os.path.join(self.output_dir, "orientations.dat")

        if self.dim == 2:
            coord_cols = ["X", "Y"]
        else:
            coord_cols = ["X", "Y", "Z"]

        # --- File creation depending on option ---
        if option == "centroidsize":
            if "GrainRadius" not in df.columns:
                raise ValueError("centroidsize mode requires 'GrainRadius' column.")

            # combine coordinates + weight into one file
            df_out = df[coord_cols + ["GrainRadius"]]
            df_out.to_csv(input_path, sep=" ", index=False, header=False)
            print(f"Neper centroidsize input file created: {input_path}")
        else:
            # standard coordinate file
            df[coord_cols].to_csv(input_path, sep=" ", index=False, header=False)
            print(f"Neper coordinate file created: {input_path}")

            # if weighted (Laguerre/centroidal)
            if self.weighted:
                if "GrainRadius" not in df.columns:
                    raise ValueError("Weighted mode requires 'GrainRadius' column.")
                if self.dim == 3:
                    df["Weight"] = (4.0 / 3.0) * np.pi * (df["GrainRadius"] ** 3)
                    weight_type = "volume"
                else:
                    df["Weight"] = np.pi * (df["GrainRadius"] ** 2)
                    weight_type = "area"

                df["Weight"] /= df["Weight"].sum()
                df["Weight"].to_csv(weight_path, sep=" ", index=False, header=False)
                print(f"Neper weight file created ({weight_type}-based): {weight_path}")

        # --- Orientation (3-column) ---
        if all(c in df.columns for c in self.angle_id):
            df[self.angle_id].to_csv(orientation_path, sep=" ", index=False, header=False)
            print(f"Neper orientation file created: {orientation_path}")
            ori_args =["-ori", f"file({orientation_path},des={self.ori_descriptor})"]
        else:
            ori_args = []

        # --- Compute domain ---
        if self.dim == 2:
            xmin, xmax, ymin, ymax = self.bounding_box
            sx, sy = xmax - xmin, ymax - ymin
            tx, ty = xmin, ymin
            domain_arg = f'square({sx},{sy}):translate({tx},{ty})'
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounding_box
            sx, sy, sz = xmax - xmin, ymax - ymin, zmax - zmin
            tx, ty, tz = xmin, ymin, zmin
            domain_arg = f'cube({sx},{sy},{sz}):translate({tx},{ty},{tz})'

        # --- Morphological arguments ---
        if option == "centroidsize":
            morpho_args = ["-morpho", f"centroidsize:file({input_path})",
                           "-morphooptistop", f"iter={CVT_iter}",
                           "-morphooptialgo", morphoalgo]
            print(f"\nCentroid-size tessellation ({self.dim}D): combined coordinate+size input.")
        elif option == "centroid":
            morpho_args = ["-morpho", f"centroid:file({input_path})",
                           "-morphooptistop", f"iter={CVT_iter}",
                           "-morphooptialgo", morphoalgo]
            print(f"\nCentroid tessellation ({self.dim}D): only coordinate input. Weighted option will be ignored.")
        elif self.weighted:
            morphoalgo = "lloyd"
            morpho_args = [
                "-morpho", f"{option}",
                "-morphooptiini", f"coo:file({input_path}),weight:file({weight_path})",
                "-morphooptistop", f"iter={CVT_iter}",
                "-morphooptialgo", morphoalgo
            ]
            print(f"\nWeighted tessellation ({self.dim}D): using weight file {weight_path}.")
        else:
            morpho_args = [
                "-morpho", f"{option}",
                "-morphooptiini", f"coo:file({input_path})",
                "-morphooptistop", f"iter={CVT_iter}",
                "-morphooptialgo", morphoalgo
            ]
            print(f"\nUnweighted tessellation ({self.dim}D): standard Poisson-Voronoi.")

        tess_name = os.path.join(self.output_dir, "voronoi")

        neper_cmd = [
            "neper", "-T",
            "-n", str(len(df)),
            "-reg", str(1),
            "-dim", str(self.dim),
            "-domain", domain_arg,
            "-oridescriptor", "rotmat",
            "-o", tess_name,
            "-format", "tess,geo"
        ] + morpho_args + ori_args

        print("\n=== Running Neper Tessellation ===\n")
        print("> " + " ".join(neper_cmd))

        log_path = os.path.join(self.output_dir, "neper_voronoi_builder.log")

        with open(log_path, "w") as logf:
            print("\n=== Running Neper Tessellation ===", file=logf)
            print("> " + " ".join(neper_cmd), file=logf)
            logf.flush()

            subprocess.run(
                neper_cmd,
                check=True,
                env=self.env,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )

        print(f"Voronoi tessellation completed: {tess_name}.tess\n")

        print(f"Evaluating tessellation correctness, in figures folder\n")

        print(f"\n=== Updating cell properties with rotation ===\n")
        self.apply_rotation_to_properties(
            tess_file=tess_name + ".tess",
        )

        self.evaluate_output_voronoi(tess_name + ".tess")

        if generate_mesh:
            self.generate_mesh(
                tess_file=tess_name + ".tess",
                output_name=tess_name,
                format_type=["msh","vtk"],
                relative_cl=relative_el_size if relative_el_size else 1.0,
                mesh_quality_min=mesh_quality_min,
            )
        
    def apply_rotation_to_properties(self,
                                     tess_file: str):
        '''
         Parameters
        ----------
        tess_file : str
            Path to the Neper .tess file to modify.
        '''

        # --- Check that data has been loaded ---
        if not hasattr(self, "data") or self.data is None or self.data.empty:
            raise RuntimeError(
                "No data loaded. Run read_input() first before applying rotation to properties."
            )

        if not os.path.exists(tess_file):
            raise FileNotFoundError(f"Tessellation file not found: {tess_file}")

        transpose = self.orientation_active_convention

        # --- Backup file ---
        backup_path = tess_file + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy(tess_file, backup_path)

        # --- Read file and locate *ori section ---
        with open(tess_file, "r") as f:
            lines = f.readlines()

        ori_start, ori_end = None, None
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("*ori"):
                ori_start = i
                continue
            if ori_start is not None and line.strip().startswith("*") and i > ori_start + 1:
                ori_end = i
                break

        if ori_start is None:
            raise RuntimeError(f"No *ori section found in {tess_file}.")
        if ori_end is None:
            ori_end = len(lines)

        # skip descriptor line
        ori_lines = lines[ori_start + 2:ori_end]
        num_cells = len(ori_lines)

        if num_cells == 0:
            raise RuntimeError("No orientation data lines found after *ori header.")

        # --- No orientation provided then identity orientation for all cells ---
        if self.angle_id is None:
            print("No orientation data provided; assigning identity matrices.")
            identity_rot = np.eye(3)
            ori_rot = np.tile(identity_rot, (num_cells, 1, 1))
        else:
            ori_data = np.array([list(map(float, ln.split())) for ln in ori_lines], dtype=float)
            if ori_data.shape[1] != 9:
                raise ValueError(
                    f"Each *ori line must have 9 numbers; found {ori_data.shape[1]}."
                    "this indicates corrupted .tess file."
                )
            ori_mats = ori_data.reshape(-1, 3, 3)
            if transpose:
                ori_mats = np.transpose(ori_mats, (0, 2, 1))
            
            ori_rot = np.array([np.dot(R, self.rotate_matrix) for R in ori_mats])

        # --- Write back new orientations ---
        new_ori_lines = [
            " ".join(f"{val:.8f}" for val in R.flatten()) + "\n" for R in ori_rot
        ]
        lines[ori_start + 2:ori_end] = new_ori_lines

        with open(tess_file, "w") as f:
            f.writelines(lines)

        print(f"\nUpdated *ori section in {tess_file}")
        print(f"by applying the rotation matrix:\n{self.rotate_matrix}")

        # Apply rotation for elastic strain if available
        if self.elastic_strain_id is not None:
            print("\nApply rotation to elastic strain tensors\n")

            strain_tensors = self.data[self.elastic_strain_id].to_numpy(dtype=float).reshape(-1, 3, 3)

            # Rotate: e' = R e R^T
            R = self.rotate_matrix
            strain_rot = np.array([np.dot(R, np.dot(E, R.T)) for E in strain_tensors])

            self.data[self.elastic_strain_id] = strain_rot.reshape(-1, 9)
        else:
            print("No elastic strain identifier provided — skipping strain rotation.")

    def generate_mesh(
        self,
        tess_file: str,
        output_name: str = "voronoi",
        format_type = ("msh4",),
        relative_cl: float = 1.0,
        mesh_quality_min: float = 0.9,
        interface_type: str = "continuous",
        partition: int = 16,
    ):
        """
        Generate a finite element mesh from a Neper tessellation (.tess)
        using strictly valid Neper -M options.

        Rules:
        - Always tri for 2D, tet for 3D.
        - Always 2nd-order elements.
        - Output: voronoi.msh4 in output directoR2.
        """

        print("\n=== Generating Mesh from Tessellation ===")

        # Normalize formats
        if isinstance(format_type, str):
            fmt_arg = format_type
        elif isinstance(format_type, (list, tuple)):
            fmt_arg = ",".join(str(f).strip() for f in format_type)

        # --- Validate tessellation file ---
        if not os.path.exists(tess_file):
            raise FileNotFoundError(f"Tessellation file not found: {tess_file}")

        # --- Select element type automatically ---
        element_type = "tri" if self.dim == 2 else "tet"

        # --- Build Neper command ---
        neper_cmd = [
            "neper", "-M",
            tess_file,
            "-elttype", element_type,
            "-rcl", str(relative_cl),
            "-order", "2",  # always 2nd order
            "-meshqualmin", str(mesh_quality_min),
            "-interface", interface_type,
            "-format", fmt_arg,
            "-o", output_name,
            "-part", str(partition),
        ]

        # --- Run meshing ---
        print("\n=== Running Neper Meshing ===\n" \
        "(this could take a while for large system)...\n")
        print("> " + " ".join(neper_cmd))
        print("\n")

        log_path = os.path.join(self.output_dir, "neper_voronoi_mesh.log")

        with open(log_path, "w") as logf:
            print("> " + " ".join(neper_cmd), file=logf)
            logf.flush()
            subprocess.run(neper_cmd,
                check=True,
                env=self.env,
                stdout=logf,
                stderr=subprocess.STDOUT)
        
        for fmt in fmt_arg.split(","):
            print(f"Generated mesh file: {output_name}.{fmt}")

        print("\n")
        
            

    def export_cell_properties(self, 
                           tess_file: str, 
                           output_cell="id,vol,w,x,y,z,radeq", 
                           output_seed="id,w,x,y,z"):
        """
        Compute cell and seed properties using Neper's built-in stats,
        then merge them into a single .dat file named <tess_file>_out.dat.
        """
        from pathlib import Path

        print("\n=== Computing Voronoi Cell and Seed Properties ===")

        if not os.path.exists(tess_file):
            raise FileNotFoundError(f"Tessellation file not found: {tess_file}")

        # Run Neper to generate .statcell and .statseed
        neper_cmd = [
            "neper", "-T",
            "-loadtess", tess_file,
            "-statcell", output_cell,
            "-statseed", output_seed,
            "-format", "ori", 
            "-oridescriptor", "rotmat"
        ]
        print("> " + " ".join(neper_cmd))
        subprocess.run(neper_cmd, check=True, env=self.env)

        # Define filenames
        base = Path(tess_file).with_suffix("")
        stat_cell_file = base.with_suffix(".stcell")
        stat_seed_file = base.with_suffix(".stseed")
        out_file = Path(str(base) + "_out.csv")

        # Validate existence
        if not stat_cell_file.exists():
            raise FileNotFoundError(f"Missing .stcell file: {stat_cell_file}")
        if not stat_seed_file.exists():
            raise FileNotFoundError(f"Missing .stseed file: {stat_seed_file}")

        # Read Neper output tables
        cols_cell = output_cell.split(",")
        cols_seed = output_seed.split(",")
        df_cell = pd.read_csv(stat_cell_file, sep=r"\s+", header=None, names=cols_cell)
        df_seed = pd.read_csv(stat_seed_file, sep=r"\s+", header=None, names=cols_seed)

        # Rename columns to distinguish them
        df_cell = df_cell.add_suffix("_cell")
        df_seed = df_seed.add_suffix("_seed")

        # Merge on IDs
        df_merged = pd.merge(df_cell, df_seed, left_on="id_cell", right_on="id_seed", how="outer")

        # Write combined output
        df_merged.to_csv(out_file, sep=",", index=False)
        print(f"Exported merged cell/seed data to {out_file}\n")

        # Export elastic strain (.ee)
        num_cells = len(df_cell)
        ee_file = Path(self.output_dir) / (base.name + ".ee")

        if self.elastic_strain_id is not None and hasattr(self, "data") and not self.data.empty:
            
            strain_data = self.data[self.elastic_strain_id].to_numpy(dtype=float)
        else:
            strain_data = np.zeros((num_cells, 9), dtype=float)

        # --- Write .ee file ---
        with open(ee_file, "w") as f:
            for row in strain_data:
                f.write(" ".join(f"{v:.8e}" for v in row) + "\n")
        
        print(f"Exported elastic strain tensors to {ee_file}\n")

    def evaluate_output_voronoi(self, tess_file: str, length_norm: bool = False):
        """
        Evaluate Neper Voronoi output for consistency and geometry errors.
        - Compares cell vs seed properties.
        - Generates centroid and statistical plots under <output_dir>/figures.
        """

        import matplotlib.pyplot as plt
        from pathlib import Path

        # check tess_file existence
        if not os.path.exists(tess_file):
            raise FileNotFoundError(f"Tessellation file not found: {tess_file}")
        self.export_cell_properties(tess_file)

        # --- Paths ---
        outputdir = self.output_dir
        os.makedirs(os.path.join(outputdir, "figures"), exist_ok=True)
        df_path = Path(str(Path(tess_file).with_suffix("")) + "_out.csv")

        # --- Load merged Neper output ---
        df = pd.read_csv(df_path)

        # --- Plot centroids for quick inspection ---
        self.plot_centroids(save_path=os.path.join(outputdir, "figures", "centroids.png"))

        # --- Consistency check between IDs ---
        id_cell = np.array(df["id_cell"])
        id_seed = np.array(df["id_seed"])
        if not np.array_equal(id_cell, id_seed):
            print("Warning: id_cell and id_seed are not identical!")


        dfdat = self.data.copy()
        if self.dim == 2:
            coord_cols = ["X", "Y"]
        elif self.dim == 3:
            coord_cols = ["X", "Y", "Z"]
        else:
            raise ValueError("Unsupported dimension (must be 2 or 3).")

        dfdat[coord_cols]

        # --- Extract fields ---
        seed_w = np.array(df["w_seed"])                      # input weight
        cell_xyz = np.array(df[["x_cell", "y_cell", "z_cell"]])
        input_xyz = np.array(dfdat[coord_cols])
        cell_radeq = np.array(df["radeq_cell"])
        seed_radeq = self.data["GrainRadius"].to_numpy()

        # --- Box size and normalization ---
        bb = self.bounding_box
        if self.dim == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = bb
            Lbox = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
        else:
            xmin, xmax, ymin, ymax = bb
            Lbox = np.array([xmax - xmin, ymax - ymin], dtype=float)

        eps = 1e-12

        Lbox = 1.0
        if length_norm:
            Lbox = np.maximum(Lbox, eps)

        # --- Position error (normalized by box lengths per-axis) ---
        
        norm_cell = cell_xyz / Lbox
        norm_seed = input_xyz / Lbox
        delta = norm_cell - norm_seed
        error_xyz = np.linalg.norm(delta, axis=1)

        # --- Equivalent radius error (%) ---
        error_radeq = (cell_radeq - seed_radeq) / seed_radeq * 100.0

        # --- Error histograms ---
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(error_xyz, bins=20)
        if length_norm:
            plt.xlabel("|(Xi_cell - Xi_input)/box_Xi| (norm-2)")
        else:
            plt.xlabel("|Xi_cell - Xi_input| (norm-2) (input length units)")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(error_radeq, bins=20)
        plt.xlabel("Relative Equivalent-Radius Error (%)")
        plt.ylabel("Frequency")

        plt.suptitle("Voronoi Cell vs Seed Point Errors")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, "figures", "voronoi_error.png"), dpi=300)
        plt.close()

        # --- Distribution comparison plots ---
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.hist(seed_w, bins=20)
        plt.xlabel("Seed Weight (from Grain Equivalent Volume)")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 2)
        plt.hist(seed_radeq, bins=20)
        plt.xlabel("Seed Equivalent Radius")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 3)
        plt.hist(cell_radeq, bins=20)
        plt.xlabel("Cell Equivalent Radius")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, "figures", "distribution_comparison.png"), dpi=300)
        plt.close()

        print(f"Evaluation figures saved in {os.path.join(outputdir, 'figures')}")
    