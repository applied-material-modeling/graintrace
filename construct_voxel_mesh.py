from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nf import convert, mesh, segment
from nf.smooth import smooth

from graph_spatial_cluster import GraphSpatialCluster
from similarity_metric_library import (
    make_misorientation_dist_edges,
    misorientation_distance,
)
from user_data_class import SimilarityMetric, WeightConfig

import matplotlib.pyplot as plt

PathLike = Union[str, Path]


class VoxelMeshBuilder:

    DEFAULT_SEGMENTATION = {
        "method": "flood",
        "params": {
            "misorientation_tol": 5.0 / 180.0 * np.pi,
            "connectivity": 26,
            "grain_threshold_final": 1000,
            # flood-only
            "batch_norm": 1000,
            "grain_threshold": 100,
            "stop_count": 500,
        },
        "graph_params": {
            "segmenter": "leiden",
            "graph_mode": "grid",
            "manhattan_radius": 2,
            "grid_tol": 1e-6,
            "n_jobs": 1,
            "weight_chunk_size": 1_000_000,
            "nodes_chunk": 250_000,
            "reduce_edges_topweights_k": None,
            "seed": 42,
            "networkit_kwargs": {"gamma": 1.0},
            "weight_cfg": {
                "mode": "inverse",
                "eps": 1e-8,
                "sigma": None,
                "sigma_auto": None,
                "power": 2.0,
            },
            "plot": True,
        },
    }

    DEFAULT_SCULPT_OPTIONS: Sequence[str] = (
        "--adapt",
        "-A",
        "7",
        "-df",
        "1",
        "-S",
        "2",
        "-CS",
        "4",
        "--void_mat",
        "0",
    )

    REQUIRED_SCULPT_KEYS = ("psculpt", "epu", "nprocs")

    def __init__(
        self,
        *,
        file_path: PathLike,
        save_dir: PathLike,
        euler_cols: Sequence[str],
        cell_id_col: Optional[str] = None,
        angle_convention: str = "bunge",
        angle_type: str = "radians",
        symmetry: str = "432",
        prefix: str = "voxel",
        write_intermediate: bool = True,
        write_vtk: bool = True,
        default_mesh_filename: str = "mesh.e",
        default_mapped_orientations_filename: str = "orientations",
    ) -> None:
        self.file_path = Path(file_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.euler_cols = tuple(euler_cols)
        if len(self.euler_cols) != 3:
            raise ValueError("euler_cols must contain exactly 3 column names")

        self.cell_id_col = cell_id_col

        self.angle_convention = str(angle_convention)
        self.angle_type = str(angle_type)
        self.symmetry = str(symmetry)

        self.prefix = str(prefix)
        self.write_intermediate = bool(write_intermediate)
        self.write_vtk = bool(write_vtk)

        self.segmented_grid_npy = self.save_dir / "segmented_fixed_grid.npy"
        self.merged_grid_npy = self.save_dir / "merged_segmented_fixed_grid.npy"

        self.spn_path = self.save_dir / f"{self.prefix}.spn"
        self.orientations_path = self.save_dir / f"{self.prefix}.orientations"
        self.mesh_path = self.save_dir / default_mesh_filename
        self.mapped_orientations_path = (
            self.save_dir / default_mapped_orientations_filename
        )

        self.df: Optional[pd.DataFrame] = None
        self.has_z: bool = True

    def _normalize_segmentation(
        self, segmentation: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        cfg = {
            "method": self.DEFAULT_SEGMENTATION["method"],
            "params": dict(self.DEFAULT_SEGMENTATION["params"]),
            "graph_params": dict(self.DEFAULT_SEGMENTATION["graph_params"]),
        }

        if segmentation:
            if "method" in segmentation:
                cfg["method"] = segmentation["method"]
            if "params" in segmentation:
                cfg["params"].update(segmentation["params"])
            if "graph_params" in segmentation:
                cfg["graph_params"].update(segmentation["graph_params"])

        if cfg["method"] not in ("flood", "graph"):
            raise ValueError("segmentation['method'] must be 'flood' or 'graph'")

        p = cfg["params"]

        # shared keys
        p["misorientation_tol"] = float(
            p.get(
                "misorientation_tol",
                self.DEFAULT_SEGMENTATION["params"]["misorientation_tol"],
            )
        )
        p["connectivity"] = int(p.get("connectivity", 26))
        p["grain_threshold_final"] = int(p.get("grain_threshold_final", 1000))

        if p["misorientation_tol"] <= 0:
            raise ValueError("segmentation['params']['misorientation_tol'] must be > 0")
        if p["connectivity"] not in (6, 26):
            raise ValueError("segmentation['params']['connectivity'] must be 6 or 26")
        if p["grain_threshold_final"] <= 0:
            raise ValueError(
                "segmentation['params']['grain_threshold_final'] must be > 0"
            )

        if cfg["method"] == "flood":
            required = {
                "misorientation_tol",
                "connectivity",
                "batch_norm",
                "grain_threshold",
                "stop_count",
                "grain_threshold_final",
            }
            missing = required - set(p.keys())
            if missing:
                raise ValueError(
                    "Missing flood segmentation keys: " + ", ".join(sorted(missing))
                )

            for k in ("batch_norm", "grain_threshold", "stop_count"):
                if int(p[k]) <= 0:
                    raise ValueError(f"segmentation['params']['{k}'] must be > 0")

            p["batch_norm"] = int(p["batch_norm"])
            p["grain_threshold"] = int(p["grain_threshold"])
            p["stop_count"] = int(p["stop_count"])

        else:
            gp = cfg["graph_params"]

            gp["segmenter"] = str(gp.get("segmenter", "leiden")).lower()
            gp["graph_mode"] = str(gp.get("graph_mode", "grid")).lower()
            gp["manhattan_radius"] = int(gp.get("manhattan_radius", 1))
            gp["grid_tol"] = float(gp.get("grid_tol", 1e-6))
            gp["n_jobs"] = int(gp.get("n_jobs", 1))
            gp["weight_chunk_size"] = int(gp.get("weight_chunk_size", 1_000_000))
            gp["nodes_chunk"] = int(gp.get("nodes_chunk", 250_000))
            gp["seed"] = int(gp.get("seed", 42))
            gp["plot"] = bool(gp.get("plot", True))

            if gp["segmenter"] not in ("leiden", "plm", "plp"):
                raise ValueError(
                    "segmentation['graph_params']['segmenter'] must be one of {'leiden','plm','plp'}"
                )
            if gp["graph_mode"] not in ("grid", "knn", "auto"):
                raise ValueError(
                    "segmentation['graph_params']['graph_mode'] must be one of {'grid','knn','auto'}"
                )
            if gp["manhattan_radius"] <= 0:
                raise ValueError(
                    "segmentation['graph_params']['manhattan_radius'] must be > 0"
                )
            if gp["grid_tol"] < 0:
                raise ValueError(
                    "segmentation['graph_params']['grid_tol'] must be >= 0"
                )
            if gp["n_jobs"] <= 0:
                raise ValueError("segmentation['graph_params']['n_jobs'] must be > 0")
            if gp["weight_chunk_size"] <= 0:
                raise ValueError(
                    "segmentation['graph_params']['weight_chunk_size'] must be > 0"
                )
            if gp.get("reduce_edges_topweights_k") is not None:
                gp["reduce_edges_topweights_k"] = int(gp["reduce_edges_topweights_k"])
                if gp["reduce_edges_topweights_k"] <= 0:
                    raise ValueError(
                        "segmentation['graph_params']['reduce_edges_topweights_k'] must be > 0"
                    )
            if gp.get("nodes_chunk") is not None:
                gp["nodes_chunk"] = int(gp["nodes_chunk"])
                if gp["nodes_chunk"] <= 0:
                    raise ValueError(
                        "segmentation['graph_params']['nodes_chunk'] must be > 0"
                    )

            if not isinstance(gp.get("networkit_kwargs", {}), dict):
                raise TypeError(
                    "segmentation['graph_params']['networkit_kwargs'] must be a dict"
                )
            gp["networkit_kwargs"] = dict(gp.get("networkit_kwargs", {}))

            weight_cfg = gp.get("weight_cfg", {})
            if not isinstance(weight_cfg, dict):
                raise TypeError(
                    "segmentation['graph_params']['weight_cfg'] must be a dict"
                )
            gp["weight_cfg"] = WeightConfig(**weight_cfg)

        return cfg

    def _validate_sculpt_config(self, sculpt_config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(sculpt_config, dict):
            raise TypeError("sculpt_config must be a dict")

        missing = [k for k in self.REQUIRED_SCULPT_KEYS if k not in sculpt_config]
        if missing:
            raise ValueError(
                "Missing sculpt_config keys: "
                + ", ".join(missing)
                + "\nRequired keys:\n"
                + "\n".join(f"  {k}" for k in self.REQUIRED_SCULPT_KEYS)
                + "\nOptional keys:\n  launcher\n  environment (dict)"
            )

        cfg = dict(sculpt_config)
        cfg["nprocs"] = int(cfg["nprocs"])
        if cfg["nprocs"] <= 0:
            raise ValueError("sculpt_config['nprocs'] must be > 0")

        env = cfg.get("environment", {})
        if env is None:
            env = {}
        if not isinstance(env, dict):
            raise TypeError("sculpt_config['environment'] must be a dict if provided")
        cfg["environment"] = env

        if "launcher" not in cfg:
            cfg["launcher"] = "mpiexec"

        return cfg

    def _load_dataframe(self) -> pd.DataFrame:
        print("Loading data...")
        df = pd.read_csv(self.file_path)

        # normalize coordinate column names
        rename_map = {}

        if "X" in df.columns and "x" not in df.columns:
            rename_map["X"] = "x"
        if "Y" in df.columns and "y" not in df.columns:
            rename_map["Y"] = "y"
        if "Z" in df.columns and "z" not in df.columns:
            rename_map["Z"] = "z"

        if rename_map:
            df = df.rename(columns=rename_map)

        for col in ("x", "y"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if "z" not in df.columns:
            print("No z column found. Promoting 2D input to 3D with z=0.")
            df["z"] = 0.0
            self.has_z = False
        else:
            self.has_z = True

        for col in self.euler_cols:
            if col not in df.columns:
                raise ValueError(f"Missing Euler column: {col}")

        required = ["x", "y", "z", *self.euler_cols]
        if df[required].isnull().any().any():
            raise ValueError("NaNs detected in required fields")

        self.df = df
        return df

    def _check_grid_spacing(self, values: np.ndarray, axis_name: str) -> None:
        if len(values) < 2:
            return
        diffs = np.diff(values)
        if not np.allclose(diffs, diffs[0]):
            print(f"Warning: non-uniform spacing detected in {axis_name}")

    def _assign_existing_ids(self, grid_t: torch.Tensor) -> torch.Tensor:
        if self.df is None:
            raise RuntimeError("Dataframe not loaded")
        if not self.cell_id_col or self.cell_id_col not in self.df.columns:
            raise ValueError("cell_id_col not found in dataframe")

        print("Using provided cell_id. Skipping segmentation.")

        df = self.df
        if df[self.cell_id_col].isnull().any():
            raise ValueError("NaNs detected in cell_id")

        values = df[self.cell_id_col].to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("cell_id must be numeric")
        if np.any(values < -1):
            raise ValueError("cell_id contains invalid values (< -1)")

        xs = np.sort(df["x"].unique())
        ys = np.sort(df["y"].unique())
        zs = np.sort(df["z"].unique())

        x_map = {v: i for i, v in enumerate(xs)}
        y_map = {v: i for i, v in enumerate(ys)}
        z_map = {v: i for i, v in enumerate(zs)}

        cell_ids = torch.zeros(grid_t.shape[:3], dtype=grid_t.dtype)

        for _, row in df.iterrows():
            i = x_map[row["x"]]
            j = y_map[row["y"]]
            k = z_map[row["z"]]

            cid = int(row[self.cell_id_col])
            if cid == -1:
                cid = 0

            cell_ids[i, j, k] = cid

        grid_t[..., 0] = cell_ids
        return grid_t

    def _graph_segmentation_dir(self) -> Path:
        out = self.save_dir / "graph_segmentation"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _build_graph_metric(self) -> SimilarityMetric:
        from similarity_metric_library import (
            misorientation_distance,
        )

        return SimilarityMetric(
            name="misorientation",
            feature_cols=list(self.euler_cols),
            func=functools.partial(
                misorientation_distance,
                angle_convention=self.angle_convention,
                input_angle_type=self.angle_type,
                symmetry=self.symmetry,
                output_unit=self.angle_type,
            ),
            dist_edges=make_misorientation_dist_edges(
                symmetry=self.symmetry,
                input_angle_type=self.angle_type,
                angle_convention=self.angle_convention,
                output_unit=self.angle_type,
            ),
        )

    def _relabel_phase_ids_contiguous(self, grid_t: torch.Tensor) -> torch.Tensor:
        grid_t = grid_t.clone()
        phase = grid_t[..., 0]

        valid = phase > 0
        if not torch.any(valid):
            return grid_t

        old_ids = torch.unique(phase[valid]).to(torch.int64)
        old_ids = torch.sort(old_ids).values

        new_phase = phase.clone()
        for new_id, old_id in enumerate(old_ids.tolist(), start=1):
            new_phase[phase == old_id] = new_id

        grid_t[..., 0] = new_phase
        return grid_t

    def _plot_graph_labels(
        self, labels: np.ndarray, out_dir: Path, filename: str
    ) -> None:
        counts = pd.Series(labels).value_counts().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(1, len(counts) + 1), counts.to_numpy())
        ax.set_xlabel("Cluster rank")
        ax.set_ylabel("Cluster size")
        ax.set_title("Graph segmentation cluster-size distribution")
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=300)
        plt.close(fig)

    def _segment_graph(
        self,
        grid_t: torch.Tensor,
        params: Dict[str, Any],
        graph_params: Dict[str, Any],
    ) -> torch.Tensor:
        if self.df is None:
            raise RuntimeError("Dataframe not loaded")

        out_dir = self._graph_segmentation_dir()

        df_graph = self.df[["x", "y", "z", *self.euler_cols]].copy()
        df_graph.insert(0, "id", np.arange(len(df_graph), dtype=np.int64))

        csv_path = out_dir / "graph_input.csv"
        df_graph.to_csv(csv_path, index=False)

        spec = SimilarityMetric(
            name="misorientation",
            feature_cols=list(self.euler_cols),
            func=functools.partial(
                misorientation_distance,
                angle_convention=self.angle_convention,
                input_angle_type=self.angle_type,
                symmetry=self.symmetry,
                output_unit=self.angle_type,
            ),
            dist_edges=make_misorientation_dist_edges(
                angle_convention=self.angle_convention,
                input_angle_type=self.angle_type,
                symmetry=self.symmetry,
                output_unit=self.angle_type,
            ),
        )

        gsc = GraphSpatialCluster(
            csv_path=str(csv_path),
            id_col="id",
            coord_cols=("x", "y", "z"),
        )

        gsc_out = gsc.run(
            spec=spec,
            graph_mode=graph_params["graph_mode"],
            manhattan_radius=graph_params["manhattan_radius"],
            grid_tol=graph_params["grid_tol"],
            n_jobs=graph_params["n_jobs"],
            weight_chunk_size=graph_params["weight_chunk_size"],
            nodes_chunk=graph_params["nodes_chunk"],
            segmenter=graph_params["segmenter"],
            seed=graph_params["seed"],
            return_labels=True,
            max_edge_distance=params["misorientation_tol"],
            weight_cfg=graph_params["weight_cfg"],
            networkit_kwargs=graph_params["networkit_kwargs"],
            reduce_edges_topweights_k=graph_params["reduce_edges_topweights_k"],
            mp_start_method="spawn",  # this is needed to avoid hanging when using multiprocessing in graph_spatial_cluster on some platforms
        )

        labels = np.asarray(gsc_out["extras"]["labels"], dtype=np.int64)

        if graph_params["plot"]:
            self._plot_graph_labels(labels, out_dir, "cluster_size_distribution.png")

        xs = np.sort(self.df["x"].unique())
        ys = np.sort(self.df["y"].unique())
        zs = np.sort(self.df["z"].unique())

        x_map = {v: i for i, v in enumerate(xs)}
        y_map = {v: i for i, v in enumerate(ys)}
        z_map = {v: i for i, v in enumerate(zs)}

        cell_ids = torch.zeros(grid_t.shape[:3], dtype=grid_t.dtype)

        for idx, (_, row) in enumerate(self.df.iterrows()):
            i = x_map[row["x"]]
            j = y_map[row["y"]]
            k = z_map[row["z"]]
            cell_ids[i, j, k] = int(labels[idx]) + 1

        grid_t[..., 0] = cell_ids
        return grid_t

    def reconstruct(
        self,
        *,
        segmentation: Optional[Dict[str, Any]] = None,
        apply_smoothing: bool = False,
    ) -> Path:
        """
        Inputs:
          - file_path containing a grid-based CSV
          - segmentation dict (optional)

        Outputs (in save_dir):
          - fixed_grid.npy (+ fixed_grid.vtk if enabled)
          - segmented_fixed_grid.npy (+ segmented_fixed_grid.vtk if enabled)
          - merged_segmented_fixed_grid.npy (+ merged_segmented_fixed_grid.vtk if enabled)

        Returns:
          - Path to merged_segmented_fixed_grid.npy
        """

        fixed_grid_npy = self.save_dir / "fixed_grid.npy"
        fixed_grid_vtk = self.save_dir / "fixed_grid.vtk"
        segmented_grid_vtk = self.save_dir / "segmented_fixed_grid.vtk"
        merged_grid_vtk = self.save_dir / "merged_segmented_fixed_grid.vtk"

        df = self._load_dataframe()
        print("Building fixed grid...")
        xs = np.sort(df["x"].unique())
        ys = np.sort(df["y"].unique())
        zs = np.sort(df["z"].unique())
        self._check_grid_spacing(xs, "x")
        self._check_grid_spacing(ys, "y")
        if self.has_z:
            self._check_grid_spacing(zs, "z")
        nx, ny, nz = len(xs), len(ys), len(zs)
        expected = nx * ny * nz
        if len(df) != expected:
            print(
                f"Warning: grid is not fully dense or contains duplicates "
                f"(rows={len(df)}, expected={expected})"
            )
        x_map = {v: i for i, v in enumerate(xs)}
        y_map = {v: i for i, v in enumerate(ys)}
        z_map = {v: i for i, v in enumerate(zs)}

        fixed_grid = np.zeros((nx, ny, nz, 7), dtype=np.float32)

        xi = df["x"].map(x_map).to_numpy()
        yi = df["y"].map(y_map).to_numpy()
        zi = df["z"].map(z_map).to_numpy()

        fixed_grid[xi, yi, zi, 0] = 1.0
        fixed_grid[xi, yi, zi, 1:4] = df[list(self.euler_cols)].to_numpy(
            dtype=np.float32
        )
        fixed_grid[xi, yi, zi, 4:7] = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

        fixed_grid = torch.from_numpy(fixed_grid)

        if self.write_intermediate:
            np.save(fixed_grid_npy, fixed_grid.cpu().numpy())
        if self.write_vtk:
            convert.fixed_grid_to_vtk(fixed_grid.cpu().numpy(), str(fixed_grid_vtk))

        if apply_smoothing:
            grid_t = smooth(
                fixed_grid.clone(),
                connectivity=6,
                symmetry=self.symmetry,
                angle_convention=self.angle_convention,
                angle_type=self.angle_type,
            )
        else:
            grid_t = fixed_grid.clone()

        if self.cell_id_col and self.cell_id_col in self.df.columns:
            grid_t = self._assign_existing_ids(grid_t)
            if self.write_intermediate:
                np.save(self.segmented_grid_npy, grid_t.cpu().numpy())
            if self.write_vtk:
                convert.fixed_grid_to_vtk(grid_t.cpu().numpy(), str(segmented_grid_vtk))

            np.save(self.merged_grid_npy, grid_t.cpu().numpy())
            if self.write_vtk:
                convert.fixed_grid_to_vtk(grid_t.cpu().numpy(), str(merged_grid_vtk))
            return self.merged_grid_npy
        else:
            seg = self._normalize_segmentation(segmentation)
            params = seg["params"]
            print(f"Segmentation method: {seg['method']}")

            if seg["method"] == "flood":
                print("Running flood segmentation...")

                grid_t[..., 0] = segment.flood(
                    grid_t[..., 1:4],
                    grid_t[..., 0],
                    params["misorientation_tol"],
                    connectivity=params["connectivity"],
                    batch_norm=params["batch_norm"],
                    grain_threshold=params["grain_threshold"],
                    stop_count=params["stop_count"],
                    angle_convention=self.angle_convention,
                    angle_type=self.angle_type,
                    symmetry=self.symmetry,
                )

            elif seg["method"] == "graph":
                print("Running graph segmentation...")
                graph_params = seg["graph_params"]
                grid_t = self._segment_graph(grid_t, params, graph_params)

            else:
                raise ValueError(f"Unknown segmentation method: {seg['method']}")

            if self.write_intermediate:
                np.save(self.segmented_grid_npy, grid_t.cpu().numpy())
            if self.write_vtk:
                convert.fixed_grid_to_vtk(grid_t.cpu().numpy(), str(segmented_grid_vtk))

            print("Removing small segments...")
            grid_t = segment.remove_small_segments(
                grid_t,
                params["grain_threshold_final"],
                connectivity=params["connectivity"],
            )

            print("Infilling segmented grid...")
            grid_t = segment.infill_nearest_neighbor(
                grid_t, connectivity=params["connectivity"]
            )

            print("Relabeling phase IDs to be contiguous...")
            grid_t = self._relabel_phase_ids_contiguous(grid_t)

            out_dir = self._graph_segmentation_dir()
            final_labels = (
                grid_t[..., 0].detach().cpu().numpy().astype(np.int64).ravel()
            )
            self._plot_graph_labels(
                final_labels,
                out_dir,
                filename="final_cluster_size_distribution.png",
            )

            np.save(self.merged_grid_npy, grid_t.cpu().numpy())
            if self.write_vtk:
                convert.fixed_grid_to_vtk(grid_t.cpu().numpy(), str(merged_grid_vtk))

            return self.merged_grid_npy

    def _load_grid(self, path: Path) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim != 4 or arr.shape[-1] != 7:
            raise ValueError(
                f"Invalid grid format at {path}. Expected (nx, ny, nz, 7), got {arr.shape}"
            )
        return torch.tensor(arr)

    def mesh(
        self,
        *,
        sculpt_config: Dict[str, Any],
        sculpt_options: Optional[Sequence[str]] = None,
        merged_grid: Optional[PathLike] = None,
        spn_path: Optional[PathLike] = None,
        orientations_path: Optional[PathLike] = None,
        mesh_path: Optional[PathLike] = None,
        mapped_orientations_path: Optional[PathLike] = None,
    ) -> Path:
        cfg = self._validate_sculpt_config(sculpt_config)
        options = (
            list(sculpt_options)
            if sculpt_options is not None
            else list(self.DEFAULT_SCULPT_OPTIONS)
        )

        grid_path = (
            Path(merged_grid) if merged_grid is not None else self.merged_grid_npy
        )
        if not grid_path.exists():
            raise FileNotFoundError(
                f"Required merged grid not found: {grid_path}\n"
                "Run reconstruct() first or pass merged_grid=... explicitly."
            )

        spn_out = Path(spn_path) if spn_path is not None else self.spn_path
        ori_out = (
            Path(orientations_path)
            if orientations_path is not None
            else self.orientations_path
        )
        mesh_out = Path(mesh_path) if mesh_path is not None else self.mesh_path
        map_out = (
            Path(mapped_orientations_path)
            if mapped_orientations_path is not None
            else self.mapped_orientations_path
        )

        print("Loading merged grid for meshing...")
        data = self._load_grid(grid_path)

        print("Writing SPN and orientations...")
        mesh.write_spn(
            data,
            str(spn_out),
            str(ori_out),
            angle_convention=self.angle_convention,
            angle_type=self.angle_type,
        )

        print("Meshing with sculpt...")
        mesh.mesh_sculpt(
            cfg,
            options,
            str(spn_out),
            data,
            str(mesh_out),
            str(map_out),
            angle_convention=self.angle_convention,
            angle_type=self.angle_type,
        )

        return mesh_out
