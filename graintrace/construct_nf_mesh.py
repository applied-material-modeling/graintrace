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

"""NF HEDM reconstruction: segment .mic layers and build a hex mesh."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch

from .nf import convert, segment, mesh

PathLike = Union[str, Path]


class NearFieldMeshBuilder:
    """Reconstruct a high-resolution 3D mesh from near-field (NF) HEDM ``.mic`` layers.

    Reads a folder of per-layer orientation maps, segments them into grains on a voxel
    grid, and generates a conformal hexahedral Exodus mesh via CUBIT/SCULPT, plus
    per-element MRP orientations for CPFE. See ``examples/demonstrate_cpfe_nfff.py`` and
    the ``/nf-reconstruction`` skill. (NF reconstruction uses multiprocessing, so call it
    under an ``if __name__ == "__main__":`` guard.)
    """

    DEFAULT_SEGMENTATION: Dict[str, Any] = {
        "misorientation_tol": 5.0 / 180.0 * np.pi,  # radians
        "connectivity": 26,
        "batch_norm": 1000,
        "grain_threshold": 100,
        "stop_count": 500,
        "grain_threshold_final": 1000,
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
        input_folder: PathLike,
        save_dir: PathLike,
        exp_file_token: str = "layer",
        angle_convention: str = "bunge",
        angle_type: str = "radians",
        symmetry: str = "432",
        prefix: str = "reconstructed",
        write_intermediate: bool = True,
        write_vtk: bool = True,
        default_mesh_filename: str = "mesh.e",
        default_mapped_orientations_filename: str = "orientations",
    ) -> None:
        """Configure an NF-HEDM reconstruction from a folder of layer files.

        Args:
            input_folder: Folder containing the per-layer NF orientation maps (.mic files).
            save_dir: Output directory for the pointcloud, grids, mesh, and mapped orientations; created if missing.
            exp_file_token: Filename token used to select the layer files within input_folder.
            angle_convention: Euler angle convention of the input maps, e.g. "bunge".
            angle_type: Euler angle unit, "radians" or "degrees".
            symmetry: Crystal symmetry group used for misorientation, e.g. "432".
            prefix: Filename prefix for the .spn and .orientations intermediate files.
            write_intermediate: If True, write intermediate .npy/.csv checkpoints to save_dir.
            write_vtk: If True, write .vtk renderings of the grids to save_dir.
            default_mesh_filename: Default output mesh filename in save_dir (e.g. "mesh.e").
            default_mapped_orientations_filename: Default basename for the per-element mapped orientation output (".csv" is appended by mesh()).
        """

        self.input_folder = Path(input_folder)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.angle_convention = str(angle_convention)
        self.angle_type = str(angle_type)
        self.symmetry = str(symmetry)

        self.prefix = str(prefix)
        self.write_intermediate = bool(write_intermediate)
        self.write_vtk = bool(write_vtk)
        self.exp_file_token = str(exp_file_token)

        self.merged_grid_npy = self.save_dir / "merged_segmented_fixed_grid.npy"

        self.spn_path = self.save_dir / f"{self.prefix}.spn"
        self.orientations_path = self.save_dir / f"{self.prefix}.orientations"
        self.mesh_path = self.save_dir / default_mesh_filename
        self.mapped_orientations_path = (
            self.save_dir / default_mapped_orientations_filename
        )

    def _normalize_segmentation(
        self, segmentation: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        cfg = dict(self.DEFAULT_SEGMENTATION)
        if segmentation:
            cfg.update(segmentation)

        required = set(self.DEFAULT_SEGMENTATION.keys())
        missing = required - set(cfg.keys())
        extra = set(cfg.keys()) - required
        if missing:
            raise ValueError(
                "Missing segmentation keys: "
                + ", ".join(sorted(missing))
                + "\nRequired keys and defaults:\n"
                + "\n".join(
                    f"  {k}: {v!r}" for k, v in self.DEFAULT_SEGMENTATION.items()
                )
            )
        if extra:
            raise ValueError("Unknown segmentation keys: " + ", ".join(sorted(extra)))

        if cfg["connectivity"] not in (6, 26):
            raise ValueError("segmentation['connectivity'] must be 6 or 26")

        for k in (
            "batch_norm",
            "grain_threshold",
            "stop_count",
            "grain_threshold_final",
        ):
            if int(cfg[k]) <= 0:
                raise ValueError(f"segmentation['{k}'] must be > 0")

        if float(cfg["misorientation_tol"]) <= 0:
            raise ValueError("segmentation['misorientation_tol'] must be > 0 (radians)")

        cfg["misorientation_tol"] = float(cfg["misorientation_tol"])
        cfg["connectivity"] = int(cfg["connectivity"])
        cfg["batch_norm"] = int(cfg["batch_norm"])
        cfg["grain_threshold"] = int(cfg["grain_threshold"])
        cfg["stop_count"] = int(cfg["stop_count"])
        cfg["grain_threshold_final"] = int(cfg["grain_threshold_final"])
        return cfg

    def _validate_sculpt_config(self, sculpt_config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(sculpt_config, dict):
            raise TypeError(
                "mesh_config must be a dict containing sculpt configuration"
            )

        missing = [k for k in self.REQUIRED_SCULPT_KEYS if k not in sculpt_config]
        if missing:
            raise ValueError(
                "Missing sculpt_config keys: "
                + ", ".join(missing)
                + "\nRequired keys:\n"
                + "\n".join(f"  {k}" for k in self.REQUIRED_SCULPT_KEYS)
                + "\nOptional keys:\n  environment (dict)"
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

        return cfg

    def _load_grid(self, path: Path) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim != 4 or arr.shape[-1] != 7:
            raise ValueError(
                f"Invalid grid format at {path}. Expected (nx, ny, nz, 7), got {arr.shape}"
            )
        return torch.tensor(arr)

    def reconstruct(
        self,
        *,
        dz: float = 5.0,
        nx: int = 300,
        ny: int = 900,
        segmentation: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Reconstruct a segmented NF voxel grid from .mic/.csv layers.

        Converts the layer files to a pointcloud, rasterizes it onto a fixed voxel grid, segments it into grains via flood-fill, infills, and removes small segments.

        Args:
            dz: Layer thickness in micrometers (z spacing between stacked layers).
            nx: In-plane grid resolution along x.
            ny: In-plane grid resolution along y.
            segmentation: Flat segmentation config dict (misorientation_tol in radians, connectivity, batch_norm, grain_threshold, stop_count, grain_threshold_final); None uses the class defaults.

        Returns:
            Path to the written merged_segmented_fixed_grid.npy.
        """
        seg = self._normalize_segmentation(segmentation)

        dz = float(dz)
        nx = int(nx)
        ny = int(ny)

        pointcloud_csv = self.save_dir / "pointcloud.csv"
        fixed_grid_npy = self.save_dir / "fixed_grid.npy"
        fixed_grid_vtk = self.save_dir / "fixed_grid.vtk"
        segmented_grid_npy = self.save_dir / "segmented_fixed_grid.npy"
        segmented_grid_vtk = self.save_dir / "segmented_fixed_grid.vtk"
        merged_grid_vtk = self.save_dir / "merged_segmented_fixed_grid.vtk"

        # NearField -> pointcloud
        pc = convert.nf_to_pointcloud(
            str(self.input_folder), dz, layer_token=self.exp_file_token
        )

        if self.write_intermediate:
            pc.to_csv(pointcloud_csv, index=False)

        # pointcloud -> fixed grid
        fixed_grid = convert.pointcloud_to_fixed_grid(pc, nx=nx, ny=ny)

        if self.write_intermediate:
            np.save(fixed_grid_npy, fixed_grid)
        if self.write_vtk:
            convert.fixed_grid_to_vtk(fixed_grid, str(fixed_grid_vtk))

        # segmentation
        grid_t = torch.from_numpy(fixed_grid)

        grid_t[..., 0] = segment.flood(
            grid_t[..., 1:4],
            grid_t[..., 0],
            seg["misorientation_tol"],
            connectivity=seg["connectivity"],
            batch_norm=seg["batch_norm"],
            grain_threshold=seg["grain_threshold"],
            stop_count=seg["stop_count"],
            angle_convention=self.angle_convention,
            angle_type=self.angle_type,
            symmetry=self.symmetry,
        )

        if self.write_intermediate:
            np.save(segmented_grid_npy, grid_t.cpu().numpy())
        if self.write_vtk:
            convert.fixed_grid_to_vtk(grid_t.cpu().numpy(), str(segmented_grid_vtk))

        grid_t = segment.infill_nearest_neighbor(grid_t)
        grid_t = segment.remove_small_segments(grid_t, seg["grain_threshold_final"])

        np.save(self.merged_grid_npy, grid_t.cpu().numpy())
        if self.write_vtk:
            convert.fixed_grid_to_vtk(grid_t.cpu().numpy(), str(merged_grid_vtk))

        return self.merged_grid_npy

    def mesh(
        self,
        *,
        sculpt_config: Dict[str, Any],
        sculpt_options: Optional[Sequence[str]] = None,
        merged_grid: Optional[PathLike] = None,
        # output overrides (optional)
        spn_path: Optional[PathLike] = None,
        orientations_path: Optional[PathLike] = None,
        mesh_path: Optional[PathLike] = None,
        mapped_orientations_path: Optional[PathLike] = None,
    ) -> Path:
        """Generate an Exodus mesh via SCULPT from the merged voxel grid.

        Uses merged_grid if given, else save_dir/merged_segmented_fixed_grid.npy. Writes the SCULPT .spn/orientations inputs, runs SCULPT, and maps per-element MRP orientations.

        Args:
            sculpt_config: SCULPT configuration dict (required keys psculpt, epu, nprocs; optional launcher, environment).
            sculpt_options: SCULPT CLI flags as a sequence of strings; None uses DEFAULT_SCULPT_OPTIONS.
            merged_grid: Path to the merged segmented grid .npy; None uses save_dir/merged_segmented_fixed_grid.npy.
            spn_path: Override for the SCULPT .spn voxel input path.
            orientations_path: Override for the SCULPT per-voxel orientations input path.
            mesh_path: Override for the output Exodus mesh path; None uses the configured default.
            mapped_orientations_path: Override for the mapped per-element orientation output basename (".csv" is appended).

        Returns:
            Path to the written Exodus mesh.
        """
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

        data = self._load_grid(grid_path)

        print("\n")

        mesh.write_spn(
            data,
            str(spn_out),
            str(ori_out),
            angle_convention=self.angle_convention,
            angle_type=self.angle_type,
        )

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
