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

"""Convert pre-gridded NF HEDM layers into a VoxelMeshBuilder-ready CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .nf import convert

PathLike = Union[str, Path]


class NFGridConversion:
    """Read a folder of NF layers and emit a flat grid CSV for VoxelMeshBuilder.

    Columns: ``cell_id`` (1 solid, -1 void), ``Eul0..2``, ``x, y, z``, ``confidence``.
    """

    def __init__(
        self,
        *,
        input_folder: PathLike,
        save_dir: PathLike,
        exp_file_token: str = "layer",
        prefix: str = "reconstructed",
        write_intermediate: bool = True,
        write_vtk: bool = True,
        output_csv_filename: str = "fixed_grid.csv",
        recursive: bool = False,
    ) -> None:
        """Configure the conversion.

        Args:
            input_folder: Folder containing the per-layer NF files (.mic or .csv).
            save_dir: Output directory; created if missing.
            exp_file_token: Filename token that precedes the layer number.
            prefix: Filename prefix (kept for parity with NearFieldMeshBuilder).
            write_intermediate: If True, write the pointcloud CSV and fixed-grid .npy.
            write_vtk: If True, write a .vtk rendering of the fixed grid.
            output_csv_filename: Name of the output grid CSV in save_dir.
            recursive: If True, search subfolders for layer files.
        """
        self.input_folder = Path(input_folder)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.exp_file_token = str(exp_file_token)
        self.prefix = str(prefix)
        self.write_intermediate = bool(write_intermediate)
        self.write_vtk = bool(write_vtk)
        self.recursive = bool(recursive)

        self.pointcloud_csv = self.save_dir / "pointcloud.csv"
        self.fixed_grid_npy = self.save_dir / "fixed_grid.npy"
        self.fixed_grid_vtk = self.save_dir / "fixed_grid.vtk"
        self.fixed_grid_csv = self.save_dir / output_csv_filename
        self.confidence_npy = self.save_dir / "fixed_grid_confidence.npy"

    def convert(
        self,
        *,
        dz: float = 5.0,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        voxel_size_um: Optional[Sequence[float]] = None,
        min_confidence: Optional[float] = None,
        max_point_distance: Optional[float] = None,
    ) -> Path:
        """Convert NF .mic/.csv layers to a gridded CSV for VoxelMeshBuilder.

        Args:
            dz: Layer thickness in micrometers.
            nx: In-plane grid count along x (default 300).
            ny: In-plane grid count along y (default 900).
            voxel_size_um: (hx, hy) in-plane spacing (um) instead of nx/ny.
            min_confidence: Confidence below which a voxel becomes void.
            max_point_distance: Solid radius (um) replacing the hull test.

        Returns:
            Path to fixed_grid.csv; the confidence is also saved as fixed_grid_confidence.npy.

        Raises:
            ValueError: if voxel_size_um is given together with nx or ny.
        """
        dz = float(dz)
        if voxel_size_um is not None:
            if nx is not None or ny is not None:
                raise ValueError("Give either nx/ny or voxel_size_um, not both")
        else:
            nx = 300 if nx is None else int(nx)
            ny = 900 if ny is None else int(ny)

        print("Converting NearField data to pointcloud...")
        pc = convert.nf_to_pointcloud(
            str(self.input_folder),
            dz,
            layer_token=self.exp_file_token,
            recursive=self.recursive,
        )

        if self.write_intermediate:
            pc.to_csv(self.pointcloud_csv, index=False)

        print("Converting pointcloud to fixed grid...")
        fixed_grid, confidence = convert.pointcloud_to_fixed_grid(
            pc,
            nx=nx,
            ny=ny,
            voxel_size_um=voxel_size_um,
            min_confidence=min_confidence,
            max_point_distance=max_point_distance,
        )
        np.save(self.confidence_npy, confidence)

        if self.write_intermediate:
            np.save(self.fixed_grid_npy, fixed_grid)

        if self.write_vtk:
            convert.fixed_grid_to_vtk(fixed_grid, str(self.fixed_grid_vtk))

        print("Writing grid CSV for VoxelMeshBuilder...")
        flat = fixed_grid.reshape(-1, fixed_grid.shape[-1])

        # fixed grid format: [phase, Eul1, Eul2, Eul3, X, Y, Z]
        df = pd.DataFrame(
            {
                "cell_id": flat[:, 0].astype(np.int64),
                "Eul0": flat[:, 1],
                "Eul1": flat[:, 2],
                "Eul2": flat[:, 3],
                "x": flat[:, 4],
                "y": flat[:, 5],
                "z": flat[:, 6],
                "confidence": confidence.reshape(-1),
            }
        )

        # normalize void convention for downstream VoxelMeshBuilder
        df.loc[df["cell_id"] <= 0, "cell_id"] = -1

        df.to_csv(self.fixed_grid_csv, index=False)

        return self.fixed_grid_csv
