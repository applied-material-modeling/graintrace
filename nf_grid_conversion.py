from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from nf import convert

PathLike = Union[str, Path]


class NFGridConversion:
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
    ) -> None:
        self.input_folder = Path(input_folder)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.exp_file_token = str(exp_file_token)
        self.prefix = str(prefix)
        self.write_intermediate = bool(write_intermediate)
        self.write_vtk = bool(write_vtk)

        self.pointcloud_csv = self.save_dir / "pointcloud.csv"
        self.fixed_grid_npy = self.save_dir / "fixed_grid.npy"
        self.fixed_grid_vtk = self.save_dir / "fixed_grid.vtk"
        self.fixed_grid_csv = self.save_dir / output_csv_filename

    def convert(
        self,
        *,
        dz: float = 5.0,
        nx: int = 300,
        ny: int = 900,
    ) -> Path:
        """
        Inputs:
          - input_folder containing *.mic or *.csv
          - dz, nx, ny

        Outputs (in save_dir):
          - pointcloud.csv
          - fixed_grid.npy
          - fixed_grid.vtk (optional)
          - fixed_grid.csv

        Returns:
          - Path to fixed_grid.csv
        """
        dz = float(dz)
        nx = int(nx)
        ny = int(ny)

        print("Converting NearField data to pointcloud...")
        pc = convert.nf_to_pointcloud(
            str(self.input_folder),
            dz,
            layer_token=self.exp_file_token,
        )

        if self.write_intermediate:
            pc.to_csv(self.pointcloud_csv, index=False)

        print("Converting pointcloud to fixed grid...")
        fixed_grid = convert.pointcloud_to_fixed_grid(pc, nx=nx, ny=ny)

        if self.write_intermediate:
            np.save(self.fixed_grid_npy, fixed_grid)

        if self.write_vtk:
            convert.fixed_grid_to_vtk(fixed_grid, str(self.fixed_grid_vtk))

        print("Writing grid CSV for VoxelMeshBuilder...")
        flat = fixed_grid.reshape(-1, fixed_grid.shape[-1])

        # fixed grid format: [phase, Eul1, Eul2, Eul3, X, Y, Z]
        import pandas as pd

        df = pd.DataFrame(
            {
                "cell_id": flat[:, 0].astype(np.int64),
                "Eul0": flat[:, 1],
                "Eul1": flat[:, 2],
                "Eul2": flat[:, 3],
                "x": flat[:, 4],
                "y": flat[:, 5],
                "z": flat[:, 6],
            }
        )

        # normalize void convention for downstream VoxelMeshBuilder
        df.loc[df["cell_id"] <= 0, "cell_id"] = -1

        df.to_csv(self.fixed_grid_csv, index=False)

        return self.fixed_grid_csv
