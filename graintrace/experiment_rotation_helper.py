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

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import os
import shutil
import numpy as np
import pandas as pd
from .construct_voronoi_mesh import VoronoiMeshBuilder
import glob


def update_experiments(
    input_files: Sequence[str],
    output_root: str,
    bounding_box: List[float],
    dim: int = 3,
    weighted: bool = False,
    gmsh_version: str = "4.12.2",
    neper_version: str = "4.10.1",
    auto_fix_bbox: bool = False,
    bbox_fix_mode: Optional[str] = None,
    bbox_tolerance: float = 0.0,
    auto_rotate: bool = False,
    rotate_angles: Tuple[float, float, float] = (0, 0, 0),
    rotate_convention: str = "xyz",
    angle_identifier: Optional[List[str]] = None,
    orientation_descriptor: str = "euler-bunge",
    orientation_active_convention: bool = False,
    unit: str = "deg",
    elastic_strain_identifier: Optional[List[str]] = None,
    strain_unit: str = "microstrain",
    env: Optional[dict] = None,
) -> None:
    """Rotate each input CSV via a Voronoi build, appending Oij columns, saving under output_root."""

    os.makedirs(output_root, exist_ok=True)

    for input_file in input_files:
        input_file = os.path.abspath(input_file)
        base_name = os.path.basename(input_file)
        outputdir = os.path.join(output_root, f"{os.path.splitext(base_name)[0]}_tmp")
        os.makedirs(outputdir, exist_ok=True)

        builder = VoronoiMeshBuilder(
            input_csv=input_file,
            output_dir=outputdir,
            bounding_box=bounding_box,
            dim=dim,
            weighted=weighted,
            gmsh_version=gmsh_version,
            neper_version=neper_version,
            auto_fix_bbox=auto_fix_bbox,
            bbox_fix_mode=bbox_fix_mode,
            bbox_tolerance=bbox_tolerance,
            auto_rotate=auto_rotate,
            rotate_angles=rotate_angles,
            rotate_convention=rotate_convention,
            angle_identifier=angle_identifier,
            orientation_descriptor=orientation_descriptor,
            orientation_active_convention=orientation_active_convention,
            unit=unit,
            elastic_strain_identifier=elastic_strain_identifier,
            strain_unit=strain_unit,
            env=env,
        )

        builder.read_input()
        builder.build_voronoi(option="voronoi", generate_mesh=False)

        ori_path = os.path.join(outputdir, "reconstruction.ori")
        ori = np.loadtxt(ori_path)
        if ori.ndim == 1:
            ori = ori.reshape(1, 9)

        ori_df = pd.DataFrame(
            ori,
            columns=["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33"],
        )

        df = builder.data.copy().reset_index(drop=True)

        # Raw FF files may already carry an (unrotated) orientation matrix; drop
        # it so the freshly rotated O columns replace it instead of duplicating.
        df = df.drop(columns=ori_df.columns, errors="ignore")

        if (
            builder.strain_unit == "microstrain"
            and builder.elastic_strain_id is not None
        ):
            df[builder.elastic_strain_id] = df[builder.elastic_strain_id] * 1e6

        n = min(len(df), len(ori_df))
        combined = pd.concat(
            [
                ori_df.iloc[:n].reset_index(drop=True),
                df.iloc[:n].reset_index(drop=True),
            ],
            axis=1,
        )

        output_path = os.path.join(output_root, base_name)
        combined.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

        shutil.rmtree(outputdir, ignore_errors=True)
        print(f"Deleted temp: {outputdir}")


def try_parse_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def collect_experiment_files(data_dir):
    """
    Collect all CSV files in a directory with numeric names.
    Returns sorted list of CSV paths and numeric stress levels.
    """
    all_csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    valid_files = []
    for f in all_csvs:
        stem = os.path.basename(f).split(".")[0]
        if try_parse_float(stem) is not None:
            valid_files.append(f)

    files = sorted(valid_files, key=lambda s: float(os.path.basename(s).split(".")[0]))
    stress_levels = [float(os.path.basename(f).split(".")[0]) for f in files]
    return files, stress_levels
