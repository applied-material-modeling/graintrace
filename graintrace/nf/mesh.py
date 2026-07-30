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

"""SCULPT hex meshing and orientation mapping for NF voxel grids."""

from __future__ import annotations

import tempfile
import os.path
import subprocess
import shutil

import torch
import numpy as np
import scipy.io as sio
import scipy.spatial as sp
import tqdm

from . import metrics


def write_spn(
    data,
    filename_spn,
    filename_orientations,
    angle_convention="bunge",
    angle_type="radians",
):
    """Write an spn and per-grain orientation file from fixed grid data.

    Args:
        data (np.ndarray): nx x ny x nz x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
        filename_spn (str): output spn filename
        filename_orientations (str): output orientations filename
        angle_convention (str): 'kocks', 'bunge', or 'roe'
        angle_type (str): 'degrees' or 'radians'
    """
    flat_data = data.reshape(-1, 7)
    phases = torch.sort(torch.unique(flat_data[:, 0])).values
    phases = phases[phases != 0]

    orientations = torch.zeros((len(phases), 3))
    orientations_sameconv = torch.zeros((len(phases), 3))
    for i, phase in tqdm.tqdm(
        enumerate(phases), total=len(phases), desc="Calculating grain orientations"
    ):
        in_phase = flat_data[:, 0] == phase
        flat_data[in_phase, 0] = i + 1

        angles = flat_data[in_phase, 1:4]
        avg_R, avg = metrics.average_rotations(
            angles,
            angle_convention=angle_convention,
            angle_type=angle_type,
        )
        orientations[i] = avg_R
        orientations_sameconv[i] = avg

    np.savetxt(filename_orientations, orientations.numpy(), delimiter=",")
    np.savetxt(
        filename_orientations + "_sameconv",
        orientations_sameconv.numpy(),
        delimiter=",",
    )
    np.savetxt(filename_spn, flat_data[:, 0].numpy().flatten(), delimiter=" ", fmt="%d")


def mesh_sculpt(
    sculpt_config,
    sculpt_options,
    input_spn,
    data,
    output_mesh_filename,
    output_angle_filename,
    split_output_prefix="output_mesh",
    angle_convention="bunge",
    angle_type="radians",
):
    """Call sculpt (+epu) to generate an Exodus mesh from an spn file.

    Args:
        sculpt_config (dict): configuration for the sculpt call
        sculpt_options (list): extra command line options for sculpt
        input_spn (str): path to input spn file
        data (np.ndarray): nx x ny x nz x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
        output_mesh_filename (str): path to output exodus mesh file
        output_angle_filename (str): path to output orientations file
        split_output_prefix (str): prefix for temporary output files
        angle_convention (str): 'kocks', 'bunge', or 'roe'
        angle_type (str): 'degrees' or 'radians'
    """
    cwd = os.getcwd()
    nx, ny, nz = data.shape[:3]
    abs_path_spn = os.path.abspath(input_spn)

    launcher = sculpt_config["launcher"]
    base = os.path.basename(launcher)

    if base in ("mpiexec", "mpiexec.hydra"):
        nflag = "-n"
    elif base == "mpirun":
        nflag = "-np"
    elif base == "srun":
        nflag = "-n"
    else:
        raise ValueError(
            f"Unsupported launcher {launcher!r}. Use mpiexec, mpirun, or srun."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        sclupt_call = [
            sculpt_config["launcher"],
            nflag,
            str(sculpt_config["nprocs"]),
            sculpt_config["psculpt"],
            "-j",
            str(sculpt_config["nprocs"]),
            "-x",
            str(nx),
            "-y",
            str(ny),
            "-z",
            str(nz),
            "-isp",
            abs_path_spn,
            "-e",
            split_output_prefix,
        ] + sculpt_options
        env = os.environ.copy()
        env.update(sculpt_config.get("environment", {}))
        subprocess.run(sclupt_call, check=True, cwd=tmpdir, env=env)

        epu_call = [
            sculpt_config["epu"],
            "-p",
            str(sculpt_config["nprocs"]),
            split_output_prefix,
        ]
        subprocess.run(epu_call, check=True, cwd=tmpdir, env=env)
        rescale_exodus_mesh(
            os.path.join(tmpdir, split_output_prefix + ".e"), data.cpu().numpy()
        )

        shutil.move(
            os.path.join(tmpdir, split_output_prefix + ".e"),
            os.path.join(cwd, output_mesh_filename),
        )

        map_orientations(
            os.path.join(cwd, output_mesh_filename),
            data.cpu().numpy(),
            output_angle_filename,
            angle_convention,
            angle_type,
        )


def rescale_exodus_mesh(exo_file, data):
    """Rescale Exodus mesh coordinates to match those in a fixed grid.

    Args:
        exo_file (str): path to Exodus mesh file
        data (np.ndarray): nx x ny x nz x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
    """
    with sio.netcdf_file(exo_file, "a") as f:
        for i, coord in enumerate(["coordx", "coordy", "coordz"]):
            coords = f.variables[coord][:]
            min_coord = data[..., 4 + i].min()
            max_coord = data[..., 4 + i].max()
            coords_min = coords.min()
            coords_max = coords.max()
            scaled_coords = (coords - coords_min) / (coords_max - coords_min)
            new_coords = scaled_coords * (max_coord - min_coord) + min_coord
            f.variables[coord][:] = new_coords


def map_orientations(
    exo_file,
    data,
    output_angle_filename,
    angle_convention="bunge",
    angle_type="radians",
):
    """Map orientations from fixed grid data onto an Exodus mesh's blocks.

    Args:
        exo_file (str): path to Exodus mesh file
        data (np.ndarray): nx x ny x nz x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
        output_angle_filename (str): path to output orientations file
        angle_convention (str): 'kocks', 'bunge', or 'roe'
        angle_type (str): 'degrees' or 'radians'
    """
    flat_data = data.reshape(-1, 7)
    kd = sp.cKDTree(flat_data[:, 4:7])

    with sio.netcdf_file(exo_file, "a") as f:
        coords = np.stack(
            [
                f.variables["coordx"][:],
                f.variables["coordy"][:],
                f.variables["coordz"][:],
            ],
            axis=1,
        )
        orientations = []
        orientations_sameconv = []
        for eb in range(f.dimensions["num_el_blk"]):
            connect = f.variables[f"connect{eb+1}"][:]
            elem_coords = coords[connect - 1, :].mean(axis=1)
            _, idx = kd.query(elem_coords)
            elem_orientations = flat_data[idx, 1:4]
            avg_R, avg_sameconv = metrics.average_rotations(
                torch.tensor(elem_orientations),
                angle_convention=angle_convention,
                angle_type=angle_type,
            )
            orientations.append(avg_R.numpy())
            orientations_sameconv.append(avg_sameconv.numpy())

        orientations = np.stack(orientations, axis=0)
        orientations_sameconv = np.stack(orientations_sameconv, axis=0)

        np.savetxt(output_angle_filename + ".csv", orientations, delimiter=",")
        np.savetxt(
            output_angle_filename + "_sameconv.csv",
            orientations_sameconv,
            delimiter=",",
        )
