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

        combined = os.path.join(tmpdir, split_output_prefix + ".e")
        if not os.path.exists(combined):
            # epu does not concatenate a single part ("no concatenation needed"), so no
            # <prefix>.e is produced; fall back to the lone per-rank Exodus file.
            single = os.path.join(
                tmpdir, f"{split_output_prefix}.e.{sculpt_config['nprocs']}.0"
            )
            if os.path.exists(single):
                shutil.move(single, combined)
        rescale_exodus_mesh(combined, data.cpu().numpy())

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


def _exodus_char_array(strings, width=33):
    """Pack a list of strings into an Exodus (n, width) S1 char array."""
    arr = np.zeros((len(strings), width), dtype="S1")
    for r, s in enumerate(strings):
        for c, ch in enumerate(s[: width - 1]):
            arr[r, c] = ch.encode()
    return arr


def write_voxel_exodus(
    data,
    output_mesh_filename,
    output_angle_filename,
    angle_convention="bunge",
    angle_type="radians",
    background_id=0,
):
    """Dump a segmented voxel grid directly to an Exodus hex mesh (no SCULPT).

    Emits one axis-aligned cube HEX8 per non-background voxel, with nodes shared
    between adjacent voxels (conformal, including across grain boundaries). Because
    every element is a perfect cube its scaled Jacobian is exactly 1, so the mesh
    has no inverted/sliver elements -- at the cost of stair-stepped grain boundaries
    and one hex per filled voxel. This is the robust alternative to mesh_sculpt when
    conformal smoothing produces bad elements.

    Args:
        data: (nx, ny, nz, 7) array [grain_id, Eul1, Eul2, Eul3, X, Y, Z] (torch or
            numpy); coordinates are voxel centers.
        output_mesh_filename (str): output Exodus (.e) path. One element block per
            grain, block ids relabeled to a contiguous 1..N (matching MOOSE/PUMA).
        output_angle_filename (str): orientation output stem; the per-block MRP file
            is written to ``output_angle_filename + ".csv"`` (one row per block, same
            order/convention as map_orientations).
        angle_convention (str): 'kocks', 'bunge', or 'roe'.
        angle_type (str): 'degrees' or 'radians'.
        background_id (int): voxel id treated as void (skipped). Default 0.

    Returns:
        dict: {'nodes', 'elements', 'blocks'} counts.
    """
    grid = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    nx, ny, nz = grid.shape[:3]
    ids = grid[..., 0].astype(np.int64)
    euler = grid[..., 1:4].astype(np.float64)

    # voxel spacing / origin from the stored voxel-center coordinates
    xs, ys, zs = grid[:, 0, 0, 4], grid[0, :, 0, 5], grid[0, 0, :, 6]
    dx = float(np.median(np.diff(xs))) if nx > 1 else 1.0
    dy = float(np.median(np.diff(ys))) if ny > 1 else 1.0
    dz = float(np.median(np.diff(zs))) if nz > 1 else 1.0
    x0, y0, z0 = float(xs[0]) - dx / 2, float(ys[0]) - dy / 2, float(zs[0]) - dz / 2

    fi = np.argwhere(ids != background_id)
    if fi.shape[0] == 0:
        raise ValueError("No non-background voxels in grid.")
    ii, jj, kk = fi[:, 0], fi[:, 1], fi[:, 2]
    gid = ids[ii, jj, kk]
    nelem = fi.shape[0]

    # 8 corner offsets in Exodus HEX8 order (bottom face CCW, then top face)
    offs = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    nxp, nyp = nx + 1, ny + 1
    lat = np.empty((nelem, 8), dtype=np.int64)
    for c, (di, dj, dk) in enumerate(offs):
        lat[:, c] = (ii + di) + nxp * ((jj + dj) + nyp * (kk + dk))

    uniq, inv = np.unique(lat, return_inverse=True)
    conn = (inv.reshape(nelem, 8) + 1).astype(np.int32)  # 1-indexed
    num_nodes = uniq.shape[0]

    li = uniq % nxp
    lj = (uniq // nxp) % nyp
    lk = uniq // (nxp * nyp)
    node_x = (x0 + li * dx).astype(np.float32)
    node_y = (y0 + lj * dy).astype(np.float32)
    node_z = (z0 + lk * dz).astype(np.float32)

    # per-grain blocks (contiguous 1..N) + averaged MRP orientation per block
    present = np.unique(gid)
    nblk = present.size
    blocks = []
    ori_rows = np.zeros((nblk, 3), dtype=np.float64)
    for b, g in enumerate(present):
        sel = gid == g
        blocks.append(conn[sel])
        mrp, _ = metrics.average_rotations(
            torch.tensor(euler[ii[sel], jj[sel], kk[sel]]),
            angle_convention=angle_convention,
            angle_type=angle_type,
        )
        ori_rows[b] = np.asarray(mrp).reshape(-1)[:3]
    np.savetxt(output_angle_filename + ".csv", ori_rows, delimiter=",")

    # write Exodus via scipy netcdf (mirrors the SCULPT mesh.e schema, v6.02)
    f = sio.netcdf_file(output_mesh_filename, "w", version=2)
    f.version = np.float32(6.02)
    f.api_version = np.float32(6.02)
    f.floating_point_word_size = np.int32(4)
    f.file_size = np.int32(1)
    f.title = "voxel_grid_to_exodus"
    # time_step (unlimited) MUST be created first (scipy: only the first dimension
    # may be unlimited). The ExodusII reader (libMesh) reads the time header on
    # open, so an empty time_step + time_whole record and QA/info records are
    # required even for a mesh-only file -- omitting them makes
    # read_num_time_steps() fail.
    f.createDimension("time_step", None)
    f.createDimension("len_string", 33)
    f.createDimension("len_line", 81)
    f.createDimension("four", 4)
    f.createDimension("len_name", 33)
    f.createDimension("num_dim", 3)
    f.createDimension("num_nodes", num_nodes)
    f.createDimension("num_elem", nelem)
    f.createDimension("num_el_blk", nblk)
    f.createDimension("num_qa_rec", 1)
    f.createDimension("num_info", 1)
    f.createVariable("time_whole", "f", ("time_step",))
    qa = np.zeros((1, 4, 33), dtype="S1")
    for c, s in enumerate(["graintrace", "voxel", "", ""]):
        for k, ch in enumerate(s[:32]):
            qa[0, c, k] = ch.encode()
    f.createVariable("qa_records", "c", ("num_qa_rec", "four", "len_string"))[:] = qa
    f.createVariable("info_records", "c", ("num_info", "len_line"))[:] = (
        _exodus_char_array(["voxel_grid_to_exodus"], width=81)
    )

    f.createVariable("coordx", "f", ("num_nodes",))[:] = node_x
    f.createVariable("coordy", "f", ("num_nodes",))[:] = node_y
    f.createVariable("coordz", "f", ("num_nodes",))[:] = node_z
    f.createVariable("coor_names", "c", ("num_dim", "len_name"))[:] = (
        _exodus_char_array(["x", "y", "z"])
    )

    ebp = f.createVariable("eb_prop1", "i", ("num_el_blk",))
    ebp[:] = np.arange(1, nblk + 1, dtype=np.int32)
    ebp.name = "ID"
    f.createVariable("eb_status", "i", ("num_el_blk",))[:] = np.ones(nblk, np.int32)
    f.createVariable("eb_names", "c", ("num_el_blk", "len_name"))[:] = (
        _exodus_char_array([""] * nblk)
    )
    f.createVariable("node_num_map", "i", ("num_nodes",))[:] = np.arange(
        1, num_nodes + 1, dtype=np.int32
    )

    for b in range(nblk):
        dn, pn = f"num_el_in_blk{b+1}", f"num_nod_per_el{b+1}"
        f.createDimension(dn, blocks[b].shape[0])
        f.createDimension(pn, 8)
        cv = f.createVariable(f"connect{b+1}", "i", (dn, pn))
        cv[:] = blocks[b]
        cv.elem_type = "HEX8"
    f.close()

    return {"nodes": num_nodes, "elements": nelem, "blocks": nblk}
