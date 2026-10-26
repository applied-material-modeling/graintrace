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
import re
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
    symmetry="1",
):
    """Write an spn and per-grain orientation file from fixed grid data.

    Args:
        data (np.ndarray): nx x ny x nz x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
        filename_spn (str): output spn filename
        filename_orientations (str): output orientations filename
        angle_convention (str): 'kocks', 'bunge', or 'roe'
        angle_type (str): 'degrees' or 'radians'
        symmetry (str): crystal symmetry for the per-grain orientation average
            (see :func:`graintrace.nf.metrics.average_rotations`). Pass the
            crystal point group (e.g. ``"432"``) so voxels of one grain reported
            in different symmetry variants average correctly.

    Raises:
        ValueError: if any voxel still carries a negative ("unsegmented") phase.
            Such a voxel is not a grain, and mapping it to void here would
            silently delete material from the mesh, so the caller decides.
    """
    flat_data = data.reshape(-1, 7)

    if bool((flat_data[:, 0] < 0).any()):
        raise ValueError(
            "write_spn: unsegmented voxels (phase < 0) remain in the grid. "
            "Run segment.infill_nearest_neighbor() after "
            "segment.remove_small_segments(), or mask the voxels to void (0), "
            "before meshing."
        )

    phases = torch.sort(torch.unique(flat_data[:, 0])).values
    phases = phases[phases > 0]

    orientations = torch.zeros((len(phases), 3))
    orientations_sameconv = torch.zeros((len(phases), 3))
    # renumber out of place: writing i + 1 back into the column being scanned
    # lets an already relabelled grain be picked up again by a later iteration
    new_phase = torch.zeros_like(flat_data[:, 0])
    for i, phase in tqdm.tqdm(
        enumerate(phases), total=len(phases), desc="Calculating grain orientations"
    ):
        in_phase = flat_data[:, 0] == phase
        new_phase[in_phase] = i + 1

        angles = flat_data[in_phase, 1:4]
        avg_R, avg = metrics.average_rotations(
            angles,
            angle_convention=angle_convention,
            angle_type=angle_type,
            symmetry=symmetry,
        )
        orientations[i] = avg_R
        orientations_sameconv[i] = avg

    flat_data[:, 0] = new_phase

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
    symmetry="1",
    layer_spacing=None,
    source_labels=None,
    block_map_filename=None,
):
    """Call sculpt (+epu) to generate an Exodus mesh from an spn file.

    After ``epu`` the mesh is post-processed in three steps:

    1. :func:`rescale_exodus_mesh` maps it onto the grid's voxel-EDGE box.
    2. :func:`renumber_blocks_contiguous` renumbers SCULPT's block ids
       (``eb_prop1``, the spn material ids, sparse once SCULPT drops a small
       grain) to a contiguous ``1..N`` in file order, as the CPFE decks require,
       and writes a block map CSV.
    3. :func:`map_orientations` writes one orientation row per block in file
       order, so row ``k`` of the orientations CSV belongs to block id ``k + 1``.

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
        symmetry (str): crystal symmetry for the per-block orientation average
            (see :func:`graintrace.nf.metrics.average_rotations`).
        layer_spacing (float, optional): z voxel size, used when the grid has a
            single layer (see :func:`voxel_spacing`).
        source_labels (array, optional): (nx, ny, nz) grain ids of the caller's
            grid before :func:`write_spn` renumbered them; adds a
            ``source_grain_id`` column to the block map.
        block_map_filename (str, optional): block map CSV path; defaults to
            ``<mesh stem>_block_map.csv`` beside the mesh.

    Returns:
        str: path to the block map CSV.
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
        grid = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        rescale_exodus_mesh(combined, grid, layer_spacing=layer_spacing)

        mesh_out = os.path.join(cwd, output_mesh_filename)
        shutil.move(os.path.join(tmpdir, split_output_prefix + ".e"), mesh_out)

    if block_map_filename is None:
        block_map_filename = os.path.splitext(mesh_out)[0] + "_block_map.csv"
    renumbered = renumber_blocks_contiguous(
        mesh_out, grid, block_map_filename, source_labels=source_labels
    )
    print(f"Block map ({len(renumbered['block_grains'])} blocks): {block_map_filename}")

    map_orientations(
        mesh_out,
        grid,
        output_angle_filename,
        angle_convention,
        angle_type,
        symmetry,
        block_grains=renumbered["block_grains"],
    )
    return block_map_filename


def voxel_spacing(data, layer_spacing=None):
    """Per-axis voxel size of a fixed grid, from its voxel-centre coordinates.

    An axis with more than one voxel takes the median step of its centres. A
    single-voxel axis has no step to measure: z takes ``layer_spacing`` when
    given (``NearFieldMeshBuilder`` knows its ``dz``), and otherwise -- as does a
    single-voxel x or y -- falls back to the mean in-plane spacing of the measured
    axes (1.0 if none can be measured).

    Args:
        data (np.ndarray): nx x ny x nz x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
        layer_spacing (float, optional): known z voxel size.

    Returns:
        np.ndarray: (3,) spacing (hx, hy, hz).
    """
    grid = np.asarray(data)
    centres = (grid[:, 0, 0, 4], grid[0, :, 0, 5], grid[0, 0, :, 6])
    h = np.full(3, np.nan)
    for i, c in enumerate(centres):
        if c.size > 1:
            h[i] = float(np.median(np.abs(np.diff(c))))
    if np.isnan(h[2]) and layer_spacing is not None:
        h[2] = float(layer_spacing)
    measured = h[:2][~np.isnan(h[:2])]
    if measured.size == 0:
        measured = h[~np.isnan(h)]
    fallback = float(measured.mean()) if measured.size else 1.0
    h[np.isnan(h)] = fallback
    return h


def voxel_edge_bounds(data, layer_spacing=None):
    """Voxel-edge bounding box of a fixed grid.

    The grid stores voxel CENTRES; a hex mesh of it spans the voxel EDGES, half a
    voxel further out on every side: ``[min - h/2, max + h/2]`` per axis.

    Args:
        data (np.ndarray): nx x ny x nz x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
        layer_spacing (float, optional): known z voxel size; see :func:`voxel_spacing`.

    Returns:
        tuple: (lo, hi), each a (3,) array.
    """
    grid = np.asarray(data)
    h = voxel_spacing(grid, layer_spacing=layer_spacing)
    lo = np.array([grid[..., 4 + i].min() for i in range(3)]) - h / 2
    hi = np.array([grid[..., 4 + i].max() for i in range(3)]) + h / 2
    return lo, hi


def rescale_exodus_mesh(exo_file, data, layer_spacing=None):
    """Rescale Exodus mesh coordinates onto a fixed grid's voxel-edge box.

    SCULPT's boundary faces lie on voxel edges, so the mesh is mapped onto
    :func:`voxel_edge_bounds`, not onto the voxel-centre extents (which shrank the
    mesh by one voxel per axis and collapsed a single-layer grid to zero
    thickness -- a 2D mesh of HEX8 elements that puma-opt cannot write).

    Args:
        exo_file (str): path to Exodus mesh file
        data (np.ndarray): nx x ny x nz x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
        layer_spacing (float, optional): z voxel size for a single-layer grid;
            without it the in-plane spacing is used (see :func:`voxel_spacing`).
    """
    lo, hi = voxel_edge_bounds(data, layer_spacing=layer_spacing)
    with sio.netcdf_file(exo_file, "a") as f:
        for i, coord in enumerate(["coordx", "coordy", "coordz"]):
            coords = f.variables[coord][:]
            coords_min = coords.min()
            coords_max = coords.max()
            span = coords_max - coords_min
            scaled_coords = (coords - coords_min) / span if span > 0 else coords * 0
            f.variables[coord][:] = scaled_coords * (hi[i] - lo[i]) + lo[i]


def _decode_names(arr):
    """Decode an Exodus (n, len_name) S1 char array into a list of str."""
    return [b"".join(row).decode(errors="replace").rstrip("\x00 ") for row in arr]


def renumber_blocks_contiguous(
    exo_file, data, block_map_filename, source_labels=None, background_id=0
):
    """Renumber Exodus block ids to a contiguous 1..N in file order + write a block map.

    SCULPT keeps the spn material ids as block ids (``eb_prop1``); once it drops a
    small grain they are sparse (e.g. 1..353 with gaps), and the CPFE decks, which
    assume blocks ``1..N``, abort with "block id 5 does not exist". Renumbering in
    FILE order is safe because :func:`map_orientations` writes one row per block in
    that same order: after this call row ``k`` of the orientations CSV belongs to
    block id ``k + 1``.

    The block-to-grain association is resolved from the ORIGINAL ids first (see
    :func:`_block_grain_ids`), falling back to each block's majority grain when the
    original ids are not grid grain ids. Block names (``eb_names``) whose trailing
    number is the old id get the new id; other names are kept.

    The block map CSV has one row per block: ``block_id`` (new, 1..N),
    ``sculpt_block_id`` (original ``eb_prop1``), ``grain_id`` (majority grid label
    of the block's element centroids, via a KD-tree over the solid voxel centres),
    ``n_elem``, and ``source_grain_id`` when ``source_labels`` is given.

    Args:
        exo_file (str): path to Exodus mesh file, modified in place.
        data (np.ndarray): nx x ny x nz x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
            the mesh was generated from.
        block_map_filename (str): output block map CSV path.
        source_labels (array, optional): (nx, ny, nz) grain ids to also report per
            block, e.g. the grid's labels before :func:`write_spn` renumbered them.
        background_id (int): grid id treated as void. Default 0.

    Returns:
        dict: ``block_grains`` (grid grain id per block, file order, for
        :func:`map_orientations`), ``renumbered`` (bool) and ``block_map`` (path).
    """
    # pylint: disable-next=import-outside-toplevel  # pandas only needed here
    import pandas as pd

    flat = np.asarray(data).reshape(-1, 7)
    solid = np.flatnonzero(flat[:, 0] != background_id)
    if solid.size == 0:
        raise ValueError("No non-void voxels in the grid.")
    tree = sp.cKDTree(flat[solid, 4:7])
    src = None if source_labels is None else np.asarray(source_labels).reshape(-1)

    with sio.netcdf_file(exo_file, "a", mmap=False) as f:
        n_blk = int(f.dimensions["num_el_blk"])
        coords = np.stack(
            [f.variables[c][:] for c in ("coordx", "coordy", "coordz")], axis=1
        )
        has_ids = "eb_prop1" in f.variables
        old = (
            np.asarray(f.variables["eb_prop1"][:]).astype(np.int64)
            if has_ids
            else np.arange(1, n_blk + 1)
        )

        rows = []
        majority = np.zeros(n_blk)
        for eb in range(n_blk):
            connect = f.variables[f"connect{eb+1}"][:]
            cent = coords[connect - 1, :].mean(axis=1)
            _, idx = tree.query(cent)
            vox = solid[idx]
            labels, counts = np.unique(flat[vox, 0], return_counts=True)
            majority[eb] = labels[np.argmax(counts)]
            row = {
                "block_id": eb + 1,
                "sculpt_block_id": int(old[eb]),
                "grain_id": int(majority[eb]),
                "n_elem": int(connect.shape[0]),
            }
            if src is not None:
                sl, sc = np.unique(src[vox], return_counts=True)
                row["source_grain_id"] = int(sl[np.argmax(sc)])
            rows.append(row)

        try:
            block_grains = _block_grain_ids(f, n_blk, np.unique(flat[solid, 0]))
        except ValueError:
            block_grains = majority

        new = np.arange(1, n_blk + 1)
        renumbered = has_ids and not np.array_equal(old, new)
        if renumbered:
            f.variables["eb_prop1"][:] = new.astype(f.variables["eb_prop1"][:].dtype)
            if "eb_names" in f.variables:
                names = _decode_names(f.variables["eb_names"][:])
                width = f.variables["eb_names"].shape[1]
                fixed = []
                for o, n, name in zip(old, new, names):
                    m = re.search(r"(\d+)$", name)
                    if m and int(m.group(1)) == int(o):
                        name = name[: m.start()] + str(int(n))
                    fixed.append(name)
                f.variables["eb_names"][:] = _exodus_char_array(fixed, width=width)

    pd.DataFrame(rows).to_csv(block_map_filename, index=False)
    return {
        "block_grains": np.asarray(block_grains, dtype=np.float64),
        "renumbered": bool(renumbered),
        "block_map": block_map_filename,
    }


def _block_grain_ids(f, n_blk, present):
    """Associate each Exodus element block with a grid grain id.

    Prefers the Exodus block ids (``eb_prop1``) when every one of them names a grain
    present in the grid; falls back to a positional association with the sorted
    non-void grain ids when the counts match.

    Args:
        f: open Exodus netcdf file handle.
        n_blk (int): number of element blocks.
        present (np.ndarray): sorted unique non-void grid grain ids.

    Returns:
        np.ndarray: one grain id per element block.

    Raises:
        ValueError: if the blocks cannot be associated by id or by position.
    """
    present_set = set(float(g) for g in present)
    if "eb_prop1" in f.variables:
        block_ids = np.asarray(f.variables["eb_prop1"][:], dtype=np.float64)
        if block_ids.size == n_blk and all(float(b) in present_set for b in block_ids):
            return block_ids
    if n_blk == present.size:
        return np.asarray(present, dtype=np.float64)
    raise ValueError(
        f"Cannot associate {n_blk} Exodus element blocks with the {present.size} "
        "grains in the grid: the block ids are not grid grain ids and the counts "
        "differ. Pass the grid the mesh was generated from."
    )


def map_orientations(
    exo_file,
    data,
    output_angle_filename,
    angle_convention="bunge",
    angle_type="radians",
    symmetry="1",
    background_id=0,
    block_grains=None,
):
    """Map orientations from fixed grid data onto an Exodus mesh's blocks.

    One row is written per block in Exodus FILE order, so with contiguous block
    ids (see :func:`renumber_blocks_contiguous`) row ``k`` belongs to block id
    ``k + 1``.

    Each Exodus element block is one grain, so a block's elements are looked up
    against **that grain's own voxels only**: void voxels (``background_id``, which
    carry a meaningless Euler (0, 0, 0)) are excluded from the search, and the
    per-block KD-tree is restricted to voxels of the block's grain. A single tree
    over every voxel lets a surface or void-adjacent element take the identity from
    a void voxel, or a neighbouring grain's orientation across the boundary.

    Block-to-grain association uses the Exodus block ids (``eb_prop1``) when they are
    all present in the grid -- the SCULPT path, where :func:`write_spn` relabels the
    grid ids to the spn material ids that become the block ids. Otherwise, when the
    block count matches the grain count, blocks are associated positionally with the
    sorted non-void grain ids.

    Args:
        exo_file (str): path to Exodus mesh file
        data (np.ndarray): nx x ny x nz x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
        output_angle_filename (str): path to output orientations file
        angle_convention (str): 'kocks', 'bunge', or 'roe'
        angle_type (str): 'degrees' or 'radians'
        symmetry (str): crystal symmetry for the per-block orientation average
            (see :func:`graintrace.nf.metrics.average_rotations`).
        background_id (int): grid id treated as void and excluded from the lookup.
            Default 0.
        block_grains (array, optional): grid grain id of each block in file order,
            overriding the ``eb_prop1``/positional association -- needed once the
            block ids have been renumbered away from the grid ids.

    Raises:
        ValueError: if the grid has no non-void voxels, or the Exodus blocks cannot
            be associated with the grid's grains by id or by position.
    """
    flat_data = np.asarray(data).reshape(-1, 7)
    solid = flat_data[:, 0] != background_id
    if not solid.any():
        raise ValueError(
            f"No non-void voxels in the grid (every id == background_id={background_id})."
        )
    grain_ids = flat_data[:, 0]
    present = np.unique(grain_ids[solid])

    with sio.netcdf_file(exo_file, "a") as f:
        coords = np.stack(
            [
                f.variables["coordx"][:],
                f.variables["coordy"][:],
                f.variables["coordz"][:],
            ],
            axis=1,
        )
        n_blk = int(f.dimensions["num_el_blk"])
        if block_grains is None:
            block_grains = _block_grain_ids(f, n_blk, present)
        elif len(block_grains) != n_blk:
            raise ValueError(
                f"block_grains has {len(block_grains)} entries for {n_blk} blocks"
            )

        # one KD-tree per grain, over that grain's voxels only
        trees = {}
        for g in present:
            sel = np.flatnonzero(grain_ids == g)
            trees[float(g)] = (sp.cKDTree(flat_data[sel, 4:7]), sel)

        orientations = []
        orientations_sameconv = []
        for eb in range(n_blk):
            connect = f.variables[f"connect{eb+1}"][:]
            elem_coords = coords[connect - 1, :].mean(axis=1)
            tree, sel = trees[float(block_grains[eb])]
            _, idx = tree.query(elem_coords)
            elem_orientations = flat_data[sel[idx], 1:4]
            avg_R, avg_sameconv = metrics.average_rotations(
                torch.tensor(elem_orientations),
                angle_convention=angle_convention,
                angle_type=angle_type,
                symmetry=symmetry,
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
    symmetry="1",
    layer_spacing=None,
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
            Voxels with a negative ("unsegmented") id are skipped as well, so an
            orphan left by ``remove_small_segments`` does not become a block of
            its own and shift every grain id.
        symmetry (str): crystal symmetry for the per-block orientation average
            (see :func:`graintrace.nf.metrics.average_rotations`).
        layer_spacing (float, optional): z voxel size for a single-layer grid. The
            mesh spans the voxel-edge box (:func:`voxel_edge_bounds`), the same
            extents :func:`mesh_sculpt` rescales to; a single-voxel axis uses this
            (z only) or else the in-plane spacing (:func:`voxel_spacing`).

    Returns:
        dict: {'nodes', 'elements', 'blocks'} counts.
    """
    grid = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    nx, ny = grid.shape[:2]
    ids = grid[..., 0].astype(np.int64)
    euler = grid[..., 1:4].astype(np.float64)

    # voxel spacing / origin from the stored voxel-center coordinates
    xs, ys, zs = grid[:, 0, 0, 4], grid[0, :, 0, 5], grid[0, 0, :, 6]
    dx, dy, dz = (float(h) for h in voxel_spacing(grid, layer_spacing=layer_spacing))
    x0, y0, z0 = float(xs[0]) - dx / 2, float(ys[0]) - dy / 2, float(zs[0]) - dz / 2

    # a negative ("unsegmented") voxel is not a grain: meshing it would give it
    # its own block, and since block ids are relabelled contiguously it would
    # sort first and shift every real grain one block along
    fi = np.argwhere((ids != background_id) & (ids >= 0))
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
            symmetry=symmetry,
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
