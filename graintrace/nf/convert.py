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

"""NF point-cloud to fixed-grid conversion and VTK export helpers."""

from __future__ import annotations

import glob
import os
import os.path
import re
import warnings

import multiprocess
import numpy as np
import pandas as pd
import scipy.spatial as sp

# vtk exposes these as C-extension names pylint cannot statically resolve
# pylint: disable=no-name-in-module,import-error
from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkIOLegacy import vtkStructuredGridWriter
from vtk.util import numpy_support

# pylint: enable=no-name-in-module,import-error

import tqdm


def layer_number(path: str, token: str = "layer") -> int:
    """Extract the layer index that follows ``token`` in a filename.

    Raises:
        ValueError if no layer number can be found.
    """
    name = os.path.splitext(os.path.basename(path))[0]

    m = re.search(rf"(?i){re.escape(token)}[\s_\-]*([0-9]+)", name)
    if not m:
        raise ValueError(f"Could not parse layer number from filename: {path}")

    return int(m.group(1))


def _find_layer_files(folder, layer_token, recursive):
    """Layer files under ``folder``: .mic preferred over .csv, flat or recursive.

    A recursive search keeps only files whose name carries ``layer_token`` plus a
    number, so unrelated CSVs elsewhere in the tree are not mistaken for layers.
    """
    pattern = "**/*" if recursive else "*"
    for ext in ("mic", "csv"):
        files = glob.glob(os.path.join(folder, f"{pattern}.{ext}"), recursive=recursive)
        if recursive:
            token = re.compile(rf"(?i){re.escape(layer_token)}[\s_\-]*[0-9]+")
            files = [
                f
                for f in files
                if token.search(os.path.splitext(os.path.basename(f))[0])
            ]
        if files:
            return files
    return []


def nf_to_pointcloud(folder, dz, layer_token="layer", recursive=False):
    """Convert nearfield data (.mic/.csv layers) to a stacked point cloud.

    Layers are ordered by the number after ``layer_token`` in each file name and
    must be contiguous. MIDAS writes one subfolder per layer
    (``Layer1/...layer1.mic``, ``Layer2/...``); set ``recursive=True`` to search
    subfolders. With ``recursive=False`` (default) the folder itself is searched,
    and only if it holds no layer files does the search fall back to subfolders.

    Args:
        folder (str): folder with nearfield data
        dz (float): spacing between layers in the z-direction
        layer_token (str): filename token that precedes the layer number
        recursive (bool): search subfolders too

    Returns:
        pd.DataFrame: the stacked point cloud, one row per measured point, with the
        layer's columns (Confidence, TriEdgeSize, OrientationRowNr when present)
        plus Z and layer.

    Raises:
        FileNotFoundError: if no layer files are found.
        ValueError: if a layer number is missing or duplicated.
    """
    files = _find_layer_files(folder, layer_token, recursive)
    if not files and not recursive:
        files = _find_layer_files(folder, layer_token, True)
        if files:
            print(
                f"No layer files directly in {folder}; using {len(files)} found in subfolders"
            )
    if not files:
        raise FileNotFoundError("No .mic or .csv files found in the specified folder")

    files = sorted(files, key=lambda f: layer_number(f, token=layer_token))

    layers = [layer_number(f, token=layer_token) for f in files]
    min_layer = min(layers)
    max_layer = max(layers)
    if layers != list(range(min_layer, max_layer + 1)):
        raise ValueError(
            f"Missing or duplicate layers in nearfield data. Found: {layers[:10]}..."
        )

    dfs = []
    print("\n")
    for i, f in tqdm.tqdm(enumerate(files), total=len(files), desc="Reading layers"):
        df = pd.read_csv(f, skiprows=3, sep="\\s+", header=0)
        df = df.rename(columns={"%OrientationRowNr": "OrientationRowNr"})
        df["Z"] = i * dz
        df["layer"] = i

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def nearest_simplex_vertex(tri, query_points, simplex_indices):
    """Index of the nearest vertex of each query point's containing simplex.

    This is the legacy source-point rule, kept for the Delaunay hull path so that
    an unmasked grid is bit-identical to the pre-confidence one. It is not the same
    as the globally nearest measured point: the nearest site to a query point need
    not be a vertex of the simplex containing it (measured on a 60-point random
    layer, 5 of 102 in-hull grid points differ, by up to 0.16 um).

    Args:
        tri (scipy.spatial.Delaunay): triangulation of the measured points
        query_points (np.ndarray): (m, 2) points, all inside the hull
        simplex_indices (np.ndarray): (m,) index of each point's containing simplex

    Returns:
        np.ndarray: (m,) indices into the triangulation's points
    """
    vertices = tri.simplices[simplex_indices]
    dists = np.linalg.norm(tri.points[vertices] - query_points[:, None], axis=-1)
    return vertices[np.arange(len(simplex_indices)), np.argmin(dists, axis=-1)]


def process_layer_pointcloud(
    df_layer, plane_points, nx, ny, min_confidence=None, max_point_distance=None
):
    """Interpolate one point-cloud layer onto a fixed nx x ny grid.

    Which grid points become solid is decided by ``max_point_distance`` (a radius
    around the measured points) when it is given, otherwise by the Delaunay hull of
    the layer. Grid points whose source point is below ``min_confidence`` become void.

    The source point is the globally nearest measured point under
    ``max_point_distance``, and the nearest vertex of the containing simplex under
    the hull test. The hull path keeps the legacy rule rather than the (strictly
    closer) global nearest so that an unmasked grid is unchanged; see
    ``nearest_simplex_vertex``.

    Args:
        df_layer (pd.DataFrame): one layer of the point cloud, columns X, Y, Z,
            Eul1, Eul2, Eul3 and optionally Confidence
        plane_points (np.ndarray): (nx*ny, 2) in-plane grid coordinates
        nx, ny (int): in-plane grid resolution
        min_confidence (float, optional): grid points whose source point has a lower
            ``Confidence`` become void (phase 0). None keeps every point that passes
            the geometric test. Ignored when the layer has no ``Confidence`` column.
        max_point_distance (float, optional): maximum distance in micrometers from a
            grid point to the nearest measured point for it to be solid. None uses the
            legacy Delaunay hull test, which fills concave gaps and holes.

    Returns:
        tuple: (nx x ny x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z],
        nx x ny array of the confidence each solid grid point took, 0 where void)
    """
    z = df_layer["Z"].iloc[0]
    points = df_layer[["X", "Y"]].to_numpy()

    if max_point_distance is None:
        tri = sp.Delaunay(points)
        simplex_indices = tri.find_simplex(plane_points)
        ok = simplex_indices >= 0
        idx = np.zeros(len(plane_points), dtype=np.intp)
        idx[ok] = nearest_simplex_vertex(tri, plane_points[ok], simplex_indices[ok])
    else:
        dist, idx = sp.cKDTree(points).query(plane_points)
        ok = dist <= float(max_point_distance)

    conf = np.ones(len(plane_points))
    # A layer stacked beside ones that have Confidence gets an all-NaN column
    if "Confidence" in df_layer and df_layer["Confidence"].notna().any():
        conf = df_layer["Confidence"].to_numpy()[idx]
        if min_confidence is not None:
            ok = ok & (conf >= float(min_confidence))

    data = np.zeros((nx * ny, 7))

    data[ok, 0] = 1
    data[ok, 1:4] = df_layer[["Eul1", "Eul2", "Eul3"]].to_numpy()[idx[ok]]

    data[:, 4:6] = plane_points
    data[:, 6] = z

    return data.reshape((nx, ny, 7)), np.where(ok, conf, 0.0).reshape((nx, ny))


def grid_axis(lo, hi, n=None, spacing=None):
    """Build one in-plane grid axis from either a point count or a spacing.

    Args:
        lo, hi (float): axis extent of the data
        n (int, optional): number of grid points, spread evenly over [lo, hi]
        spacing (float, optional): grid spacing in micrometers; the axis starts at
            ``lo`` and is extended until it covers ``hi``

    Returns:
        np.ndarray: the 1D axis coordinates

    Raises:
        ValueError: if neither or both of ``n`` and ``spacing`` are given, or if they
            are not positive.
    """
    if (n is None) == (spacing is None):
        raise ValueError("Give exactly one of a grid count or a voxel size, not both")

    if spacing is None:
        if int(n) < 1:
            raise ValueError(f"Grid counts must be >= 1, got {n}")
        return np.linspace(lo, hi, int(n))

    spacing = float(spacing)
    if spacing <= 0:
        raise ValueError(f"Voxel sizes must be > 0, got {spacing}")

    # Cover the footprint: round the span up to a whole number of voxels
    count = int(np.ceil((hi - lo) / spacing - 1e-9)) + 1
    return lo + np.arange(count) * spacing


def pointcloud_to_fixed_grid(
    pointcloud,
    nx=None,
    ny=None,
    voxel_size_um=None,
    min_confidence=None,
    max_point_distance=None,
):
    """Convert a point cloud to an nx x ny x nz x 7 fixed grid.

    Args:
        pointcloud (pd.DataFrame): columns [phase, Eul1, Eul2, Eul3, X, Y, Z, layer]
            and optionally Confidence
        nx, ny (int, optional): in-plane grid resolution; mutually exclusive with
            voxel_size_um
        voxel_size_um (tuple, optional): (hx, hy) in-plane voxel size in micrometers,
            as an alternative to nx/ny. A fixed count over a rectangular footprint
            gives anisotropic voxels; a spacing gives an isotropic grid independent of
            the footprint.
        min_confidence (float, optional): confidence threshold below which a grid point
            becomes void; see process_layer_pointcloud
        max_point_distance (float, optional): maximum distance in micrometers to the
            nearest measured point; see process_layer_pointcloud

    Returns:
        tuple: (nx x ny x nz x 7 grid [phase, Eul1, Eul2, Eul3, X, Y, Z],
        nx x ny x nz array of the confidence each solid voxel took, 0 where void)

    Raises:
        ValueError: if voxel_size_um is given together with nx or ny, if neither is
            given, or if voxel_size_um is not a pair.
    """
    if voxel_size_um is not None:
        if nx is not None or ny is not None:
            raise ValueError("Give either nx/ny or voxel_size_um, not both")
        if len(voxel_size_um) != 2:
            raise ValueError(
                f"voxel_size_um must be (hx, hy), got {voxel_size_um!r}",
            )
        hx, hy = voxel_size_um
    elif nx is None or ny is None:
        raise ValueError("Give both nx and ny, or voxel_size_um instead")
    else:
        hx = hy = None

    layers = pointcloud["layer"].unique()
    if min_confidence is not None:
        if "Confidence" not in pointcloud.columns:
            warnings.warn(
                "min_confidence was given but the NF layers have no Confidence column; "
                "no voxel is masked by confidence",
                stacklevel=2,
            )
        else:
            missing = [
                int(layer)
                for layer in layers
                if pointcloud.loc[pointcloud["layer"] == layer, "Confidence"]
                .isna()
                .all()
            ]
            if missing:
                warnings.warn(
                    f"min_confidence was given but layers {missing[:10]} have no "
                    "Confidence values; those layers are not masked by confidence",
                    stacklevel=2,
                )
    x = grid_axis(pointcloud["X"].min(), pointcloud["X"].max(), n=nx, spacing=hx)
    y = grid_axis(pointcloud["Y"].min(), pointcloud["Y"].max(), n=ny, spacing=hy)
    nx, ny = len(x), len(y)

    X, Y = np.meshgrid(x, y, indexing="ij")
    plane_points = np.vstack([X.ravel(), Y.ravel()]).T

    with multiprocess.Pool() as tp:
        layer_collection = list(
            tqdm.tqdm(
                tp.imap(
                    lambda df_layer: process_layer_pointcloud(
                        df_layer,
                        plane_points,
                        nx,
                        ny,
                        min_confidence=min_confidence,
                        max_point_distance=max_point_distance,
                    ),
                    [pointcloud[pointcloud["layer"] == layer] for layer in layers],
                ),
                total=len(layers),
                desc="Processing layers",
            )
        )

    grids, confidences = zip(*layer_collection)

    return np.stack(grids, axis=2), np.stack(confidences, axis=2)


def fixed_grid_to_vtk(  # pylint: disable=dangerous-default-value
    # varlabels is a read-only list literal, never mutated inside the function
    fixed_grid,
    filename,
    varlabels=["phase", "Eul1", "Eul2", "Eul3", "X", "Y", "Z"],
):
    """Save an nx x ny x nz x 7 fixed grid to a structured-grid VTK file.

    Args:
        fixed_grid (np.ndarray): grid [phase, Eul1, Eul2, Eul3, X, Y, Z]
        filename (str): output VTK filename
        varlabels (list): labels for the 7 columns
    """
    varray = vtkStructuredGrid()
    nx, ny, nz, _nvars = fixed_grid.shape
    if nz == 1:
        dz = 1.0
    else:
        dz = fixed_grid[0, 0, 1, 6] - fixed_grid[0, 0, 0, 6]

    if nx == 1:
        dx = 1.0
    else:
        dx = fixed_grid[1, 0, 0, 4] - fixed_grid[0, 0, 0, 4]

    if ny == 1:
        dy = 1.0
    else:
        dy = fixed_grid[0, 1, 0, 5] - fixed_grid[0, 0, 0, 5]

    x = np.linspace(
        fixed_grid[0, 0, 0, 4] - dx / 2, fixed_grid[-1, 0, 0, 4] + dx / 2, nx + 1
    )
    y = np.linspace(
        fixed_grid[0, 0, 0, 5] - dy / 2, fixed_grid[0, -1, 0, 5] + dy / 2, ny + 1
    )
    z = np.linspace(
        fixed_grid[0, 0, 0, 6] - dz / 2, fixed_grid[0, 0, -1, 6] + dz / 2, nz + 1
    )
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points_array = np.vstack(
        [X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")]
    ).T

    varray.SetDimensions(nx + 1, ny + 1, nz + 1)
    points = vtkPoints()

    vtk_points = numpy_support.numpy_to_vtk(
        points_array, deep=True, array_type=vtkDoubleArray().GetDataType()
    )
    points.SetData(vtk_points)
    varray.SetPoints(points)

    for ii, l in enumerate(varlabels):
        data = vtkDoubleArray()
        data.SetName(l)
        data.SetNumberOfComponents(1)
        data.SetNumberOfTuples(nx * ny * nz)

        array = numpy_support.numpy_to_vtk(
            fixed_grid[:, :, :, ii].ravel(order="F"),
            deep=True,
            array_type=vtkDoubleArray().GetDataType(),
        )
        array.SetName(l)

        varray.GetCellData().AddArray(array)

    writer = vtkStructuredGridWriter()
    writer.SetFileTypeToBinary()
    writer.SetFileName(filename)
    writer.SetInputData(varray)
    writer.Write()
