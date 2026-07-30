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


def nf_to_pointcloud(folder, dz, layer_token="layer"):
    """Convert nearfield data (.mic/.csv layers) to a stacked point cloud.

    Args:
        folder (str): folder with nearfield data
        dz (float): spacing between layers in the z-direction
    """

    mic_files = glob.glob(f"{folder}/*.mic")
    csv_files = glob.glob(f"{folder}/*.csv")

    if mic_files:
        files = sorted(mic_files, key=lambda f: layer_number(f, token=layer_token))
    elif csv_files:
        files = sorted(csv_files, key=lambda f: layer_number(f, token=layer_token))
    else:
        raise FileNotFoundError("No .mic or .csv files found in the specified folder")

    layers = [layer_number(f, token=layer_token) for f in files]
    min_layer = min(layers)
    max_layer = max(layers)
    if layers != list(range(min_layer, max_layer + 1)):
        raise ValueError(f"Missing layers in nearfield data. Found: {layers[:10]}...")

    dfs = []
    print("\n")
    for i, f in tqdm.tqdm(enumerate(files), total=len(files), desc="Reading layers"):
        df = pd.read_csv(f, skiprows=3, sep="\\s+", header=0)
        df = df.rename(columns={"%OrientationRowNr": "OrientationRowNr"})
        df["Z"] = i * dz
        df["layer"] = i

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def process_layer_pointcloud(df_layer, plane_points, nx, ny):
    """Interpolate one point-cloud layer onto a fixed nx x ny grid.

    Returns:
        np.ndarray: nx x ny x 7 array [phase, Eul1, Eul2, Eul3, X, Y, Z]
    """
    z = df_layer["Z"].iloc[0]
    points = df_layer[["X", "Y"]].to_numpy()

    tri = sp.Delaunay(points)
    simplex_indices = tri.find_simplex(plane_points)

    def min_ind(p, simp):
        vertices = tri.simplices[simp]
        vert_points = tri.points[vertices]
        dists = np.linalg.norm(vert_points - p[:, None], axis=-1)
        which = np.argmin(dists, axis=-1)
        return [vertices[i, which[i]] for i in range(len(simp))]

    found = np.where(simplex_indices >= 0)[0]

    data = np.zeros((nx * ny, 7))
    idx = min_ind(plane_points[found], simplex_indices[found])

    data[found, 0] = 1
    data[found, 1:4] = df_layer.iloc[idx][["Eul1", "Eul2", "Eul3"]].to_numpy()

    data[:, 4:6] = plane_points
    data[:, 6] = z

    return data.reshape((nx, ny, 7))


def pointcloud_to_fixed_grid(pointcloud, nx, ny):
    """Convert a point cloud to an nx x ny x nz x 7 fixed grid.

    Args:
        pointcloud (pd.DataFrame): columns [phase, Eul1, Eul2, Eul3, X, Y, Z, layer]
        nx, ny (int): in-plane grid resolution

    Returns:
        np.ndarray: nx x ny x nz x 7 grid [phase, Eul1, Eul2, Eul3, X, Y, Z]
    """
    layers = pointcloud["layer"].unique()
    x = np.linspace(
        pointcloud["X"].min(),
        pointcloud["X"].max(),
        nx,
    )
    y = np.linspace(
        pointcloud["Y"].min(),
        pointcloud["Y"].max(),
        ny,
    )
    X, Y = np.meshgrid(x, y, indexing="ij")
    plane_points = np.vstack([X.ravel(), Y.ravel()]).T

    tp = multiprocess.Pool()

    layer_collection = list(
        tqdm.tqdm(
            tp.imap(
                lambda df_layer: process_layer_pointcloud(
                    df_layer, plane_points, nx, ny
                ),
                [pointcloud[pointcloud["layer"] == layer] for layer in layers],
            ),
            total=len(layers),
            desc="Processing layers",
        )
    )

    return np.stack(layer_collection, axis=2)


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
