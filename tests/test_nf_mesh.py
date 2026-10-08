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

"""Tests for the NF Exodus writers' domain extents (graintrace.nf.mesh)."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.io as sio

from graintrace.nf import mesh

# nx, ny, nz voxels at these spacings, first voxel centered at ORIGIN
NX, NY, NZ = 5, 3, 2
DX, DY, DZ = 2.0, 4.0, 10.0
ORIGIN = (-7.0, 100.0, 0.5)


def _fixed_grid():
    """A fully populated (NX, NY, NZ, 7) fixed grid with voxel-center coords."""
    grid = np.zeros((NX, NY, NZ, 7), dtype=np.float64)
    grid[..., 0] = 1.0  # single grain, no void
    xs = ORIGIN[0] + DX * np.arange(NX)
    ys = ORIGIN[1] + DY * np.arange(NY)
    zs = ORIGIN[2] + DZ * np.arange(NZ)
    grid[..., 4] = xs[:, None, None]
    grid[..., 5] = ys[None, :, None]
    grid[..., 6] = zs[None, None, :]
    return grid


def _write_sculpt_like_exodus(path):
    """A minimal Exodus carrying only node coordinates.

    SCULPT meshes the ``.spn`` voxel occupancy grid, so its node coordinates sit
    on the voxel-face lattice ``0..n`` in index units -- which is exactly what
    ``rescale_exodus_mesh`` is given to map onto physical coordinates.
    """
    ii, jj, kk = np.meshgrid(
        np.arange(NX + 1, dtype=np.float64),
        np.arange(NY + 1, dtype=np.float64),
        np.arange(NZ + 1, dtype=np.float64),
        indexing="ij",
    )
    f = sio.netcdf_file(str(path), "w", version=2)
    f.createDimension("num_nodes", ii.size)
    for name, arr in (("coordx", ii), ("coordy", jj), ("coordz", kk)):
        v = f.createVariable(name, "d", ("num_nodes",))
        v[:] = arr.ravel()
    f.close()


def _bbox_extent(path):
    """(dx, dy, dz) extent of an Exodus file's node bounding box."""
    with sio.netcdf_file(str(path), "r") as f:
        return tuple(
            float(f.variables[c][:].max() - f.variables[c][:].min())
            for c in ("coordx", "coordy", "coordz")
        )


def _bbox_min(path):
    with sio.netcdf_file(str(path), "r") as f:
        return tuple(
            float(f.variables[c][:].min()) for c in ("coordx", "coordy", "coordz")
        )


class TestRescaleExodusMeshExtents:
    """Regression for issue #15: ``rescale_exodus_mesh`` mapped the SCULPT mesh
    onto the voxel-*center* min/max, so the meshed domain came out ``(n-1)*d``
    instead of ``n*d`` -- a silent ``(n-1)/n`` error in every gauge-length
    quantity. The sibling writer ``write_voxel_exodus`` in the same module has
    always used voxel faces; the two disagreed."""

    def test_extent_is_n_voxels_times_spacing(self, tmp_path):
        exo = tmp_path / "mesh.e"
        _write_sculpt_like_exodus(exo)
        mesh.rescale_exodus_mesh(str(exo), _fixed_grid())

        got = _bbox_extent(exo)
        want = (NX * DX, NY * DY, NZ * DZ)
        assert got == pytest.approx(want, rel=0, abs=1e-9)

        # and the lower face sits half a voxel below the first voxel center
        assert _bbox_min(exo) == pytest.approx(
            (ORIGIN[0] - DX / 2, ORIGIN[1] - DY / 2, ORIGIN[2] - DZ / 2), abs=1e-9
        )

    def test_old_center_to_center_extent_is_not_produced(self, tmp_path):
        """The pre-fix behaviour, stated explicitly so it cannot come back."""
        exo = tmp_path / "mesh.e"
        _write_sculpt_like_exodus(exo)
        mesh.rescale_exodus_mesh(str(exo), _fixed_grid())

        center_to_center = ((NX - 1) * DX, (NY - 1) * DY, (NZ - 1) * DZ)
        assert _bbox_extent(exo) != pytest.approx(center_to_center, abs=1e-9)

    def test_agrees_with_write_voxel_exodus(self, tmp_path):
        """The two Exodus writers in this module must bound the same domain."""
        pytest.importorskip("neml2")

        grid = _fixed_grid()
        rescaled = tmp_path / "rescaled.e"
        _write_sculpt_like_exodus(rescaled)
        mesh.rescale_exodus_mesh(str(rescaled), grid)

        direct = tmp_path / "direct.e"
        mesh.write_voxel_exodus(grid, str(direct), str(tmp_path / "ori"))

        assert _bbox_extent(rescaled) == pytest.approx(_bbox_extent(direct), abs=1e-6)
        assert _bbox_min(rescaled) == pytest.approx(_bbox_min(direct), abs=1e-6)


class TestVoxelSpacing:
    def test_reads_spacing_from_centers(self):
        assert mesh.voxel_spacing(_fixed_grid()) == pytest.approx((DX, DY, DZ))

    def test_single_voxel_axis_falls_back_to_unit_spacing(self):
        grid = _fixed_grid()[:, :, :1]  # one voxel along z
        assert mesh.voxel_spacing(grid) == pytest.approx((DX, DY, 1.0))
