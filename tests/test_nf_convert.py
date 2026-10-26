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

"""Tests for NF confidence masking, distance masking and voxel sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.spatial as sp
import torch

from graintrace.nf import convert, metrics, segment


def _legacy_process_layer(df_layer, plane_points, nx, ny):
    """The pre-confidence gridding, frozen as a backward-compatibility reference.

    Every grid point inside the layer's Delaunay hull becomes solid and takes the
    Euler angles of the nearest vertex of the simplex containing it.
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


def _plane_points(x, y):
    """Grid coordinates for the two axes, in the layout the converter uses."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.vstack([X.ravel(), Y.ravel()]).T


def _stripe_layer():
    """One layer: two grains split by a low-confidence stripe.

    Measured points sit on the integer grid x = 0..9, y = 0..2. Orientation is
    0 for x < 4, 45 across the stripe at x in {4, 5}, and 90 for x > 5, so a
    flood at a loose tolerance can hop grain -> stripe -> grain. Only the stripe
    points carry a low confidence.
    """
    xs, ys = np.meshgrid(np.arange(10.0), np.arange(3.0), indexing="ij")
    xs = xs.ravel()
    ys = ys.ravel()

    eul1 = np.where(xs < 4, 0.0, np.where(xs > 5, 90.0, 45.0))
    stripe = (xs >= 4) & (xs <= 5)

    return pd.DataFrame(
        {
            "X": xs,
            "Y": ys,
            "Z": np.zeros_like(xs),
            "Eul1": eul1,
            "Eul2": np.zeros_like(xs),
            "Eul3": np.zeros_like(xs),
            "Confidence": np.where(stripe, 0.2, 0.95),
        }
    )


def _stripe_plane_points():
    return _plane_points(np.arange(10.0), np.arange(3.0))


def _segment_counts(grid_plane, misorientation_tol=50.0):
    """Flood one gridded plane and return the number of segments found.

    ``flood``'s misorientation is neml2-backed; these tests are about which
    voxels are solid, not the orientation math, so the first Euler component's
    absolute difference stands in for it, as in ``test_nf_segment``.
    """
    grid = torch.from_numpy(grid_plane).unsqueeze(2)

    original = metrics.misorientation
    metrics.misorientation = lambda e1, e2, **kwargs: torch.abs(e1[:, 0] - e2[:, 0])
    try:
        phase = segment.flood(
            grid[..., 1:4],
            grid[..., 0],
            misorientation_tol,
            connectivity=6,
            grain_threshold=1,
            stop_count=1,
        )
    finally:
        metrics.misorientation = original

    return phase


class TestConfidenceMask:
    """``min_confidence`` voids low-confidence grid points."""

    def test_default_reproduces_the_legacy_grid(self):
        """All-default arguments must reproduce the pre-confidence grid exactly."""
        rng = np.random.default_rng(42)
        n = 60
        df = pd.DataFrame(
            {
                "X": rng.uniform(0.0, 10.0, n),
                "Y": rng.uniform(0.0, 10.0, n),
                "Z": np.full(n, 3.0),
                "Eul1": rng.uniform(0.0, 360.0, n),
                "Eul2": rng.uniform(0.0, 180.0, n),
                "Eul3": rng.uniform(0.0, 360.0, n),
                "Confidence": rng.uniform(0.0, 1.0, n),
            }
        )
        pts = _plane_points(np.linspace(0.0, 10.0, 12), np.linspace(0.0, 10.0, 13))

        new, _conf = convert.process_layer_pointcloud(df, pts, 12, 13)
        legacy = _legacy_process_layer(df, pts, 12, 13)

        assert np.array_equal(new, legacy)

    def test_low_confidence_points_become_void(self):
        """Grid points below the threshold lose both their phase and their angles."""
        df = _stripe_layer()
        pts = _stripe_plane_points()

        grid, conf = convert.process_layer_pointcloud(
            df, pts, 10, 3, min_confidence=0.5
        )

        stripe = slice(4, 6)
        assert np.all(grid[stripe, :, 0] == 0)
        assert np.all(grid[stripe, :, 1:4] == 0)
        assert np.all(conf[stripe, :] == 0.0)

        # Everything else is untouched and keeps its measured confidence
        assert np.all(grid[:4, :, 0] == 1)
        assert np.all(grid[6:, :, 0] == 1)
        assert np.all(conf[:4, :] == pytest.approx(0.95))

    def test_confidence_is_kept_without_a_threshold(self):
        """With no threshold the sidecar still reports the measured confidence."""
        grid, conf = convert.process_layer_pointcloud(
            _stripe_layer(), _stripe_plane_points(), 10, 3
        )

        assert np.all(grid[..., 0] == 1)
        assert np.all(conf[4:6, :] == pytest.approx(0.2))

    def test_a_layer_without_confidence_is_all_ones(self):
        """Layers with no Confidence column keep every point and report 1.0."""
        df = _stripe_layer().drop(columns=["Confidence"])

        grid, conf = convert.process_layer_pointcloud(
            df, _stripe_plane_points(), 10, 3, min_confidence=0.5
        )

        assert np.all(grid[..., 0] == 1)
        assert np.all(conf == 1.0)


class TestConfidenceMaskStopsPercolation:
    """The low-confidence stripe is the path segmentation leaks through."""

    def test_flood_bridges_the_grains_without_the_mask(self):
        grid, _conf = convert.process_layer_pointcloud(
            _stripe_layer(), _stripe_plane_points(), 10, 3
        )
        phase = _segment_counts(grid)

        assert torch.unique(phase[phase > 0]).numel() == 1

    def test_the_mask_keeps_the_grains_apart(self):
        grid, _conf = convert.process_layer_pointcloud(
            _stripe_layer(), _stripe_plane_points(), 10, 3, min_confidence=0.5
        )
        phase = _segment_counts(grid)

        assert torch.unique(phase[phase > 0]).numel() == 2
        # The stripe stays void; flood never seeds or floods into phase 0
        assert torch.all(phase[4:6, :, :] == 0)


class TestMaxPointDistance:
    """``max_point_distance`` replaces the hull test, which fills concave gaps."""

    def test_the_concave_notch_is_voided(self):
        """A C-shaped outline: the hull fills the notch, the radius does not."""
        xs, ys = np.meshgrid(np.arange(9.0), np.arange(9.0), indexing="ij")
        xs = xs.ravel()
        ys = ys.ravel()

        # Carve a notch out of the right-hand side of the square
        notch = (xs >= 4) & (ys >= 3) & (ys <= 5)
        xs, ys = xs[~notch], ys[~notch]

        df = pd.DataFrame(
            {
                "X": xs,
                "Y": ys,
                "Z": np.zeros_like(xs),
                "Eul1": np.zeros_like(xs),
                "Eul2": np.zeros_like(xs),
                "Eul3": np.zeros_like(xs),
            }
        )
        pts = _plane_points(np.arange(9.0), np.arange(9.0))

        hull, _c = convert.process_layer_pointcloud(df, pts, 9, 9)
        radius, _c2 = convert.process_layer_pointcloud(
            df, pts, 9, 9, max_point_distance=1.0
        )

        # The notch interior is inside the hull, so the legacy test fills it
        assert np.all(hull[6:9, 4, 0] == 1)
        # It is more than one voxel from any measured point, so the radius voids it
        assert np.all(radius[6:9, 4, 0] == 0)
        # Measured points themselves stay solid under both
        assert radius[0, 0, 0] == 1
        assert radius[8, 0, 0] == 1


class TestGridAxis:
    """``voxel_size_um`` is an alternative to a fixed grid count."""

    def test_spacing_is_honored_and_the_footprint_is_covered(self):
        axis = convert.grid_axis(2.0, 10.5, spacing=2.0)

        assert np.allclose(np.diff(axis), 2.0)
        assert axis[0] == pytest.approx(2.0)
        assert axis[-1] >= 10.5

    def test_an_exact_span_is_not_padded(self):
        axis = convert.grid_axis(0.0, 10.0, spacing=2.0)

        assert len(axis) == 6
        assert axis[-1] == pytest.approx(10.0)

    def test_a_count_spreads_evenly_over_the_extent(self):
        axis = convert.grid_axis(0.0, 10.0, n=5)

        assert np.allclose(axis, np.linspace(0.0, 10.0, 5))

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"n": 5, "spacing": 2.0}],
    )
    def test_exactly_one_of_count_or_spacing_is_required(self, kwargs):
        with pytest.raises(ValueError, match="exactly one"):
            convert.grid_axis(0.0, 10.0, **kwargs)

    def test_a_nonpositive_spacing_is_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            convert.grid_axis(0.0, 10.0, spacing=0.0)


class TestPointcloudToFixedGrid:
    """End-to-end gridding of a stacked point cloud."""

    @staticmethod
    def _two_layer_cloud():
        layers = []
        for i in range(2):
            df = _stripe_layer()
            df["Z"] = float(i) * 5.0
            df["layer"] = i
            layers.append(df)
        return pd.concat(layers, ignore_index=True)

    def test_the_confidence_sidecar_matches_the_grid(self):
        grid, conf = convert.pointcloud_to_fixed_grid(
            self._two_layer_cloud(), nx=10, ny=3, min_confidence=0.5
        )

        assert grid.shape == (10, 3, 2, 7)
        assert conf.shape == (10, 3, 2)
        # Void exactly where the grid is void, measured confidence elsewhere
        assert np.array_equal(conf > 0.0, grid[..., 0] > 0.0)
        assert np.all(conf[grid[..., 0] > 0.0] == pytest.approx(0.95))

    def test_voxel_size_sets_the_spacing(self):
        grid, conf = convert.pointcloud_to_fixed_grid(
            self._two_layer_cloud(), voxel_size_um=(0.5, 1.0)
        )

        # x spans 0..9 at 0.5 um, y spans 0..2 at 1.0 um
        assert grid.shape == (19, 3, 2, 7)
        assert conf.shape == (19, 3, 2)
        assert grid[1, 0, 0, 4] - grid[0, 0, 0, 4] == pytest.approx(0.5)
        assert grid[0, 1, 0, 5] - grid[0, 0, 0, 5] == pytest.approx(1.0)

    def test_counts_and_voxel_size_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            convert.pointcloud_to_fixed_grid(
                self._two_layer_cloud(), nx=10, ny=3, voxel_size_um=(1.0, 1.0)
            )

    def test_a_grid_size_is_required(self):
        with pytest.raises(ValueError, match="voxel_size_um instead"):
            convert.pointcloud_to_fixed_grid(self._two_layer_cloud())
