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

"""Tests for the label bookkeeping of the NF mesh writers.

A segmented grid can legitimately carry a negative ("unsegmented") phase:
:func:`graintrace.nf.segment.remove_small_segments` reassigns a sub-threshold
segment to ``-1`` when it has no kept neighbour, so a builder that runs the
infill *before* the removal (``NearFieldMeshBuilder.reconstruct``) hands those
orphans straight to the mesh writers. These tests pin down what the writers do
with such a grid.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graintrace.nf import mesh, metrics


def _stub_average_rotations(monkeypatch):
    """Replace the neml2-backed per-grain average with a pure-torch stand-in.

    The writers consume ``average_rotations`` only to fill the per-grain
    orientation rows; these tests are about label bookkeeping, not orientation
    math, and the real metric imports neml2, which is not part of the base
    install. Returning the mean Euler triple keeps the tests runnable on a
    plain checkout (the same stand-in approach as ``tests/test_nf_segment.py``).
    """
    monkeypatch.setattr(
        metrics,
        "average_rotations",
        lambda angles, **kwargs: (
            torch.as_tensor(angles, dtype=torch.float64).mean(dim=0),
            torch.as_tensor(angles, dtype=torch.float64).mean(dim=0),
        ),
    )


def _grid_from_labels(labels):
    """Build a minimal (n,1,1,7) fixed grid carrying ``labels`` as the phase."""
    n = len(labels)
    grid = torch.zeros(n, 1, 1, 7, dtype=torch.float64)
    grid[..., 0] = torch.tensor(labels, dtype=torch.float64).reshape(n, 1, 1)
    grid[..., 4] = torch.arange(n, dtype=torch.float64).reshape(n, 1, 1)
    return grid


def test_write_spn_raises_on_unsegmented_voxels(monkeypatch, tmp_path):
    """A leftover ``-1`` must stop the mesh, loudly, naming the remedy.

    Historically this input was written out silently, and catastrophically: the
    in-place renumber turned the phase column ``[0, -1, 1, 1, 2, 2, 3, 3]`` into
    the spn ``[0, 4, 4, 4, 4, 4, 4, 4]`` -- every grain in the microstructure
    collapsed into one SCULPT id. Mapping the orphans to void instead would
    silently delete material from the mesh, so the caller has to decide.
    """
    _stub_average_rotations(monkeypatch)
    grid = _grid_from_labels([0, -1, 1, 1, 2, 2, 3, 3])

    with pytest.raises(ValueError) as excinfo:
        mesh.write_spn(
            grid,
            str(tmp_path / "grid.spn"),
            str(tmp_path / "orientations.dat"),
            angle_convention="bunge",
            angle_type="degrees",
        )

    message = str(excinfo.value)
    assert "infill_nearest_neighbor" in message
    assert "remove_small_segments" in message
    assert not (tmp_path / "grid.spn").exists()


def test_write_spn_renumber_is_collision_proof(monkeypatch, tmp_path):
    """Renumbering must not fold two grains together for *any* label set.

    The in-place renumber was only safe for a contiguous ``1..N`` phase column:
    writing ``i + 1`` back into the column being scanned lets an already
    relabelled grain be picked up again by a later iteration. A label below its
    own rank is enough to trigger it -- here ``0.5`` is relabelled to ``1`` and
    is then swept up by the pass that handles the real grain ``1``.
    """
    _stub_average_rotations(monkeypatch)
    grid = _grid_from_labels([0, 0.5, 0.5, 1, 1, 2, 2])

    spn = tmp_path / "grid.spn"
    mesh.write_spn(
        grid,
        str(spn),
        str(tmp_path / "orientations.dat"),
        angle_convention="bunge",
        angle_type="degrees",
    )

    written = np.loadtxt(spn).astype(np.int64)
    grains = np.unique(written[written != 0])
    assert grains.tolist() == [1, 2, 3]
    assert np.count_nonzero(written == 0) == 1


def test_write_voxel_exodus_skips_unsegmented_voxels(monkeypatch, tmp_path):
    """An orphan voxel must not become a block and shift every grain id.

    Block ids are relabelled to a contiguous ``1..N`` over the ids present, so a
    ``-1`` voxel sorts first and takes block 1, pushing every real grain one
    block along and mismatching the per-block orientation rows.
    """
    _stub_average_rotations(monkeypatch)
    grid = _grid_from_labels([0, -1, 1, 1, 2, 2])

    info = mesh.write_voxel_exodus(
        grid,
        str(tmp_path / "mesh.e"),
        str(tmp_path / "orientations"),
        angle_convention="bunge",
        angle_type="degrees",
    )

    assert info["blocks"] == 2
    # two voxels per grain, and neither the void nor the orphan is meshed
    assert info["elements"] == 4
    assert np.loadtxt(tmp_path / "orientations.csv", delimiter=",").shape == (2, 3)
