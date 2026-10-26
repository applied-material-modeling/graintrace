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


# --------------------------------------------------------------------------- #
# SCULPT post-processing: voxel-edge rescale + contiguous block renumbering
# --------------------------------------------------------------------------- #


def _physical_grid(labels):
    """A fixed grid over ``labels`` (nx, ny, nz) with anisotropic spacing.

    Centres sit at x = 100 + 10 i, y = 5 j, z = 1 + 2 k, and grain ``g`` carries
    the Euler triple ``(g, 0, 0)`` so a mapped orientation row names its grain.
    """
    labels = np.asarray(labels, dtype=np.float64)
    nx, ny, nz = labels.shape
    grid = np.zeros((nx, ny, nz, 7))
    grid[..., 0] = labels
    grid[..., 1] = labels
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    grid[..., 4] = 100.0 + 10.0 * i
    grid[..., 5] = 5.0 * j
    grid[..., 6] = 1.0 + 2.0 * k
    return torch.from_numpy(grid)


def _index_grid(labels):
    """Same labels on unit voxels at i + 0.5, i.e. SCULPT's index-space output."""
    labels = np.asarray(labels, dtype=np.float64)
    grid = np.zeros((*labels.shape, 7))
    grid[..., 0] = labels
    ijk = np.meshgrid(*[np.arange(n) for n in labels.shape], indexing="ij")
    for a in range(3):
        grid[..., 4 + a] = ijk[a] + 0.5
    return torch.from_numpy(grid)


def _write_fake_sculpt_output(path, sculpt_labels, block_ids, tmp_path):
    """An Exodus mesh shaped like SCULPT's: index-space coords, given block ids."""
    import scipy.io as sio  # pylint: disable=import-outside-toplevel

    mesh.write_voxel_exodus(
        _index_grid(sculpt_labels), str(path), str(tmp_path / "unused_ori")
    )
    with sio.netcdf_file(str(path), "a", mmap=False) as f:
        f.variables["eb_prop1"][:] = np.asarray(block_ids, dtype=np.int32)
        f.variables["eb_names"][:] = mesh._exodus_char_array(
            [f"block_{b}" for b in block_ids]
        )


def _run_mesh_sculpt(monkeypatch, tmp_path, data, fake, **kwargs):
    """Run mesh_sculpt with SCULPT/epu stubbed to drop ``fake`` in place."""
    import shutil  # pylint: disable=import-outside-toplevel

    def fake_run(call, check, cwd, env):  # pylint: disable=unused-argument
        if call[0] == "mpiexec":
            shutil.copy(fake, f"{cwd}/output_mesh.e")

    monkeypatch.setattr(mesh.subprocess, "run", fake_run)
    out = tmp_path / "mesh.e"
    block_map = mesh.mesh_sculpt(
        {"launcher": "mpiexec", "psculpt": "psculpt", "epu": "epu", "nprocs": 2},
        [],
        str(tmp_path / "grid.spn"),
        data,
        str(out),
        str(tmp_path / "orientations"),
        **kwargs,
    )
    return out, block_map


def _read_exodus(path):
    import scipy.io as sio  # pylint: disable=import-outside-toplevel

    with sio.netcdf_file(str(path), "r", mmap=False) as f:
        coords = np.stack(
            [f.variables[c][:].copy() for c in ("coordx", "coordy", "coordz")], 1
        )
        ids = f.variables["eb_prop1"][:].copy()
        names = mesh._decode_names(f.variables["eb_names"][:].copy())
    return coords, ids, names


def test_mesh_sculpt_renumbers_sparse_block_ids(monkeypatch, tmp_path):
    """SCULPT dropped a grain, so eb_prop1 is sparse; the CPFE decks need 1..N.

    The grid has grains 2, 5, 7, 9 (merged-grid ids); write_spn renumbers them to
    spn ids 1, 2, 3, 4, and the one-voxel grain 7 (spn 3) is lost by SCULPT, so
    the mesh comes back with blocks 1, 2, 4.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    _stub_average_rotations(monkeypatch)
    labels = np.full((6, 4, 3), 9.0)
    labels[:2] = 2.0
    labels[2:4] = 5.0
    labels[5, 3, 2] = 7.0
    data = _physical_grid(labels)
    source = data[..., 0].clone().numpy()
    mesh.write_spn(
        data, str(tmp_path / "grid.spn"), str(tmp_path / "grid.ori"), symmetry="1"
    )

    sculpt_labels = data[..., 0].numpy().copy()
    sculpt_labels[sculpt_labels == 3] = 4  # SCULPT absorbs the tiny grain
    fake = tmp_path / "fake_sculpt.e"
    _write_fake_sculpt_output(fake, sculpt_labels, [1, 2, 4], tmp_path)

    out, block_map = _run_mesh_sculpt(
        monkeypatch, tmp_path, data, fake, source_labels=source
    )

    _, ids, names = _read_exodus(out)
    assert ids.tolist() == [1, 2, 3]
    assert names == ["block_1", "block_2", "block_3"]

    bm = pd.read_csv(block_map)
    assert block_map == str(tmp_path / "mesh_block_map.csv")
    assert bm["block_id"].tolist() == [1, 2, 3]
    assert bm["sculpt_block_id"].tolist() == [1, 2, 4]
    assert bm["grain_id"].tolist() == [1, 2, 4]
    assert bm["source_grain_id"].tolist() == [2, 5, 9]
    assert bm["n_elem"].tolist() == [24, 24, 24]

    # row k of the orientations CSV is block k + 1; Eul1 encodes the source grain
    ori = np.loadtxt(tmp_path / "orientations.csv", delimiter=",")
    assert ori[:, 0].tolist() == [2.0, 5.0, 9.0]


def test_mesh_sculpt_leaves_contiguous_ids_alone(monkeypatch, tmp_path):
    """Already-contiguous block ids pass through, and the block map is written."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    _stub_average_rotations(monkeypatch)
    labels = np.ones((4, 2, 2))
    labels[2:] = 2.0
    data = _physical_grid(labels)
    fake = tmp_path / "fake_sculpt.e"
    _write_fake_sculpt_output(fake, labels, [1, 2], tmp_path)

    out, block_map = _run_mesh_sculpt(monkeypatch, tmp_path, data, fake)
    _, ids, _ = _read_exodus(out)
    assert ids.tolist() == [1, 2]
    bm = pd.read_csv(block_map)
    assert bm["grain_id"].tolist() == [1, 2]
    assert "source_grain_id" not in bm.columns


def test_mesh_sculpt_rescales_to_voxel_edges(monkeypatch, tmp_path):
    """The mesh spans the voxel-EDGE box, half a voxel beyond the centres."""
    _stub_average_rotations(monkeypatch)
    labels = np.ones((6, 4, 3))
    data = _physical_grid(labels)
    fake = tmp_path / "fake_sculpt.e"
    _write_fake_sculpt_output(fake, labels, [1], tmp_path)

    out, _ = _run_mesh_sculpt(monkeypatch, tmp_path, data, fake)
    coords, _, _ = _read_exodus(out)
    np.testing.assert_allclose(coords.min(0), [95.0, -2.5, 0.0], atol=1e-4)
    np.testing.assert_allclose(coords.max(0), [155.0, 17.5, 6.0], atol=1e-4)

    lo, hi = mesh.voxel_edge_bounds(data.numpy())
    np.testing.assert_allclose(coords.min(0), lo, atol=1e-4)
    np.testing.assert_allclose(coords.max(0), hi, atol=1e-4)


@pytest.mark.parametrize("layer_spacing, thickness", [(None, 7.5), (3.0, 3.0)])
def test_single_layer_mesh_keeps_a_thickness(
    monkeypatch, tmp_path, layer_spacing, thickness
):
    """A 1-layer grid must not collapse to coordz == 0 (puma-opt aborts on it).

    Without a known layer spacing the z thickness falls back to the mean
    in-plane spacing, (10 + 5) / 2.
    """
    _stub_average_rotations(monkeypatch)
    labels = np.ones((6, 4, 1))
    data = _physical_grid(labels)
    fake = tmp_path / "fake_sculpt.e"
    _write_fake_sculpt_output(fake, labels, [1], tmp_path)

    out, _ = _run_mesh_sculpt(
        monkeypatch, tmp_path, data, fake, layer_spacing=layer_spacing
    )
    coords, _, _ = _read_exodus(out)
    np.testing.assert_allclose(
        [coords[:, 2].min(), coords[:, 2].max()],
        [1.0 - thickness / 2, 1.0 + thickness / 2],
        atol=1e-4,
    )


@pytest.mark.parametrize("nz", [1, 3])
def test_voxel_exodus_agrees_with_the_sculpt_extents(monkeypatch, tmp_path, nz):
    """Both meshers span the same voxel-edge box, single-layer included."""
    _stub_average_rotations(monkeypatch)
    data = _physical_grid(np.ones((6, 4, nz)))
    path = tmp_path / "voxel.e"
    mesh.write_voxel_exodus(data, str(path), str(tmp_path / "ori"))
    coords, _, _ = _read_exodus(path)
    lo, hi = mesh.voxel_edge_bounds(data.numpy())
    np.testing.assert_allclose(coords.min(0), lo, atol=1e-4)
    np.testing.assert_allclose(coords.max(0), hi, atol=1e-4)
    assert hi[2] - lo[2] > 0
