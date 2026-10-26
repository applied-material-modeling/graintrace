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

import json
import os
import warnings
from pathlib import Path

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
# SCULPT post-processing: void flag, affine rescale, contiguous block ids
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parent.parent


def _physical_grid(labels, h=(10.0, 5.0, 2.0), origin=(100.0, 0.0, 1.0)):
    """A fixed grid over ``labels`` (nx, ny, nz) with anisotropic spacing.

    Grain ``g`` carries the Euler triple ``(g, 0, 0)`` so a mapped orientation
    row names its grain.
    """
    labels = np.asarray(labels, dtype=np.float64)
    grid = np.zeros((*labels.shape, 7))
    grid[..., 0] = labels
    grid[..., 1] = labels
    ijk = np.meshgrid(*[np.arange(n) for n in labels.shape], indexing="ij")
    for a in range(3):
        grid[..., 4 + a] = origin[a] + h[a] * ijk[a]
    return torch.from_numpy(grid)


def _index_grid(labels):
    """``labels`` on unit voxels centred at i + 0.5 (SCULPT's index space)."""
    labels = np.asarray(labels, dtype=np.float64)
    grid = np.zeros((*labels.shape, 7))
    grid[..., 0] = labels
    ijk = np.meshgrid(*[np.arange(n) for n in labels.shape], indexing="ij")
    for a in range(3):
        grid[..., 4 + a] = ijk[a] + 0.5
    return torch.from_numpy(grid)


def _flag(call, name):
    return call[call.index(name) + 1] if name in call else None


class FakeSculpt:
    """psculpt/epu stand-in that follows SCULPT's measured conventions.

    Measured with CUBIT psculpt on a void-bordered toy grid: material/block ids
    are the 1-based RANK of each spn value (void present: spn 0 -> 1, grain g ->
    g + 1; no void: g -> g); ``--void_mat v`` removes the material with id ``v``;
    coordinates are voxel-INDEX units, 0 at the first voxel edge. ``absorb``
    maps an spn value to the spn value whose block swallows it, standing in for
    SCULPT dropping a small grain (which leaves the block ids sparse).
    """

    def __init__(self, tmp_path, absorb=None):
        self.tmp_path = tmp_path
        self.absorb = absorb or {}
        self.calls = []

    def __call__(self, call, check, cwd, env):  # pylint: disable=unused-argument
        self.calls.append(list(call))
        if "-isp" not in call:
            return  # epu: nothing to concatenate in the stand-in
        shape = tuple(int(_flag(call, k)) for k in ("-x", "-y", "-z"))
        spn = np.loadtxt(_flag(call, "-isp")).astype(np.int64).reshape(shape)
        values = np.unique(spn)
        rank = {int(v): r + 1 for r, v in enumerate(values)}
        ids = np.vectorize(rank.get)(spn)
        for src, dst in self.absorb.items():
            ids[spn == src] = rank[dst]
        void_mat = _flag(call, "--void_mat")
        if void_mat is not None:
            ids[ids == int(void_mat)] = 0
        out = f"{cwd}/{_flag(call, '-e')}.e"
        mesh.write_voxel_exodus(_index_grid(ids), out, str(self.tmp_path / "fake_ori"))
        import scipy.io as sio  # pylint: disable=import-outside-toplevel

        kept = np.unique(ids[ids > 0]).astype(np.int32)
        with sio.netcdf_file(out, "a", mmap=False) as f:
            f.variables["eb_prop1"][:] = kept
            f.variables["eb_names"][:] = mesh._exodus_char_array(
                [f"block_{b}" for b in kept]
            )

    @property
    def psculpt_call(self):
        return next(c for c in self.calls if "-isp" in c)


CFG = {"launcher": "mpiexec", "psculpt": "psculpt", "epu": "epu", "nprocs": 2}


def _run_mesh_sculpt(monkeypatch, tmp_path, data, options=(), absorb=None, **kw):
    """write_spn + mesh_sculpt with SCULPT stubbed; returns (mesh, map, fake)."""
    fake = FakeSculpt(tmp_path, absorb)
    monkeypatch.setattr(mesh.subprocess, "run", fake)
    mesh.write_spn(
        data, str(tmp_path / "grid.spn"), str(tmp_path / "grid.ori"), symmetry="1"
    )
    out = tmp_path / "mesh.e"
    block_map = mesh.mesh_sculpt(
        CFG,
        list(options),
        str(tmp_path / "grid.spn"),
        data,
        str(out),
        str(tmp_path / "orientations"),
        **kw,
    )
    return out, block_map, fake


def _read_exodus(path):
    import scipy.io as sio  # pylint: disable=import-outside-toplevel

    with sio.netcdf_file(str(path), "r", mmap=False) as f:
        coords = np.stack(
            [f.variables[c][:].copy() for c in ("coordx", "coordy", "coordz")], 1
        )
        ids = f.variables["eb_prop1"][:].copy()
        names = mesh._decode_names(f.variables["eb_names"][:].copy())
    return coords, ids, names


def _bordered_labels():
    """Grains 2, 5, 7 (one voxel), 9 inside an asymmetric 2-voxel void border."""
    labels = np.zeros((10, 8, 3))
    labels[2:5, 2:5] = 2.0
    labels[5:8, 2:5] = 5.0
    labels[2:8, 5:6] = 9.0
    labels[7, 5, 2] = 7.0
    return labels


class TestSculptVoidOptions:
    def test_void_forces_void_mat_one(self):
        with pytest.warns(UserWarning, match="--void_mat 1"):
            out = mesh.sculpt_void_options(["-A", "7", "--void_mat", "0"], True)
        assert out == ["-A", "7", "--void_mat", "1"]

    @pytest.mark.parametrize("opts", [["-VM", "1"], ["--void_mat=1"], []])
    def test_void_mat_one_is_silent(self, opts):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert mesh.sculpt_void_options(opts, True) == ["--void_mat", "1"]

    def test_no_void_drops_the_flag(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert mesh.sculpt_void_options(["--void_mat", "0", "-S", "2"], False) == [
                "-S",
                "2",
            ]
        with pytest.warns(UserWarning, match="remove grain 1"):
            assert mesh.sculpt_void_options(["--void_mat", "1"], False) == []


def test_mesh_sculpt_removes_void_and_renumbers_sparse_ids(monkeypatch, tmp_path):
    """Void present + a grain SCULPT drops: the realistic NF case.

    write_spn renumbers grains 2, 5, 7, 9 to spn 1..4; SCULPT ranks spn 0..4 to
    materials 1..5, --void_mat 1 removes the void, and the one-voxel grain (spn
    3, material 4) is absorbed, so the blocks come back as 2, 3, 5.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    _stub_average_rotations(monkeypatch)
    data = _physical_grid(_bordered_labels())
    source = data[..., 0].clone().numpy()
    with pytest.warns(UserWarning, match="--void_mat 1"):
        out, block_map, fake = _run_mesh_sculpt(
            monkeypatch,
            tmp_path,
            data,
            options=["--void_mat", "0"],
            absorb={3: 4},
            source_labels=source,
        )
    assert _flag(fake.psculpt_call, "--void_mat") == "1"
    assert fake.psculpt_call.count("--void_mat") == 1

    _, ids, names = _read_exodus(out)
    assert ids.tolist() == [1, 2, 3]
    assert names == ["block_1", "block_2", "block_3"]

    bm = pd.read_csv(block_map)
    assert block_map == str(tmp_path / "mesh_block_map.csv")
    assert bm["block_id"].tolist() == [1, 2, 3]
    assert bm["sculpt_block_id"].tolist() == [2, 3, 5]
    assert bm["grain_id"].tolist() == [1, 2, 4]
    assert bm["source_grain_id"].tolist() == [2, 5, 9]
    assert bm["n_elem"].tolist() == [27, 27, 18]

    # row k of the orientations CSV is block k + 1; Eul1 encodes the source grain
    ori = np.loadtxt(tmp_path / "orientations.csv", delimiter=",")
    assert ori[:, 0].tolist() == [2.0, 5.0, 9.0]


def test_mesh_sculpt_without_void_passes_ids_through(monkeypatch, tmp_path):
    """No void: no flag is passed and SCULPT's ids already equal the spn ids."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    _stub_average_rotations(monkeypatch)
    labels = np.ones((4, 2, 2))
    labels[2:] = 2.0
    out, block_map, fake = _run_mesh_sculpt(
        monkeypatch, tmp_path, _physical_grid(labels), options=["--void_mat", "0"]
    )
    assert "--void_mat" not in fake.psculpt_call
    _, ids, _ = _read_exodus(out)
    assert ids.tolist() == [1, 2]
    bm = pd.read_csv(block_map)
    assert bm["sculpt_block_id"].tolist() == [1, 2]
    assert bm["grain_id"].tolist() == [1, 2]
    assert "source_grain_id" not in bm.columns


def test_a_meshed_void_is_an_error(monkeypatch, tmp_path):
    """If the void is meshed anyway (--void_mat 0 forced), say so, loudly."""
    _stub_average_rotations(monkeypatch)
    monkeypatch.setattr(mesh, "sculpt_void_options", lambda opts, has_void: opts)
    with pytest.raises(ValueError, match="void_mat 1"):
        _run_mesh_sculpt(
            monkeypatch,
            tmp_path,
            _physical_grid(_bordered_labels()),
            options=["--void_mat", "0"],
        )


def test_mesh_sculpt_maps_index_coordinates_exactly(monkeypatch, tmp_path):
    """The sample keeps its own voxel-edge box; it is not stretched to the grid's.

    The solid is i 2..7, j 2..5 of a 10 x 8 x 3 grid with spacing (10, 5, 2) and
    origin (100, 0, 1): edges x 115..175, y 7.5..27.5, z 0..6.
    """
    _stub_average_rotations(monkeypatch)
    data = _physical_grid(_bordered_labels())
    out, _, _ = _run_mesh_sculpt(monkeypatch, tmp_path, data)
    coords, _, _ = _read_exodus(out)
    np.testing.assert_allclose(coords.min(0), [115.0, 7.5, 0.0], atol=1e-4)
    np.testing.assert_allclose(coords.max(0), [175.0, 27.5, 6.0], atol=1e-4)


@pytest.mark.parametrize("layer_spacing, thickness", [(None, 7.5), (3.0, 3.0)])
def test_single_layer_mesh_keeps_a_thickness(
    monkeypatch, tmp_path, layer_spacing, thickness
):
    """A 1-layer grid must not collapse to coordz == 0 (puma-opt aborts on it).

    Without a known layer spacing the z thickness falls back to the mean
    in-plane spacing, (10 + 5) / 2.
    """
    _stub_average_rotations(monkeypatch)
    data = _physical_grid(np.ones((6, 4, 1)))
    out, _, _ = _run_mesh_sculpt(
        monkeypatch, tmp_path, data, layer_spacing=layer_spacing
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


def _real_sculpt_config():
    """sculpt_config from $GRAINTRACE_TOOLS_JSON or deploy/tools.json, if usable."""
    for path in (os.environ.get("GRAINTRACE_TOOLS_JSON"), REPO / "deploy/tools.json"):
        if path and Path(path).is_file():
            cfg = json.loads(Path(path).read_text(encoding="utf-8")).get(
                "sculpt_config"
            )
            if cfg and all(Path(cfg.get(k, "")).exists() for k in ("psculpt", "epu")):
                return cfg
    return None


@pytest.mark.skipif(
    _real_sculpt_config() is None,
    reason="needs CUBIT psculpt: set GRAINTRACE_TOOLS_JSON or deploy/tools.json",
)
def test_real_sculpt_void_bordered_toy(monkeypatch, tmp_path):
    """psculpt on a void-bordered asymmetric toy: void removed, ids and box right.

    Solid i 3..16, j 3..11 of a 20 x 20 x 10 grid (h = 4, 4, 5 um): voxel edges
    x 10..66, y 10..46, z -2.5..47.5. SCULPT smoothing moves the boundary by
    under ~0.1 voxel.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    _stub_average_rotations(monkeypatch)
    n = (20, 20, 10)
    i, j, _ = np.meshgrid(*[np.arange(k) for k in n], indexing="ij")
    inner = (i >= 3) & (i < 17) & (j >= 3) & (j < 12)
    labels = np.where(inner, 1 + (i >= 10) + 2 * (j >= 7), 0)
    data = _physical_grid(labels, h=(4.0, 4.0, 5.0), origin=(0.0, 0.0, 0.0))
    mesh.write_spn(data, str(tmp_path / "t.spn"), str(tmp_path / "t.ori"))
    monkeypatch.chdir(tmp_path)
    block_map = mesh.mesh_sculpt(
        _real_sculpt_config(),
        ["--void_mat", "1"],
        str(tmp_path / "t.spn"),
        data,
        str(tmp_path / "t.e"),
        str(tmp_path / "ori"),
        layer_spacing=5.0,
    )
    coords, ids, _ = _read_exodus(tmp_path / "t.e")
    assert ids.tolist() == [1, 2, 3, 4]
    assert pd.read_csv(block_map)["grain_id"].tolist() == [1, 2, 3, 4]
    tol = 0.15 * np.array([4.0, 4.0, 5.0])
    lo, hi = coords.min(0), coords.max(0)
    assert (np.abs(lo - [10.0, 10.0, -2.5]) <= tol).all(), lo
    assert (np.abs(hi - [66.0, 46.0, 47.5]) <= tol).all(), hi
