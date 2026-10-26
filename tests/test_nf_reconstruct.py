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

"""End-to-end NF reconstruction: island voiding, layouts, graph segmentation."""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

# pylint: disable=wrong-import-position  # must follow the torch importorskip
from graintrace.construct_nf_mesh import NearFieldMeshBuilder  # noqa: E402
from graintrace.nf import convert, mesh, segment  # noqa: E402
from graintrace.nf_grid_conversion import NFGridConversion  # noqa: E402

# pylint: enable=wrong-import-position

requires_neml2 = pytest.mark.skipif(
    importlib.util.find_spec("neml2") is None, reason="requires neml2"
)
requires_networkit = pytest.mark.skipif(
    importlib.util.find_spec("networkit") is None, reason="requires networkit"
)

MIC_HEADER = (
    "%header1\n%header2\n%header3\n"
    "%OrientationRowNr OrientationID RunTime X Y TriEdgeSize UpDown "
    "Eul1 Eul2 Eul3 Confidence PhaseNr\n"
)


def _write_mic(path, xy, eul, conf):
    """Write one MIDAS-style .mic layer."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(MIC_HEADER)
        for i, (p, e, c) in enumerate(zip(xy, eul, conf)):
            fh.write(
                f"{i} {i} 0 {p[0]:.4f} {p[1]:.4f} 5 1 "
                f"{e[0]:.6f} {e[1]:.6f} {e[2]:.6f} {c:.3f} 1\n"
            )


def _island_layers(folder, n_layers=3):
    """An L-shaped two-grain sample with one stray confident point in the notch.

    Points in the top-right quadrant are air (dropped). A low-confidence stripe
    at x = 100 splits the grains; one Confidence 0.99 point sits alone at
    (180, 180), deep in the notch, so it grids to a small island in void.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for layer in range(n_layers):
        xy = rng.uniform(0, 200, (400, 2))
        xy = xy[~((xy[:, 0] > 120) & (xy[:, 1] > 120))]
        grain = (xy[:, 0] > 100)[:, None]
        eul = np.where(grain, [0.3, 0.4, 0.5], [1.0, 0.2, 0.1])
        eul = eul + rng.normal(0, 0.002, eul.shape)
        conf = np.where(np.abs(xy[:, 0] - 100) < 8, 0.2, 0.95)
        xy = np.vstack([xy, [[180.0, 180.0]]])
        eul = np.vstack([eul, [[0.3, 0.4, 0.5]]])
        conf = np.append(conf, 0.99)
        _write_mic(folder / f"sample_layer{layer}.mic", xy, eul, conf)


def _builder(inp, out, **kwargs):
    return NearFieldMeshBuilder(
        input_folder=inp,
        save_dir=out,
        exp_file_token="layer",
        angle_type="radians",
        symmetry="432",
        write_intermediate=False,
        write_vtk=False,
        **kwargs,
    )


FLOOD = {
    "misorientation_tol": 5.0 / 180.0 * np.pi,
    "connectivity": 6,
    "batch_norm": 200_000,
    "grain_threshold": 20,
    "stop_count": 50,
    "grain_threshold_final": 50,
}


class TestIsolatedIslands:
    """A solid island with no grain to join must end up void, not -1."""

    def test_void_unsegmented_zeroes_leftovers(self):
        grid = torch.zeros((3, 1, 1, 7))
        grid[:, 0, 0, 0] = torch.tensor([1.0, -1.0, 0.0])
        grid[:, 0, 0, 1:4] = 0.5
        out = segment.void_unsegmented(grid)
        assert out[:, 0, 0, 0].tolist() == [1.0, 0.0, 0.0]
        assert out[1, 0, 0, 1:4].tolist() == [0.0, 0.0, 0.0]
        assert out[0, 0, 0, 1:4].tolist() == [0.5, 0.5, 0.5]
        assert grid[1, 0, 0, 0] == -1  # input untouched

    @requires_neml2
    def test_reconstruct_then_write_spn(self, tmp_path):
        """The reported repro: 27 island voxels used to reach write_spn at -1."""
        torch.manual_seed(42)
        inp = tmp_path / "mic"
        _island_layers(inp)
        builder = _builder(inp, tmp_path / "out")
        path = builder.reconstruct(
            dz=5.0,
            nx=40,
            ny=50,
            min_confidence=0.5,
            max_point_distance=7.5,
            segmentation=FLOOD,
        )
        grid = np.load(path)
        assert (grid[..., 0] >= 0).all()
        # the island in the notch is void
        near = np.hypot(grid[..., 4] - 180.0, grid[..., 5] - 180.0) < 7.5
        assert near.any() and (grid[near, 0] == 0).all()

        mesh.write_spn(
            torch.from_numpy(grid),
            str(tmp_path / "x.spn"),
            str(tmp_path / "x.ori"),
            symmetry="432",
        )
        assert (tmp_path / "x.spn").exists()


class TestArgumentChecks:
    def test_voxel_size_and_counts_conflict(self, tmp_path):
        builder = _builder(tmp_path / "missing", tmp_path / "out")
        with pytest.raises(ValueError, match="not both"):
            builder.reconstruct(nx=10, voxel_size_um=(5.0, 5.0))

    def test_nf_grid_conversion_conflict(self, tmp_path):
        conv = NFGridConversion(input_folder=tmp_path, save_dir=tmp_path / "out")
        with pytest.raises(ValueError, match="not both"):
            conv.convert(ny=10, voxel_size_um=(5.0, 5.0))

    def test_unknown_graph_key(self, tmp_path):
        builder = _builder(tmp_path, tmp_path / "out")
        with pytest.raises(ValueError, match="Unknown graph segmentation keys"):
            builder.reconstruct(segmentation={"method": "graph", "bogus": 1})

    def test_unknown_method(self, tmp_path):
        builder = _builder(tmp_path, tmp_path / "out")
        with pytest.raises(ValueError, match="'flood' or 'graph'"):
            builder.reconstruct(segmentation={"method": "watershed"})


class TestMissingConfidence:
    @staticmethod
    def _cloud(with_conf):
        xs, ys = np.meshgrid(np.arange(4.0), np.arange(3.0), indexing="ij")
        df = pd.DataFrame(
            {
                "X": xs.ravel(),
                "Y": ys.ravel(),
                "Z": 0.0,
                "Eul1": 0.0,
                "Eul2": 0.0,
                "Eul3": 0.0,
                "layer": 0,
            }
        )
        if with_conf:
            df["Confidence"] = 0.9
        return df

    def test_min_confidence_without_column_warns(self):
        with pytest.warns(UserWarning, match="no Confidence column"):
            grid, _ = convert.pointcloud_to_fixed_grid(
                self._cloud(False), nx=4, ny=3, min_confidence=0.5
            )
        assert (grid[..., 0] == 1).all()

    def test_layer_without_confidence_values_warns_and_is_not_voided(self):
        a = self._cloud(True)
        b = self._cloud(False).assign(layer=1, Z=1.0)
        cloud = pd.concat([a, b], ignore_index=True)
        with pytest.warns(UserWarning, match=r"layers \[1\]"):
            grid, _ = convert.pointcloud_to_fixed_grid(
                cloud, nx=4, ny=3, min_confidence=0.5
            )
        assert (grid[..., 0] == 1).all()

    def test_no_warning_with_confidence(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            convert.pointcloud_to_fixed_grid(
                self._cloud(True), nx=4, ny=3, min_confidence=0.5
            )


def _tiny_layers(root, nested, n_layers=3):
    """Three tiny .mic layers, flat or one MIDAS-style subfolder per layer."""
    for layer in range(1, n_layers + 1):
        folder = root / f"Layer{layer}" if nested else root
        folder.mkdir(parents=True, exist_ok=True)
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        _write_mic(
            folder / f"sample_layer{layer}.mic", xy, np.zeros((4, 3)), np.ones(4)
        )


class TestLayerDiscovery:
    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("recursive", [False, True])
    def test_flat_and_per_layer_subfolders(self, tmp_path, nested, recursive):
        _tiny_layers(tmp_path, nested)
        pc = convert.nf_to_pointcloud(str(tmp_path), 5.0, recursive=recursive)
        assert sorted(pc["layer"].unique()) == [0, 1, 2]
        assert sorted(pc["Z"].unique()) == [0.0, 5.0, 10.0]
        assert "Confidence" in pc.columns

    def test_layers_sort_numerically_across_subfolders(self, tmp_path):
        _tiny_layers(tmp_path, nested=True, n_layers=11)
        files = convert._find_layer_files(str(tmp_path), "layer", True)
        ordered = sorted(files, key=convert.layer_number)
        assert [convert.layer_number(f) for f in ordered] == list(range(1, 12))
        pc = convert.nf_to_pointcloud(str(tmp_path), 1.0, recursive=True)
        assert pc["layer"].max() == 10

    def test_missing_nested_layer_is_reported(self, tmp_path):
        _tiny_layers(tmp_path, nested=True)
        for f in (tmp_path / "Layer2").iterdir():
            f.unlink()
        with pytest.raises(ValueError, match="Missing or duplicate layers"):
            convert.nf_to_pointcloud(str(tmp_path), 5.0, recursive=True)

    def test_flat_files_win_over_subfolders(self, tmp_path):
        """With recursive=False a populated folder is searched flat, as before."""
        _tiny_layers(tmp_path, nested=False)
        _tiny_layers(tmp_path / "copy", nested=True)
        pc = convert.nf_to_pointcloud(str(tmp_path), 5.0)
        assert pc["layer"].nunique() == 3

    def test_unrelated_csvs_are_ignored_recursively(self, tmp_path):
        _tiny_layers(tmp_path, nested=True)
        (tmp_path / "Layer1" / "notes.csv").write_text("a,b\n1,2\n")
        pc = convert.nf_to_pointcloud(str(tmp_path), 5.0, recursive=True)
        assert pc["layer"].nunique() == 3

    def test_nothing_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert.nf_to_pointcloud(str(tmp_path), 5.0)


class TestNFGridConversion:
    def test_confidence_mask_and_sidecar(self, tmp_path):
        inp = tmp_path / "mic"
        inp.mkdir()
        xs, ys = np.meshgrid(np.arange(6.0), np.arange(4.0), indexing="ij")
        xy = np.stack([xs.ravel(), ys.ravel()], 1)
        conf = np.where(xy[:, 0] < 3, 0.9, 0.1)
        for layer in range(2):
            _write_mic(inp / f"s_layer{layer}.mic", xy, np.zeros((len(xy), 3)), conf)

        conv = NFGridConversion(
            input_folder=inp, save_dir=tmp_path / "out", write_vtk=False
        )
        csv = conv.convert(
            dz=2.0, voxel_size_um=(1.0, 1.0), min_confidence=0.5, max_point_distance=0.5
        )
        df = pd.read_csv(csv)
        assert "confidence" in df.columns
        solid = df["cell_id"] == 1
        assert solid.sum() == 3 * 4 * 2
        assert (df.loc[solid, "confidence"] == 0.9).all()
        assert (df.loc[~solid, "cell_id"] == -1).all()
        side = np.load(conv.confidence_npy)
        assert side.shape == (6, 4, 2)
        np.testing.assert_allclose(side.reshape(-1), df["confidence"].to_numpy())


def _three_grain_grid():
    """A 12 x 10 x 2 grid: void border, three grains split along x."""
    nx, ny, nz = 12, 10, 2
    grid = np.zeros((nx, ny, nz, 7))
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    grid[..., 4] = 5.0 * i
    grid[..., 5] = 5.0 * j
    grid[..., 6] = 5.0 * k
    solid = (i > 0) & (i < nx - 1) & (j > 0) & (j < ny - 1)
    euler = np.zeros((nx, ny, nz, 3))
    euler[(i >= 4) & (i < 8)] = [0.0, np.deg2rad(30.0), 0.0]
    euler[i >= 8] = [np.deg2rad(40.0), np.deg2rad(20.0), np.deg2rad(10.0)]
    grid[..., 1:4] = np.where(solid[..., None], euler, 0.0)
    grid[..., 0] = solid
    return torch.from_numpy(grid), solid


def _same_partition(a, b, solid):
    """True when two labelings split the solid voxels identically."""
    pairs = set(zip(a[solid].tolist(), b[solid].tolist()))
    return len(pairs) == len(set(a[solid].tolist())) == len(set(b[solid].tolist()))


GRAPH = {"method": "graph", "gamma": 0.1, "grain_threshold_final": 10}


@requires_neml2
@requires_networkit
class TestGraphSegmentation:
    def test_graph_matches_flood_and_keeps_void(self, tmp_path):
        torch.manual_seed(42)
        grid, solid = _three_grain_grid()
        builder = _builder(tmp_path, tmp_path / "out")

        cfg = builder._normalize_graph_segmentation(GRAPH)
        graph = builder._segment_graph(grid, cfg)[..., 0].numpy()

        flood = segment.flood(
            grid[..., 1:4].clone(),
            grid[..., 0].clone(),
            np.deg2rad(5.0),
            connectivity=6,
            grain_threshold=1,
            stop_count=10,
            angle_convention="bunge",
            angle_type="radians",
            symmetry="432",
        ).numpy()

        assert (graph[~solid] == 0).all()
        assert sorted(np.unique(graph[solid]).tolist()) == [1.0, 2.0, 3.0]
        assert _same_partition(graph, flood, solid)
        assert len(np.unique(flood[solid])) == 3

    def test_downsample_assigns_every_solid_voxel(self, tmp_path):
        grid, solid = _three_grain_grid()
        builder = _builder(tmp_path, tmp_path / "out")
        full = builder._segment_graph(
            grid, builder._normalize_graph_segmentation(GRAPH)
        )[..., 0].numpy()
        cfg = builder._normalize_graph_segmentation(
            {**GRAPH, "downsample": (2, 2, 1), "grain_threshold_final": 5}
        )
        down = builder._segment_graph(grid, cfg)[..., 0].numpy()
        assert (down[~solid] == 0).all()
        assert (down[solid] > 0).all()
        # An odd-x voxel between two grains is equidistant from kept voxels of
        # both, so the boundary may shift by one voxel; compare the even-x columns.
        even_x = solid.copy()
        even_x[1::2] = False
        assert _same_partition(full, down, even_x)
        assert len(np.unique(down[solid])) == 3

    def test_reconstruct_graph_end_to_end(self, tmp_path):
        inp = tmp_path / "mic"
        _island_layers(inp)
        builder = _builder(inp, tmp_path / "out")
        path = builder.reconstruct(
            dz=5.0,
            nx=40,
            ny=50,
            min_confidence=0.5,
            max_point_distance=7.5,
            segmentation={**GRAPH, "grain_threshold_final": 30},
        )
        grid = np.load(path)
        conf = np.load(tmp_path / "out" / "fixed_grid_confidence.npy")
        assert (grid[conf == 0, 0] == 0).all()
        assert (grid[conf > 0, 0] > 0).all()
        assert len(np.unique(grid[grid[..., 0] > 0, 0])) >= 2
        mesh.write_spn(
            torch.from_numpy(grid),
            str(tmp_path / "g.spn"),
            str(tmp_path / "g.ori"),
            symmetry="432",
        )
