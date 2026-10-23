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

"""Both VoxelMeshBuilder segmentation paths must exclude void voxels.

Issue #58 item 6. A confidence-masked NF voxel reaches ``VoxelMeshBuilder`` as
a ``NFGridConversion`` row with ``cell_id = -1`` and a zero orientation triple.
The builder used to stamp phase 1 on every CSV row, so those voxels entered
both segmenters as solid material -- and, sharing one orientation, formed a
zero-misorientation network that bridges real grains. That is the percolation
the confidence mask exists to remove.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

# torch, matplotlib and networkit are graintrace runtime dependencies, so CI's
# ``pip install ".[dev]"`` has all three and everything above runs there. neml2
# is not pip-installed by that command, and the misorientation math inside
# reconstruct() needs it -- so the end-to-end tests below carry their own guard
# rather than failing the run. The mask tests are pure numpy and always run.
requires_neml2 = pytest.mark.skipif(
    importlib.util.find_spec("neml2") is None, reason="requires neml2"
)

# pylint: disable-next=wrong-import-position  # must follow the torch importorskip
from graintrace.construct_voxel_mesh import VoxelMeshBuilder  # noqa: E402

EULER_COLS = ["Eul0", "Eul1", "Eul2"]

# Two orientations far enough apart that no tolerance merges them.
GRAIN_A = (0.0, 0.0, 0.0)
GRAIN_B = (0.0, np.deg2rad(45.0), 0.0)


def _slab_table(void_slab=True, void_value=-1):
    """A 5 x 3 x 1 grid: grain A, a middle slab, grain B along x.

    With ``void_slab`` the middle column is marked void and carries the zero
    orientation triple -- which is also grain A's, so an unmasked run has a
    continuous zero-misorientation path from A through the slab to nothing
    that separates it from B except the slab itself.
    """
    rows = []
    for i in range(5):
        for j in range(3):
            if i < 2:
                cid, eul = 1, GRAIN_A
            elif i == 2:
                cid, eul = (void_value, (0.0, 0.0, 0.0)) if void_slab else (1, GRAIN_A)
            else:
                cid, eul = 2, GRAIN_B
            rows.append(
                {
                    "x": float(i),
                    "y": float(j),
                    "z": 0.0,
                    "Eul0": eul[0],
                    "Eul1": eul[1],
                    "Eul2": eul[2],
                    "cell_id": cid,
                }
            )
    return pd.DataFrame(rows)


def _builder(tmp_path, df, name="grid.csv", **kwargs):
    """Write ``df`` and return a builder over it, with plotting output off."""
    path = tmp_path / name
    df.to_csv(path, index=False)
    return VoxelMeshBuilder(
        file_path=path,
        save_dir=tmp_path / "out",
        euler_cols=EULER_COLS,
        angle_type="radians",
        write_intermediate=False,
        write_vtk=False,
        **kwargs,
    )


class TestSolidMask:
    """``_solid_mask`` reads the two void conventions this repo writes."""

    def test_plain_table_is_all_solid(self, tmp_path):
        """An EBSD ``x, y, z, Eul*`` table has no marker, so nothing is void."""
        df = _slab_table().drop(columns=["cell_id"])
        builder = _builder(tmp_path, df)
        assert builder._solid_mask(df).all()

    def test_non_positive_cell_id_is_void(self, tmp_path):
        """``NFGridConversion`` normalises masked voxels to ``cell_id = -1``."""
        df = _slab_table()
        builder = _builder(tmp_path, df)
        mask = builder._solid_mask(df)
        assert mask.sum() == 12
        assert not mask[df["cell_id"].to_numpy() == -1].any()

    @pytest.mark.parametrize("void_value", [-1, 0])
    def test_zero_and_minus_one_both_read_as_void(self, tmp_path, void_value):
        """Both halves of the ``<= 0`` convention are honoured."""
        df = _slab_table(void_value=void_value)
        builder = _builder(tmp_path, df)
        assert builder._solid_mask(df).sum() == 12

    def test_phase_column_is_honoured(self, tmp_path):
        """A ``phase`` column marks void with ``0``, the name issue #58 uses."""
        df = _slab_table().rename(columns={"cell_id": "phase"})
        df["phase"] = (df["phase"] > 0).astype(int)
        builder = _builder(tmp_path, df)
        assert builder._solid_mask(df).sum() == 12


class TestVoidExclusion:
    """Void voxels stay void through both segmenters."""

    @staticmethod
    def _grain_ids(grid):
        """The (5, 3) plane of grain ids from a reconstructed grid file."""
        return np.load(grid)[..., 0].reshape(5, 3)

    @requires_neml2
    def test_flood_leaves_void_voxels_at_zero(self, tmp_path):
        """flood() seeds from phase > 0, so the slab must never be stamped."""
        builder = _builder(tmp_path, _slab_table())
        ids = self._grain_ids(
            builder.reconstruct(
                segmentation={
                    "method": "flood",
                    "params": {
                        "misorientation_tol": np.deg2rad(5.0),
                        "connectivity": 6,
                        "grain_threshold": 1,
                        "grain_threshold_final": 1,
                        "batch_norm": 1000,
                        "stop_count": 10,
                    },
                }
            )
        )
        assert (ids[2] == 0).all()
        assert ids[0][0] != ids[4][0]

    @requires_neml2
    def test_graph_leaves_void_voxels_at_zero(self, tmp_path):
        """The graph builder must drop void rows before the graph is built."""
        pytest.importorskip("networkit")

        builder = _builder(tmp_path, _slab_table())
        ids = self._grain_ids(
            builder.reconstruct(
                segmentation={
                    "method": "graph",
                    "params": {
                        "misorientation_tol": np.deg2rad(5.0),
                        "connectivity": 6,
                        "grain_threshold_final": 1,
                    },
                    "graph_params": {
                        "segmenter": "leiden",
                        "graph_mode": "grid",
                        "manhattan_radius": 1,
                        "grid_tol": 1e-6,
                        "n_jobs": 1,
                        "weight_chunk_size": 100_000,
                        "reduce_edges_topweights_k": 6,
                        "nodes_chunk": 100_000,
                        "seed": 42,
                        "networkit_kwargs": {"gamma": 1.0},
                        # Fixed sigma at half the cutoff; sigma_auto collapses
                        # on near-identical intra-grain edges (CLAUDE.md §10).
                        "weight_cfg": {
                            "mode": "rbf",
                            "sigma": np.deg2rad(2.5),
                            "power": 2.0,
                        },
                        "plot": False,
                    },
                }
            )
        )
        assert (ids[2] == 0).all()
        assert ids[0][0] != ids[4][0]

    def test_an_all_void_grid_is_refused(self, tmp_path):
        """Segmenting nothing is a configuration error, not an empty result."""
        pytest.importorskip("networkit")

        df = _slab_table()
        df["cell_id"] = -1
        builder = _builder(tmp_path, df)
        builder._load_dataframe()
        with pytest.raises(ValueError, match="every voxel is marked void"):
            builder._segment_graph(torch.zeros((5, 3, 1, 7)), {}, {})


class TestUnmarkedInputIsUnchanged:
    """A table with no marker column must segment exactly as it did before."""

    @requires_neml2
    def test_phase_slot_is_all_ones(self, tmp_path):
        """The regression guard: no marker column means no voxel is dropped."""
        df = _slab_table(void_slab=False).drop(columns=["cell_id"])
        builder = _builder(tmp_path, df)
        grid = builder.reconstruct(
            segmentation={
                "method": "flood",
                "params": {
                    "misorientation_tol": np.deg2rad(5.0),
                    "connectivity": 6,
                    "grain_threshold": 1,
                    "grain_threshold_final": 1,
                    "batch_norm": 1000,
                    "stop_count": 10,
                },
            }
        )
        assert (np.load(grid)[..., 0] > 0).all()
