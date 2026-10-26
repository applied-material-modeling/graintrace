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
    """``_solid_mask``: explicit void markers plus the unambiguous ``-1`` sentinel."""

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

    def test_cell_id_zero_is_not_auto_void(self, tmp_path):
        """Only the ``-1`` sentinel is automatic; 0 could be a 0-based grain id."""
        df = _slab_table(void_value=0)
        builder = _builder(tmp_path, df)
        assert builder._solid_mask(df).all()

    def test_cell_id_col_zero_is_void(self, tmp_path):
        """A named grain-id column uses the ``<= 0`` rule of _assign_existing_ids."""
        df = _slab_table(void_value=0)
        builder = _builder(tmp_path, df, cell_id_col="cell_id")
        assert builder._solid_mask(df).sum() == 12

    def test_zero_based_ang_phase_is_not_void(self, tmp_path):
        """An EBSD .ang phase is 0-based: phase 0 is a real phase, not void.

        Reading ``phase <= 0`` as void, as the implicit ``phase`` marker did,
        turned a single-phase EBSD map into an all-void grid.
        """
        df = _slab_table(void_slab=False).drop(columns=["cell_id"])
        df["phase"] = 0
        builder = _builder(tmp_path, df)
        assert builder._solid_mask(df).all()

    def test_phase_column_is_honoured_when_opted_in(self, tmp_path):
        """``void_marker_col="phase"`` reads a 1-based phase where 0 is void."""
        df = _slab_table().rename(columns={"cell_id": "phase"})
        df["phase"] = (df["phase"] > 0).astype(int)
        builder = _builder(tmp_path, df, void_marker_col="phase")
        assert builder._solid_mask(df).sum() == 12

    def test_missing_void_marker_col_is_an_error(self, tmp_path):
        df = _slab_table()
        builder = _builder(tmp_path, df, void_marker_col="phase")
        with pytest.raises(ValueError, match="void_marker_col"):
            builder._solid_mask(df)


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


class TestIsolatedIsland:
    """A small island with no solid neighbour must end up void, not -1."""

    @requires_neml2
    def test_island_is_voided_and_meshable(self, tmp_path):
        """remove_small_segments marks the island -1 and infill cannot reach it."""
        from graintrace.nf import mesh  # pylint: disable=import-outside-toplevel

        rows = []
        for i in range(9):
            for j in range(3):
                if i < 5:
                    cid, eul = 1, GRAIN_A
                elif i == 7 and j == 1:
                    cid, eul = 1, GRAIN_B  # one-voxel island, all-void neighbours
                else:
                    cid, eul = -1, (0.0, 0.0, 0.0)
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
        builder = _builder(tmp_path, pd.DataFrame(rows))
        path = builder.reconstruct(
            segmentation={
                "method": "flood",
                "params": {
                    "misorientation_tol": np.deg2rad(5.0),
                    "connectivity": 6,
                    "grain_threshold": 1,
                    "grain_threshold_final": 3,
                    "batch_norm": 1000,
                    "stop_count": 10,
                },
            }
        )
        grid = np.load(path)
        ids = grid[..., 0].reshape(9, 3)
        assert (ids >= 0).all()
        assert ids[7, 1] == 0
        assert (ids[:5] == 1).all()
        mesh.write_spn(
            torch.from_numpy(grid), str(tmp_path / "v.spn"), str(tmp_path / "v.ori")
        )
