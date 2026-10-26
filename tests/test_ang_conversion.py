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

"""Tests for the ``.ang`` reader and the two conversion targets."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graintrace.ang_conversion import ANG_COLUMNS, AngConverter, read_ang

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "mwe_data" / "ang_ebsd"
N_LAYERS = 3
N_POINTS = 30  # 6 x 5 square grid per layer


def _write_ang(folder, rows, name="layer1.ang", header=""):
    """Write an 8-column ``.ang`` file (optionally with ``#`` header lines)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    lines = ["  ".join(f"{v:.6f}" for v in row) for row in rows]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return path


# phi1, PHI, phi2, x, y, IQ, CI, phase (0-based, as TSL single-phase files write it)
ROWS = [
    (0.1, 0.2, 0.3, 0.0, 0.0, 100.0, 0.9, 0.0),
    (0.1, 0.2, 0.3, 0.5, 0.0, 100.0, 0.4, 0.0),
    (1.1, 0.7, 0.3, 0.0, 0.5, 100.0, -1.0, 0.0),
    (1.1, 0.7, 0.3, 0.5, 0.5, 100.0, 0.8, 0.0),
]


def _converter(input_path, tmp_path, **kwargs):
    return AngConverter(input_path=input_path, save_dir=tmp_path / "out", **kwargs)


class TestReadAng:
    """Reading the 10-column headed and 8-column headerless layouts."""

    def test_ten_column_fixture(self):
        df, header = read_ang(FIXTURE_DIR / "layer1.ang")
        assert list(df.columns) == list(ANG_COLUMNS)
        assert len(df) == N_POINTS
        assert header["GRID"] == "SqrGrid"
        assert float(header["XSTEP"]) == pytest.approx(0.5)

    def test_eight_column_headerless(self, tmp_path):
        df, header = read_ang(_write_ang(tmp_path, ROWS))
        assert header == {}
        np.testing.assert_allclose(df["CI"], [0.9, 0.4, -1.0, 0.8])

    def test_too_few_columns_raises(self, tmp_path):
        path = tmp_path / "layer1.ang"
        path.write_text("0.1 0.2 0.3 0.0 0.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected at least 8"):
            read_ang(path)

    def test_unknown_unit_raises(self, tmp_path):
        with pytest.raises(ValueError, match="angle_type"):
            _converter(FIXTURE_DIR, tmp_path, angle_type="grad")


class TestToEbsdCsv:
    """The merged CSV target for ``VoxelMeshBuilder``."""

    def test_schema_and_z_stacking(self, tmp_path):
        df = pd.read_csv(_converter(FIXTURE_DIR, tmp_path).to_ebsd_csv(zstep=2.0))
        # No phase/cell_id column: nothing a builder could read as a void marker.
        assert list(df.columns) == ["x", "y", "z", "Eul0", "Eul1", "Eul2"]
        assert len(df) == N_LAYERS * N_POINTS
        assert sorted(df["z"].unique()) == [0.0, 2.0, 4.0]

    def test_units(self, tmp_path):
        raw, _ = read_ang(FIXTURE_DIR / "layer1.ang")
        deg = pd.read_csv(
            _converter(FIXTURE_DIR / "layer1.ang", tmp_path).to_ebsd_csv(zstep=1.0)
        )
        rad = pd.read_csv(
            _converter(
                FIXTURE_DIR / "layer1.ang", tmp_path, output_angle_type="radians"
            ).to_ebsd_csv(zstep=1.0, output_name="rad.csv")
        )
        np.testing.assert_allclose(rad["Eul0"], raw["phi1"])
        np.testing.assert_allclose(deg["Eul0"], np.rad2deg(raw["phi1"]))

    def test_degrees_file_read_as_radians_raises(self, tmp_path):
        rows = [(r[0] * 100, *r[1:]) for r in ROWS]  # phi1 up to 110: not radians
        path = _write_ang(tmp_path / "in", rows)
        with pytest.raises(ValueError, match="cannot be radians"):
            _converter(path, tmp_path).to_ebsd_csv(zstep=1.0)
        out = _converter(path, tmp_path, angle_type="degrees").to_ebsd_csv(zstep=1.0)
        np.testing.assert_allclose(pd.read_csv(out)["Eul0"], [10.0, 10.0, 110.0, 110.0])

    def test_unindexed_four_pi_marker_is_accepted(self, tmp_path):
        rows = ROWS + [(4 * np.pi, 4 * np.pi, 4 * np.pi, 1.0, 1.0, 0.0, -1.0, 0.0)]
        out = _converter(_write_ang(tmp_path / "in", rows), tmp_path).to_ebsd_csv(
            zstep=1.0
        )
        assert len(pd.read_csv(out)) == len(rows)

    def test_confidence_filter(self, tmp_path):
        path = _write_ang(tmp_path / "in", ROWS)
        out = _converter(path, tmp_path, confidence_min=0.0).to_ebsd_csv(zstep=1.0)
        assert len(pd.read_csv(out)) == 3
        with pytest.raises(ValueError, match="no points left"):
            _converter(path, tmp_path, confidence_min=0.99).to_ebsd_csv(zstep=1.0)

    def test_hex_grid_raises(self, tmp_path):
        path = _write_ang(
            tmp_path / "in", ROWS, header="# GRID: HexGrid\n# XSTEP: 0.5\n"
        )
        with pytest.raises(ValueError, match="HexGrid"):
            _converter(path, tmp_path).to_ebsd_csv(zstep=1.0)
        assert (
            len(_converter(path, tmp_path).to_mic(zstep=1.0)) == 1
        )  # scattered points are fine for NF

    def test_voxel_mesh_builder_recovers_the_planted_grains(self, tmp_path):
        pytest.importorskip("neml2")
        pytest.importorskip("networkit")
        # pylint: disable=import-outside-toplevel
        from graintrace.construct_voxel_mesh import VoxelMeshBuilder

        merged = _converter(FIXTURE_DIR, tmp_path).to_ebsd_csv(zstep=0.5)
        builder = VoxelMeshBuilder(
            file_path=merged,
            save_dir=tmp_path / "voxel",
            euler_cols=["Eul0", "Eul1", "Eul2"],
            angle_convention="bunge",
            angle_type="degrees",
            symmetry="432",
        )
        grid_path = builder.reconstruct(
            apply_smoothing=False,
            segmentation={
                "method": "graph",
                "params": {
                    "misorientation_tol": 5.0,
                    "connectivity": 26,
                    "grain_threshold_final": 5,
                },
                "graph_params": {
                    "segmenter": "leiden",
                    "graph_mode": "grid",
                    "manhattan_radius": 2,
                    "grid_tol": 1e-6,
                    "n_jobs": 1,
                    "seed": 42,
                    "reduce_edges_topweights_k": 8,
                    "networkit_kwargs": {"gamma": 0.1},
                    "weight_cfg": {"mode": "rbf", "sigma": 2.5, "power": 2.0},
                    "plot": False,
                },
            },
        )
        grid = np.load(grid_path)
        assert grid.shape == (6, 5, 3, 7)
        assert sorted(set(grid[..., 0].ravel().tolist())) == [1.0, 2.0, 3.0]


class TestToMic:
    """The per-layer ``.mic`` target for ``NearFieldMeshBuilder``."""

    def test_nf_reader_round_trip(self, tmp_path):
        nf_convert = pytest.importorskip("graintrace.nf.convert")
        written = _converter(FIXTURE_DIR, tmp_path).to_mic(zstep=2.0)
        assert [p.name for p in written] == ["layer1.mic", "layer2.mic", "layer3.mic"]
        lines = written[1].read_text(encoding="utf-8").splitlines()
        assert lines[:3] == [
            "%TriEdgeSize 0.500000",
            "%NumPhases 1",
            "%GlobalPosition 2.000000",
        ]

        pc = nf_convert.nf_to_pointcloud(
            str(tmp_path / "out"), 2.0, layer_token="layer"
        )
        assert len(pc) == N_LAYERS * N_POINTS
        assert sorted(pc["Z"].unique()) == [0.0, 2.0, 4.0]
        raw, _ = read_ang(FIXTURE_DIR / "layer1.ang")
        first = pc[pc["layer"] == 0]
        np.testing.assert_allclose(first["Confidence"], raw["CI"])
        np.testing.assert_allclose(first["Eul1"], np.rad2deg(raw["phi1"]), atol=1e-5)
        assert (first["TriEdgeSize"] == 0.5).all()

    def test_zero_based_phase_is_written_as_phase_one(self, tmp_path):
        written = _converter(_write_ang(tmp_path / "in", ROWS), tmp_path).to_mic(
            zstep=1.0
        )
        df = pd.read_csv(written[0], skiprows=3, sep=r"\s+")
        assert (df["PhaseNr"] == 1).all()
        np.testing.assert_allclose(df["Confidence"], [0.9, 0.4, -1.0, 0.8])

    def test_tri_edge_size_and_prefix_override(self, tmp_path):
        written = _converter(FIXTURE_DIR / "layer1.ang", tmp_path).to_mic(
            zstep=1.0, tri_edge_size=3.5, prefix="scan"
        )
        assert written[0].name == "scan1.mic"
        assert (
            written[0].read_text(encoding="utf-8").startswith("%TriEdgeSize 3.500000")
        )


class TestLayerOrdering:
    """Folder inputs are ordered by the integer after ``file_token``."""

    @staticmethod
    def _layers(tmp_path, indices):
        source = tmp_path / "in"
        for i in indices:
            _write_ang(
                source,
                [(0.1 * i, 0.2, 0.3, 0.0, 0.0, 1.0, 1.0, 0.0)],
                name=f"layer{i}.ang",
            )
        return source

    def test_numeric_not_lexical_order(self, tmp_path):
        source = self._layers(tmp_path, (1, 2, 10))
        df = pd.read_csv(
            _converter(source, tmp_path, output_angle_type="radians").to_ebsd_csv(
                zstep=1.0
            )
        )
        np.testing.assert_allclose(df["Eul0"], [0.1, 0.2, 1.0], atol=1e-9)
        np.testing.assert_allclose(df["z"], [0.0, 1.0, 9.0])

    def test_gap_kept_in_z_and_mic_names(self, tmp_path):
        nf_convert = pytest.importorskip("graintrace.nf.convert")
        source = self._layers(tmp_path, (1, 3))
        conv = _converter(source, tmp_path)
        assert conv.layer_offsets() == [
            (0, source / "layer1.ang"),
            (2, source / "layer3.ang"),
        ]
        np.testing.assert_allclose(
            pd.read_csv(conv.to_ebsd_csv(zstep=2.0))["z"], [0.0, 4.0]
        )
        written = conv.to_mic(zstep=2.0)
        assert [p.name for p in written] == ["layer1.mic", "layer3.mic"]
        with pytest.raises(ValueError, match="Missing layers"):
            nf_convert.nf_to_pointcloud(str(tmp_path / "out"), dz=2.0)

    def test_empty_folder_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="No .ang files"):
            _converter(tmp_path / "empty", tmp_path).layer_offsets()
