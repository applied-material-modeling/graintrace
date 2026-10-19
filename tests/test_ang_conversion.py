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

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graintrace.ang_conversion import AngConverter, read_ang, read_ang_header
from graintrace.construct_voxel_mesh import VoxelMeshBuilder
from graintrace.nf import convert as nf_convert

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "mwe_data" / "ang_ebsd"
N_LAYERS = 3
N_POINTS = 30  # 6 x 5 square grid per layer


def _headerless(tmp_path, rows, name="layer1.ang"):
    """Write a minimal 8-column, comment-free .ang file and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    lines = ["  ".join(f"{v:.6f}" for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


ROWS_8COL = [
    # phi1, PHI, phi2, x, y, IQ, CI, phase
    (0.1, 0.2, 0.3, 0.0, 0.0, 100.0, 0.9, 1.0),
    (0.1, 0.2, 0.3, 0.5, 0.0, 100.0, 0.4, 1.0),
    (1.1, 0.7, 0.3, 0.0, 0.5, 100.0, -1.0, 1.0),
    (1.1, 0.7, 0.3, 0.5, 0.5, 100.0, 0.8, 1.0),
]


class TestAngHeader:
    """Parsing the ``#`` comment block."""

    def test_fixture_header_pins_units_convention_and_step(self):
        header = read_ang_header(FIXTURE_DIR / "layer1.ang")

        assert header.angle_type == "radians"
        assert header.angle_convention == "bunge"
        assert header.xstep == pytest.approx(0.5)
        assert header.ystep == pytest.approx(0.5)
        assert header.grid == "SqrGrid"
        # Whitespace-separated keys are captured alongside the colon form.
        assert header.keys["MaterialName"] == "Nickel"

    def test_headerless_file_pins_nothing(self, tmp_path):
        header = read_ang_header(_headerless(tmp_path, ROWS_8COL))

        assert header.angle_type is None
        assert header.angle_convention is None
        assert header.keys == {}

    def test_unrecognised_unit_in_header_raises(self, tmp_path):
        path = tmp_path / "bad.ang"
        path.write_text("# ANGLE_UNITS: gradians\n0 0 0 0 0 1 1 1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="unrecognised angle unit"):
            read_ang_header(path)

    def test_unrecognised_convention_in_header_raises(self, tmp_path):
        path = tmp_path / "bad.ang"
        path.write_text(
            "# EULER_CONVENTION: matthies\n0 0 0 0 0 1 1 1\n", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="unrecognised Euler convention"):
            read_ang_header(path)


class TestReadAng:
    """Reading the data block of both layouts."""

    def test_ten_column_header_variant(self):
        df, header = read_ang(FIXTURE_DIR / "layer1.ang")

        assert len(df) == N_POINTS
        assert list(df.columns)[:8] == [
            "phi1",
            "PHI",
            "phi2",
            "x",
            "y",
            "IQ",
            "CI",
            "phase",
        ]
        assert list(df.columns)[8:] == ["SEM", "fit"]
        assert header.angle_type == "radians"
        assert header.angle_convention == "bunge"

    def test_eight_column_headerless_needs_explicit_settings(self, tmp_path):
        path = _headerless(tmp_path, ROWS_8COL)

        df, header = read_ang(path, angle_convention="bunge", angle_type="radians")

        assert list(df.columns) == [
            "phi1",
            "PHI",
            "phi2",
            "x",
            "y",
            "IQ",
            "CI",
            "phase",
        ]
        assert len(df) == len(ROWS_8COL)
        assert header.angle_type == "radians"

    def test_refuses_to_guess_the_unit(self, tmp_path):
        path = _headerless(tmp_path, ROWS_8COL)

        with pytest.raises(ValueError, match="angle_type is not pinned"):
            read_ang(path, angle_convention="bunge")

    def test_refuses_to_guess_the_convention(self, tmp_path):
        path = _headerless(tmp_path, ROWS_8COL)

        with pytest.raises(ValueError, match="angle_convention is not pinned"):
            read_ang(path, angle_type="radians")

    def test_explicit_argument_overrides_the_header(self):
        _df, header = read_ang(FIXTURE_DIR / "layer1.ang", angle_type="degrees")

        assert header.angle_type == "degrees"
        assert header.angle_convention == "bunge"

    def test_too_few_columns_raises(self, tmp_path):
        path = tmp_path / "short.ang"
        path.write_text("0.1 0.2 0.3 0.0 0.0\n", encoding="utf-8")

        with pytest.raises(ValueError, match="expected at least 8"):
            read_ang(path, angle_convention="bunge", angle_type="radians")

    def test_column_headers_key_names_the_columns(self, tmp_path):
        path = tmp_path / "named.ang"
        path.write_text(
            "# ANGLE_UNITS: radians\n"
            "# EULER_CONVENTION: bunge\n"
            "# COLUMN_HEADERS: phi1, PHI, phi2, x, y, IQ, CI, Phase index\n"
            "0.1 0.2 0.3 0.0 0.0 100.0 0.9 1\n",
            encoding="utf-8",
        )

        df, _header = read_ang(path)

        assert list(df.columns) == [
            "phi1",
            "PHI",
            "phi2",
            "x",
            "y",
            "IQ",
            "CI",
            "phase",
        ]

    def test_column_headers_count_mismatch_raises(self, tmp_path):
        path = tmp_path / "mismatch.ang"
        path.write_text(
            "# ANGLE_UNITS: radians\n"
            "# EULER_CONVENTION: bunge\n"
            "# COLUMN_HEADERS: phi1, PHI, phi2, x, y\n"
            "0.1 0.2 0.3 0.0 0.0 100.0 0.9 1\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="declares 5 columns"):
            read_ang(path)


class TestToEbsdCsv:
    """The merged CSV target for ``VoxelMeshBuilder``."""

    def test_merged_csv_schema_and_z_stacking(self, tmp_path):
        converter = AngConverter(input_path=FIXTURE_DIR, save_dir=tmp_path)

        out = converter.to_ebsd_csv(zstep=2.0)
        df = pd.read_csv(out)

        assert out.name == "ebsd_merged.csv"
        assert list(df.columns) == ["x", "y", "z", "Eul0", "Eul1", "Eul2"]
        assert len(df) == N_LAYERS * N_POINTS
        # z is the layer index times the step, which is the decision this
        # converter exists to make once.
        assert sorted(df["z"].unique().tolist()) == [0.0, 2.0, 4.0]

    def test_output_is_degrees_by_default(self, tmp_path):
        converter = AngConverter(input_path=FIXTURE_DIR, save_dir=tmp_path)
        df = pd.read_csv(converter.to_ebsd_csv(zstep=1.0))

        # The fixture is radians in file; degrees out means values well past 2*pi.
        assert df[["Eul0", "Eul1", "Eul2"]].to_numpy().max() > 2 * math.pi

    def test_radian_output_round_trips_the_input(self, tmp_path):
        converter = AngConverter(
            input_path=FIXTURE_DIR / "layer1.ang",
            save_dir=tmp_path,
            output_angle_type="radians",
        )
        df = pd.read_csv(converter.to_ebsd_csv(zstep=1.0))
        source, _header = read_ang(FIXTURE_DIR / "layer1.ang")

        np.testing.assert_allclose(
            df[["Eul0", "Eul1", "Eul2"]].to_numpy(),
            source[["phi1", "PHI", "phi2"]].to_numpy(),
            atol=1e-9,
        )
        assert (df["z"] == 0.0).all()

    def test_kocks_input_is_converted_to_bunge(self, tmp_path):
        path = _headerless(tmp_path / "in", [(0.4, 0.5, 0.6, 0.0, 0.0, 1.0, 1.0, 1.0)])
        converter = AngConverter(
            input_path=path,
            save_dir=tmp_path / "out",
            angle_convention="kocks",
            angle_type="radians",
            output_angle_type="radians",
        )

        df = pd.read_csv(converter.to_ebsd_csv(zstep=1.0))

        # orientation_helper._to_bunge: (psi, theta, phi) ->
        # (psi + pi/2, theta, pi/2 - phi)
        assert df["Eul0"][0] == pytest.approx(0.4 + math.pi / 2)
        assert df["Eul1"][0] == pytest.approx(0.5)
        assert df["Eul2"][0] == pytest.approx(math.pi / 2 - 0.6)

    def test_confidence_filter_drops_low_ci_points(self, tmp_path):
        path = _headerless(tmp_path / "in", ROWS_8COL)
        converter = AngConverter(
            input_path=path,
            save_dir=tmp_path / "out",
            angle_convention="bunge",
            angle_type="radians",
            confidence_min=0.5,
        )

        df = pd.read_csv(converter.to_ebsd_csv(zstep=1.0))

        assert len(df) == 2  # the 0.4 and the -1.0 CI rows are dropped

    def test_confidence_filter_emptying_a_layer_raises(self, tmp_path):
        path = _headerless(tmp_path / "in", ROWS_8COL)
        converter = AngConverter(
            input_path=path,
            save_dir=tmp_path / "out",
            angle_convention="bunge",
            angle_type="radians",
            confidence_min=0.99,
        )

        with pytest.raises(ValueError, match="no points left"):
            converter.to_ebsd_csv(zstep=1.0)


class TestToMic:
    """The per-layer ``.mic`` target for ``NearFieldMeshBuilder``."""

    def test_written_layers_match_the_nf_reader_schema(self, tmp_path):
        converter = AngConverter(input_path=FIXTURE_DIR, save_dir=tmp_path)

        written = converter.to_mic(zstep=2.0)

        assert [p.name for p in written] == ["layer1.mic", "layer2.mic", "layer3.mic"]
        text = written[1].read_text(encoding="utf-8").splitlines()
        assert text[0].startswith("%TriEdgeSize")
        assert text[1] == "%NumPhases 1"
        assert text[2] == "%GlobalPosition 2.000000"
        # The NF reader does skiprows=3, header=0 on a whitespace split.
        assert text[3].split("\t")[0] == "%OrientationRowNr"

    def test_nf_reader_round_trips_the_written_layers(self, tmp_path):
        converter = AngConverter(input_path=FIXTURE_DIR, save_dir=tmp_path)
        converter.to_mic(zstep=2.0)

        # Read them exactly the way graintrace.nf.convert.nf_to_pointcloud does.
        df = pd.read_csv(tmp_path / "layer2.mic", skiprows=3, sep=r"\s+", header=0)
        df = df.rename(columns={"%OrientationRowNr": "OrientationRowNr"})

        assert len(df) == N_POINTS
        for column in ("OrientationRowNr", "X", "Y", "Eul1", "Eul2", "Eul3"):
            assert column in df.columns
        assert df["TriEdgeSize"].iloc[0] == pytest.approx(0.5)  # from the header XSTEP

    def test_tri_edge_size_override(self, tmp_path):
        converter = AngConverter(
            input_path=FIXTURE_DIR / "layer1.ang", save_dir=tmp_path
        )

        written = converter.to_mic(zstep=1.0, tri_edge_size=3.5, prefix="scan")

        assert written[0].name == "scan1.mic"
        assert (
            written[0].read_text(encoding="utf-8").startswith("%TriEdgeSize 3.500000")
        )


class TestDownstreamBuilders:
    """Both outputs are read by the builder they were written for.

    The fixture holds three grains planted by construction (two columns of the
    6 x 5 grid each, repeated over three layers), so a correct conversion is
    one the downstream reader recovers them from.
    """

    def test_nf_pointcloud_reader_ingests_the_mic_folder(self, tmp_path):
        converter = AngConverter(input_path=FIXTURE_DIR, save_dir=tmp_path)
        converter.to_mic(zstep=2.0)

        pointcloud = nf_convert.nf_to_pointcloud(
            str(tmp_path), 2.0, layer_token="layer"
        )

        assert len(pointcloud) == N_LAYERS * N_POINTS
        assert sorted(pointcloud["Z"].unique().tolist()) == [0.0, 2.0, 4.0]
        assert sorted(pointcloud["layer"].unique().tolist()) == [0, 1, 2]

    def test_voxel_mesh_builder_recovers_the_planted_grains(self, tmp_path):
        converter = AngConverter(input_path=FIXTURE_DIR, save_dir=tmp_path)
        merged = converter.to_ebsd_csv(zstep=0.5)

        builder = VoxelMeshBuilder(
            file_path=merged,
            save_dir=tmp_path / "voxel",
            euler_cols=["Eul0", "Eul1", "Eul2"],
            angle_convention="bunge",
            # to_ebsd_csv writes degrees by default; the builder must be told.
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


class TestLayerOrdering:
    """Folder inputs are ordered by the integer after ``file_token``."""

    def test_layers_sort_numerically_not_lexically(self, tmp_path):
        source = tmp_path / "in"
        source.mkdir()
        for index in (1, 2, 10):
            _headerless(
                source,
                [(0.1 * index, 0.2, 0.3, 0.0, 0.0, 1.0, 1.0, 1.0)],
                name=f"layer{index}.ang",
            )
        converter = AngConverter(
            input_path=source,
            save_dir=tmp_path / "out",
            angle_convention="bunge",
            angle_type="radians",
            output_angle_type="radians",
        )

        df = pd.read_csv(converter.to_ebsd_csv(zstep=1.0))

        # Lexical order would put layer10 second; layer order puts it last.
        np.testing.assert_allclose(df["Eul0"].to_numpy(), [0.1, 0.2, 1.0], atol=1e-9)
        np.testing.assert_allclose(df["z"].to_numpy(), [0.0, 1.0, 2.0])

    def test_empty_folder_raises(self, tmp_path):
        source = tmp_path / "empty"
        source.mkdir()
        converter = AngConverter(
            input_path=source,
            save_dir=tmp_path / "out",
            angle_convention="bunge",
            angle_type="radians",
        )

        with pytest.raises(FileNotFoundError, match="No .ang files"):
            converter.layer_files()
