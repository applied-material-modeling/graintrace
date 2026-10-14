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

"""Tests for graintrace.experiment_rotation_helper.update_experiments."""

import os

import numpy as np
import pandas as pd
import pytest


def _install_fake_builder(monkeypatch, n_grains, n_orientations):
    """Replace VoronoiMeshBuilder with a stub that needs no NEPER.

    The stub writes ``n_orientations`` rows to ``reconstruction.ori`` and
    exposes a grain table of ``n_grains`` rows, so the two can be made to
    disagree the way a bounding-box filter makes them disagree in practice.
    """
    from graintrace import experiment_rotation_helper as erh

    class _FakeBuilder:  # pylint: disable=too-few-public-methods
        def __init__(self, input_csv, output_dir, **_kwargs):
            self.input_csv = input_csv
            self.output_dir = output_dir
            self.strain_unit = None
            self.elastic_strain_id = None
            self.data = pd.DataFrame(
                {
                    "GrainID": np.arange(n_grains),
                    "X": np.arange(n_grains, dtype=float),
                    "Y": np.zeros(n_grains),
                    "Z": np.zeros(n_grains),
                }
            )

        def read_input(self):
            return None

        def build_voronoi(self, **_kwargs):
            os.makedirs(self.output_dir, exist_ok=True)
            ori = np.tile(np.arange(9, dtype=float), (n_orientations, 1))
            ori += np.arange(n_orientations, dtype=float).reshape(-1, 1)
            np.savetxt(os.path.join(self.output_dir, "reconstruction.ori"), ori)

    monkeypatch.setattr(erh, "VoronoiMeshBuilder", _FakeBuilder)
    return erh


def _run(monkeypatch, tmp_path, n_grains, n_orientations):
    erh = _install_fake_builder(monkeypatch, n_grains, n_orientations)
    src = tmp_path / "scan_0.csv"
    src.write_text("X,Y,Z\n0,0,0\n")
    out_root = tmp_path / "rotated"
    erh.update_experiments(
        input_files=[str(src)],
        output_root=str(out_root),
        bounding_box=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
    )
    return out_root / "scan_0.csv"


class TestRowCountMismatchRaises:
    """A positional join must refuse a count mismatch, not truncate it (#51).

    The output feeds a physically registered MaterialCalibration fit. A silent
    off-by-N row shift gives every grain the wrong orientation, the fit still
    converges, and the calibrated parameters are wrong with no diagnostic.
    """

    def test_matching_counts_still_write_the_file(self, monkeypatch, tmp_path):
        out = _run(monkeypatch, tmp_path, n_grains=6, n_orientations=6)
        written = pd.read_csv(out)
        assert len(written) == 6
        assert list(written.columns[:9]) == [
            f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)
        ]

    @pytest.mark.parametrize(
        "n_grains,n_orientations", [(6, 4), (4, 6), (100, 99), (1, 2)]
    )
    def test_mismatched_counts_raise(
        self, monkeypatch, tmp_path, n_grains, n_orientations
    ):
        with pytest.raises(ValueError) as excinfo:
            _run(monkeypatch, tmp_path, n_grains, n_orientations)
        message = str(excinfo.value)
        assert str(n_grains) in message and str(n_orientations) in message
        assert "reconstruction.ori" in message

    def test_nothing_is_written_when_the_counts_disagree(self, monkeypatch, tmp_path):
        """The old code wrote a truncated, mis-paired file. It must not."""
        with pytest.raises(ValueError):
            _run(monkeypatch, tmp_path, n_grains=6, n_orientations=4)
        assert not (tmp_path / "rotated" / "scan_0.csv").exists()
