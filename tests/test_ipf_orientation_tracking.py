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

"""Tests for IPF orientation-tracking (trajectory) plotting and track builders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

MWE = Path(__file__).parent.parent / "mwe_data"


class TestComposeTracks:
    def test_chain_three_steps(self):
        from graintrace.ipf_orientation_tracking import OrientationTracker

        tracks = OrientationTracker.compose_tracks([[1, 2, 0], [2, 0, 1]])
        assert tracks[0] == [0, 1, 0]
        assert tracks[1] == [1, 2, 1]
        assert tracks[2] == [2, 0, 2]

    def test_truncates_on_unmatched(self):
        from graintrace.ipf_orientation_tracking import OrientationTracker

        tracks = OrientationTracker.compose_tracks([[1, -1, 0]], seed_ids=[0, 1, 2])
        assert tracks[0] == [0, 1]
        assert tracks[1] == [1]  # truncated at the dropout
        assert tracks[2] == [2, 0]

    def test_single_step_no_maps(self):
        from graintrace.ipf_orientation_tracking import OrientationTracker

        assert OrientationTracker.compose_tracks([], seed_ids=[0, 3, 7]) == [
            [0],
            [3],
            [7],
        ]

    def test_seed_default_is_all_first_rows(self):
        from graintrace.ipf_orientation_tracking import OrientationTracker

        assert len(OrientationTracker.compose_tracks([[2, 0, 1]])) == 3


class TestGetIpfPoints:
    def test_count_and_order_preserved(self):
        pytest.importorskip("neml2")
        import torch

        from graintrace.ipf_postprocess import IPFProcessor
        from graintrace.orientation_helper import mrp_to_matrix

        torch.set_default_dtype(torch.float64)
        # identity is a triangle corner the ragged neml2 reducer drops; here it must
        # be kept, in order, so the count matches the input.
        mrp = torch.tensor(
            [[0.0, 0, 0], [0.1, 0, 0], [-0.32, -0.19, 0.47], [0.05, 0.05, 0.05]]
        )
        ipf = IPFProcessor(crystal_symmetry="432", sample_symmetry="1")
        pts = np.asarray(ipf.get_ipf_points(mrp_to_matrix(mrp), [0.0, 0.0, 1.0]))
        assert pts.shape == (4, 2)
        assert np.all(np.isfinite(pts))

    def test_matches_color_reduction_length(self):
        pytest.importorskip("neml2")
        import torch

        from graintrace.ipf_postprocess import IPFProcessor
        from graintrace.orientation_helper import mrp_to_matrix

        torch.set_default_dtype(torch.float64)
        R = mrp_to_matrix(torch.tensor([[-0.32, -0.19, 0.47], [0.4, 0.03, 0.09]]))
        ipf = IPFProcessor(crystal_symmetry="432", sample_symmetry="1")
        pts = np.asarray(ipf.get_ipf_points(R, [0.0, 0.0, 1.0]))
        rgb = np.asarray(ipf.get_ipf_color(R, [0.0, 0.0, 1.0]))
        assert pts.shape[0] == rgb.shape[0] == 2


@pytest.fixture
def sim_results():
    if not (MWE / "out.csv").exists():
        pytest.skip("mwe_data/out.csv not found")
    from graintrace.simulation_postprocessing import FieldFileNaming, SimulationResults

    naming = FieldFileNaming(
        prefix="out_element_centroid", index_width=4, sep="_", suffix=".csv"
    )
    return SimulationResults(
        block_csv=MWE / "out.csv", field_dir=MWE / "grid_out", field_naming=naming
    )


class TestExperimentGrainTracksById:
    def test_tracks_by_stable_id(self):
        exp = MWE / "synthetic_load_exp"
        if not exp.exists():
            pytest.skip("mwe_data/synthetic_load_exp not found")
        from graintrace.ipf_orientation_tracking import OrientationTracker

        csvs = [str(exp / f"expsyn_{t}time.csv") for t in (100, 130, 160)]
        tracks = OrientationTracker().experiment_grain_tracks_by_id(
            csvs, seed_grain_ids=[1, 2, 3]
        )
        assert len(tracks) == 3
        assert tracks[0].shape == (3, 3)


class TestSimulationGrainTracks:
    def test_shapes(self, sim_results):
        from graintrace.ipf_orientation_tracking import OrientationTracker

        gids = sim_results.grain_ids[:5]
        tracks = OrientationTracker().simulation_grain_tracks(
            sim_results, grain_ids=gids
        )
        assert len(tracks) == 5
        assert tracks[0].shape == (sim_results.n_steps, 3)

    def test_step_subset(self, sim_results):
        from graintrace.ipf_orientation_tracking import OrientationTracker

        tracks = OrientationTracker().simulation_grain_tracks(
            sim_results, grain_ids=sim_results.grain_ids[:2], steps=[1, 3, 5]
        )
        assert tracks[0].shape == (3, 3)


class TestPlotIpfOrientationTracking:
    def test_angle_convention_is_required(self):
        # angle_convention has no default (guards the Euler-as-MRP footgun): calling
        # without it must fail up front, before any heavy import.
        from graintrace.plot_postprocessing import plot_ipf_orientation_tracking

        with pytest.raises(TypeError):
            plot_ipf_orientation_tracking([])

    def test_writes_png(self, sim_results, tmp_path):
        pytest.importorskip("neml2")
        import matplotlib

        matplotlib.use("Agg")
        from graintrace.ipf_orientation_tracking import OrientationTracker
        from graintrace.plot_postprocessing import plot_ipf_orientation_tracking

        tracks = OrientationTracker().simulation_grain_tracks(
            sim_results, grain_ids=sim_results.grain_ids[:4], steps=range(1, 6)
        )
        out = plot_ipf_orientation_tracking(
            tracks,
            angle_convention="mrp",
            crystal_symmetry="432",
            output_folder=str(tmp_path),
            savefig_name="track.png",
        )
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0
