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


class TestIpfTrackPoints:
    """``get_ipf_track_points``: one sector operator for the whole track."""

    @staticmethod
    def _track(crossing_deg=6.0, steps=13):
        """Crystal direction R^T z sweeping across the [001]-[101] edge (y = 0)."""
        import torch

        from graintrace.orientation_helper import euler_to_matrix

        # R^T z = (sinP sin p2, sinP cos p2, cosP): p2 = 90 +/- a puts the
        # direction at (sinP cos a, -/+ sinP sin a, cosP), crossing y = 0.
        a = np.linspace(-crossing_deg, crossing_deg, steps)
        eul = np.stack([np.full(steps, 23.0), np.full(steps, 30.0), 90.0 + a], 1)
        return euler_to_matrix(torch.as_tensor(eul), "bunge", "degrees")

    def test_track_crosses_the_edge_continuously(self):
        pytest.importorskip("neml2")
        from graintrace.ipf_postprocess import IPFProcessor

        ipf = IPFProcessor(crystal_symmetry="432", sample_symmetry="1")
        r = self._track()
        indep = np.asarray(ipf.get_ipf_points(r, [0.0, 0.0, 1.0]))
        track = np.asarray(ipf.get_ipf_track_points(r, [0.0, 0.0, 1.0]))
        assert track.shape == indep.shape == (13, 2)
        # step 0 is exactly where the per-orientation reduction puts it
        assert np.allclose(track[0], indep[0], atol=1e-12)
        # independent reduction bounces off the edge (y >= 0 everywhere) ...
        assert np.all(indep[:, 1] >= -1e-9)
        # ... the track keeps going: y is monotonic and changes sign
        dy = np.diff(track[:, 1])
        assert np.all(dy > 0) or np.all(dy < 0)
        assert track[0, 1] * track[-1, 1] < 0
        # step size stays uniform (no jump)
        step = np.linalg.norm(np.diff(track, axis=0), axis=1)
        assert step.max() < 1.5 * step.min()

    def test_variant_reported_steps_do_not_jump(self):
        """Steps reported in other symmetry variants land on the same track."""
        pytest.importorskip("neml2")
        import torch

        from graintrace.ipf_postprocess import IPFProcessor
        from graintrace.orientation_helper import symmetry_operators

        ipf = IPFProcessor(crystal_symmetry="432", sample_symmetry="1")
        r = self._track()
        ops = symmetry_operators("432")
        mixed = r.clone()
        mixed[1::2] = torch.matmul(mixed[1::2], ops[5])  # R S: same orientation
        a = np.asarray(ipf.get_ipf_track_points(r, [0.0, 0.0, 1.0]))
        b = np.asarray(ipf.get_ipf_track_points(mixed, [0.0, 0.0, 1.0]))
        assert np.allclose(a, b, atol=1e-9)

    def test_empty_track(self):
        pytest.importorskip("neml2")
        import torch

        from graintrace.ipf_postprocess import IPFProcessor

        ipf = IPFProcessor(crystal_symmetry="432", sample_symmetry="1")
        pts = ipf.get_ipf_track_points(torch.zeros(0, 3, 3), [0.0, 0.0, 1.0])
        assert tuple(pts.shape) == (0, 2)


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
