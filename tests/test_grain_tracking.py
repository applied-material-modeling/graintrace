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

"""End-to-end grain tracking across load steps, on a tiny synthetic series.

One four-frame, four-grain series (``_write_series``) drives every test here, so
the happy path and the degenerate cases are asserted against the *same* input:

* frames 0-3: four grains drift a little and rotate a little, keeping identity;
* frame 2: grain 3 disappears (death) and grain 5 appears (birth);
* a separate two-into-one fixture covers the distance tie.

The file is split into two tiers deliberately.

``TestTrackIdentitiesById`` / ``TestComposeTracksDegenerate`` are pure
pandas/numpy and run everywhere, including CI.

``TestDetectSplitsAcrossFrames`` and ``TestGraphMatchedTracksAcrossFrames``
need ``neml2`` (and ``torch_geometric`` for the matcher) and self-skip. That is
not an oversight: every orientation primitive on the tracking path --
``euler_to_matrix``, ``mrp_to_matrix`` and ``symmetry_operators`` -- is a thin
wrapper over ``neml2.types`` / ``neml2.ops``, so an orientation-aware cross-step
link cannot be computed without it. See issue #23.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Four well-separated orientations (Bunge, degrees). Pairwise misorientations are
# far above the 5 deg cross-step tolerance used below, so a frame-to-frame link is
# decided by geometry, never by an orientation coincidence.
BASE = {
    1: (0.0, 0.0, 0.0, 10.0, 20.0, 30.0),
    2: (60.0, 0.0, 0.0, 80.0, 40.0, 120.0),
    3: (0.0, 60.0, 0.0, 200.0, 70.0, 250.0),
    4: (60.0, 60.0, 0.0, 300.0, 30.0, 60.0),
}
NEWCOMER = (120.0, 120.0, 0.0, 150.0, 60.0, 200.0)

# Per-frame drift: small next to the 60 um grain spacing, so every grain stays
# nearest to its own self in the next frame.
DRIFT_XY = 2.0
DRIFT_EUL = 0.4

COLUMNS = ["grain_id", "X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]


def _frame(step: int) -> pd.DataFrame:
    """The synthetic grain table for one load step."""
    rows = []
    for gid, (x, y, z, e0, e1, e2) in BASE.items():
        if gid == 3 and step >= 2:
            continue  # grain 3 dies at frame 2
        rows.append(
            [
                gid,
                x + step * DRIFT_XY,
                y + step * DRIFT_XY,
                z,
                15.0,
                e0 + step * DRIFT_EUL,
                e1,
                e2,
            ]
        )
    if step >= 2:  # grain 5 is born at frame 2
        x, y, z, e0, e1, e2 = NEWCOMER
        rows.append([5, x, y, z, 15.0, e0, e1, e2])
    return pd.DataFrame(rows, columns=COLUMNS)


def _write_series(tmp_path, n_steps: int = 4):
    """Write the four-frame series as FF-style CSVs; return the paths."""
    paths = []
    for step in range(n_steps):
        path = tmp_path / f"step{step:02d}.csv"
        _frame(step).to_csv(path, index=False)
        paths.append(str(path))
    return paths


def _expected_euler(gid: int, step: int) -> np.ndarray:
    """The Euler triple ``_frame`` writes for one grain at one step."""
    _, _, _, e0, e1, e2 = BASE[gid]
    return np.array([e0 + step * DRIFT_EUL, e1, e2])


class TestTrackIdentitiesById:
    """``experiment_grain_tracks_by_id`` over the four-frame series."""

    def test_happy_path_tracks_every_surviving_grain(self, tmp_path):
        from graintrace.ipf_orientation_tracking import OrientationTracker

        csvs = _write_series(tmp_path)
        tracks = OrientationTracker().experiment_grain_tracks_by_id(
            csvs, seed_grain_ids=[1, 2, 4]
        )

        assert len(tracks) == 3
        for track, gid in zip(tracks, (1, 2, 4)):
            assert track.shape == (4, 3)
            for step in range(4):
                np.testing.assert_allclose(track[step], _expected_euler(gid, step))

    def test_disappearing_grain_truncates_its_track(self, tmp_path):
        csvs = _write_series(tmp_path)
        from graintrace.ipf_orientation_tracking import OrientationTracker

        (track,) = OrientationTracker().experiment_grain_tracks_by_id(
            csvs, seed_grain_ids=[3]
        )
        # Present in frames 0 and 1 only; the track stops where the grain does.
        assert track.shape == (2, 3)
        np.testing.assert_allclose(track[1], _expected_euler(3, 1))

    def test_grain_absent_from_the_first_frame_yields_no_track(self, tmp_path):
        csvs = _write_series(tmp_path)
        from graintrace.ipf_orientation_tracking import OrientationTracker

        # Grain 5 is born at frame 2. Seeding is first-frame only, so it has no
        # track at all -- it is dropped, not returned as an empty array.
        assert (
            OrientationTracker().experiment_grain_tracks_by_id(csvs, seed_grain_ids=[5])
            == []
        )

    def test_default_seeds_are_the_first_frame_ids(self, tmp_path):
        csvs = _write_series(tmp_path)
        from graintrace.ipf_orientation_tracking import OrientationTracker

        tracks = OrientationTracker().experiment_grain_tracks_by_id(csvs)
        # Four grains in frame 0; grain 3's track is the short one.
        assert len(tracks) == 4
        assert sorted(t.shape[0] for t in tracks) == [2, 4, 4, 4]


class TestComposeTracksDegenerate:
    """Degenerate ``compose_tracks`` inputs: ties and unmatched rows."""

    def test_two_seeds_tying_on_one_row_both_follow_it(self):
        from graintrace.ipf_orientation_tracking import OrientationTracker

        # Rows 0 and 1 of step 0 both match row 2 of step 1 -- compose_tracks
        # resolves nothing, it chains whatever the matcher produced.
        tracks = OrientationTracker.compose_tracks(
            [[2, 2, 0], [1, 1, 1]], seed_ids=[0, 1]
        )
        assert tracks == [[0, 2, 1], [1, 2, 1]]

    def test_out_of_range_row_truncates_instead_of_raising(self):
        from graintrace.ipf_orientation_tracking import OrientationTracker

        # Step 1 has fewer rows than step 0 points at; the track stops there.
        assert OrientationTracker.compose_tracks([[5], [0]], seed_ids=[0]) == [[0, 5]]


class TestDetectSplitsAcrossFrames:
    """Orientation-aware cross-step lineage over the whole series (needs neml2)."""

    @staticmethod
    def _chain(csvs):
        from graintrace.fragmentation import FragmentationAnalyzer

        analyzer = FragmentationAnalyzer(symmetry="432")
        return [
            analyzer.detect_splits(
                csvs[k],
                csvs[k + 1],
                d_tol=15.0,
                theta_tol_deg=5.0,
                angle_type="degrees",
            )
            for k in range(len(csvs) - 1)
        ]

    def test_identities_are_preserved_frame_to_frame(self, tmp_path):
        pytest.importorskip("neml2")
        csvs = _write_series(tmp_path)

        frame0, frame1, frame2 = self._chain(csvs)

        # 0 -> 1: nothing happens, every grain matches itself.
        assert (frame0["n_matches"], frame0["n_births"], frame0["n_deaths"]) == (
            4,
            0,
            0,
        )
        pairs = {(tuple(c["a_ids"]), tuple(c["b_ids"])) for c in frame0["components"]}
        assert pairs == {((g,), (g,)) for g in (1, 2, 3, 4)}
        assert frame0["n_splits"] == 0 and frame0["n_merges"] == 0

        # 1 -> 2: grain 3 dies, grain 5 is born, the other three still match.
        assert frame1["n_matches"] == 3
        assert frame1["n_deaths"] == 1
        assert frame1["n_births"] == 1
        deaths = [c["a_ids"] for c in frame1["components"] if c["kind"] == "death"]
        births = [c["b_ids"] for c in frame1["components"] if c["kind"] == "birth"]
        assert deaths == [[3]] and births == [[5]]

        # 2 -> 3: steady again, now with four grains (1, 2, 4, 5).
        assert (frame2["n_matches"], frame2["n_births"], frame2["n_deaths"]) == (
            4,
            0,
            0,
        )

    def test_no_spurious_splits_anywhere_in_the_series(self, tmp_path):
        pytest.importorskip("neml2")
        csvs = _write_series(tmp_path)

        for result in self._chain(csvs):
            assert result["n_splits"] == 0
            assert result["n_merges"] == 0
            assert len(result["split_correspondences"]) == 0

    def test_distance_tie_between_two_parents_is_one_merge(self, tmp_path):
        pytest.importorskip("neml2")

        # Two step-A grains sitting at exactly the same distance (10 um) from the
        # single step-B grain, with the same orientation: the link is a genuine
        # tie, and both survive the filters, so the component is a 2 -> 1 merge
        # rather than an arbitrary winner plus a death.
        shared_euler = (10.0, 20.0, 30.0)
        a_rows = [
            [1, 0.0, 0.0, 0.0, 15.0, *shared_euler],
            [2, 20.0, 0.0, 0.0, 15.0, *shared_euler],
        ]
        b_rows = [[1, 10.0, 0.0, 0.0, 15.0, *shared_euler]]
        nodes_a = pd.DataFrame(a_rows, columns=COLUMNS)
        nodes_b = pd.DataFrame(b_rows, columns=COLUMNS)

        from graintrace.fragmentation import FragmentationAnalyzer

        result = FragmentationAnalyzer(symmetry="432").detect_splits(
            nodes_a, nodes_b, d_tol=15.0, theta_tol_deg=5.0, angle_type="degrees"
        )
        assert result["n_merges"] == 1
        assert result["n_deaths"] == 0
        (component,) = result["components"]
        assert sorted(component["a_ids"]) == [1, 2]
        assert component["b_ids"] == [1]


class TestGraphMatchedTracksAcrossFrames:
    """``experiment_grain_tracks``: matcher-driven tracking (needs the gnn extra)."""

    @staticmethod
    def _graph(coords, euler, edges):
        import torch
        from torch_geometric.data import Data

        x = torch.tensor(
            np.hstack([np.asarray(coords, float), np.asarray(euler, float)]),
            dtype=torch.float64,
        )
        graph = Data(
            x=x,
            edge_index=torch.tensor(np.asarray(edges).T, dtype=torch.long),
        )
        graph.feature_slices = {
            "X": (0, 1),
            "Y": (1, 2),
            "Z": (2, 3),
            "Eul0": (3, 4),
            "Eul1": (4, 5),
            "Eul2": (5, 6),
        }
        graph.feature_names = ["X", "Y", "Z", "Eul0", "Eul1", "Eul2"]
        graph.num_nodes = x.shape[0]
        return graph

    def test_tracks_follow_grains_through_renumbered_frames(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pytest.importorskip("neml2")
        from graintrace.ipf_orientation_tracking import OrientationTracker

        gids = [1, 2, 4]  # the three grains that live through every frame
        coords = [BASE[g][:3] for g in gids]
        euler = [BASE[g][3:] for g in gids]
        edges = [(0, 1), (1, 2), (0, 2)]

        # Each later frame renumbers its nodes, which is exactly the case the
        # matcher exists for: tracking has to recover the permutation.
        perms = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
        graphs = []
        for step, perm in enumerate(perms):
            shifted = [
                (x + step * DRIFT_XY, y + step * DRIFT_XY, z) for x, y, z in coords
            ]
            rotated = [(e0 + step * DRIFT_EUL, e1, e2) for e0, e1, e2 in euler]
            c = [None] * 3
            e = [None] * 3
            for i, j in enumerate(perm):
                c[j], e[j] = shifted[i], rotated[i]
            graphs.append(self._graph(c, e, [(perm[u], perm[v]) for u, v in edges]))

        tracks = OrientationTracker().experiment_grain_tracks(
            graphs,
            seed_grain_ids=[0, 1, 2],
            message_passing_iter=3,
            neighbor_selection_param={
                "lambda": 0.125,
                "iterations": 5,
                "tolerance": 1e-6,
                "topk": 3,
                "chunk": 16,
            },
            output_dir=str(tmp_path / "tracking"),
        )

        assert len(tracks) == 3
        for i, gid in enumerate(gids):
            assert tracks[i].shape == (3, 3)
            for step in range(3):
                # Node i of frame 0 is grain gid; the track must hold that
                # grain's Euler angles at every frame despite the renumbering.
                np.testing.assert_allclose(
                    tracks[i][step], _expected_euler(gid, step), atol=1e-9
                )

    def test_each_step_pair_gets_its_own_result_directory(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pytest.importorskip("neml2")
        from graintrace.ipf_orientation_tracking import OrientationTracker

        coords = [BASE[g][:3] for g in (1, 2, 4)]
        euler = [BASE[g][3:] for g in (1, 2, 4)]
        edges = [(0, 1), (1, 2), (0, 2)]
        graphs = [self._graph(coords, euler, edges) for _ in range(3)]

        out = tmp_path / "tracking"
        OrientationTracker().experiment_grain_tracks(
            graphs,
            message_passing_iter=1,
            neighbor_selection_param={"iterations": 2, "topk": 3, "chunk": 16},
            output_dir=str(out),
        )
        # Two consecutive pairs -> two directories, so neither overwrites the other.
        assert sorted(p.name for p in out.iterdir()) == ["pair_0", "pair_1"]
        assert (out / "pair_0" / "out_matches.csv").is_file()
        assert (out / "pair_1" / "out_matches.csv").is_file()

    def test_empty_graph_sequence_raises(self):
        pytest.importorskip("torch_geometric")
        from graintrace.ipf_orientation_tracking import OrientationTracker

        with pytest.raises(ValueError, match="non-empty"):
            OrientationTracker().experiment_grain_tracks([])
