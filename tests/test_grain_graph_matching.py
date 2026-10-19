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

"""Tests for graintrace.grain_graph_matching (first tests for this module)."""

import json
import os

import pytest


def _matcher(tmp_path, prefix=None):
    """Build a GraphGrainMatcher with stub graphs (write_results never reads them)."""
    from graintrace.grain_graph_matching import GraphGrainMatcher

    class _Stub:  # pylint: disable=too-few-public-methods
        feature_names = ["X", "Y", "Z"]

    kwargs = {} if prefix is None else {"output_prefix": prefix}
    return GraphGrainMatcher(
        graph_a=_Stub(), graph_b=_Stub(), output_dir=str(tmp_path), **kwargs
    )


def _result(cost):
    """Minimal result payload of the shape match_grains hands to write_results."""
    import torch

    return {
        "graph_a_features": torch.zeros(2, 3),
        "graph_b_features": torch.ones(2, 3),
        "graph_a_ctx": {"required_node_features": ["X", "Y", "Z"]},
        "match": {
            "matches": torch.tensor([[0, 1]]),
            "costs": torch.tensor([float(cost)]),
            "a_to_b": torch.tensor([1, 0]),
            "params": {"lambda": 0.125},
            "mean_cost_history": [float(cost)],
        },
    }


class TestWriteResultsOutputPrefix:
    """write_results must honour output_prefix so runs do not overwrite."""

    EXPECTED = ("matches.csv", "a_to_b.csv", "Fa.pt", "Fb.pt", "meta.json")

    def test_prefix_is_used_for_every_file(self, tmp_path):
        pytest.importorskip("torch_geometric")
        _matcher(tmp_path, "step01_").write_results(_result(3))
        assert sorted(os.listdir(tmp_path)) == sorted(
            f"step01_{name}" for name in self.EXPECTED
        )

    def test_default_prefix_is_out(self, tmp_path):
        pytest.importorskip("torch_geometric")
        _matcher(tmp_path).write_results(_result(3))
        assert sorted(os.listdir(tmp_path)) == sorted(
            f"out_{name}" for name in self.EXPECTED
        )

    def test_empty_prefix_gives_bare_names(self, tmp_path):
        pytest.importorskip("torch_geometric")
        _matcher(tmp_path, "").write_results(_result(3))
        assert sorted(os.listdir(tmp_path)) == sorted(self.EXPECTED)

    def test_two_prefixes_do_not_overwrite_each_other(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pd = pytest.importorskip("pandas")
        _matcher(tmp_path, "step01_").write_results(_result(3))
        _matcher(tmp_path, "step02_").write_results(_result(7))

        first = pd.read_csv(tmp_path / "step01_matches.csv")["cost"].tolist()
        second = pd.read_csv(tmp_path / "step02_matches.csv")["cost"].tolist()
        assert first == [3.0]
        assert second == [7.0]

    def test_output_dir_is_created(self, tmp_path):
        pytest.importorskip("torch_geometric")
        nested = tmp_path / "a" / "b"
        _matcher(nested, "p_").write_results(_result(1))
        assert (nested / "p_matches.csv").is_file()

    def test_written_contents_round_trip(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pd = pytest.importorskip("pandas")
        torch = pytest.importorskip("torch")

        _matcher(tmp_path, "r_").write_results(_result(5))

        matches = pd.read_csv(tmp_path / "r_matches.csv")
        assert matches["i_in_A"].tolist() == [0]
        assert matches["j_in_B"].tolist() == [1]
        assert matches["cost"].tolist() == [5.0]

        mapping = pd.read_csv(tmp_path / "r_a_to_b.csv")
        assert mapping["i_in_A"].tolist() == [0, 1]
        assert mapping["j_in_B"].tolist() == [1, 0]

        assert torch.load(tmp_path / "r_Fa.pt").shape == (2, 3)
        assert torch.load(tmp_path / "r_Fb.pt").shape == (2, 3)

        with open(tmp_path / "r_meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["neighbor_selection_params"] == {"lambda": 0.125}
        assert meta["mean_cost_history"] == [5.0]
        assert meta["graph_a_feature_names"] == ["X", "Y", "Z"]
        assert meta["message_passing_required_features"] == ["X", "Y", "Z"]


def _chain_matcher(tmp_path, n=4):
    """Matcher over two identical 1-D chain graphs of ``n`` nodes."""
    import torch

    from graintrace.grain_graph_matching import GraphGrainMatcher

    class _Graph:  # pylint: disable=too-few-public-methods
        def __init__(self):
            self.edge_index = torch.tensor(
                [list(range(n - 1)), list(range(1, n))], dtype=torch.long
            )
            self.num_nodes = n

    return GraphGrainMatcher(
        graph_a=_Graph(), graph_b=_Graph(), output_dir=str(tmp_path)
    )


def _run_selection(tmp_path, tolerance, iterations, n=4):
    """Run neighbor_selection on a fixed, well-separated feature pair."""
    import torch

    from graintrace.grain_graph_matching import GraphGrainMatcher

    torch.manual_seed(42)
    Fa = torch.arange(n, dtype=torch.float64).reshape(n, 1) * 10.0
    Fb = Fa.clone() + 0.25

    return _chain_matcher(tmp_path, n).neighbor_selection(
        Fa,
        Fb,
        GraphGrainMatcher.default_neighbor_selection_cost_function,
        {
            "lambda": 0.125,
            "iterations": iterations,
            "tolerance": tolerance,
            "topk": n,
            "chunk": 16,
        },
    )


class TestNeighborSelectionTolerance:
    """``tolerance`` must act as a convergence criterion, not a dead echo (#44)."""

    def test_large_tolerance_stops_after_two_rounds(self, tmp_path):
        """A tolerance larger than any cost change exits at the first comparison."""
        pytest.importorskip("torch_geometric")
        out = _run_selection(tmp_path, tolerance=1e9, iterations=25)
        # Round 1 records a cost; round 2 compares against it and breaks.
        assert len(out["mean_cost_history"]) == 2

    def test_zero_tolerance_runs_the_full_budget(self, tmp_path):
        """``tolerance=0.0`` disables the early exit, preserving the old behaviour."""
        pytest.importorskip("torch_geometric")
        out = _run_selection(tmp_path, tolerance=0.0, iterations=7)
        assert len(out["mean_cost_history"]) == 7

    def test_tolerance_does_not_change_the_converged_assignment(self, tmp_path):
        """Exiting early must give the same matching as running the full budget."""
        pytest.importorskip("torch_geometric")
        early = _run_selection(tmp_path, tolerance=1e-6, iterations=25)
        full = _run_selection(tmp_path, tolerance=0.0, iterations=25)
        assert early["a_to_b"].tolist() == full["a_to_b"].tolist()
        assert len(early["mean_cost_history"]) < len(full["mean_cost_history"])

    def test_tolerance_is_still_echoed_in_params(self, tmp_path):
        """The echoed value keeps reporting what was requested."""
        pytest.importorskip("torch_geometric")
        out = _run_selection(tmp_path, tolerance=0.5, iterations=5)
        assert out["params"]["tolerance"] == 0.5


class TestPositionChannelsArePreserved:
    """The zero X/Y/Z messages are an invariant, not a dead branch (#45).

    ``message_passing`` updates additively (``F = F + m``), so a zero message
    is the identity and is exactly what carries each node's measured centroid
    through every round untouched. Filling those channels with a
    source-to-destination offset -- the change #45 proposes -- would
    accumulate into the coordinates and break the ``cdist`` candidate pool.
    These tests fail if anyone makes that change.
    """

    @staticmethod
    def _graph(n=5):
        pytest.importorskip("torch_geometric")
        pytest.importorskip("neml2")
        import torch
        from torch_geometric.data import Data

        torch.manual_seed(42)
        xyz = torch.arange(n, dtype=torch.float64).reshape(n, 1) * torch.ones(
            1, 3, dtype=torch.float64
        )
        xyz = xyz + torch.tensor([0.0, 10.0, 100.0], dtype=torch.float64)
        euler = torch.tensor(
            [[13.0 * i, 7.0 * i, 29.0 * i] for i in range(n)], dtype=torch.float64
        )
        graph = Data(
            x=torch.cat([xyz, euler], dim=1),
            edge_index=torch.tensor(
                [list(range(n - 1)), list(range(1, n))], dtype=torch.long
            ),
        )
        graph.feature_slices = {
            "X": (0, 1),
            "Y": (1, 2),
            "Z": (2, 3),
            "Eul0": (3, 4),
            "Eul1": (4, 5),
            "Eul2": (5, 6),
        }
        return graph, xyz

    def _run(self, iters, n=5):
        graph, xyz = self._graph(n)

        from graintrace.grain_graph_matching import GraphGrainMatcher

        F, ctx = GraphGrainMatcher.message_passing(
            None,
            graph,
            GraphGrainMatcher.default_message_passing_function,
            message_passing_iter=iters,
            use_default=True,
        )
        return F, ctx, xyz

    @pytest.mark.parametrize("iters", [1, 3, 6])
    def test_centroids_survive_every_round_unchanged(self, iters):
        import torch

        F, ctx, xyz = self._run(iters)
        for axis, col in (("X", 0), ("Y", 1), ("Z", 2)):
            a, b = ctx["F_slices"][axis]
            assert torch.allclose(F[:, a:b], xyz[:, col : col + 1]), (
                f"{axis} drifted after {iters} rounds; the zero message is the "
                "identity of an additive update and must stay zero"
            )

    def test_the_message_channel_does_move(self):
        """Positive control: M must accumulate, or the test above proves nothing."""
        import torch

        F, ctx, _ = self._run(3)
        a, b = ctx["F_slices"]["M"]
        assert not torch.allclose(F[:, a:b], torch.zeros_like(F[:, a:b]))
