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

"""Tests for graintrace.grain_graph_matching (first tests for this module).

``TestWriteResultsOutputPrefix`` is the regression suite for PR #40 (the prefix
was ignored and every run wrote ``run_*``). The rest covers the normal path:
message passing, neighbour selection, and ``match_grains`` end to end on a
six-grain graph whose second frame is a known relabelling of the first.

Everything here needs ``torch_geometric``, which lives in the ``gnn`` extra that
CI does not install, so every test gates on it. The two tests that exercise the
*default* message-passing scheme additionally need ``neml2`` -- that scheme calls
``euler_to_matrix``/``misorientation_matrix``. A custom scheme does not, which is
why the end-to-end coverage exists in both flavours. See issue #31.
"""

import json
import os

import pytest

# Six grains on a path, each with its own position and orientation.
COORDS = [
    (0.0, 0.0, 0.0),
    (10.0, 1.0, 0.0),
    (20.0, 0.0, 1.0),
    (30.0, 2.0, 0.0),
    (40.0, 0.0, 2.0),
    (50.0, 3.0, 0.0),
]
EULER = [
    (10.0, 20.0, 30.0),
    (80.0, 40.0, 120.0),
    (200.0, 70.0, 250.0),
    (300.0, 30.0, 60.0),
    (150.0, 60.0, 200.0),
    (45.0, 15.0, 95.0),
]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
# Frame B lists the same grains in this order: grain i of A is node PERM[i] of B.
PERM = [3, 0, 5, 1, 4, 2]
FEATURES = ("X", "Y", "Z", "Eul0", "Eul1", "Eul2")


def _frame(perm=None):
    """A six-grain graph, optionally with its nodes relabelled by ``perm``."""
    import torch
    from torch_geometric.data import Data

    order = list(range(len(COORDS))) if perm is None else perm
    rows = [None] * len(order)
    for grain, node in enumerate(order):
        rows[node] = list(COORDS[grain]) + list(EULER[grain])
    edges = [(order[u], order[v]) for u, v in EDGES]

    graph = Data(
        x=torch.tensor(rows, dtype=torch.float64),
        edge_index=torch.tensor(edges, dtype=torch.long).T,
    )
    graph.num_nodes = len(order)
    graph.feature_names = list(FEATURES)
    graph.feature_slices = {name: (k, k + 1) for k, name in enumerate(FEATURES)}
    return graph


def _neighbour_sum_scheme():
    """A message-passing scheme that needs no orientation maths (so no neml2)."""

    def factory():
        spec = {"required_node_features": list(FEATURES)}

        def phi_operator(F_src, _F_dst, _ctx, _k):
            return F_src

        return spec, phi_operator

    return factory


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


class TestMessagePassing:
    """Feature propagation over the (symmetrised) edge list."""

    def test_zero_iterations_just_packs_the_required_features(self, tmp_path):
        pytest.importorskip("torch_geometric")
        import torch

        graph = _frame()
        F, ctx = _matcher(tmp_path).message_passing(
            graph, _neighbour_sum_scheme(), message_passing_iter=0
        )

        assert torch.equal(F, graph.x)
        assert ctx["F_slices"] == {name: (k, k + 1) for k, name in enumerate(FEATURES)}
        assert ctx["required_node_features"] == list(FEATURES)
        # Edges are walked in both directions, so the symmetrised list is 2E.
        assert ctx["edge_src"].shape == (2 * len(EDGES),)

    def test_one_iteration_adds_the_sum_over_both_edge_directions(self, tmp_path):
        pytest.importorskip("torch_geometric")
        import torch

        graph = _frame()
        F, _ = _matcher(tmp_path).message_passing(
            graph, _neighbour_sum_scheme(), message_passing_iter=1
        )

        # phi returns F_src, so every node gains the sum of its neighbours.
        expected = graph.x.clone()
        for u, v in EDGES:
            expected[v] += graph.x[u]
            expected[u] += graph.x[v]
        assert torch.allclose(F, expected)

    def test_result_is_equivariant_under_relabelling(self, tmp_path):
        pytest.importorskip("torch_geometric")
        import torch

        matcher = _matcher(tmp_path)
        Fa, _ = matcher.message_passing(
            _frame(), _neighbour_sum_scheme(), message_passing_iter=3
        )
        Fb, _ = matcher.message_passing(
            _frame(PERM), _neighbour_sum_scheme(), message_passing_iter=3
        )
        assert torch.allclose(Fa, Fb[PERM])

    def test_non_data_input_is_rejected(self, tmp_path):
        pytest.importorskip("torch_geometric")
        with pytest.raises(TypeError, match="torch_geometric.data.Data"):
            _matcher(tmp_path).message_passing({}, _neighbour_sum_scheme())

    def test_edge_index_must_be_two_rows(self, tmp_path):
        pytest.importorskip("torch_geometric")
        graph = _frame()
        graph.edge_index = graph.edge_index.T
        with pytest.raises(ValueError, match=r"shape \[2, E\]"):
            _matcher(tmp_path).message_passing(graph, _neighbour_sum_scheme())

    def test_missing_feature_slices_is_rejected(self, tmp_path):
        pytest.importorskip("torch_geometric")
        graph = _frame()
        del graph.feature_slices
        with pytest.raises(ValueError, match="feature_slices missing"):
            _matcher(tmp_path).message_passing(graph, _neighbour_sum_scheme())

    def test_required_feature_absent_from_the_graph_is_named(self, tmp_path):
        pytest.importorskip("torch_geometric")

        def factory():
            return {"required_node_features": ["NotAFeature"]}, lambda *a: None

        with pytest.raises(KeyError, match="NotAFeature"):
            _matcher(tmp_path).message_passing(_frame(), factory)

    def test_empty_required_feature_list_is_rejected(self, tmp_path):
        pytest.importorskip("torch_geometric")

        def factory():
            return {"required_node_features": []}, lambda *a: None

        with pytest.raises(ValueError, match="non-empty list"):
            _matcher(tmp_path).message_passing(_frame(), factory)

    def test_spec_must_be_a_dict(self, tmp_path):
        pytest.importorskip("torch_geometric")

        def factory():
            return ["X"], lambda *a: None

        with pytest.raises(TypeError, match="spec: dict"):
            _matcher(tmp_path).message_passing(_frame(), factory)

    def test_negative_iteration_count_is_rejected(self, tmp_path):
        pytest.importorskip("torch_geometric")
        with pytest.raises(ValueError, match="must be >= 0"):
            _matcher(tmp_path).message_passing(
                _frame(), _neighbour_sum_scheme(), message_passing_iter=-1
            )

    def test_phi_operator_must_return_the_declared_shape(self, tmp_path):
        pytest.importorskip("torch_geometric")
        import torch

        def factory():
            spec = {"required_node_features": list(FEATURES)}
            return spec, lambda F_src, *_: torch.zeros(F_src.shape[0], 2)

        with pytest.raises(ValueError, match="must return tensor of shape"):
            _matcher(tmp_path).message_passing(
                _frame(), factory, message_passing_iter=1
            )

    def test_euler_slices_are_required_even_for_a_custom_scheme(self, tmp_path):
        pytest.importorskip("torch_geometric")

        # message_passing reads Eul0/1/2 unconditionally, before it looks at the
        # scheme's own required_node_features, so a scheme that never touches
        # orientation still cannot run on a graph without them. Pinned as
        # current behaviour, not endorsed; reported alongside issue #31.
        graph = _frame()
        graph.feature_slices = {"X": (0, 1), "Y": (1, 2), "Z": (2, 3)}

        def factory():
            return {"required_node_features": ["X"]}, lambda F_src, *_: F_src[:, :1]

        with pytest.raises(KeyError):
            _matcher(tmp_path).message_passing(graph, factory)


class TestNeighborSelection:
    """Assignment of A's grains to B's."""

    PARAM = {"lambda": 0.125, "iterations": 5, "tolerance": 1e-6, "topk": 6}

    def test_recovers_a_known_relabelling(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from graintrace.grain_graph_matching import GraphGrainMatcher

        matcher = GraphGrainMatcher(
            graph_a=_frame(), graph_b=_frame(PERM), output_dir=str(tmp_path)
        )
        Fa, _ = matcher.message_passing(
            matcher.graph_a, _neighbour_sum_scheme(), message_passing_iter=2
        )
        Fb, _ = matcher.message_passing(
            matcher.graph_b, _neighbour_sum_scheme(), message_passing_iter=2
        )
        out = matcher.neighbor_selection(
            Fa,
            Fb,
            neighbor_selection_cost_function=GraphGrainMatcher.default_neighbor_selection_cost_function,
            neighbor_selection_param=self.PARAM,
        )

        assert out["a_to_b"].tolist() == PERM
        assert out["matches"].tolist() == [[i, j] for i, j in enumerate(PERM)]
        # An exact relabelling is a zero-distance match.
        assert out["costs"].abs().max().item() == pytest.approx(0.0, abs=1e-9)
        assert out["params"]["topk"] == 6
        assert len(out["mean_cost_history"]) == self.PARAM["iterations"]

    def test_feature_dimension_mismatch_is_rejected(self, tmp_path):
        pytest.importorskip("torch_geometric")
        import torch

        from graintrace.grain_graph_matching import GraphGrainMatcher

        matcher = GraphGrainMatcher(
            graph_a=_frame(), graph_b=_frame(PERM), output_dir=str(tmp_path)
        )
        with pytest.raises(ValueError, match="Feature dims mismatch"):
            matcher.neighbor_selection(
                torch.zeros(6, 4),
                torch.zeros(6, 3),
                neighbor_selection_cost_function=GraphGrainMatcher.default_neighbor_selection_cost_function,
            )

    def test_non_tensor_features_are_rejected(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from graintrace.grain_graph_matching import GraphGrainMatcher

        matcher = GraphGrainMatcher(
            graph_a=_frame(), graph_b=_frame(PERM), output_dir=str(tmp_path)
        )
        with pytest.raises(TypeError, match="must be tensors"):
            matcher.neighbor_selection(
                [[0.0]],
                [[0.0]],
                neighbor_selection_cost_function=GraphGrainMatcher.default_neighbor_selection_cost_function,
            )


class TestDefaultNeighborSelectionCostFunction:
    """The base term and the neighbour-consistency adjustment."""

    @staticmethod
    def _features():
        import torch

        Fa = torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]], dtype=torch.float64)
        Fb = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]], dtype=torch.float64)
        return Fa, Fb

    def test_isolated_nodes_cost_the_squared_feature_distance(self):
        pytest.importorskip("torch_geometric")
        from graintrace.grain_graph_matching import GraphGrainMatcher

        Fa, Fb = self._features()
        cost = GraphGrainMatcher.default_neighbor_selection_cost_function(
            Fa,
            Fb,
            [set(), set(), set()],
            [set(), set(), set()],
            [-1, -1, -1],
            0,
            0,
            0.5,
        )
        assert cost == pytest.approx(1.0**2 + 2.0**2)

    def test_unmatched_neighbours_leave_the_base_cost_alone(self):
        pytest.importorskip("torch_geometric")
        from graintrace.grain_graph_matching import GraphGrainMatcher

        Fa, Fb = self._features()
        neigh_a = [{1}, {0}, set()]
        neigh_b = [{1}, {0}, set()]
        cost = GraphGrainMatcher.default_neighbor_selection_cost_function(
            Fa, Fb, neigh_a, neigh_b, [-1, -1, -1], 0, 0, 0.5
        )
        assert cost == pytest.approx(1.0**2 + 2.0**2)

    def test_a_matched_shared_neighbour_adjusts_the_cost(self):
        pytest.importorskip("torch_geometric")
        from graintrace.grain_graph_matching import GraphGrainMatcher

        Fa, Fb = self._features()
        neigh_a = [{1}, {0}, set()]
        neigh_b = [{1}, {0}, set()]
        base = GraphGrainMatcher.default_neighbor_selection_cost_function(
            Fa, Fb, neigh_a, neigh_b, [-1, -1, -1], 0, 0, 0.5
        )
        # A's node 1 is already matched to B's node 1, which neighbours B's 0.
        # Fa[1] == Fb[1], so the consistency term contributes zero magnitude.
        agreeing = GraphGrainMatcher.default_neighbor_selection_cost_function(
            Fa, Fb, neigh_a, neigh_b, [-1, 1, -1], 0, 0, 0.5
        )
        assert agreeing == pytest.approx(base)

        # Match A's node 1 to B's node 2 instead: the pair now disagrees by
        # ||Fa[1] - Fb[2]||^2 = 9 + 9, scaled by lambda and the neighbour count.
        neigh_b_alt = [{2}, set(), {0}]
        disagreeing = GraphGrainMatcher.default_neighbor_selection_cost_function(
            Fa, Fb, neigh_a, neigh_b_alt, [-1, 2, -1], 0, 0, 0.5
        )
        # Assert on the magnitude only. The sign of this term is a separate open
        # question the maintainer excluded from the current round of fixes, so
        # pinning it here would block whichever way it is settled.
        assert abs(disagreeing - base) == pytest.approx(0.5 * 18.0)


class TestMatchGrainsEndToEnd:
    """The whole pipeline: two frames in, correspondence files out."""

    PARAM = {"lambda": 0.125, "iterations": 5, "tolerance": 1e-6, "topk": 6}

    def test_custom_scheme_recovers_the_relabelling_and_writes_results(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pd = pytest.importorskip("pandas")
        from graintrace.grain_graph_matching import GraphGrainMatcher

        matcher = GraphGrainMatcher(
            graph_a=_frame(),
            graph_b=_frame(PERM),
            output_dir=str(tmp_path),
            output_prefix="pair_0_",
        )
        result = matcher.match_grains(
            message_passing_function=_neighbour_sum_scheme(),
            message_passing_iter=2,
            neighbor_selection_param=self.PARAM,
        )

        assert result["match"]["a_to_b"].tolist() == PERM
        assert result["graph_a_features"].shape == (6, len(FEATURES))

        mapping = pd.read_csv(tmp_path / "pair_0_a_to_b.csv")
        assert mapping["j_in_B"].tolist() == PERM
        with open(tmp_path / "pair_0_meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["message_passing_required_features"] == list(FEATURES)

    def test_default_scheme_recovers_the_relabelling(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pytest.importorskip("neml2")
        from graintrace.grain_graph_matching import GraphGrainMatcher

        matcher = GraphGrainMatcher(
            graph_a=_frame(), graph_b=_frame(PERM), output_dir=str(tmp_path)
        )
        result = matcher.match_grains(
            message_passing_iter=2,
            neighbor_selection_param=self.PARAM,
            angle_convention="bunge",
            angle_type="degrees",
            symmetry="432",
        )

        assert result["match"]["a_to_b"].tolist() == PERM
        # The default scheme packs X, Y, Z and the misorientation accumulator M.
        assert result["graph_a_ctx"]["F_slices"]["M"] == (3, 4)
        assert result["graph_a_features"].shape == (6, 4)
