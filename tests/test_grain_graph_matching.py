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
