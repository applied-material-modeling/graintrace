"""Tests for rare_criteria_selection_library functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from graintrace.rare_criteria_selection_library import (
    select_highest_norm_3x3_tensor,
    select_highest_scalar,
    select_highest_von_mises_from_components,
    select_smallest_cluster,
)


def _make_cluster_df(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cluster_label": np.arange(n),
            "n": rng.integers(1, 100, n),
            "sxx_mean_mean": rng.normal(100, 20, n),
            "syy_mean_mean": rng.normal(50, 10, n),
            "szz_mean_mean": rng.normal(30, 5, n),
            "sxy_mean_mean": rng.normal(0, 5, n),
            "syz_mean_mean": rng.normal(0, 5, n),
            "sxz_mean_mean": rng.normal(0, 5, n),
            "scalar_val": rng.uniform(0, 1, n),
            **{f"t{i}{j}_mean_mean": rng.normal(0, 1, n) for i in range(1, 4) for j in range(1, 4)},
        }
    )


class TestSelectSmallest:
    def test_returns_array(self):
        df = _make_cluster_df()
        result = select_smallest_cluster(df, nsmallest=3)
        assert isinstance(result, np.ndarray)
        assert len(result) <= 3

    def test_min_size_filter(self):
        df = _make_cluster_df()
        result = select_smallest_cluster(df, nsmallest=100, min_size=50)
        labels = result.tolist()
        for lbl in labels:
            assert df.loc[df["cluster_label"] == lbl, "n"].values[0] >= 50

    def test_returns_smallest(self):
        df = pd.DataFrame({"cluster_label": [0, 1, 2], "n": [5, 20, 1]})
        result = select_smallest_cluster(df, nsmallest=1)
        assert result[0] == 2


class TestSelectHighestVonMises:
    def test_returns_k_labels(self):
        df = _make_cluster_df()
        result = select_highest_von_mises_from_components(df, k=2)
        assert len(result) == 2

    def test_missing_cols_raises(self):
        df = pd.DataFrame({"cluster_label": [0], "n": [1]})
        with pytest.raises(ValueError, match="Missing"):
            select_highest_von_mises_from_components(df, k=1)


class TestSelectHighestScalar:
    def test_returns_k_labels(self):
        df = _make_cluster_df()
        result = select_highest_scalar(df, required_cols="scalar_val", k=3)
        assert len(result) <= 3

    def test_missing_col_raises(self):
        df = _make_cluster_df()
        with pytest.raises(ValueError, match="Missing"):
            select_highest_scalar(df, required_cols="nonexistent")

    def test_picks_highest(self):
        df = pd.DataFrame(
            {"cluster_label": [0, 1, 2], "n": [1, 1, 1], "scalar_val": [0.1, 0.9, 0.5]}
        )
        result = select_highest_scalar(df, required_cols="scalar_val", k=1)
        assert result[0] == 1


class TestSelectHighestNorm3x3:
    def test_returns_k_labels(self):
        df = _make_cluster_df()
        result = select_highest_norm_3x3_tensor(df, k=2)
        assert len(result) == 2

    def test_missing_cols_raises(self):
        df = pd.DataFrame({"cluster_label": [0], "n": [1]})
        with pytest.raises(ValueError, match="Missing"):
            select_highest_norm_3x3_tensor(df, k=1)
