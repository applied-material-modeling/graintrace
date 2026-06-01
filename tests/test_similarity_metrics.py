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

"""Tests for similarity_metric_library distance functions and SimilarityMetricLibrary."""
from __future__ import annotations

import numpy as np
import pytest
from graintrace.similarity_metric_library import (
    SimilarityMetricLibrary,
    diff_norm_3x3,
    diff_norm_3x3_batch,
    von_mises_stress_distance,
    von_mises_stress_distance_batch,
)
from graintrace.user_data_class import SimilarityMetric


class TestVonMisesDistance:
    def test_identical_stress_zero(self):
        u = np.array([100.0, 50.0, 30.0, 5.0, 3.0, 2.0])
        assert von_mises_stress_distance(u, u) == pytest.approx(0.0, abs=1e-10)

    def test_different_stress_positive(self):
        u = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v = np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        d = von_mises_stress_distance(u, v)
        assert d > 0.0
        assert isinstance(d, float)

    def test_batch_consistent_with_scalar(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 100, (20, 6))
        edges = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)])
        batch_vals = von_mises_stress_distance_batch(X, edges)
        for k, (i, j) in enumerate(edges):
            scalar = von_mises_stress_distance(X[i], X[j])
            assert batch_vals[k] == pytest.approx(scalar, rel=1e-6)


class TestDiffNorm3x3:
    def test_identical_tensor_zero(self):
        t = np.eye(3).ravel()
        assert diff_norm_3x3(t, t) == pytest.approx(0.0, abs=1e-12)

    def test_different_tensor_positive(self):
        u = np.eye(3).ravel()
        v = np.zeros(9)
        d = diff_norm_3x3(u, v)
        assert d == pytest.approx(np.sqrt(3), rel=1e-6)

    def test_batch_consistent_with_scalar(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (10, 9))
        edges = np.array([[0, 1], [2, 3], [4, 5]])
        batch_vals = diff_norm_3x3_batch(X, edges)
        for k, (i, j) in enumerate(edges):
            scalar = diff_norm_3x3(X[i], X[j])
            assert batch_vals[k] == pytest.approx(scalar, rel=1e-6)


class TestSimilarityMetricLibrary:
    def setup_method(self):
        self.lib = SimilarityMetricLibrary()

    def test_von_mises_stress_returns_metric(self):
        m = self.lib.von_mises_stress()
        assert isinstance(m, SimilarityMetric)
        assert m.name == "von_mises_stress"
        assert set(m.feature_cols) == {"sxx", "syy", "szz", "sxy", "syz", "sxz"}
        assert m.func is not None
        assert m.dist_edges is not None

    def test_von_mises_stress_custom_cols(self):
        cols = ["s11", "s22", "s33", "s12", "s23", "s13"]
        m = self.lib.von_mises_stress(cols=cols)
        assert m.feature_cols == cols

    def test_von_mises_func_callable(self):
        m = self.lib.von_mises_stress()
        u = np.array([100.0, 50.0, 30.0, 5.0, 3.0, 2.0])
        result = m.func(u, u)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_misorientation_returns_metric(self):
        m = self.lib.misorientation()
        assert isinstance(m, SimilarityMetric)
        assert m.name == "misorientation"
        assert len(m.feature_cols) == 3

    def test_misorientation_wrong_cols_raises(self):
        with pytest.raises(ValueError, match="exactly 3"):
            self.lib.misorientation(feature_cols=["a", "b"])

    def test_nye_tensor_norm_returns_metric(self):
        m = self.lib.nye_tensor_norm()
        assert isinstance(m, SimilarityMetric)
        assert m.name == "nye_tensor_norm"
        assert len(m.feature_cols) == 9
