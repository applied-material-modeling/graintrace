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

"""Tests for user_data_class dataclasses."""
from __future__ import annotations

import numpy as np
import pytest
from graintrace.user_data_class import RareCriteria, SimilarityMetric, WeightConfig


def _dummy_func(u, v):
    return float(np.linalg.norm(u - v))


def test_similarity_metric_fields():
    m = SimilarityMetric(name="test", feature_cols=["a", "b"], func=_dummy_func)
    assert m.name == "test"
    assert m.feature_cols == ["a", "b"]
    assert m.func is _dummy_func
    assert m.dist_edges is None


def test_similarity_metric_with_batch():
    def batch(X, e):
        return np.zeros(len(e))

    m = SimilarityMetric(
        name="test", feature_cols=["x"], func=_dummy_func, dist_edges=batch
    )
    assert m.dist_edges is batch


def test_weight_config_defaults():
    cfg = WeightConfig()
    assert cfg.mode == "inverse"
    assert cfg.eps == pytest.approx(1e-8)
    assert cfg.sigma is None
    assert cfg.sigma_auto is None
    assert cfg.power == pytest.approx(2.0)


def test_weight_config_rbf():
    cfg = WeightConfig(mode="rbf", sigma=1.0, power=1.5)
    assert cfg.mode == "rbf"
    assert cfg.sigma == pytest.approx(1.0)
    assert cfg.power == pytest.approx(1.5)


def test_weight_config_frozen():
    cfg = WeightConfig()
    with pytest.raises(Exception):
        cfg.mode = "rbf"  # type: ignore[misc]


def test_rare_criteria_defaults():
    rc = RareCriteria()
    assert rc.selector is None
    assert rc.size_quantile == pytest.approx(0.05)
    assert rc.min_size == 1
    assert rc.max_rare is None


def test_rare_criteria_custom():
    rc = RareCriteria(size_quantile=0.1, min_size=5, max_rare=3)
    assert rc.size_quantile == pytest.approx(0.1)
    assert rc.min_size == 5
    assert rc.max_rare == 3
