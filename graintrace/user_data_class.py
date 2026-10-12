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

"""User-facing dataclasses for similarity metrics, weighting, and rare criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

DistanceFunction = Callable[[np.ndarray, np.ndarray], float]
BatchDistanceFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SimilarityMetric:
    """A named feature-space distance metric and its required columns.

    Attributes:
        name: Label for the metric. Used only in error messages, the progress
            line, and the graph-stage checkpoint metadata, where resuming
            compares it against the stored value. It is **not** a column stem
            and does not appear in any output frame.
        feature_cols: Column names the metric consumes, **in the order its
            distance function expects them**. They must exist in the stage's
            input frame.

            The graph stage's reduction emits one ``<col>_mean`` per feature
            column (plus ``cluster_id``, ``n``, ``x``, ``y``, ``z``, and a
            ``<prefix>_norm_mean`` holding the mean per-point Frobenius norm
            whenever the feature columns contain a complete ``<prefix>_11``
            ... ``<prefix>_33`` set). The indicator stage that reads that frame
            must therefore name the ``_mean`` columns, not these -- which is
            why a two-stage Nye-norm run ends up selecting on
            ``nye_tensor_norm_mean_mean``: the graph stage emits
            ``nye_tensor_norm_mean``, and the indicator stage's own reduction
            suffixes it again.
        func: Scalar distance between two feature vectors, ``func(u, v) ->
            float``. Used by the hierarchical indicator stage.
        dist_edges: Optional vectorized form, ``dist_edges(X, edges) -> (E,)``,
            evaluating one distance per graph edge. When present the graph stage
            uses it instead of looping ``func``; both must agree numerically.
    """

    name: str
    feature_cols: List[str]  # required feature names
    func: DistanceFunction  # metric(u, v) -> float
    dist_edges: Optional[BatchDistanceFunction] = None  # vectorized: X,edges -> (E,)


@dataclass(frozen=True)
class WeightConfig:
    """Configuration for converting edge distances into graph weights.

    A larger weight means "more similar", so every mode is a decreasing function
    of the edge distance ``d``.

    Attributes:
        mode: Weight kernel, one of ``"inverse"``, ``"rbf"``, ``"exp"`` or
            ``"log_inv"``. Any other value raises ``ValueError``.
        eps: Regularizer that keeps a zero distance finite. Read by ``"inverse"``
            (:math:`1/(d + \\varepsilon)`) and ``"log_inv"``
            (:math:`-\\log(d + \\varepsilon)`); ignored otherwise.
        sigma: Length scale of ``"rbf"`` (:math:`e^{-(d/\\sigma)^{p}}`) and
            ``"exp"`` (:math:`e^{-d/\\sigma}`), **in the units of the metric** --
            degrees for a misorientation metric, MPa for a stress metric. When
            ``None``, it is estimated from the edge distances via ``sigma_auto``.
        sigma_auto: Estimation spec used only when ``mode`` is ``"rbf"`` or
            ``"exp"`` and ``sigma`` is ``None``. Only the ``"quantile"`` key is
            read; ``sigma`` becomes that quantile of the surviving edge
            distances.
        power: The exponent :math:`p` of ``"rbf"`` only; ``"inverse"``,
            ``"exp"`` and ``"log_inv"`` ignore it.
    """

    mode: str = "inverse"  # "inverse" | "rbf" | "exp" | "log_inv"
    eps: float = 1e-8  # used by inverse/log_inv
    sigma: Optional[float] = None  # used by rbf/exp
    sigma_auto: Optional[Dict[str, Any]] = (
        None  # estimate sigma from edge distances when sigma is None (rbf/exp)
    )
    power: float = 2.0  # rbf exponent: exp(-(d/sigma)^power)


@dataclass
class RareCriteria:
    """Select rare merged clusters via ``selector``, or by the size quantile.

    ``selector`` and the quantile fields are alternatives: when ``selector`` is
    given it decides entirely which clusters are rare, and the three fields
    below are not consulted.

    Attributes:
        selector: ``selector(df) -> ids``, taking the merged per-cluster stats
            frame and returning the ids of the rare clusters. The helpers in
            :mod:`graintrace.rare_criteria_selection_library` have this shape.
        size_quantile: Without a ``selector``, clusters in the bottom quantile
            of cluster size ``n`` are rare -- the built-in notion of "rare" is
            *small*, not *extreme*. Use a ``selector`` to rank by a field value.
        min_size: Absolute floor on cluster size, applied as well as the
            quantile, so single-point noise is not reported as a rare event.
        max_rare: Optional cap on how many clusters are returned, smallest
            first. ``None`` means no cap.
    """

    selector: Optional[
        Callable[[pd.DataFrame], Union[np.ndarray, List[int], List[str]]]
    ] = None

    size_quantile: float = 0.05  # default: bottom quantile by cluster size 'n'
    min_size: int = 1  # enforce absolute minimum
    max_rare: Optional[int] = None  # cap number of rare clusters (smallest first)
