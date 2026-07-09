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

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any, Union
import numpy as np
import pandas as pd

DistanceFunction = Callable[[np.ndarray, np.ndarray], float]
BatchDistanceFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SimilarityMetric:
    name: str
    feature_cols: List[str]  # required feature names
    func: DistanceFunction  # metric(u, v) -> float
    dist_edges: Optional[BatchDistanceFunction] = None  # vectorized: X,edges -> (E,)


@dataclass(frozen=True)
class WeightConfig:
    mode: str = "inverse"  # "inverse" | "rbf" | "exp" | "identity" | "log_inv"
    eps: float = 1e-8  # used by inverse/log_inv
    sigma: Optional[float] = None  # used by rbf/exp
    sigma_auto: Optional[Dict[str, Any]] = (
        None  # estimate sigma from edge distances when sigma is None (rbf/exp)
    )
    power: float = 2.0  # rbf exponent: exp(-(d/sigma)^power)


@dataclass
class RareCriteria:
    """Select rare merged clusters via `selector`, or the built-in size-quantile defaults."""

    selector: Optional[
        Callable[[pd.DataFrame], Union[np.ndarray, List[int], List[str]]]
    ] = None

    size_quantile: float = 0.05  # default: bottom quantile by cluster size 'n'
    min_size: int = 1  # enforce absolute minimum
    max_rare: Optional[int] = None  # cap number of rare clusters (smallest first)
