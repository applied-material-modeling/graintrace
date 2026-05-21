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
    feature_cols: List[str]  # requried feature names
    func: DistanceFunction  # metric(u, v) -> float
    dist_edges: Optional[BatchDistanceFunction] = None
    # X,edges -> (E,) batch version of func, used for vectorized computations.


@dataclass(frozen=True)
class WeightConfig:
    mode: str = "inverse"  # "inverse" | "rbf" | "exp" | "identity" | "log_inv"
    eps: float = 1e-8  # used by inverse/log_inv
    sigma: Optional[float] = None  # used by rbf/exp
    sigma_auto: Optional[Dict[str, Any]] = (
        None  # if sigma is None and mode is rbf/exp, use this config to estimate sigma from graph edge distances
    )
    power: float = 2.0  # rbf exponent: exp(-(d/sigma)^power)


@dataclass
class RareCriteria:
    """
    Define how to select rare *merged* clusters.
    Either provide `selector` or use the built-in defaults.
    """

    selector: Optional[
        Callable[[pd.DataFrame], Union[np.ndarray, List[int], List[str]]]
    ] = None

    # Built-in default: pick bottom quantile by 'n' (cluster size) from indicator_clusters_df
    size_quantile: float = 0.05  # bottom 5%
    min_size: int = 1  # enforce absolute minimum
    max_rare: Optional[int] = None  # cap number of rare clusters (smallest first)
