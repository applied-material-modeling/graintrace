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

from typing import List, Optional

import numpy as np
import pandas as pd


def select_smallest_cluster(
    df: pd.DataFrame,
    nsmallest: int = 10,
    min_size: int = 1,
) -> np.ndarray:
    d = df[df["n"] >= min_size]
    d = d.nsmallest(nsmallest, "n")
    return d["cluster_label"].to_numpy()


def select_highest_von_mises_from_components(
    df: pd.DataFrame,
    k: int = 1,
    required_cols: Optional[List[str]] = None,
    min_size: int = 1,
) -> np.ndarray:

    if required_cols is None:
        required_cols = [
            "sxx_mean_mean",
            "syy_mean_mean",
            "szz_mean_mean",
            "sxy_mean_mean",
            "syz_mean_mean",
            "sxz_mean_mean",
        ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for von Mises: {missing}")

    d = df[df["n"] >= min_size].copy()
    for c in required_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=required_cols)

    sxx, syy, szz, sxy, syz, sxz = (d[c].to_numpy(dtype=float) for c in required_cols)

    vm = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + sxz**2)
    )

    d["_von_mises_from_means"] = vm
    return d.nlargest(k, "_von_mises_from_means")["cluster_label"].to_numpy()


def select_highest_scalar(
    df: pd.DataFrame,
    required_cols: str,
    k: int = 1,
    min_size: int = 1,
) -> np.ndarray:
    col = required_cols
    if col not in df.columns:
        raise ValueError(f"Missing required column for scalar selection: {col}")

    d = df[df["n"] >= min_size].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[col])

    return d.nlargest(k, col)["cluster_label"].to_numpy()


def select_highest_norm_3x3_tensor(
    df: pd.DataFrame,
    k: int = 1,
    required_cols: Optional[List[str]] = None,
    min_size: int = 1,
) -> np.ndarray:
    if required_cols is None:
        required_cols = [
            "t11_mean_mean",
            "t12_mean_mean",
            "t13_mean_mean",
            "t21_mean_mean",
            "t22_mean_mean",
            "t23_mean_mean",
            "t31_mean_mean",
            "t32_mean_mean",
            "t33_mean_mean",
        ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for 3x3 tensor norm: {missing}")

    d = df[df["n"] >= min_size].copy()
    for c in required_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=required_cols)

    t11, t12, t13, t21, t22, t23, t31, t32, t33 = (
        d[c].to_numpy(dtype=float) for c in required_cols
    )

    norm = np.sqrt(
        t11**2 + t12**2 + t13**2 + t21**2 + t22**2 + t23**2 + t31**2 + t32**2 + t33**2
    )

    d["_norm_3x3"] = norm
    return d.nlargest(k, "_norm_3x3")["cluster_label"].to_numpy()
