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
from .user_data_class import SimilarityMetric
import functools
from dataclasses import dataclass


# Resident orientation cache (GPU): keep the full orientation array on-device and
# gather per chunk, avoiding repeated PCIe copies. Only the latest array is held.
_RESIDENT_X = {"key": None, "tensor": None}


def _resident_orientations(X: np.ndarray, device, dtype):
    import torch

    key = (id(X), X.shape, str(device), str(dtype))
    if _RESIDENT_X["key"] != key:
        _RESIDENT_X["tensor"] = torch.as_tensor(X, dtype=dtype, device=device)
        _RESIDENT_X["key"] = key
    return _RESIDENT_X["tensor"]


def von_mises_stress_distance(u: np.ndarray, v: np.ndarray) -> float:
    sxx_u, syy_u, szz_u, sxy_u, syz_u, sxz_u = u
    sxx_v, syy_v, szz_v, sxy_v, syz_v, sxz_v = v

    def von_mises(sxx, syy, szz, sxy, syz, sxz):
        term1 = ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) / 2
        term2 = 3 * (sxy**2 + syz**2 + sxz**2)
        return np.sqrt(term1 + term2)

    vm_u = von_mises(sxx_u, syy_u, szz_u, sxy_u, syz_u, sxz_u)
    vm_v = von_mises(sxx_v, syy_v, szz_v, sxy_v, syz_v, sxz_v)

    return float(abs(vm_u - vm_v) / (abs(vm_u) + abs(vm_v) + 1e-8))


def von_mises_stress_distance_batch(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    sxx = X[:, 0]
    syy = X[:, 1]
    szz = X[:, 2]
    sxy = X[:, 3]
    syz = X[:, 4]
    sxz = X[:, 5]
    term1 = ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) / 2.0
    term2 = 3.0 * (sxy**2 + syz**2 + sxz**2)
    vm = np.sqrt(term1 + term2)

    I = edges[:, 0]
    J = edges[:, 1]
    a = vm[I]
    b = vm[J]
    return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8)


def misorientation_distance(
    u: np.ndarray,
    v: np.ndarray,
    angle_convention: str = "bunge",
    input_angle_type: str = "degrees",
    symmetry: str = "432",
    output_unit: str = "degrees",
) -> float:
    import torch
    from .orientation_helper import (
        euler_to_matrix,
        mrp_to_matrix,
        misorientation_matrix,
    )

    valid_euler = {"bunge", "kocks", "roe"}
    valid_special = {"mrp"}

    if angle_convention not in (valid_euler | valid_special):
        raise ValueError(f"Unsupported angle_convention: {angle_convention}")
    if input_angle_type not in {"degrees", "radians"}:
        raise ValueError(f"Unsupported input_angle_type: {input_angle_type}")
    if output_unit not in {"degrees", "radians"}:
        raise ValueError(f"Unsupported output_unit: {output_unit}")

    e1 = torch.as_tensor(u, dtype=torch.float64).reshape(1, 3)
    e2 = torch.as_tensor(v, dtype=torch.float64).reshape(1, 3)

    if angle_convention == "mrp":
        r1 = mrp_to_matrix(e1)
        r2 = mrp_to_matrix(e2)
    else:
        r1 = euler_to_matrix(e1, angle_convention, input_angle_type)
        r2 = euler_to_matrix(e2, angle_convention, input_angle_type)

    mis = misorientation_matrix(r1, r2, symmetry, angle_type=output_unit)

    return float(mis.detach().cpu().numpy().reshape(-1)[0])


@dataclass(frozen=True)
class MisorientationDistEdges:
    angle_convention: str = "bunge"
    input_angle_type: str = "degrees"
    symmetry: str = "432"
    output_unit: str = "degrees"
    device: str = "cpu"  # "cpu" or "cuda"/"cuda:N"

    def __call__(self, X: np.ndarray, edges: np.ndarray) -> np.ndarray:
        import torch
        from .orientation_helper import (
            euler_to_matrix,
            mrp_to_matrix,
            misorientation_matrix,
        )

        valid_euler = {"bunge", "kocks", "roe"}
        valid_special = {"mrp"}

        if self.angle_convention not in (valid_euler | valid_special):
            raise ValueError(f"Unsupported angle_convention: {self.angle_convention}")
        if self.input_angle_type not in {"degrees", "radians"}:
            raise ValueError(f"Unsupported input_angle_type: {self.input_angle_type}")
        if self.output_unit not in {"degrees", "radians"}:
            raise ValueError(f"Unsupported output_unit: {self.output_unit}")

        I = edges[:, 0]
        J = edges[:, 1]

        if str(self.device) != "cpu":
            # Gather edge endpoints from the on-device resident orientation array.
            Xg = _resident_orientations(X, self.device, torch.float64)
            Ig = torch.as_tensor(np.ascontiguousarray(I), device=self.device)
            Jg = torch.as_tensor(np.ascontiguousarray(J), device=self.device)
            e1 = Xg[Ig]
            e2 = Xg[Jg]
        else:
            e1 = torch.as_tensor(X[I], dtype=torch.float64, device=self.device)
            e2 = torch.as_tensor(X[J], dtype=torch.float64, device=self.device)

        if self.angle_convention == "mrp":
            r1 = mrp_to_matrix(e1)
            r2 = mrp_to_matrix(e2)
        else:
            r1 = euler_to_matrix(e1, self.angle_convention, self.input_angle_type)
            r2 = euler_to_matrix(e2, self.angle_convention, self.input_angle_type)

        mis = misorientation_matrix(r1, r2, self.symmetry, angle_type=self.output_unit)

        return mis.detach().cpu().numpy().astype(np.float64, copy=False)


def make_misorientation_dist_edges(
    angle_convention: str = "bunge",
    input_angle_type: str = "degrees",
    symmetry: str = "432",
    output_unit: str = "degrees",
    device: str = "cpu",
) -> MisorientationDistEdges:
    return MisorientationDistEdges(
        angle_convention=angle_convention,
        input_angle_type=input_angle_type,
        symmetry=symmetry,
        output_unit=output_unit,
        device=device,
    )


def diff_norm_3x3(u: np.ndarray, v: np.ndarray) -> float:
    nye_u = u.reshape((3, 3))
    nye_v = v.reshape((3, 3))

    diff = nye_u - nye_v
    diff_norm = np.linalg.norm(diff, ord="fro")

    return diff_norm


def diff_norm_3x3_batch(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    I = edges[:, 0]
    J = edges[:, 1]

    D = X[I] - X[J]
    diff_norm = np.sqrt(np.sum(D * D, axis=1))
    return diff_norm


class SimilarityMetricLibrary:
    """Metrics returning SimilarityMetric objects (distance functions where smaller = more similar).

    Each method defines the required feature columns and a distance func. The order
    of feature_cols maps to index order: u[i] corresponds to feature_cols[i]. Distance
    funcs must be top-level (picklable). Provide dist_edges for a vectorized batch path.
    """

    def von_mises_stress(
        self,
        cols: Optional[List[str]] = None,
    ) -> SimilarityMetric:

        default = ["sxx", "syy", "szz", "sxy", "syz", "sxz"]

        use_cols = cols if cols is not None else default

        return SimilarityMetric(
            name="von_mises_stress",
            feature_cols=use_cols,
            func=von_mises_stress_distance,
            dist_edges=von_mises_stress_distance_batch,
        )

    def misorientation(
        self,
        feature_cols: Optional[List[str]] = None,
        symmetry: str = "432",
        input_angle_type: str = "degrees",
        angle_convention: str = "mrp",
        output_unit: str = "degrees",
        device: str = "cpu",
    ) -> SimilarityMetric:

        if feature_cols is None:
            feature_cols = ["ori_rodrigues_x", "ori_rodrigues_y", "ori_rodrigues_z"]

        if len(feature_cols) != 3:
            raise ValueError("misorientation requires exactly 3 feature columns")

        return SimilarityMetric(
            name="misorientation",
            feature_cols=list(feature_cols),
            func=functools.partial(
                misorientation_distance,
                angle_convention=angle_convention,
                input_angle_type=input_angle_type,
                symmetry=symmetry,
                output_unit=output_unit,
            ),
            dist_edges=make_misorientation_dist_edges(
                symmetry=symmetry,
                input_angle_type=input_angle_type,
                angle_convention=angle_convention,
                output_unit=output_unit,
                device=device,
            ),
        )

    def nye_tensor_norm(self, cols: Optional[List[str]] = None) -> SimilarityMetric:

        cols = [
            "nye_tensor_11",
            "nye_tensor_12",
            "nye_tensor_13",
            "nye_tensor_21",
            "nye_tensor_22",
            "nye_tensor_23",
            "nye_tensor_31",
            "nye_tensor_32",
            "nye_tensor_33",
        ]

        return SimilarityMetric(
            name="nye_tensor_norm",
            feature_cols=cols,
            func=diff_norm_3x3,
            dist_edges=diff_norm_3x3_batch,
        )
