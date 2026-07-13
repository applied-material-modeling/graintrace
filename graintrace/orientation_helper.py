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

from math import sqrt
from typing import Optional, Union

import numpy as np
import torch

inds = ["11", "22", "33", "23", "13", "12"]
facts = [1.0, 1.0, 1.0, sqrt(2.0), sqrt(2.0), sqrt(2.0)]


# Orientation primitives in pure torch (plus neml2.ops.symmetry). Euler->Bunge
# (phi1, Phi, phi2) mapping:
#   bunge : used directly
#   kocks : (psi, theta, phi) -> (psi + pi/2, theta, pi/2 - phi)
#   roe   : (psi, theta, phi) -> (psi + pi/2, theta, phi - pi/2)

_HALF_PI = torch.pi / 2.0


def _to_bunge(euler: torch.Tensor, convention: str) -> tuple:
    """Map Euler angles (radians) in the given convention to Bunge angles."""
    convention = convention.lower()
    a, b, c = euler[..., 0], euler[..., 1], euler[..., 2]
    if convention == "bunge":
        return a, b, c
    if convention == "kocks":
        return a + _HALF_PI, b, _HALF_PI - c
    if convention == "roe":
        return a + _HALF_PI, b, c - _HALF_PI
    raise ValueError(f"Unknown angle_convention: {convention!r}")


def _from_bunge(phi1: torch.Tensor, Phi: torch.Tensor, phi2: torch.Tensor,
                convention: str) -> torch.Tensor:
    """Inverse of :func:`_to_bunge` — Bunge angles (radians) back to convention."""
    convention = convention.lower()
    if convention == "bunge":
        a, b, c = phi1, Phi, phi2
    elif convention == "kocks":
        a, b, c = phi1 - _HALF_PI, Phi, _HALF_PI - phi2
    elif convention == "roe":
        a, b, c = phi1 - _HALF_PI, Phi, phi2 + _HALF_PI
    else:
        raise ValueError(f"Unknown angle_convention: {convention!r}")
    return torch.stack((a, b, c), dim=-1)


def euler_to_matrix(
    euler: Union[np.ndarray, list, torch.Tensor],
    convention: str = "bunge",
    angle_type: str = "degrees",
) -> torch.Tensor:
    """Convert Euler angles to rotation matrices.

    Bunge angles are applied as a Z-X-Z composition of ``MRP.from_axis_angle``
    rotations; the matrix comes from ``euler_rodrigues``.

    Args:
        euler: (..., 3) Euler angles.
        convention: 'bunge', 'kocks', or 'roe'.
        angle_type: 'degrees' or 'radians'.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    from neml2.types import MRP, Vec, Scalar, euler_rodrigues, compose

    e = torch.as_tensor(euler, dtype=torch.float64)
    if angle_type == "degrees":
        e = torch.deg2rad(e)
    elif angle_type != "radians":
        raise ValueError(f"Unknown angle_type: {angle_type!r}")

    phi1, Phi, phi2 = _to_bunge(e, convention)
    zaxis = Vec(torch.tensor([0.0, 0.0, 1.0], dtype=e.dtype, device=e.device))
    xaxis = Vec(torch.tensor([1.0, 0.0, 0.0], dtype=e.dtype, device=e.device))
    r = compose(
        compose(
            MRP.from_axis_angle(zaxis, Scalar(phi1.contiguous())),
            MRP.from_axis_angle(xaxis, Scalar(Phi.contiguous())),
        ),
        MRP.from_axis_angle(zaxis, Scalar(phi2.contiguous())),
    )
    # euler_rodrigues gives the active rotation; the Bunge orientation matrix g is
    # its transpose (matches the v2 convention).
    return euler_rodrigues(r).data.transpose(-2, -1)


def matrix_to_euler(
    M: torch.Tensor,
    convention: str = "bunge",
    angle_type: str = "degrees",
) -> torch.Tensor:
    """Convert rotation matrices to Euler angles (inverse of :func:`euler_to_matrix`).

    Args:
        M: (..., 3, 3) rotation matrices.
        convention: 'bunge', 'kocks', or 'roe'.
        angle_type: 'degrees' or 'radians'.

    Returns:
        (..., 3) Euler angles.
    """
    M = torch.as_tensor(M, dtype=torch.float64)
    Phi = torch.arccos(torch.clamp(M[..., 2, 2], -1.0, 1.0))
    sP = torch.sin(Phi)

    phi1 = torch.atan2(M[..., 2, 0], -M[..., 2, 1])
    phi2 = torch.atan2(M[..., 0, 2], M[..., 1, 2])

    # Gimbal lock (Phi ~ 0 or pi): only phi1 + phi2 is defined; put it all in phi1.
    gimbal = sP.abs() < 1e-8
    if torch.any(gimbal):
        phi1_g = torch.atan2(M[..., 0, 1], M[..., 0, 0])
        phi1 = torch.where(gimbal, phi1_g, phi1)
        phi2 = torch.where(gimbal, torch.zeros_like(phi2), phi2)

    euler = _from_bunge(phi1, Phi, phi2, convention)
    if angle_type == "degrees":
        return torch.rad2deg(euler)
    return euler


def matrix_to_mrp(M: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert rotation matrices (..., 3, 3) to neml2 v3 MRP (..., 3)."""
    from neml2 import types as _t

    M = torch.as_tensor(M, dtype=torch.float64).contiguous()
    return _t.MRP.from_matrix(_t.R2(M, 0)).data


def euler_to_mrp(
    euler: Union[np.ndarray, list, torch.Tensor],
    convention: str = "bunge",
    angle_type: str = "degrees",
) -> torch.Tensor:
    """Convert Euler angles to neml2 v3 modified Rodrigues parameters (..., 3)."""
    return matrix_to_mrp(euler_to_matrix(euler, convention, angle_type))


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert a scalar-first unit quaternion (..., 4) ``(w, x, y, z)`` to a
    rotation matrix (..., 3, 3), via neml2 ``quaternion_rotation_matrix``.
    """
    from neml2 import types as _t

    q = torch.as_tensor(q, dtype=torch.float64).contiguous()
    return _t.quaternion_rotation_matrix(_t.Quaternion(q, 0)).data


def mrp_to_matrix(p: Union[np.ndarray, list, torch.Tensor]) -> torch.Tensor:
    """Convert neml2 v3 MRP orientations (..., 3) to rotation matrices (..., 3, 3)."""
    from neml2 import types as _t

    p = torch.as_tensor(p, dtype=torch.float64).contiguous()
    return _t.euler_rodrigues(_t.MRP(p, 0)).data


def mrp_to_euler(
    p: Union[np.ndarray, list, torch.Tensor],
    convention: str = "bunge",
    angle_type: str = "degrees",
) -> torch.Tensor:
    """Convert neml2 v3 MRP orientations (..., 3) to Euler angles (..., 3)."""
    return matrix_to_euler(mrp_to_matrix(p), convention, angle_type)


def symmetry_operators(symmetry: str) -> torch.Tensor:
    """Crystal symmetry operators as rotation matrices (nops, 3, 3)."""
    from neml2.ops import symmetry as _symmetry

    ops = _symmetry(symmetry).data
    return ops.reshape(-1, 3, 3).to(torch.float64)


def misorientation_matrix(
    R1: torch.Tensor,
    R2: torch.Tensor,
    symmetry: str = "1",
    angle_type: str = "degrees",
) -> torch.Tensor:
    """Misorientation angle between two sets of rotation matrices under symmetry.

    Args:
        R1, R2: (N, 3, 3) rotation matrices.
        symmetry: crystal symmetry in orbifold notation.
        angle_type: 'degrees' or 'radians' for the returned angle.

    Returns:
        (N,) misorientation angles.
    """
    symmetry_ops = symmetry_operators(symmetry).to(device=R1.device, dtype=R1.dtype)

    dR = torch.matmul(R1, R2.transpose(-2, -1))

    options = torch.matmul(
        torch.matmul(symmetry_ops.unsqueeze(0), dR.unsqueeze(1)).unsqueeze(2),
        symmetry_ops.transpose(-2, -1).unsqueeze(0).unsqueeze(0),
    )

    rad_mis = (
        torch.arccos(
            torch.clamp(
                (options.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0, -1.0, 1.0
            )
        )
        .reshape(R1.shape[0], -1)
        .min(dim=1)
    ).values

    if angle_type == "degrees":
        return torch.rad2deg(rad_mis)
    return rad_mis


def move_to_fundamental_zone(R: torch.Tensor, symmetry: str = "1") -> torch.Tensor:
    """Reduce rotation matrices to the fundamental zone under crystal symmetry.

    Picks, per orientation, the symmetry-equivalent with the smallest rotation
    angle (largest trace of ``O @ R``).

    Args:
        R: (..., 3, 3) rotation matrices.
        symmetry: crystal symmetry in orbifold notation.

    Returns:
        (..., 3, 3) reduced rotation matrices.
    """
    ops = symmetry_operators(symmetry).to(device=R.device, dtype=R.dtype)  # (nops, 3, 3)
    cand = torch.matmul(ops, R.unsqueeze(-3))  # (..., nops, 3, 3)
    trace = cand.diagonal(dim1=-2, dim2=-1).sum(-1)  # (..., nops)
    idx = torch.argmax(trace, dim=-1)  # (...,)
    idx_exp = idx[..., None, None, None].expand(idx.shape + (1, 3, 3))
    return torch.gather(cand, -3, idx_exp).squeeze(-3)


def average_orientation(
    euler: Union[np.ndarray, list, torch.Tensor],
    weights: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    convention: str = "bunge",
    angle_type: str = "degrees",
    symmetry: str = "1",
) -> torch.Tensor:
    """Symmetry-aware weighted mean of a set of Euler orientations.

    Every orientation is brought into the symmetry-equivalent closest to the
    highest-weight orientation, matrix-averaged (weighted), and re-projected onto
    SO(3) via SVD (the polar/chordal mean); the result is converted back to Euler
    angles in the given convention/units.

    Args:
        euler: (N, 3) Euler angles, or (3,) for a single orientation.
        weights: (N,) non-negative weights (default: uniform).
        convention: 'bunge', 'kocks', or 'roe'.
        angle_type: 'degrees' or 'radians'.
        symmetry: crystal symmetry in orbifold notation.

    Returns:
        (3,) averaged Euler angles in the given convention/units.
    """
    e = torch.as_tensor(euler, dtype=torch.float64)
    if e.ndim == 1:
        e = e.unsqueeze(0)
    n = e.shape[0]

    if weights is None:
        w = torch.ones(n, dtype=torch.float64)
    else:
        w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if w.shape[0] != n:
        raise ValueError(
            f"weights length ({w.shape[0]}) must match number of orientations ({n})."
        )

    R = euler_to_matrix(e, convention, angle_type)  # (N, 3, 3)
    if n == 1:
        return matrix_to_euler(R[0], convention, angle_type)

    ops = symmetry_operators(symmetry).to(dtype=R.dtype)  # (nops, 3, 3)

    # Reference = highest-weight orientation; align every orientation to it by
    # picking the symmetry op O maximizing <ref, O @ R_i>.
    ref = R[int(torch.argmax(w))]                         # (3, 3)
    cand = torch.matmul(ops.unsqueeze(0), R.unsqueeze(1))  # (N, nops, 3, 3)
    score = torch.einsum("ij,nkij->nk", ref, cand)        # (N, nops)
    best = torch.argmax(score, dim=1)                     # (N,)
    R_aligned = cand[torch.arange(n), best]              # (N, 3, 3)

    # Weighted matrix mean, re-projected onto SO(3).
    M = torch.einsum("n,nij->ij", w, R_aligned) / w.sum()
    U, _, Vh = torch.linalg.svd(M)
    Ravg = U @ Vh
    if torch.det(Ravg) < 0:                               # guard against reflection
        U = U.clone()
        U[:, -1] = -U[:, -1]
        Ravg = U @ Vh

    return matrix_to_euler(Ravg, convention, angle_type)


def misorientation(
    e1: Union[np.ndarray, list],
    e2: Union[np.ndarray, list],
    angle_convention: str = "kocks",
    angle_type: str = "degrees",
    symmetry: str = "1",
) -> torch.Tensor:
    """Compute misorientation between two sets of Euler angles

    Args:
        e1: Nx3 array of Euler angles
        e2: Nx3 array of Euler angles

    Keyword Args:
        angle_convention (str): 'kocks', 'bunge', or 'roe' (default: 'kocks')
        angle_type (str): 'degrees' or 'radians' (default: 'degrees')
        symmetry (str): crystal symmetry in orbifold notation (default: '1')

    Returns:
        Nx1 array of misorientation angles
    """
    e1 = torch.as_tensor(e1, dtype=torch.float64)
    e2 = torch.as_tensor(e2, dtype=torch.float64)

    if e1.ndim == 1:
        e1 = e1.unsqueeze(0)
    if e2.ndim == 1:
        e2 = e2.unsqueeze(0)

    R1 = euler_to_matrix(e1, angle_convention, angle_type)
    R2 = euler_to_matrix(e2, angle_convention, angle_type)

    return misorientation_matrix(R1, R2, symmetry, angle_type=angle_type)


if __name__ == "__main__":

    def check(e1, e2, expected, tol=1e-3):
        val = misorientation(
            e1,
            e2,
            angle_convention="bunge",
            angle_type="degrees",
            symmetry="432",
        )

        val = val.item() if isinstance(val, torch.Tensor) else val

        print(f"e1={e1}, e2={e2} -> {val:.4f} deg")

        if expected is not None:
            assert abs(val - expected) < tol, f"Expected {expected} deg, got {val} deg"

    check([0, 0, 0], [0, 0, 0], 0.0)
    check([0, 0, 0], [90, 0, 0], 0.0)
    check([0, 0, 0], [5, 0, 0], 5.0)
    check([12.0, 0, 27.0], [102.0, 0, 27.0], 0.0)

    val = misorientation(
        [10, 20, 30],
        [40, 50, 60],
        angle_convention="kocks",
        angle_type="degrees",
        symmetry="432",
    )
    val = val.item() if isinstance(val, torch.Tensor) else val
    print(f"random -> {val:.4f} deg")
    assert val > 0.0, "Random orientations should not give zero misorientation"

    val = misorientation(
        [0, 0, 0],
        [45.0, 45.0, 0.0],
        angle_convention="kocks",
        angle_type="degrees",
        symmetry="432",
    )
    val = val.item() if isinstance(val, torch.Tensor) else val
    print(f"max test -> {val:.4f} deg")
    assert 62.5 <= val <= 63.0, "Cubic max misorientation out of range"

    print("\nAll misorientation tests PASSED.")


def load_orientation_matrices(
    df: "pd.DataFrame",
    field: Optional[str] = "O",
) -> torch.Tensor:
    """Load per-row rotation matrices (..., 3, 3) from a dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing orientation data.
        field (str, optional): Column prefix for the 3x3 orientation matrix
            (columns ``{field}{i}{j}``). If ``None``, the first 9 columns are
            used positionally. Defaults to 'O'.

    Returns:
        torch.Tensor: (n, 3, 3) rotation matrices.
    """
    if field is None:
        n = len(df)
        matrix = torch.tensor(df.iloc[:, :9].values, dtype=torch.float32)
        matrix = matrix.view(n, 3, 3)
    else:
        matrix = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(
                    torch.tensor(df[field + str(i + 1) + str(j + 1)].dropna().values)
                )
            matrix.append(torch.stack(row, dim=-1))
        matrix = torch.stack(matrix, dim=-2)
    return matrix


def load_orientations(
    df: "pd.DataFrame",
    field: Optional[str] = "O",
) -> torch.Tensor:
    """Load orientations as neml2 v3 modified Rodrigues parameters (n, 3).

    Args:
        df (pd.DataFrame): Dataframe containing orientation data.
        field (str, optional): Column name for orientation matrix. Defaults to 'O'.

    Returns:
        torch.Tensor: (n, 3) neml2 MRP orientations.
    """
    return matrix_to_mrp(load_orientation_matrices(df, field))


# Backwards-compatible alias.
load_orientations_mrp = load_orientations


def matrix_to_quat(M: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices (..., 3, 3) to scalar-first unit quaternions
    (..., 4) ``(w, x, y, z)``, via neml2 (``MRP.from_matrix`` -> ``to_quaternion``).
    """
    from neml2 import types as _t

    M = torch.as_tensor(M, dtype=torch.float64).contiguous()
    return _t.to_quaternion(_t.MRP.from_matrix(_t.R2(M, 0))).data


def load_strains(
    df: "pd.DataFrame",
    field: str = "eKen",
    factor: float = 1e-6,
) -> torch.Tensor:
    """Load strains from dataframe and convert to torch tensor.

    Args:
        df (pd.DataFrame): Dataframe containing strain data.
        field (str, optional): Column name for strain data. Defaults to 'eKen'.
        factor (float, optional): Conversion factor. Defaults to 1e6.

    Returns:
        torch.Tensor: Tensor containing strain data.
    """
    return torch.stack(
        [
            torch.tensor(df[field + i].dropna().values) * factor * f
            for i, f in zip(inds, facts)
        ],
        dim=-1,
    )


def load_weights(
    df: "pd.DataFrame",
    field: str = "GrainRadius",
) -> torch.Tensor:
    """Load the grain volumes and convert to a torch tensor of weights.

    Args:
        df (pd.DataFrame): Dataframe containing grain volume data.
        field (str, optional): Column name for grain volume data. Defaults to 'GrainRadius'.

    Returns:
        torch.Tensor: Normalized weight tensor.
    """
    weights = torch.tensor(df[field].dropna().values) ** 3.0
    return weights / torch.sum(weights)
