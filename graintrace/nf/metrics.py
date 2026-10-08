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

"""Orientation averaging and misorientation metrics for NF grids."""

from __future__ import annotations

import torch

import tqdm

from ..orientation_helper import (
    euler_to_matrix,
    matrix_to_euler,
    matrix_to_mrp,
    matrix_to_quat,
    quat_to_matrix,
    misorientation_matrix,
    symmetry_operators,
)


def fold_to_reference_variant(R, symmetry):
    """Fold rotations onto the symmetry variant closest to the first of the set.

    A crystal orientation has one rotation matrix per symmetry operator (24 for
    cubic ``432``), and a reconstruction may report neighbouring voxels of one
    grain in different variants. Averaging across variants is meaningless, so
    every member is first replaced by the symmetry-equivalent that is closest
    (largest Frobenius inner product) to the reference member ``R[..., 0, :, :]``.

    Args:
        R: ``(..., N, 3, 3)`` rotation matrices; the set is averaged over axis -3.
        symmetry (str): crystal symmetry in orbifold notation; ``"1"`` disables
            folding and returns ``R`` unchanged.

    Returns:
        ``(..., N, 3, 3)`` rotation matrices, all in a common symmetry variant.
    """
    if symmetry == "1" or R.shape[-3] < 2:
        return R
    ops = symmetry_operators(symmetry).to(device=R.device, dtype=R.dtype)
    # (..., N, nops, 3, 3): every symmetry-equivalent of every member
    cand = torch.matmul(ops, R.unsqueeze(-3))
    ref = R[..., 0, :, :]
    # Frobenius inner product with the reference; maximal == smallest rotation apart
    score = (cand * ref[..., None, None, :, :]).sum(dim=(-2, -1))
    best = score.argmax(dim=-1)
    index = best[..., None, None, None].expand(*best.shape, 1, 3, 3)
    return torch.gather(cand, -3, index).squeeze(-3)


def average_rotations(e, angle_convention="kocks", angle_type="radians", symmetry="1"):
    """Average a set of Euler angles via quaternion (eigenvector) mean.

    Args:
        e: Nx...x3 array of Euler angles
        angle_convention (str): 'kocks', 'bunge', or 'roe'
        angle_type (str): 'degrees' or 'radians'
        symmetry (str): crystal symmetry in orbifold notation (e.g. ``"432"``).
            Members are folded into a common symmetry variant before averaging,
            so a grain whose voxels are reported in different variants still
            yields the correct mean. ``"1"`` (the default) disables folding and
            gives the plain quaternion eigenvector (Markley) mean.

    Returns:
        tuple ``(mrp, euler)``: averaged orientation as a neml2 v3 MRP (..., 3)
        and as Euler angles (..., 3) in the given convention.
    """
    R = euler_to_matrix(e, angle_convention, angle_type)

    R = fold_to_reference_variant(R, symmetry)

    Q = matrix_to_quat(R)

    QQt = torch.matmul(Q.transpose(-2, -1), Q)

    _, eigvecs = torch.linalg.eigh(QQt)

    new_Q = eigvecs[..., -1]

    new_R = quat_to_matrix(new_Q)

    new_mrp = matrix_to_mrp(new_R)
    new_eulers = matrix_to_euler(new_R, angle_convention, angle_type)

    return new_mrp, new_eulers


def batched_norm(v1, v2, norm, chunk_size, **kwargs):
    """Compute ``norm(v1, v2)`` in chunks of ``chunk_size`` to save memory."""
    n = v1.shape[0]
    results = []
    for start in tqdm.trange(0, n, chunk_size, desc="Precalculating norms"):
        end = min(start + chunk_size, n)
        res_chunk = norm(v1[start:end], v2[start:end], **kwargs)
        results.append(res_chunk)
    return torch.cat(results, dim=0)


def misorientation(
    e1, e2, angle_convention="kocks", angle_type="degrees", symmetry="1"
):
    """Compute misorientation angles between two sets of Euler angles.

    Args:
        e1, e2: Nx3 arrays of Euler angles
        angle_convention (str): 'kocks', 'bunge', or 'roe'
        angle_type (str): 'degrees' or 'radians'
        symmetry (str): crystal symmetry in orbifold notation

    Returns:
        Nx1 array of misorientation angles in degrees
    """
    R1 = euler_to_matrix(e1, angle_convention, angle_type)
    R2 = euler_to_matrix(e2, angle_convention, angle_type)

    return misorientation_matrix(R1, R2, symmetry, angle_type=angle_type)
