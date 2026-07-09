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

import torch

import tqdm

from ..orientation_helper import (
    euler_to_matrix,
    matrix_to_euler,
    matrix_to_mrp,
    matrix_to_quat,
    quat_to_matrix,
    misorientation_matrix,
)


def average_rotations(e, angle_convention="kocks", angle_type="radians"):
    """Average a set of Euler angles via quaternion (eigenvector) mean.

    Args:
        e: Nx...x3 array of Euler angles
        angle_convention (str): 'kocks', 'bunge', or 'roe'
        angle_type (str): 'degrees' or 'radians'

    Returns:
        tuple ``(mrp, euler)``: averaged orientation as a neml2 v3 MRP (..., 3)
        and as Euler angles (..., 3) in the given convention.
    """
    R = euler_to_matrix(e, angle_convention, angle_type)

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
