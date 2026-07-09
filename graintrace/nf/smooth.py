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

from ..orientation_helper import (
    euler_to_matrix,
    matrix_to_euler,
    matrix_to_quat,
    quat_to_matrix,
    move_to_fundamental_zone,
)

from .image import get_neighbor_indices, connectivity_options


def smooth(
    data, connectivity=6, symmetry="1", angle_convention="bunge", angle_type="radians"
):
    """Smooth voxel orientations via a neighbor quaternion mean.

    Args:
        data (nx,ny,nz,7) tensor: fixed grid [phase, Eul1, Eul2, Eul3, X, Y, Z]
        connectivity (int): 6 or 26
        symmetry (str): crystal symmetry in orbifold notation
        angle_convention (str): 'kocks', 'bunge', or 'roe'
        angle_type (str): 'degrees' or 'radians'
    """
    print("Smoothing...")
    R = euler_to_matrix(data[..., 1:4], angle_convention, angle_type)

    R = move_to_fundamental_zone(R, symmetry)

    Q1 = matrix_to_quat(R)

    not_void = data[..., 0] > 0

    offsets = connectivity_options[connectivity]
    Xk, Yk, Zk, valid = get_neighbor_indices(offsets, data.shape[:3])

    valid = valid & not_void.unsqueeze(0)

    Q = Q1[Xk, Yk, Zk].permute(1, 2, 3, 4, 0)
    mask = valid.permute(1, 2, 3, 0).unsqueeze(-2)
    Q_masked = Q * mask
    QQt = torch.matmul(Q_masked, Q_masked.transpose(-2, -1))

    _, eigvecs = torch.linalg.eigh(QQt)

    new_Q = eigvecs[..., -1]

    new_R = quat_to_matrix(new_Q)

    new_eulers = matrix_to_euler(new_R, angle_convention, angle_type)

    data[..., 1:4] = new_eulers

    return data
