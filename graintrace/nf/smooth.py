from __future__ import annotations

import torch

from neml2 import tensors, crystallography

from .image import get_neighbor_indices, connectivity_options


def smooth(
    data, connectivity=6, symmetry="1", angle_convention="bunge", angle_type="radians"
):
    """Smooth orientations

    Args:
        data (nx,ny,nz,7) tensor: fixed grid data with columns [phase, Eul1, Eul2, Eul3, X, Y, Z]

    Keyword Args:
        connectivity (int): 6 or 26 (default: 6) determines neighbor connectivity
        symmetry (str): crystal symmetry in orbifold notation (default: '1')
        angle_convention (str): 'kocks', 'bunge', or 'roe' (default: 'bunge')
        angle_type (str): 'degrees' or 'radians' (default: 'radians')
    """
    print("Smoothing...")
    R = tensors.Rot.fill_euler_angles(
        tensors.Vec(data[..., 1:4]), angle_convention, angle_type
    )

    R = crystallography.move_to_fundamental_zone(R, symmetry)

    Q1 = tensors.Quaternion(R).torch()

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

    new_R = tensors.Rot.fill_matrix(tensors.Quaternion(new_Q).rotation_matrix())

    new_eulers = new_R.to_euler_angles(angle_convention, angle_type).torch()

    data[..., 1:4] = new_eulers

    return data
