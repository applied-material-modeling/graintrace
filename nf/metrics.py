import torch

import tqdm

from neml2 import tensors, crystallography


def average_rotations(e, angle_convention="kocks", angle_type="radians"):
    """Average two sets of Euler angles

    Args:
        e: Nx3 array of Euler angles

    Keyword Args:
        angle_convention (str): 'kocks', 'bunge', or 'roe' (default: 'kocks')
        angle_type (str): 'degrees' or 'radians' (default: 'radians')

    Returns:
        Nx3 array of averaged Euler angles
    """
    R = tensors.Rot.fill_euler_angles(tensors.Vec(e), angle_convention, angle_type)

    Q = tensors.Quaternion(R).torch()

    QQt = torch.matmul(Q.transpose(-2, -1), Q)

    _, eigvecs = torch.linalg.eigh(QQt)

    new_Q = eigvecs[..., -1]

    new_R = tensors.Rot.fill_matrix(tensors.Quaternion(new_Q).rotation_matrix())

    new_eulers = new_R.to_euler_angles(angle_convention, angle_type)

    return new_R.torch(), new_eulers.torch()


def batched_norm(v1, v2, norm, chunk_size, **kwargs):
    """Compute norms between two large arrays in chunks to save memory

    Args:
        v1: NxM array
        v2: NxM array
        norm: function to compute norm between two arrays
        chunk_size: size of chunks to process

    Returns:
        Nx1 array of norms
    """
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
    """Compute misorientation between two sets of Euler angles

    Args:
        e1: Nx3 array of Euler angles
        e2: Nx3 array of Euler angles

    Keyword Args:
        angle_convention (str): 'kocks', 'bunge', or 'roe' (default: 'kocks')
        angle_type (str): 'degrees' or 'radians' (default: 'degrees')
        symmetry (str): crystal symmetry in orbifold notation (default: '1')

    Returns:
        Nx1 array of misorientation angles in degrees
    """
    R1 = tensors.Rot.fill_euler_angles(tensors.Vec(e1), angle_convention, angle_type)
    R2 = tensors.Rot.fill_euler_angles(tensors.Vec(e2), angle_convention, angle_type)

    rad_mis = crystallography.misorientation(R1, R2, symmetry).torch()

    if angle_type == "degrees":
        return torch.rad2deg(rad_mis)

    return rad_mis
