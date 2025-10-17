from math import sqrt
import torch

inds = ["11", "22", "33", "23", "13", "12"]
facts = [1.0, 1.0, 1.0, sqrt(2.0), sqrt(2.0), sqrt(2.0)]


def load_orientations(df, field="O"):
    """Load orientations from dataframe and convert to torch tensor giving the modified Rodrigues parameters.

    The assumption here is we have a rotation matrix givin the *active* convention rotation from the crystal to the sample frame.

    Args:
        df (pd.DataFrame): Dataframe containing orientation data.
        field (str, optional): Column name for orientation matrix. Defaults to 'O'.

    Returns:
        torch.Tensor: Tensor containing orientation data.
    """
    matrix = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(
                torch.tensor(df[field + str(i + 1) + str(j + 1)].dropna().values)
            )
        matrix.append(torch.stack(row, dim=-1))
    matrix = torch.stack(matrix, dim=-2)

    quat = matrix_to_quat(matrix)

    return quat[..., 1:] / quat[..., 0:1]


def matrix_to_quat(M):
    """Convert rotation matrix to quaternion

    Args:
        M (torch.Tensor): Rotation matrix of shape (..., 3, 3)
    """
    tr = M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]
    conds = [
        tr > 0,
        (M[..., 0, 0] > M[..., 1, 1]) & (M[..., 0, 0] > M[..., 2, 2]),
        M[..., 1, 1] > M[..., 2, 2],
    ]

    funcs = [quat_option_1, quat_option_2, quat_option_3, quat_option_4]
    return torch.where(
        conds[0][..., None],
        funcs[0](M),
        torch.where(
            conds[1][..., None],
            funcs[1](M),
            torch.where(conds[2][..., None], funcs[2](M), funcs[3](M)),
        ),
    )


def quat_option_1(M):
    """First case of matrix to quaternion

    Args:
        M (torch.Tensor): Rotation matrix of shape (..., 3, 3)
    """
    tr = M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]
    S = torch.sqrt(tr + 1.0) * 2
    s = 0.25 * S
    x = (M[..., 2, 1] - M[..., 1, 2]) / S
    y = (M[..., 0, 2] - M[..., 2, 0]) / S
    z = (M[..., 1, 0] - M[..., 0, 1]) / S
    return torch.stack((s, x, y, z), dim=-1)


def quat_option_2(M):
    """Second case of matrix to quaternion

    Args:
        M (torch.Tensor): Rotation matrix of shape (..., 3, 3)
    """
    S = torch.sqrt(1.0 + M[..., 0, 0] - M[..., 1, 1] - M[..., 2, 2]) * 2
    s = (M[..., 2, 1] - M[..., 1, 2]) / S
    x = 0.25 * S
    y = (M[..., 0, 1] + M[..., 1, 0]) / S
    z = (M[..., 0, 2] + M[..., 2, 0]) / S
    return torch.stack((s, x, y, z), dim=-1)


def quat_option_3(M):
    """Third case of matrix to quaternion

    Args:
        M (torch.Tensor): Rotation matrix of shape (..., 3, 3)
    """
    S = torch.sqrt(1.0 + M[..., 1, 1] - M[..., 0, 0] - M[..., 2, 2]) * 2
    s = (M[..., 0, 2] - M[..., 2, 0]) / S
    x = (M[..., 0, 1] + M[..., 1, 0]) / S
    y = 0.25 * S
    z = (M[..., 1, 2] + M[..., 2, 1]) / S
    return torch.stack((s, x, y, z), dim=-1)


def quat_option_4(M):
    """Fourth case of matrix to quaternion

    Args:
        M (torch.Tensor): Rotation matrix of shape (..., 3, 3)
    """
    S = torch.sqrt(1.0 + M[..., 2, 2] - M[..., 0, 0] - M[..., 1, 1]) * 2
    s = (M[..., 1, 0] - M[..., 0, 1]) / S
    x = (M[..., 0, 2] + M[..., 2, 0]) / S
    y = (M[..., 1, 2] + M[..., 2, 1]) / S
    z = 0.25 * S
    return torch.stack((s, x, y, z), dim=-1)


def load_strains(df, field="eKen", factor=1e-6):
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


def load_weights(df, field="GrainRadius"):
    """Load the grain volumes and convert to a torch tensor of weights.

    Args:
        df (pd.DataFrame): Dataframe containing grain volume data.
        field (str, optional): Column name for grain volume data. Defaults to 'GrainRadius'.

    Returns:

    """
    weights = torch.tensor(df[field].dropna().values) ** 3.0
    return weights / torch.sum(weights)
