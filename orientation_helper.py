from math import sqrt
import torch

inds = ["11", "22", "33", "23", "13", "12"]
facts = [1.0, 1.0, 1.0, sqrt(2.0), sqrt(2.0), sqrt(2.0)]


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
        Nx1 array of misorientation angles
    """
    import neml2
    from neml2 import tensors
    from neml2 import crystallography

    e1 = torch.tensor(e1, dtype=torch.float64)
    e2 = torch.tensor(e2, dtype=torch.float64)

    if e1.ndim == 1:
        e1 = e1.unsqueeze(0)
    if e2.ndim == 1:
        e2 = e2.unsqueeze(0)

    e1 = tensors.Vec(e1)
    e2 = tensors.Vec(e2)

    R1 = (
        tensors.Rot.fill_euler_angles(tensors.Vec(e1), angle_convention, angle_type)
        .euler_rodrigues()
        .torch()
    )
    R2 = (
        tensors.Rot.fill_euler_angles(tensors.Vec(e2), angle_convention, angle_type)
        .euler_rodrigues()
        .torch()
    )

    ####
    symmetry_ops = crystallography.symmetry(symmetry).torch()
    # symmetry_ops = crystallography.symmetry_operations_from_orbifold(symmetry).torch()
    ####

    dR = torch.matmul(R1, R2.transpose(-2, -1))

    # print(symmetry_ops.shape)
    # print(dR.shape)
    # print(symmetry_ops.unsqueeze(0).shape)
    # print(dR.unsqueeze(1).shape)

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

    # output
    if angle_type == "degrees":
        return torch.rad2deg(rad_mis)

    return rad_mis


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


def load_orientations(df, field="O"):
    """Load orientations from dataframe and convert to torch tensor giving the modified Rodrigues parameters.

    The assumption here is we have a rotation matrix givin the *active* convention rotation from the crystal to the sample frame.

    Args:
        df (pd.DataFrame): Dataframe containing orientation data.
        field (str, optional): Column name for orientation matrix. Defaults to 'O'.

    Returns:
        torch.Tensor: Tensor containing orientation data.
    """
    if field is None:
        # assume columns are ordered row-major
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

    q = torch.where(
        conds[0][..., None],
        funcs[0](M),
        torch.where(
            conds[1][..., None],
            funcs[1](M),
            torch.where(conds[2][..., None], funcs[2](M), funcs[3](M)),
        ),
    )
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


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
