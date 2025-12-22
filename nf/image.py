import torch

connectivity_options = {
    6: torch.tensor(
        [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    ),
    26: torch.tensor(
        [
            (i, j, k)
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            for k in [-1, 0, 1]
            if not (i == 0 and j == 0 and k == 0)
        ]
    ),
}


def get_neighbor_indices(offsets, shape):
    """Calculate the indices of all neighbors according to the given offsets

    Args:
        offsets: list of (dx, dy, dz) tuples
        shape: (nx, ny, nz) tuple

    Returns:
        dx, dy, dz: tensors of shape (n_offsets,) with neighbor offsets
        Xk, Yk, Zk: tensors of shape (n_offsets, nx, ny, nz) with neighbor indices
        valid: tensor of shape (n_offsets, nx, ny, nz) indicating valid neighbors
    """
    dx = offsets[:, 0]
    dy = offsets[:, 1]
    dz = offsets[:, 2]

    X, Y, Z = torch.meshgrid(
        torch.arange(shape[0]),
        torch.arange(shape[1]),
        torch.arange(shape[2]),
        indexing="ij",
    )
    Xk = X.unsqueeze(0) + dx[:, None, None, None]
    Yk = Y.unsqueeze(0) + dy[:, None, None, None]
    Zk = Z.unsqueeze(0) + dz[:, None, None, None]

    valid = (
        (Xk >= 0)
        & (Xk < shape[0])
        & (Yk >= 0)
        & (Yk < shape[1])
        & (Zk >= 0)
        & (Zk < shape[2])
    )

    Xk = Xk.clamp(0, shape[0] - 1)
    Yk = Yk.clamp(0, shape[1] - 1)
    Zk = Zk.clamp(0, shape[2] - 1)

    return Xk, Yk, Zk, valid
