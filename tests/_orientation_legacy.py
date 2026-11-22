# Copyright 2026, UChicago Argonne, LLC -- MIT (see package LICENSE)
"""Frozen reference: the orientation primitives exactly as they stood at 0544c83.

At 0544c83 graintrace's ``euler_to_matrix`` returned the Bunge *passive* matrix
``g`` (sample -> crystal) and applied crystal symmetry on the LEFT (``S g``).
The convention flip makes the matrix the active ``R = g^T`` (crystal -> sample,
MIDAS / neml2) with symmetry on the RIGHT (``R S``). These copies are the
pre-change implementation, copied verbatim (bodies untouched; only a ``legacy_``
prefix added to each name and to the internal calls between them) so
``tests/test_convention_invariance.py`` can prove the flip changes nothing but
the documented side:

* matrices: ``new == legacy^T``; Euler outputs and misorientation angles equal;
* MRP outputs: ``mrp_to_matrix(new) == mrp_to_matrix(legacy)^T`` (``new == -legacy``).

Do not refactor these, and do not "fix" them to match the current code: their
value is that they do not change. Same pattern as ``tests/_prune_original.py``.
Conversions the flip leaves untouched (``matrix_to_mrp``, ``mrp_to_matrix``,
quaternion helpers, ``symmetry_operators``) are imported, not duplicated.
"""

# pylint: disable=import-outside-toplevel,invalid-name
from typing import Optional, Tuple, Union

import numpy as np
import torch

from graintrace.orientation_helper import (
    _from_bunge,
    _to_bunge,
    _wrap,
    matrix_to_mrp,
    matrix_to_quat,
    quat_to_matrix,
    symmetry_operators,
)
from graintrace.fragmentation import (
    _FOLD_CHUNK,
    _matrix_to_quat_np,
    _moments_mean_matrix,
    _mrp_to_matrix_np,
    _rotation_vector_deg,
)
from graintrace.frame_transform import _short_mrp


# --- graintrace/orientation_helper.py @ 0544c83 ----------------------------
def legacy_euler_to_matrix(
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
    # pylint: disable-next=import-outside-toplevel  # neml2 is a heavy optional dep
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


def legacy_matrix_to_euler(
    M: torch.Tensor,
    convention: str = "bunge",
    angle_type: str = "degrees",
) -> torch.Tensor:
    """Convert rotation matrices to Euler angles (inverse of :func:`euler_to_matrix`).

    First and third angles are wrapped to ``[0, 360)`` deg (``[0, 2*pi)`` rad). At
    gimbal lock ``Phi`` snaps to 0 or 180 and the third Bunge angle is 0.

    Args:
        M: (..., 3, 3) rotation matrices.
        convention: 'bunge', 'kocks', or 'roe'.
        angle_type: 'degrees' or 'radians'.

    Returns:
        (..., 3) Euler angles.
    """
    M = torch.as_tensor(M, dtype=torch.float64)
    # atan2 keeps precision near the poles, where arccos does not.
    sP = torch.hypot(M[..., 2, 0], M[..., 2, 1])
    Phi = torch.atan2(sP, M[..., 2, 2])

    phi1 = torch.atan2(M[..., 2, 0], -M[..., 2, 1])
    phi2 = torch.atan2(M[..., 0, 2], M[..., 1, 2])

    # Gimbal lock (Phi ~ 0 or pi): only phi1 +/- phi2 is defined; put it all in phi1.
    # atan2(M01, M00) recovers phi1+phi2 at Phi=0 and phi1-phi2 at Phi=pi; both rebuild
    # the same matrix, so the round-trip stays exact.
    gimbal = sP < 1e-8
    if torch.any(gimbal):
        phi1_g = torch.atan2(M[..., 0, 1], M[..., 0, 0])
        phi1 = torch.where(gimbal, phi1_g, phi1)
        phi2 = torch.where(gimbal, torch.zeros_like(phi2), phi2)
        Phi_g = torch.where(
            M[..., 2, 2] > 0, torch.zeros_like(Phi), torch.full_like(Phi, torch.pi)
        )
        Phi = torch.where(gimbal, Phi_g, Phi)

    euler = _from_bunge(phi1, Phi, phi2, convention)
    period = 2.0 * torch.pi
    if angle_type == "degrees":
        euler = torch.rad2deg(euler)
        period = 360.0
    elif angle_type != "radians":
        raise ValueError(f"Unknown angle_type: {angle_type!r}")
    return torch.stack(
        (_wrap(euler[..., 0], period), euler[..., 1], _wrap(euler[..., 2], period)),
        dim=-1,
    )


def legacy_euler_to_mrp(
    euler: Union[np.ndarray, list, torch.Tensor],
    convention: str = "bunge",
    angle_type: str = "degrees",
) -> torch.Tensor:
    """Convert Euler angles to neml2 v3 modified Rodrigues parameters (..., 3)."""
    return matrix_to_mrp(legacy_euler_to_matrix(euler, convention, angle_type))


def legacy_perturb_orientation(
    euler: Union[np.ndarray, list, torch.Tensor],
    sigma_deg: float,
    convention: str = "bunge",
    angle_type: str = "degrees",
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Apply an independent random misorientation to each orientation.

    Each orientation is composed with a random rotation about a uniformly
    distributed axis, with rotation angle drawn from ``N(0, sigma_deg)`` (degrees).
    ``sigma_deg`` is therefore the standard deviation of the applied rotation angle;
    the resulting fundamental-zone misorientation magnitude is ``|angle|``. This is a
    proper (gimbal-free) perturbation, unlike scaling each Euler angle independently.
    ``sigma_deg <= 0`` returns the input orientations unchanged.

    Args:
        euler: (..., 3) Euler angles.
        sigma_deg: std of the applied rotation angle, in degrees.
        convention: 'bunge', 'kocks', or 'roe'.
        angle_type: 'degrees' or 'radians' (applies to both input and output).
        rng: optional numpy Generator for reproducibility (default: ``np.random``).

    Returns:
        (..., 3) perturbed Euler angles in the same convention/units.
    """
    e = torch.as_tensor(euler, dtype=torch.float64)
    if sigma_deg <= 0.0:
        return e.clone()

    R = legacy_euler_to_matrix(e, convention, angle_type)  # (..., 3, 3)
    flat = R.reshape(-1, 3, 3)
    n = flat.shape[0]

    draw = rng if rng is not None else np.random
    # uniform random axes on the unit sphere + gaussian rotation angle (radians)
    axis = torch.as_tensor(draw.normal(0.0, 1.0, size=(n, 3)), dtype=torch.float64)
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp_min(1e-300)
    angle = torch.as_tensor(
        np.deg2rad(draw.normal(0.0, sigma_deg, size=n)), dtype=torch.float64
    )

    # Rodrigues delta rotations: dR = I + sin(a) K + (1 - cos(a)) K^2
    K = torch.zeros(n, 3, 3, dtype=torch.float64)
    ax, ay, az = axis[:, 0], axis[:, 1], axis[:, 2]
    K[:, 0, 1], K[:, 0, 2] = -az, ay
    K[:, 1, 0], K[:, 1, 2] = az, -ax
    K[:, 2, 0], K[:, 2, 1] = -ay, ax
    eye = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
    s = torch.sin(angle).reshape(n, 1, 1)
    c = torch.cos(angle).reshape(n, 1, 1)
    dR = eye + s * K + (1.0 - c) * torch.matmul(K, K)

    R_new = torch.matmul(dR, flat).reshape(R.shape)
    return legacy_matrix_to_euler(R_new, convention, angle_type)


def legacy_misorientation_matrix(
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


def legacy_move_to_fundamental_zone(
    R: torch.Tensor, symmetry: str = "1"
) -> torch.Tensor:
    """Reduce rotation matrices to the fundamental zone under crystal symmetry.

    Picks, per orientation, the symmetry-equivalent with the smallest rotation
    angle (largest trace of ``O @ R``).

    Args:
        R: (..., 3, 3) rotation matrices.
        symmetry: crystal symmetry in orbifold notation.

    Returns:
        (..., 3, 3) reduced rotation matrices.
    """
    ops = symmetry_operators(symmetry).to(
        device=R.device, dtype=R.dtype
    )  # (nops, 3, 3)
    cand = torch.matmul(ops, R.unsqueeze(-3))  # (..., nops, 3, 3)
    trace = cand.diagonal(dim1=-2, dim2=-1).sum(-1)  # (..., nops)
    idx = torch.argmax(trace, dim=-1)  # (...,)
    idx_exp = idx[..., None, None, None].expand(idx.shape + (1, 3, 3))
    return torch.gather(cand, -3, idx_exp).squeeze(-3)


def legacy_average_orientation(
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

    R = legacy_euler_to_matrix(e, convention, angle_type)  # (N, 3, 3)
    if n == 1:
        return legacy_matrix_to_euler(R[0], convention, angle_type)

    ops = symmetry_operators(symmetry).to(dtype=R.dtype)  # (nops, 3, 3)

    # Reference = highest-weight orientation; align every orientation to it by
    # picking the symmetry op O maximizing <ref, O @ R_i>.
    ref = R[int(torch.argmax(w))]  # (3, 3)
    cand = torch.matmul(ops.unsqueeze(0), R.unsqueeze(1))  # (N, nops, 3, 3)
    score = torch.einsum("ij,nkij->nk", ref, cand)  # (N, nops)
    best = torch.argmax(score, dim=1)  # (N,)
    R_aligned = cand[torch.arange(n), best]  # (N, 3, 3)

    # Weighted matrix mean, re-projected onto SO(3).
    M = torch.einsum("n,nij->ij", w, R_aligned) / w.sum()
    U, _, Vh = torch.linalg.svd(M)
    Ravg = U @ Vh
    if torch.det(Ravg) < 0:  # guard against reflection
        U = U.clone()
        U[:, -1] = -U[:, -1]
        Ravg = U @ Vh

    return legacy_matrix_to_euler(Ravg, convention, angle_type)


def legacy_misorientation(
    e1: Union[np.ndarray, list],
    e2: Union[np.ndarray, list],
    angle_convention: str = "bunge",
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

    R1 = legacy_euler_to_matrix(e1, angle_convention, angle_type)
    R2 = legacy_euler_to_matrix(e2, angle_convention, angle_type)

    return legacy_misorientation_matrix(R1, R2, symmetry, angle_type=angle_type)


# --- graintrace/nf/metrics.py @ 0544c83 ------------------------------------
def legacy_fold_to_reference_variant(R, symmetry):
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


def legacy_average_rotations(
    e, angle_convention="kocks", angle_type="radians", symmetry="1"
):
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
    R = legacy_euler_to_matrix(e, angle_convention, angle_type)

    R = legacy_fold_to_reference_variant(R, symmetry)

    Q = matrix_to_quat(R)

    QQt = torch.matmul(Q.transpose(-2, -1), Q)

    _, eigvecs = torch.linalg.eigh(QQt)

    new_Q = eigvecs[..., -1]

    new_R = quat_to_matrix(new_Q)

    new_mrp = matrix_to_mrp(new_R)
    new_eulers = legacy_matrix_to_euler(new_R, angle_convention, angle_type)

    return new_mrp, new_eulers


# --- graintrace/fragmentation.py @ 0544c83 ---------------------------------
def legacy_left_quat_matrix(q: np.ndarray) -> np.ndarray:
    """``L(q)`` with ``q (x) p == L(q) @ p`` (Hamilton product, scalar first).

    Folding a cluster by symmetry operator ``S`` maps its moments to ``L M L^T``.
    """
    w, x, y, z = q
    return np.array(
        [[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]], dtype=float
    )


def legacy_fold_to_reference(
    rmat: np.ndarray, ref: np.ndarray, ops: np.ndarray
) -> np.ndarray:
    """Fold each ``rmat[i]`` to the variant ``ops[k] @ rmat[i]`` closest to ``ref[i]``."""
    # sum_ij ref_ij (O_k r)_ij == sum_il O_kil (r ref^T)_li: one (n,9)x(9,k) GEMM
    ops_flat = np.ascontiguousarray(np.swapaxes(ops, 1, 2)).reshape(len(ops), 9).T
    out = np.empty_like(rmat)
    for s in range(0, len(rmat), _FOLD_CHUNK):
        r = rmat[s : s + _FOLD_CHUNK]
        prod = np.matmul(r, np.swapaxes(ref[s : s + _FOLD_CHUNK], -1, -2))
        best = np.argmax(prod.reshape(len(r), 9) @ ops_flat, axis=1)
        out[s : s + _FOLD_CHUNK] = np.matmul(ops[best], r)
    return out


def legacy_cluster_moments(
    rmat: np.ndarray, inv: np.ndarray, n_clusters: int, ops: np.ndarray
) -> np.ndarray:
    """Per-cluster quaternion moments ``M_k = sum q q^T`` ``(K, 4, 4)``.

    Points are folded to their cluster's first point, as in
    :meth:`FragmentationAnalyzer.mean_orientation_mrp`.
    """
    _, first = np.unique(inv, return_index=True)
    folded = legacy_fold_to_reference(rmat, rmat[first][inv], ops)
    q = _matrix_to_quat_np(folded)
    mom = np.zeros((n_clusters, 4, 4))
    for i in range(4):
        for j in range(i, 4):
            col = np.bincount(inv, weights=q[:, i] * q[:, j], minlength=n_clusters)
            mom[:, i, j] = col
            mom[:, j, i] = col
    return mom


def legacy_sym_miso_deg(ra: np.ndarray, rb: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Symmetry-reduced misorientation (deg) between row-aligned matrix stacks."""
    dr = np.matmul(ra, np.swapaxes(rb, -1, -2))
    tr = np.einsum("kij,nji->nk", ops, dr).max(axis=1)
    return np.rad2deg(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def legacy_cluster_tangent(
    mrp: np.ndarray, inv: np.ndarray, n_clusters: int, ops: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster mean matrices ``(K, 3, 3)`` and per-point rotation vectors (deg) from them."""
    rmat = _mrp_to_matrix_np(mrp)
    mean_r = _moments_mean_matrix(legacy_cluster_moments(rmat, inv, n_clusters, ops))
    ref = mean_r[inv]
    folded = legacy_fold_to_reference(rmat, ref, ops)
    return mean_r, _rotation_vector_deg(np.matmul(folded, np.swapaxes(ref, -1, -2)))


# --- graintrace/fragmentation.py @ 0544c83 (FragmentationAnalyzer) ---------
def legacy_mean_orientation_mrp(mrp_arr: np.ndarray, symmetry: str = "1") -> np.ndarray:
    """Symmetry-aware quaternion (eigenvector) mean of neml2 MRP orientations.

    A proper orientation mean (NOT an arithmetic mean of the MRP components).
    When ``symmetry`` names a crystal point group, every orientation is first
    folded to the symmetry-equivalent variant closest to the first (reference)
    orientation, so a set whose members are reported in different symmetry
    variants (which the plain quaternion mean would average into a meaningless
    result) still yields the correct mean. With ``symmetry="1"`` (the default)
    this reduces to the plain quaternion eigenvector (Markley) mean.

    Args:
        mrp_arr: ``(N, 3)`` neml2 MRP orientations.
        symmetry: crystal symmetry (orbifold notation) to fold variants under;
            ``"1"`` disables folding.

    Returns:
        ``(3,)`` mean neml2 MRP orientation.
    """
    # pylint: disable=import-outside-toplevel
    import torch

    from graintrace.orientation_helper import (
        matrix_to_mrp,
        matrix_to_quat,
        mrp_to_matrix,
        quat_to_matrix,
        symmetry_operators,
    )

    rmat = mrp_to_matrix(torch.as_tensor(np.asarray(mrp_arr), dtype=torch.float64))
    if symmetry != "1" and rmat.shape[0] > 1:
        # fold each orientation to the symmetry variant closest to the reference
        ops = symmetry_operators(symmetry).to(rmat.dtype)  # (nops, 3, 3)
        ref = rmat[0]
        cand = torch.matmul(ops.unsqueeze(0), rmat.unsqueeze(1))  # (N, nops, 3, 3)
        score = torch.einsum("ij,nkij->nk", ref, cand)  # (N, nops)
        best = torch.argmax(score, dim=1)
        rmat = cand[torch.arange(rmat.shape[0]), best]
    quat = matrix_to_quat(rmat)  # (N, 4)
    qqt = torch.matmul(quat.transpose(-2, -1), quat)
    _, eigvecs = torch.linalg.eigh(qqt)
    mean_r = quat_to_matrix(eigvecs[..., -1])
    return matrix_to_mrp(mean_r).cpu().numpy().reshape(3)


# --- graintrace/frame_transform.py @ 0544c83 (FrameTransform, Q as argument) --
def legacy_frame_orientation_matrices(self, g):
    """Transform orientation matrices on the sample side: ``g' = g Q^T``.

    Args:
        g: (..., 3, 3) matrices in :func:`~graintrace.orientation_helper.euler_to_matrix`'s
            convention; a tensor keeps its dtype and device.

    Returns:
        torch.Tensor: (..., 3, 3) transformed matrices.
    """
    # pylint: disable-next=import-outside-toplevel
    import torch

    if isinstance(g, torch.Tensor):
        gt = g
    else:
        # np.array, not asarray: a read-only input makes torch warn.
        gt = torch.as_tensor(np.array(g, dtype=float), dtype=torch.float64)
    Qt = torch.as_tensor(self.Q.T, dtype=gt.dtype, device=gt.device)
    return gt @ Qt


def legacy_frame_mrp(self, p):
    """Transform neml2 v3 MRP orientations in neml2; returns the short set ``|p'| <= 1``.

    Args:
        p: (..., 3) neml2 v3 MRP orientations (either shadow set).

    Returns:
        torch.Tensor: (..., 3) transformed MRP, ``|p'| <= 1``.
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from neml2.types import MRP, compose

    from graintrace.orientation_helper import matrix_to_mrp

    pt = _short_mrp(torch.as_tensor(np.asarray(p, dtype=float), dtype=torch.float64))
    q_frame = _short_mrp(
        matrix_to_mrp(torch.as_tensor(self.Q.T, dtype=torch.float64))
    ).expand_as(pt)
    # compose(a, b) has matrix(a) @ matrix(b), so g Q^T is compose(p, Q^T).
    # Its denominator 1 + |a|^2|b|^2 - 2 a.b stays >= 1 when a.b <= 0.
    flip = (pt * q_frame).sum(dim=-1, keepdim=True) > 0.0
    q_frame = torch.where(flip, _shadow(q_frame), q_frame)
    composed = compose(MRP(pt.contiguous(), 0), MRP(q_frame.contiguous(), 0))
    return _short_mrp(composed.data)
