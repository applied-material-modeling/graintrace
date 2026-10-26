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

"""Grain fragmentation and reorientation analysis.

:class:`FragmentationAnalyzer` groups the reorientation and fragmentation
capabilities that share the same crystal symmetry and column conventions:

* reorientation (misorientation-from-a-reference-step) as a rare-event scalar;
* orientation-based point-set segmentation (graph/Leiden) and its reduction to a
  per-grain table, shared by the simulation and experiment paths;
* fragmentation detection: simulation intragranular sub-grains
  (:meth:`FragmentationAnalyzer.grain_fragments`) and experiment grain splits
  across load steps (:meth:`FragmentationAnalyzer.detect_splits`).

The segmenter uses a fixed broad RBF sigma (``sigma_auto`` collapses on
near-identical intra-grain edges and shatters grains), a hard misorientation edge
cutoff and a cap-free gamma pick (the lowest swept gamma, or the lowest one the
physical percolation check -- :meth:`FragmentationAnalyzer.cluster_leak_check` --
does not flag). It then absorbs small fragments into an orientation-compatible
neighbour and runs an agglomerative adjacency-only re-merge that rejects unions
failing the percolation check. No step caps grain size by default; the legacy
3 % size cap (``percolation_max_frac``) is an explicit opt-in.

Heavy dependencies (torch, neml2, scipy, networkit via ``GraphSpatialCluster``)
are imported lazily inside the methods.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


# --------------------------------------------------------------------------- #
# Module-level private helpers (stateless numeric utilities)
# --------------------------------------------------------------------------- #
def _pair_misorientation_deg(
    mrp_a: np.ndarray, mrp_b: np.ndarray, symmetry: str
) -> float:
    """Symmetry-reduced misorientation (deg) between two single MRP orientations."""
    # pylint: disable=import-outside-toplevel
    import torch

    from .orientation_helper import misorientation_matrix, mrp_to_matrix

    ra = mrp_to_matrix(torch.as_tensor(mrp_a.reshape(1, 3), dtype=torch.float64))
    rb = mrp_to_matrix(torch.as_tensor(mrp_b.reshape(1, 3), dtype=torch.float64))
    return float(misorientation_matrix(ra, rb, symmetry, "degrees").item())


def _as_label_set(background_label: Union[int, Iterable[int]]) -> set:
    """Normalise a background-label argument to a set of ints.

    Accepts a single integer or any iterable of integers, so a caller working
    on a raw segmented voxel grid can exclude both non-grain sentinels --
    ``0`` (void) and ``-1`` (material-but-unsegmented) -- in one call.

    Raises:
        TypeError: If the argument is neither an integer nor an iterable of
            integers.
    """
    if isinstance(background_label, (bool, np.bool_)):
        raise TypeError("background_label must be an int or an iterable of ints")
    if isinstance(background_label, (int, np.integer)):
        return {int(background_label)}
    try:
        return {int(v) for v in background_label}
    except TypeError as exc:
        raise TypeError(
            "background_label must be an int or an iterable of ints, got "
            f"{type(background_label).__name__}"
        ) from exc


def _knn_indices(coords: np.ndarray, k_neighbors: int) -> np.ndarray:
    """``(N, k+1)`` kNN indices (self first) used to measure cluster contact."""
    # pylint: disable=import-outside-toplevel
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    kq = min(k_neighbors + 1, len(coords))
    _, idx = tree.query(coords, k=kq)
    return np.atleast_2d(idx).reshape(len(coords), -1)


def _contact_pairs(
    nbr_idx: np.ndarray, labels: np.ndarray, ignore: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Touching label pairs ``(P, 2)`` (``a < b``) and their contact counts.

    Pairs are returned in order of first appearance in a row-major scan of
    ``nbr_idx`` (point ``i``, then its neighbours in order), which is the order
    the original per-point loop inserted them in. Downstream tie-breaks depend on
    that order, so keeping it keeps the legacy labels bit-identical. Points whose
    ``ignore`` flag is set (unsegmented / background) never form a contact.
    """
    li = labels[:, None]
    lj = labels[nbr_idx[:, 1:]]
    valid = li != lj
    if ignore is not None:
        valid &= ~ignore[:, None]
        valid &= ~ignore[nbr_idx[:, 1:]]
    a = np.minimum(li, lj)[valid].astype(np.int64)
    b = np.maximum(li, lj)[valid].astype(np.int64)
    if a.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    off = int(min(a.min(), 0))
    span = int(b.max()) - off + 1
    key = (a - off) * span + (b - off)
    ukey, first, counts = np.unique(key, return_index=True, return_counts=True)
    order = np.argsort(first, kind="stable")
    ukey, counts = ukey[order], counts[order]
    pairs = np.stack([ukey // span + off, ukey % span + off], axis=1)
    return pairs.astype(np.int64), counts.astype(np.int64)


def _knn_cluster_adjacency(
    coords: np.ndarray,
    labels: np.ndarray,
    k_neighbors: int,
    ignore: Optional[np.ndarray] = None,
    nbr_idx: Optional[np.ndarray] = None,
) -> Dict[Tuple[int, int], int]:
    """Adjacent (touching) cluster-label pairs and their contact counts, via kNN.

    Two clusters are adjacent when a point of one is among the ``k_neighbors``
    nearest neighbours of a point of the other. Works for both voxel grids and
    scattered mesh element centroids (no regular-grid assumption). Vectorised;
    the dict preserves the first-appearance order of a per-point scan.
    """
    if nbr_idx is None:
        nbr_idx = _knn_indices(coords, k_neighbors)
    pairs, counts = _contact_pairs(nbr_idx, np.asarray(labels), ignore)
    return {(int(a), int(b)): int(n) for (a, b), n in zip(pairs, counts)}


def _ignored_mask(
    labels: np.ndarray, background_label: Union[int, Iterable[int], None]
) -> np.ndarray:
    """Points that are not part of any cluster: every label ``< 0`` plus the
    ``background_label`` set (e.g. raw-grid void ``0``)."""
    mask = labels < 0
    if background_label is not None:
        extra = [v for v in _as_label_set(background_label) if v >= 0]
        if extra:
            mask |= np.isin(labels, extra)
    return mask


def _relabel_contiguous(labels: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """Relabel the non-ignored clusters contiguously from 0; ignored points keep
    their label. Ids equal to an ignored non-negative label (raw-grid void ``0``)
    are skipped so a background value never collides with a cluster id."""
    out = labels.astype(np.int64).copy()
    if not ignore.any():
        return np.unique(out, return_inverse=True)[1].astype(np.int64).reshape(-1)
    keep = ~ignore
    uniq, inv = np.unique(out[keep], return_inverse=True)
    taken = set(int(v) for v in np.unique(out[ignore]) if v >= 0)
    ids = [i for i in range(len(uniq) + len(taken)) if i not in taken][: len(uniq)]
    out[keep] = np.asarray(ids, dtype=np.int64)[inv.reshape(-1)]
    return out


# --- vectorised symmetry-aware orientation statistics ---------------------- #
_FOLD_CHUNK = 50_000


def _symmetry_ops_np(symmetry: str) -> np.ndarray:
    """Crystal symmetry operators as a ``(nops, 3, 3)`` numpy array."""
    # pylint: disable=import-outside-toplevel
    from .orientation_helper import symmetry_operators

    return symmetry_operators(symmetry).cpu().numpy().astype(np.float64)


def _mrp_to_matrix_np(mrp: np.ndarray) -> np.ndarray:
    """neml2 MRP ``(N, 3)`` -> rotation matrices ``(N, 3, 3)`` (numpy)."""
    # pylint: disable=import-outside-toplevel
    import torch

    from .orientation_helper import mrp_to_matrix

    tens = torch.as_tensor(np.asarray(mrp, dtype=np.float64).reshape(-1, 3))
    return mrp_to_matrix(tens).cpu().numpy().reshape(-1, 3, 3)


def _matrix_to_quat_np(rmat: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(N, 3, 3)`` -> scalar-first quaternions ``(N, 4)``."""
    # pylint: disable=import-outside-toplevel
    import torch

    from .orientation_helper import matrix_to_quat

    return matrix_to_quat(torch.as_tensor(rmat)).cpu().numpy().reshape(-1, 4)


def _quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """Unit quaternions ``(..., 4)`` (w, x, y, z) -> matrices, neml2 convention."""
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1
            ),
            np.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1
            ),
            np.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        -2,
    )


def _left_quat_matrix(q: np.ndarray) -> np.ndarray:
    """``L(q)`` with ``q (x) p == L(q) @ p`` (Hamilton product, scalar first).

    ``matrix_to_quat(S @ R) == +-(q_S (x) q_R)``, so folding a cluster's
    quaternions by a symmetry operator ``S`` maps its moment matrix
    ``M = sum q q^T`` to ``L(q_S) M L(q_S)^T``.
    """
    w, x, y, z = q
    return np.array(
        [[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]], dtype=float
    )


def _fold_to_reference(
    rmat: np.ndarray, ref: np.ndarray, ops: np.ndarray
) -> np.ndarray:
    """Fold each ``rmat[i]`` to the variant ``ops[k] @ rmat[i]`` closest to ``ref[i]``.

    Same rule as :meth:`FragmentationAnalyzer.mean_orientation_mrp` (maximise
    ``sum(ref * cand)``, i.e. the trace of ``ref^T cand``), chunked for memory.
    """
    # sum_ij ref_ij (O_k r)_ij == sum_il O_kil (r ref^T)_li: one (n,9)x(9,k) GEMM
    ops_flat = np.ascontiguousarray(np.swapaxes(ops, 1, 2)).reshape(len(ops), 9).T
    out = np.empty_like(rmat)
    for s in range(0, len(rmat), _FOLD_CHUNK):
        r = rmat[s : s + _FOLD_CHUNK]
        prod = np.matmul(r, np.swapaxes(ref[s : s + _FOLD_CHUNK], -1, -2))
        best = np.argmax(prod.reshape(len(r), 9) @ ops_flat, axis=1)
        out[s : s + _FOLD_CHUNK] = np.matmul(ops[best], r)
    return out


def _cluster_moments(
    rmat: np.ndarray, inv: np.ndarray, n_clusters: int, ops: np.ndarray
) -> np.ndarray:
    """Per-cluster quaternion moment matrices ``(K, 4, 4)``, ``M_k = sum q q^T``.

    Every point is first folded to the symmetry variant closest to its cluster's
    first point (the :meth:`FragmentationAnalyzer.mean_orientation_mrp` rule), so
    the top eigenvector of ``M_k`` is that method's mean, for all clusters at once.
    """
    _, first = np.unique(inv, return_index=True)
    folded = _fold_to_reference(rmat, rmat[first][inv], ops)
    q = _matrix_to_quat_np(folded)
    mom = np.zeros((n_clusters, 4, 4))
    for i in range(4):
        for j in range(i, 4):
            col = np.bincount(inv, weights=q[:, i] * q[:, j], minlength=n_clusters)
            mom[:, i, j] = col
            mom[:, j, i] = col
    return mom


def _moments_mean_matrix(mom: np.ndarray) -> np.ndarray:
    """Mean rotation matrices from moment matrices (Markley top eigenvector)."""
    _, vecs = np.linalg.eigh(mom)
    return _quat_to_matrix_np(vecs[..., -1])


def _sym_miso_deg(ra: np.ndarray, rb: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Symmetry-reduced misorientation (deg) between row-aligned matrix stacks.

    Equals :func:`~graintrace.orientation_helper.misorientation_matrix`: the
    two-sided ``S dR S'^T`` minimum reduces to one-sided ``T dR`` by trace
    invariance under conjugation.
    """
    dr = np.matmul(ra, np.swapaxes(rb, -1, -2))
    tr = np.einsum("kij,nji->nk", ops, dr).max(axis=1)
    return np.rad2deg(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _rotation_vector_deg(rmat: np.ndarray) -> np.ndarray:
    """Rotation vectors (axis * angle, degrees) of ``(N, 3, 3)`` matrices (< 180 deg)."""
    tr = np.trace(rmat, axis1=-2, axis2=-1)
    theta = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    axis = np.stack(
        [
            rmat[:, 2, 1] - rmat[:, 1, 2],
            rmat[:, 0, 2] - rmat[:, 2, 0],
            rmat[:, 1, 0] - rmat[:, 0, 1],
        ],
        axis=1,
    )
    sin = np.sin(theta)
    scale = np.where(sin > 1e-12, theta / np.maximum(2.0 * sin, 1e-300), 0.5)
    return np.rad2deg(axis * scale[:, None])


def _cluster_tangent(
    mrp: np.ndarray, inv: np.ndarray, n_clusters: int, ops: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster mean matrices ``(K, 3, 3)`` and each point's tangent vector.

    The tangent vector of a point is the rotation vector (degrees) taking its
    cluster mean to the point's closest symmetry variant, so its norm is the
    symmetry-aware misorientation to the mean and distances between tangent
    vectors approximate point-to-point misorientations within a cluster.
    """
    rmat = _mrp_to_matrix_np(mrp)
    mean_r = _moments_mean_matrix(_cluster_moments(rmat, inv, n_clusters, ops))
    ref = mean_r[inv]
    folded = _fold_to_reference(rmat, ref, ops)
    return mean_r, _rotation_vector_deg(np.matmul(folded, np.swapaxes(ref, -1, -2)))


def _group_offsets(inv: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """Counts and start offsets of each cluster in a ``inv``-sorted array."""
    counts = np.bincount(inv, minlength=n_clusters)
    return counts, np.cumsum(counts) - counts


def _group_percentile(
    values: np.ndarray, inv: np.ndarray, n_clusters: int, q: float
) -> np.ndarray:
    """Per-cluster ``q``-th percentile (``np.percentile`` linear rule), vectorised."""
    order = np.lexsort((values, inv))
    sv = values[order]
    counts, starts = _group_offsets(inv, n_clusters)
    pos = np.maximum(counts - 1, 0) * (q / 100.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(counts - 1, 0))
    frac = pos - lo
    out = np.zeros(n_clusters)
    has = counts > 0
    out[has] = (
        sv[starts[has] + lo[has]] * (1.0 - frac[has])
        + sv[starts[has] + hi[has]] * frac[has]
    )
    return out


def _subsample_per_cluster(
    inv: np.ndarray, max_points: int, rng: np.random.Generator
) -> np.ndarray:
    """Sorted point indices keeping at most ``max_points`` random points per cluster."""
    n = len(inv)
    if not max_points:
        return np.arange(n)
    order = np.lexsort((rng.random(n), inv))
    sorted_inv = inv[order]
    rank = np.arange(n) - np.searchsorted(sorted_inv, sorted_inv, side="left")
    return np.sort(order[rank < max_points])


def _separated_populations(
    vecs: np.ndarray, tol_deg: float, min_size: int, min_cell_points: int = 2
) -> Tuple[int, float]:
    """Points in, and separation of, orientation populations split off by a gap.

    ``vecs`` are one cluster's tangent vectors (degrees). They are binned on a
    grid of ``tol_deg / 4`` cells; a cell with at least ``min_cell_points`` points
    is dense, and dense cells that touch (26-connectivity) form one population.
    A population of at least ``min_size`` points whose centroid sits more than
    ``tol_deg`` from the largest population's is a leak. A continuous gradient
    fills a connected run of cells (one population); a leaked grain is a second
    dense blob behind an empty gap; a speckled bridge of isolated noisy points
    fills no dense cell, so it does not join the two.

    Returns:
        ``(n_leaked_points, max_separation_deg)``; ``(0, 0.0)`` when no leak.
    """
    # pylint: disable=import-outside-toplevel
    from scipy import ndimage

    h = tol_deg / 4.0
    lo = vecs.min(axis=0)
    span = vecs.max(axis=0) - lo
    while np.prod(np.floor(span / h) + 1.0) > 8e6:
        h *= 2.0
    cell = np.floor((vecs - lo) / h).astype(np.int64)
    dims = tuple(int(v) for v in cell.max(axis=0) + 1)
    flat = np.ravel_multi_index(tuple(cell.T), dims)
    dense = (
        np.bincount(flat, minlength=int(np.prod(dims))) >= min_cell_points
    ).reshape(dims)
    comp_grid, n_comp = ndimage.label(dense, structure=np.ones((3, 3, 3)))
    if n_comp < 2:
        return 0, 0.0
    comp = comp_grid.reshape(-1)[flat]
    sizes = np.bincount(comp, minlength=n_comp + 1)
    sizes[0] = 0
    main = int(np.argmax(sizes))
    cent = (
        np.stack(
            [
                np.bincount(comp, weights=vecs[:, i], minlength=n_comp + 1)
                for i in range(3)
            ],
            axis=1,
        )
        / np.maximum(sizes, 1)[:, None]
    )
    sep = np.linalg.norm(cent - cent[main], axis=1)
    leak = (sizes >= min_size) & (sep > tol_deg)
    leak[0] = False
    leak[main] = False
    if not leak.any():
        return 0, 0.0
    return int(sizes[leak].sum()), float(sep[leak].max())


def _cluster_stats(mrp, labels, symmetry, q, max_points, seed, background_label):
    """Shared engine: (labels, tangent vecs, cluster index, spread, counts)."""
    mrp = np.asarray(mrp, dtype=float).reshape(-1, 3)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    keep = np.flatnonzero(~_ignored_mask(labels, background_label))
    if keep.size == 0:
        return None
    uniq, inv = np.unique(labels[keep], return_inverse=True)
    inv = inv.reshape(-1)
    sub = _subsample_per_cluster(inv, max_points, np.random.default_rng(seed))
    inv = inv[sub]
    _, vecs = _cluster_tangent(
        mrp[keep[sub]], inv, len(uniq), _symmetry_ops_np(symmetry)
    )
    spread = _group_percentile(np.linalg.norm(vecs, axis=1), inv, len(uniq), q)
    return uniq, vecs, inv, spread, np.bincount(inv, minlength=len(uniq))


def _voxel_volume(coords: np.ndarray) -> float:
    """Approximate per-point volume from the median spacing along each axis."""
    vol = 1.0
    for kax in range(3):
        u = np.unique(np.round(coords[:, kax], 3))
        d = np.diff(u)
        vol *= float(np.median(d)) if len(d) else 1.0
    return vol if vol > 0 else 1.0


def _load_nodes(nodes) -> "object":
    """Accept a DataFrame or a CSV path; return a DataFrame."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    if isinstance(nodes, str):
        return pd.read_csv(nodes)
    return nodes


def _orientation_matrices(df, ori_cols, angle_convention, angle_type):
    """Rotation matrices ``(N, 3, 3)`` from a node table's orientation columns."""
    # pylint: disable=import-outside-toplevel
    import torch

    from .orientation_helper import euler_to_matrix, mrp_to_matrix

    # np.array (not ascontiguousarray) forces a writable copy -> avoids torch's
    # non-writable-tensor warning on a read-only pandas buffer.
    vals = np.array(df[list(ori_cols)].to_numpy(dtype=float), dtype=np.float64)
    tens = torch.as_tensor(vals, dtype=torch.float64)
    if angle_convention == "mrp":
        return mrp_to_matrix(tens)
    return euler_to_matrix(tens, angle_convention, angle_type)


class FragmentationAnalyzer:
    """Reorientation and fragmentation analysis for simulation and experiment data.

    Holds the crystal symmetry and the simulation orientation/coordinate column
    conventions shared by the methods. The experiment split detector
    (:meth:`detect_splits`) takes its own column params because grain tables use a
    different schema (``Eul0/1/2``, ``X/Y/Z``).

    Args:
        symmetry: crystal symmetry in orbifold notation (default ``"432"``).
        ori_cols: per-element neml2 MRP orientation columns in the simulation field
            CSVs (default ``ori_rodrigues_x/y/z``).
        coord_cols: per-element coordinate columns in the simulation field CSVs
            (default ``x/y/z``).
    """

    def __init__(
        self,
        symmetry: str = "432",
        ori_cols: Sequence[str] = (
            "ori_rodrigues_x",
            "ori_rodrigues_y",
            "ori_rodrigues_z",
        ),
        coord_cols: Sequence[str] = ("x", "y", "z"),
    ) -> None:
        self.symmetry = symmetry
        self.ori_cols = tuple(ori_cols)
        self.coord_cols = tuple(coord_cols)
        #: per-gamma diagnostics from the most recent :meth:`segment` sweep
        self.last_gamma_diagnostics: List[dict] = []

    # ----------------------------------------------------------------------- #
    # Reorientation from a reference step
    # ----------------------------------------------------------------------- #
    def reorientation_from_reference(
        self,
        results,
        step: int,
        ref_step: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-element misorientation angle (deg) from a reference load step.

        Joins the per-element field CSVs at ``ref_step`` and ``step`` on ``id`` and
        returns the symmetry-reduced misorientation between the two orientations at
        each shared element. The ``ori_cols`` hold neml2 v3 MRP orientations.

        Args:
            results: a :class:`~graintrace.simulation_postprocessing.SimulationResults`
                whose ``field_dir`` holds per-element CSVs.
            step: field-step index to measure.
            ref_step: reference field-step index (defaults to the first field step).

        Returns:
            ``(ids, deg)``: element ids ``(N,)`` and misorientation angles in degrees
            ``(N,)``, aligned to the ids common to both steps.
        """
        # pylint: disable=import-outside-toplevel  # heavy optional deps
        import torch

        from .orientation_helper import misorientation_matrix, mrp_to_matrix

        field_steps = sorted(results.field_files.keys())
        if not field_steps:
            raise ValueError("results has no per-element field CSVs.")
        if ref_step is None:
            ref_step = field_steps[0]

        df_ref = results.load_field_data(ref_step).set_index("id")
        df_cur = results.load_field_data(step).set_index("id")
        common = df_ref.index.intersection(df_cur.index)
        if len(common) == 0:
            raise ValueError(
                "No element ids shared between the reference and target steps."
            )

        cols = list(self.ori_cols)
        mrp0 = df_ref.loc[common, cols].to_numpy(dtype=float)
        mrp_n = df_cur.loc[common, cols].to_numpy(dtype=float)

        r0 = mrp_to_matrix(torch.as_tensor(mrp0, dtype=torch.float64))
        rn = mrp_to_matrix(torch.as_tensor(mrp_n, dtype=torch.float64))
        deg = misorientation_matrix(rn, r0, self.symmetry, "degrees").cpu().numpy()

        return common.to_numpy(dtype=np.int64), np.asarray(deg, dtype=float)

    def write_reorientation_field(
        self,
        results,
        step: int,
        out_csv: str,
        ref_step: Optional[int] = None,
        extra_cols: Sequence[str] = (),
    ) -> str:
        """Write ``id,x,y,z,reorientation_deg`` (+ optional passthrough) for REI.

        The output is consumed unchanged by
        :class:`~graintrace.rare_cluster_indicator.IdentifyRareClusters`. Passing
        ``extra_cols`` (e.g. ``nye_tensor_*`` or stress columns) carries those
        features through so the same CSV can drive a *combined* rare-event criterion.

        Args:
            results: a :class:`~graintrace.simulation_postprocessing.SimulationResults`.
            step: field-step index to measure.
            out_csv: output CSV path.
            ref_step: reference field-step index (defaults to the first field step).
            extra_cols: additional feature columns to copy through for combined REI.

        Returns:
            The output CSV path.
        """
        # pylint: disable=import-outside-toplevel
        import pandas as pd

        ids, deg = self.reorientation_from_reference(results, step, ref_step=ref_step)
        df_cur = results.load_field_data(step).set_index("id")
        sub = df_cur.loc[ids]

        out = {"id": ids}
        for c in self.coord_cols:
            out[c] = sub[c].to_numpy()
        out["reorientation_deg"] = deg
        for c in extra_cols:
            if c not in sub.columns:
                raise KeyError(f"extra_cols column '{c}' not in the field CSV.")
            out[c] = sub[c].to_numpy()

        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        pd.DataFrame(out).to_csv(out_csv, index=False)
        return out_csv

    # ----------------------------------------------------------------------- #
    # Orientation-based segmentation (graph / Leiden) + stateless helpers
    # ----------------------------------------------------------------------- #
    @staticmethod
    def mean_orientation_mrp(mrp_arr: np.ndarray, symmetry: str = "1") -> np.ndarray:
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

        from .orientation_helper import (
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

    @staticmethod
    def cluster_orientation_spread(
        mrp: np.ndarray,
        labels: np.ndarray,
        symmetry: str = "432",
        q: float = 95.0,
        max_points: int = 5000,
        seed: int = 0,
        background_label: Union[int, Iterable[int], None] = -1,
    ) -> Dict[int, float]:
        """Per-cluster intragranular orientation spread, in degrees.

        The spread of a cluster is the ``q``-th percentile (symmetry-aware)
        misorientation of its points to the cluster's mean orientation. It is a
        descriptive statistic; the percolation *flag* used by :meth:`pick_gamma`
        and :meth:`adjacency_remerge` is :meth:`cluster_leak_check`, which uses
        this spread only as a coarse backstop (a high percentile misses a leak of
        a few percent of the points, and flags a real grain with a smooth
        intragranular gradient). Vectorised over all clusters at once.

        Args:
            mrp: ``(N, 3)`` neml2 MRP orientations.
            labels: ``(N,)`` integer cluster labels aligned to ``mrp``.
            symmetry: crystal symmetry (orbifold notation).
            q: percentile of the within-cluster misorientation distribution.
            max_points: clusters larger than this are randomly subsampled (the mean
                and the percentile use the subsample); ``0`` disables subsampling.
            seed: RNG seed for that subsampling.
            background_label: label, or iterable of labels, that is not a cluster
                (e.g. raw-grid void ``0``), on top of every negative label.

        Returns:
            ``{label: spread_deg}`` for every cluster label present. Negative
            labels are never clusters: ``-1`` marks material-but-unsegmented points
            (see :meth:`~graintrace.nf.segment.remove_small_segments`), scattered
            orphans with no shared orientation whose "spread" would be large by
            construction and flag every :meth:`pick_gamma` ``"spread"`` candidate.
        """
        stats = _cluster_stats(
            mrp, labels, symmetry, q, max_points, seed, background_label
        )
        if stats is None:
            return {}
        uniq, _, _, spread, counts = stats
        return {
            int(u): (float(s) if c >= 2 else 0.0)
            for u, s, c in zip(uniq, spread, counts)
        }

    @staticmethod
    def cluster_leak_check(
        mrp: np.ndarray,
        labels: np.ndarray,
        tol_deg: float,
        symmetry: str = "432",
        min_leak_frac: float = 0.02,
        min_leak_points: int = 10,
        q: float = 95.0,
        backstop_factor: float = 2.0,
        max_points: int = 5000,
        seed: int = 0,
        background_label: Union[int, Iterable[int], None] = -1,
    ) -> Dict[int, Dict[str, Any]]:
        """Flag clusters that leaked across a grain boundary (physical percolation).

        A cluster is **flagged** when either:

        * **a separated orientation population** -- at least ``min_leak_frac`` of
          its points (and at least ``min_leak_points``) form a population that is
          split off from the dominant one by an orientation *gap* and sits more
          than ``tol_deg`` from it; or
        * **the backstop** -- its ``q``-th percentile spread
          (:meth:`cluster_orientation_spread`) exceeds ``backstop_factor *
          tol_deg``, i.e. the cluster is too wide to be one grain at this cutoff
          whatever its internal structure.

        *Why not the spread percentile alone.* A percentile to the mean is blind
        to the shape of the distribution. 1000 points of grain A plus 45 points
        of a grain 30 deg away (a 4 % leak) give a p95 of only ~2 deg, while one
        real grain with a smooth 12 deg orientation gradient gives ~5.7 deg, so a
        percentile threshold either misses the leak or splits the grain. What
        separates them is *continuity*: a gradient has small orientation steps
        between neighbouring points, a leak has a jump. The check therefore
        embeds each point as its rotation vector from the cluster mean (the
        tangent space, where distance approximates misorientation), bins it on a
        ``tol_deg / 4`` grid, and connects dense, touching cells into
        populations. A gradient fills one connected run of cells; a leaked grain
        is a second dense blob behind an empty gap; isolated noisy voxels (a
        speckled low-confidence bridge) fill no dense cell, so they cannot join
        two grains. Only a population that is both sizeable (``min_leak_frac`` /
        ``min_leak_points``, so a few outlier voxels never flag) and farther than
        ``tol_deg`` from the dominant one counts. Orientation-only, so it needs
        no coordinates and is identical for voxel grids and mesh centroids.

        Args:
            mrp: ``(N, 3)`` neml2 MRP orientations.
            labels: ``(N,)`` integer cluster labels aligned to ``mrp``.
            tol_deg: the segmentation misorientation cutoff (degrees).
            symmetry: crystal symmetry (orbifold notation).
            min_leak_frac: minimum fraction of the cluster a separated population
                must hold to count as a leak.
            min_leak_points: minimum number of points it must hold.
            q: percentile for the reported spread and the backstop.
            backstop_factor: the backstop flags a spread above
                ``backstop_factor * tol_deg``; ``None`` or ``inf`` disables it.
            max_points: per-cluster random subsample size (``0`` = all points).
            seed: RNG seed for that subsampling.
            background_label: label(s) that are not clusters, on top of every
                negative label (unsegmented points are never checked).

        Returns:
            ``{label: {"flagged", "spread_deg", "leak_frac", "leak_sep_deg"}}``
            for every cluster label present.
        """
        stats = _cluster_stats(
            mrp, labels, symmetry, q, max_points, seed, background_label
        )
        if stats is None:
            return {}
        uniq, vecs, inv, spread, counts = stats
        dist = np.linalg.norm(vecs, axis=1)
        max_dist = np.zeros(len(uniq))
        np.maximum.at(max_dist, inv, dist)
        order = np.argsort(inv, kind="stable")
        _, starts = _group_offsets(inv, len(uniq))
        backstop = (
            np.inf
            if backstop_factor is None
            else float(backstop_factor) * float(tol_deg)
        )
        out: Dict[int, Dict[str, Any]] = {}
        for c, lab in enumerate(uniq):
            n_c = int(counts[c])
            sp = float(spread[c]) if n_c >= 2 else 0.0
            n_leak, sep = 0, 0.0
            min_size = max(int(min_leak_points), int(np.ceil(min_leak_frac * n_c)))
            # a sizeable second population needs >= 2 * min_size points, and two
            # centroids farther than tol apart need a point beyond tol / 2
            if n_c >= 2 * min_size and max_dist[c] > tol_deg / 2.0:
                sel = order[starts[c] : starts[c] + n_c]
                n_leak, sep = _separated_populations(vecs[sel], tol_deg, min_size)
            out[int(lab)] = {
                "flagged": bool(n_leak > 0 or sp > backstop),
                "spread_deg": sp,
                "leak_frac": float(n_leak / n_c) if n_c else 0.0,
                "leak_sep_deg": float(sep),
            }
        return out

    @staticmethod
    def merge_fragments_by_misorientation(
        mrp: np.ndarray,
        labels: np.ndarray,
        tol_deg: float,
        symmetry: str = "432",
    ) -> np.ndarray:
        """Merge sub-grain fragments whose mean orientations are within ``tol_deg``.

        Agglomerative, orientation-only merge for intragranular sub-grains: the two
        fragments with the smallest (symmetry-aware) mean misorientation are merged
        first, the merged mean is recomputed, and the process repeats until the
        closest remaining pair is at least ``tol_deg`` apart. Recomputing the mean
        each step avoids the single-linkage *chaining* that a fixed-mean union-find
        would suffer. Use it to impose a **minimum sub-grain misorientation** so a
        smooth intragranular gradient is not reported as several near-identical
        fragments.

        Args:
            mrp: ``(N, 3)`` neml2 MRP orientations aligned to ``labels``.
            labels: ``(N,)`` integer fragment labels.
            tol_deg: minimum mean misorientation (degrees) to keep two fragments
                distinct; closer pairs merge.
            symmetry: crystal symmetry (orbifold notation).

        Returns:
            ``(N,)`` contiguous 0-based labels after merging.
        """
        # pylint: disable=import-outside-toplevel
        import torch

        from .orientation_helper import misorientation_matrix, mrp_to_matrix

        labels = np.asarray(labels, dtype=np.int64).copy()
        mrp = np.asarray(mrp, dtype=float)
        while True:
            uniq = np.unique(labels)
            if len(uniq) < 2:
                break
            means = {
                int(u): FragmentationAnalyzer.mean_orientation_mrp(
                    mrp[labels == u], symmetry
                )
                for u in uniq
            }
            best_pair, best_deg = None, float("inf")
            for i, ui in enumerate(uniq):
                for j in range(i + 1, len(uniq)):
                    a, b = int(ui), int(uniq[j])
                    deg = float(
                        misorientation_matrix(
                            mrp_to_matrix(
                                torch.as_tensor(means[a][None, :], dtype=torch.float64)
                            ),
                            mrp_to_matrix(
                                torch.as_tensor(means[b][None, :], dtype=torch.float64)
                            ),
                            symmetry=symmetry,
                        ).item()
                    )
                    if deg < best_deg:
                        best_deg, best_pair = deg, (a, b)
            if best_pair is None or best_deg >= tol_deg:
                break
            labels[labels == best_pair[1]] = best_pair[0]
        return np.unique(labels, return_inverse=True)[1].astype(np.int64)

    @staticmethod
    def absorb_fragments(
        coords: np.ndarray,
        labels: np.ndarray,
        min_size: int,
        k_neighbors: int = 8,
        max_passes: int = 25,
        mrp: Optional[np.ndarray] = None,
        max_miso_deg: Optional[float] = None,
        symmetry: str = "432",
        background_label: Union[int, Iterable[int], None] = -1,
    ) -> np.ndarray:
        """Absorb sub-``min_size`` clusters into a touching neighbour.

        Uses kNN contact counts for adjacency, so it works for any point set (voxel
        grids or scattered mesh centroids). Iterates until no sub-``min_size``
        cluster can be absorbed (or ``max_passes``), then relabels contiguously
        from 0. Vectorised over points (the per-pass work is a kNN contact count,
        not a Python loop over voxels).

        **Orientation gate.** Without ``mrp``/``max_miso_deg`` (the legacy rule) a
        small cluster goes to its *largest-contact* neighbour whatever their
        misorientation, which on real NF data swallows every small, differently
        oriented grain into its neighbours and drags their mean orientations. With
        both given, a small cluster is absorbed only into a touching neighbour
        whose (symmetry-aware) mean orientation is within ``max_miso_deg``,
        choosing the **closest misorientation** among the touching neighbours
        (ties go to the larger contact); with no such neighbour it stays its own
        small grain. Cluster means are recomputed each pass, and a small cluster
        whose target is itself being absorbed waits for the next pass, so gated
        absorption never chains past ``max_miso_deg``.

        Args:
            coords: ``(N, 3)`` point coordinates.
            labels: ``(N,)`` integer cluster labels.
            min_size: clusters with fewer points are absorbed.
            k_neighbors: neighbours per point used to measure cluster contact.
            max_passes: safety cap on absorption passes.
            mrp: ``(N, 3)`` neml2 MRP orientations, enabling the orientation gate.
            max_miso_deg: gate tolerance (degrees); ``None`` disables the gate.
            symmetry: crystal symmetry (orbifold notation) for the gate.
            background_label: label(s) that are not clusters, on top of every
                negative label. Such points are never absorbed, never absorb
                anything, and keep their label (e.g. pass ``(0, -1)`` on a raw
                voxel grid with void ``0``; the clusters are then numbered from 1).

        Returns:
            ``(N,)`` labels: clusters contiguous from 0 (skipping any background
            value), unsegmented/background points unchanged.
        """
        labels = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
        ignore = _ignored_mask(labels, background_label)
        gated = mrp is not None and max_miso_deg is not None
        nbr_idx = _knn_indices(np.asarray(coords, dtype=float), k_neighbors)
        ops = _symmetry_ops_np(symmetry) if gated else None
        rmat = _mrp_to_matrix_np(mrp) if gated else None

        for _ in range(max_passes):
            uniq, counts = np.unique(labels[~ignore], return_counts=True)
            small = set(int(u) for u, c in zip(uniq, counts) if c < min_size)
            if not small or len(uniq) == 1:
                break

            pairs, ncontact = _contact_pairs(nbr_idx, labels, ignore)
            if len(pairs) == 0:
                break
            # directed (src -> dst) rows in the legacy dict-scan order
            src = np.stack([pairs[:, 0], pairs[:, 1]], axis=1).reshape(-1)
            dst = np.stack([pairs[:, 1], pairs[:, 0]], axis=1).reshape(-1)
            cnt = np.repeat(ncontact, 2)
            seq = np.arange(len(src))
            if not gated:
                # largest contact, first-seen on ties (legacy `n > best` rule)
                pick = np.lexsort((seq, -cnt, src))
                first = np.ones(len(pick), dtype=bool)
                first[1:] = src[pick][1:] != src[pick][:-1]
                best = dict(zip(src[pick][first].tolist(), dst[pick][first].tolist()))
                remap = {}
                for s in small:
                    if s in best:
                        remap[s] = best[s]
                if not remap:
                    break
                # resolve chains (small -> small) to a stable target
                for s in list(remap):
                    seen = {s}
                    tgt = remap[s]
                    while tgt in remap and tgt not in seen:
                        seen.add(tgt)
                        tgt = remap[tgt]
                    remap[s] = tgt
            else:
                remap = FragmentationAnalyzer._gated_absorb_targets(
                    labels, ignore, rmat, ops, small, src, dst, cnt, max_miso_deg
                )
                if not remap:
                    break
            vals = np.unique(labels)
            new = np.array([remap.get(int(v), int(v)) for v in vals], dtype=np.int64)
            labels = new[np.searchsorted(vals, labels)]

        return _relabel_contiguous(labels, ignore)

    @staticmethod
    def _gated_absorb_targets(
        labels, ignore, rmat, ops, small, src, dst, cnt, max_miso_deg
    ) -> Dict[int, int]:
        """One pass of misorientation-gated absorption targets (``small -> dst``)."""
        is_small = np.isin(src, np.fromiter(small, dtype=np.int64))
        src, dst, cnt = src[is_small], dst[is_small], cnt[is_small]
        if len(src) == 0:
            return {}
        involved = np.unique(np.concatenate([src, dst]))
        pts = np.flatnonzero(np.isin(labels, involved) & ~ignore)
        inv = np.searchsorted(involved, labels[pts])
        mean_r = _moments_mean_matrix(
            _cluster_moments(rmat[pts], inv, len(involved), ops)
        )
        a = np.searchsorted(involved, src)
        b = np.searchsorted(involved, dst)
        miso = _sym_miso_deg(mean_r[a], mean_r[b], ops)
        ok = miso <= float(max_miso_deg)
        if not ok.any():
            return {}
        src, dst, cnt, miso = src[ok], dst[ok], cnt[ok], miso[ok]
        pick = np.lexsort((-cnt, miso, src))  # closest, then largest contact
        first = np.ones(len(pick), dtype=bool)
        first[1:] = src[pick][1:] != src[pick][:-1]
        tgt = dict(zip(src[pick][first].tolist(), dst[pick][first].tolist()))
        size = dict(zip(*np.unique(labels[~ignore], return_counts=True)))
        remap = {}
        for s, t in sorted(tgt.items()):
            if t not in tgt:
                remap[s] = t
            elif tgt[t] == s and (size[s], -s) < (size[t], -t):
                remap[s] = t  # mutual pair: the smaller joins the larger
        if not remap:  # a cycle of >= 3 small clusters: break it at one edge
            s0 = min(tgt)
            remap[s0] = tgt[s0]
        return remap

    @staticmethod
    def adjacency_remerge(
        coords: np.ndarray,
        mrp: np.ndarray,
        labels: np.ndarray,
        tol_deg: float,
        symmetry: str = "432",
        k_neighbors: int = 8,
        percolation_max_frac: Optional[float] = None,
        spread_tol_deg: Optional[float] = None,
        background_label: Union[int, Iterable[int], None] = -1,
    ) -> np.ndarray:
        """Merge spatially adjacent clusters within ``tol_deg`` misorientation.

        Adjacency is kNN contact, so it works for both voxel grids and scattered
        mesh element centroids. Bounded by **misorientation only** by default:
        grain size is not a percolation signal, so no merge is rejected for being
        large.

        **Agglomerative (default, no size cap).** The closest adjacent pair (by
        mean misorientation) merges first, the union's mean is recomputed, and its
        misorientation to every neighbour is re-evaluated before the next merge
        (as :meth:`merge_fragments_by_misorientation` does). A fixed-mean
        union-find links every pair inside ``tol_deg`` and so *chains*: eight
        adjacent blocks 2.5 deg apart (17.5 deg end to end) collapse into one at
        ``tol_deg=3``. Recomputing the mean stops that: once two blocks merge,
        their mean is 3.75 deg from the next block, so the chain cannot propagate.
        The union mean comes from summed quaternion moments (folded by the
        symmetry operator that brings the two means together), so each merge is
        O(1), not O(cluster size).

        Optional guards:

        * ``spread_tol_deg`` -- the physical check: reject a merge whose union
          would be flagged by :meth:`cluster_leak_check` at this tolerance (a
          separated orientation population, or a spread far beyond it). Costs one
          check per candidate merge. :meth:`segment` turns it on by default.
        * ``percolation_max_frac`` -- the legacy size cap, kept as an explicit
          opt-in for unmasked data. It also selects the **legacy** fixed-mean
          union-find (tightest pairs first), so the opt-in reproduces the old
          labels exactly. Any union above this fraction of the point count is
          skipped, which makes the method inert on small point sets (one grain's
          ~100 elements in :meth:`grain_fragments`).

        Args:
            coords: ``(N, 3)`` point coordinates.
            mrp: ``(N, 3)`` neml2 MRP orientations.
            labels: ``(N,)`` integer cluster labels.
            tol_deg: merge adjacent clusters whose mean orientations are within this.
            symmetry: crystal symmetry (orbifold notation).
            k_neighbors: neighbours per point used to measure cluster contact.
            percolation_max_frac: legacy size cap (and legacy union-find); ``None``
                (the default) applies no size cap and merges agglomeratively.
            spread_tol_deg: reject a merge whose union fails
                :meth:`cluster_leak_check` at this tolerance (degrees); ``None``
                (the default) skips the check.
            background_label: label(s) that are not clusters, on top of every
                negative label; such points never merge and keep their label.

        Returns:
            ``(N,)`` labels: clusters contiguous from 0 (skipping any background
            value), unsegmented/background points unchanged.
        """
        labels = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
        mrp = np.asarray(mrp, dtype=float).reshape(-1, 3)
        ignore = _ignored_mask(labels, background_label)
        if len(np.unique(labels[~ignore])) < 2:
            return labels
        contacts = _knn_cluster_adjacency(coords, labels, k_neighbors, ignore=ignore)
        if not contacts:
            return labels
        if percolation_max_frac is not None:
            return FragmentationAnalyzer._remerge_union_find(
                mrp,
                labels,
                ignore,
                contacts,
                tol_deg,
                symmetry,
                percolation_max_frac * float(len(labels)),
                spread_tol_deg,
            )
        return FragmentationAnalyzer._remerge_agglomerative(
            mrp, labels, ignore, contacts, tol_deg, symmetry, spread_tol_deg
        )

    @staticmethod
    def _union_is_leaky(mrp_union: np.ndarray, spread_tol_deg: float, symmetry: str):
        """True when the union of two clusters fails :meth:`cluster_leak_check`."""
        chk = FragmentationAnalyzer.cluster_leak_check(
            mrp_union,
            np.zeros(len(mrp_union), dtype=np.int64),
            spread_tol_deg,
            symmetry=symmetry,
        )
        return bool(chk[0]["flagged"])

    @staticmethod
    def _remerge_union_find(
        mrp, labels, ignore, contacts, tol_deg, symmetry, size_cap, spread_tol_deg
    ) -> np.ndarray:
        """Legacy fixed-mean union-find re-merge (``percolation_max_frac`` path)."""
        uniq, counts = np.unique(labels[~ignore], return_counts=True)
        id2idx = {int(u): k for k, u in enumerate(uniq)}
        means = np.array(
            [
                FragmentationAnalyzer.mean_orientation_mrp(mrp[labels == u], symmetry)
                for u in uniq
            ]
        )
        pairs = list(contacts.keys())
        misos = np.array(
            [
                _pair_misorientation_deg(means[id2idx[a]], means[id2idx[b]], symmetry)
                for a, b in pairs
            ]
        )
        comp_size = counts.astype(np.float64).copy()
        parent = list(range(len(uniq)))
        members: Optional[Dict[int, np.ndarray]] = None
        if spread_tol_deg is not None:
            members = {k: np.flatnonzero(labels == int(u)) for k, u in enumerate(uniq)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for k in np.argsort(misos):
            if misos[k] >= tol_deg:
                break
            a, b = pairs[k]
            ra, rb = find(id2idx[a]), find(id2idx[b])
            if ra == rb:
                continue
            if comp_size[ra] + comp_size[rb] > size_cap:
                continue
            union_idx = None
            if members is not None:
                union_idx = np.concatenate([members[ra], members[rb]])
                if FragmentationAnalyzer._union_is_leaky(
                    mrp[union_idx], spread_tol_deg, symmetry
                ):
                    continue
            parent[rb] = ra
            comp_size[ra] += comp_size[rb]
            if members is not None:
                members[ra] = union_idx
                members.pop(rb, None)

        roots = np.array([find(k) for k in range(len(uniq))])
        out = labels.copy()
        valid = ~ignore
        out[valid] = roots[np.searchsorted(uniq, labels[valid])]
        return _relabel_contiguous(out, ignore)

    @staticmethod
    def _remerge_agglomerative(
        mrp, labels, ignore, contacts, tol_deg, symmetry, spread_tol_deg
    ) -> np.ndarray:
        """Closest-pair-first re-merge with the union mean recomputed per merge."""
        # pylint: disable=import-outside-toplevel
        import heapq

        valid = np.flatnonzero(~ignore)
        uniq, inv = np.unique(labels[valid], return_inverse=True)
        inv = inv.reshape(-1)
        n_c = len(uniq)
        ops = _symmetry_ops_np(symmetry)
        op_quat = _matrix_to_quat_np(ops)
        mom = _cluster_moments(_mrp_to_matrix_np(mrp[valid]), inv, n_c, ops)
        mean_r = _moments_mean_matrix(mom)
        size = np.bincount(inv, minlength=n_c).astype(np.int64)
        members: Optional[List[np.ndarray]] = None
        if spread_tol_deg is not None:
            order = np.argsort(inv, kind="stable")
            _, starts = _group_offsets(inv, n_c)
            members = [order[starts[c] : starts[c] + size[c]] for c in range(n_c)]
        mrp_valid = mrp[valid]

        pairs = np.searchsorted(uniq, np.array(list(contacts.keys()), dtype=np.int64))
        nbrs: List[set] = [set() for _ in range(n_c)]
        for a, b in pairs.tolist():
            nbrs[a].add(b)
            nbrs[b].add(a)
        misos = _sym_miso_deg(mean_r[pairs[:, 0]], mean_r[pairs[:, 1]], ops)
        version = np.zeros(n_c, dtype=np.int64)
        heap = [
            (float(m), int(a), int(b), 0, 0)
            for m, (a, b) in zip(misos, pairs.tolist())
            if m < tol_deg
        ]
        heapq.heapify(heap)
        alive = np.ones(n_c, dtype=bool)
        parent = np.arange(n_c)

        while heap:
            _, a, b, va, vb = heapq.heappop(heap)
            if not (alive[a] and alive[b]) or version[a] != va or version[b] != vb:
                continue
            if members is not None and FragmentationAnalyzer._union_is_leaky(
                mrp_valid[np.concatenate([members[a], members[b]])],
                spread_tol_deg,
                symmetry,
            ):
                continue
            if size[b] > size[a]:
                a, b = b, a
            # fold b's moments into a's variant frame, then add
            cand = np.einsum("kij,jl->kil", ops, mean_r[b])
            s_op = int(np.argmax(np.einsum("ij,kij->k", mean_r[a], cand)))
            lq = _left_quat_matrix(op_quat[s_op])
            mom[a] = mom[a] + lq @ mom[b] @ lq.T
            mean_r[a] = _moments_mean_matrix(mom[a][None])[0]
            size[a] += size[b]
            alive[b] = False
            parent[b] = a
            version[a] += 1
            if members is not None:
                members[a] = np.concatenate([members[a], members[b]])
                members[b] = members[b][:0]
            for n in nbrs[b]:
                nbrs[n].discard(b)
                if n != a:
                    nbrs[n].add(a)
                    nbrs[a].add(n)
            nbrs[a].discard(b)
            nbrs[a].discard(a)
            nbrs[b] = set()
            if nbrs[a]:
                arr = np.fromiter(nbrs[a], dtype=np.int64)
                mm = _sym_miso_deg(
                    np.repeat(mean_r[a][None], len(arr), 0), mean_r[arr], ops
                )
                for m, n in zip(mm.tolist(), arr.tolist()):
                    if m < tol_deg:
                        heapq.heappush(
                            heap, (m, a, n, int(version[a]), int(version[n]))
                        )

        roots = parent.copy()
        for c in range(n_c):
            r = c
            while roots[r] != r:
                r = roots[r]
            roots[c] = r
        out = labels.copy()
        out[valid] = roots[inv]
        return _relabel_contiguous(out, ignore)

    @staticmethod
    def gamma_sweep_leiden(
        csv_path: str,
        ckpt_base: str,
        gammas: Sequence[float],
        n_points: int,
        spec,
        weight_cfg,
        max_edge_distance: float,
        graph_mode: str = "grid",
        manhattan_radius: int = 2,
        k: int = 16,
        reduce_topk: Optional[int] = 12,
        n_jobs: int = 1,
        n_threads: int = 1,
        seed: int = 42,
    ) -> List[dict]:
        """Build the misorientation graph once, then run Leiden per gamma.

        The edges/weights depend only on the graph + misorientation cutoff (not
        gamma), so they are checkpointed and reused across the sweep.

        Returns:
            list of dicts with keys ``gamma``, ``n_grains``, ``max_frac``, ``labels``.
        """
        # pylint: disable=import-outside-toplevel
        from .graph_spatial_cluster import GraphSpatialCluster

        gsc = GraphSpatialCluster(
            csv_path=csv_path, id_col="id", coord_cols=("x", "y", "z")
        )
        # _load_checkpoint requires the meta file too, so a partially written
        # checkpoint must read as absent rather than crashing the resume.
        ckpt_exists = all(
            os.path.exists(ckpt_base + suffix)
            for suffix in (".edges.npy", ".weights.npy", ".meta.json")
        )
        results: List[dict] = []
        for i, g in enumerate(gammas):
            out = gsc.run(
                spec=spec,
                graph_mode=graph_mode,
                k=k,
                manhattan_radius=manhattan_radius,
                grid_tol=1e-6,
                n_jobs=n_jobs,
                n_threads=n_threads,
                weight_chunk_size=1_000_000,
                segmenter="leiden",
                seed=seed,
                return_labels=True,
                max_edge_distance=max_edge_distance,
                weight_cfg=weight_cfg,
                reduce_edges_topweights_k=reduce_topk,
                networkit_kwargs={"gamma": float(g)},
                checkpoint_base_path=ckpt_base,
                resume_from_checkpoint=(i > 0) or ckpt_exists,
            )
            labels = np.asarray(out["extras"]["labels"], dtype=np.int64)
            _, cnt = np.unique(labels, return_counts=True)
            results.append(
                {
                    "gamma": float(g),
                    "n_grains": int(len(cnt)),
                    "max_frac": float(cnt.max() / max(n_points, 1)),
                    "labels": labels,
                }
            )
        return results

    @staticmethod
    def pick_gamma_by_percolation(
        sweep: Sequence[dict], percolation_max_frac: Optional[float] = None
    ) -> dict:
        """Pick the smallest gamma whose biggest cluster is below a size cap.

        **Legacy size-cap pick.** Grain size is not a valid percolation signal: a
        legitimately large grain is forced below the cap and cut into arbitrary
        pieces, so the chosen gamma reflects the cap rather than the data. Prefer
        :meth:`pick_gamma`, which selects on the physical orientation-spread check.
        This remains available as an explicit opt-in for legacy, unmasked data.

        The raw Leiden count is near gamma-invariant here (all gammas over-segment);
        gamma controls only the biggest ("percolating") cluster, and the final grain
        count is set by absorption. So choose the least over-resolved, still
        de-percolated gamma; if none qualifies, the overall minimum ``max_frac``.

        Args:
            sweep: per-gamma results from :meth:`gamma_sweep_leiden`.
            percolation_max_frac: size cap as a fraction of the point count;
                ``None`` (the default) applies no cap, so the lowest swept gamma
                wins.

        Returns:
            The selected sweep entry.
        """
        if percolation_max_frac is None:
            return min(sweep, key=lambda r: r["gamma"])
        ok = [r for r in sweep if r["max_frac"] < percolation_max_frac]
        if ok:
            return min(ok, key=lambda r: r["gamma"])
        return min(sweep, key=lambda r: r["max_frac"])

    @staticmethod
    def pick_gamma(
        sweep: Sequence[dict],
        mode: str = "lowest",
        mrp: Optional[np.ndarray] = None,
        spread_tol_deg: Optional[float] = None,
        symmetry: str = "432",
        percolation_max_frac: Optional[float] = None,
        spread_q: float = 95.0,
    ) -> dict:
        """Choose a Leiden resolution from a gamma sweep, without a size cap.

        Two cap-free modes:

        * ``"lowest"`` (the default) -- the lowest swept gamma, i.e. the least
          over-resolved segmentation. Correct when the data has already been masked
          (confidence/void), so there is no bridge for a cluster to percolate along.
        * ``"spread"`` -- the lowest gamma for which **no** cluster is flagged by
          the percolation check (:meth:`cluster_leak_check` at
          ``spread_tol_deg``). If every gamma flags something, the one with the
          fewest flagged clusters wins (lowest gamma on a tie).

        Passing ``percolation_max_frac`` restores the legacy size-cap pick
        (:meth:`pick_gamma_by_percolation`) and ignores ``mode``.

        **Diagnostics.** Whenever ``mrp`` and ``spread_tol_deg`` are given, every
        entry in ``sweep`` -- in every mode, including the legacy cap -- is
        annotated in place with ``n_flagged`` (clusters flagged by the check) and
        ``max_spread_deg`` (largest ``spread_q`` percentile spread), so the
        choice is inspectable. The annotation never changes the ``"lowest"`` or
        legacy pick.

        Args:
            sweep: per-gamma results from :meth:`gamma_sweep_leiden`.
            mode: ``"lowest"`` or ``"spread"``.
            mrp: ``(N, 3)`` neml2 MRP orientations; required for ``"spread"``.
            spread_tol_deg: the percolation-check tolerance (degrees); required for
                ``"spread"``. Set it to the segmentation misorientation cutoff.
            symmetry: crystal symmetry (orbifold notation).
            percolation_max_frac: legacy size cap opt-in; ``None`` means no cap.
            spread_q: percentile for the reported spread and the check's backstop.

        Returns:
            The selected sweep entry.

        Raises:
            ValueError: on an unknown ``mode``, an empty ``sweep``, or ``"spread"``
                without ``mrp`` and ``spread_tol_deg``.
        """
        if not sweep:
            raise ValueError("pick_gamma got an empty sweep.")
        # Validated before the legacy short-circuit below, not after it: the cap
        # path ignores mode, so a typo'd mode combined with percolation_max_frac
        # would otherwise be accepted silently and the caller would believe a
        # pick they never asked for was the one they configured.
        if mode not in ("lowest", "spread"):
            raise ValueError(
                f"Unknown gamma pick mode '{mode}'; expected 'lowest' or 'spread'."
            )
        if mode == "spread" and percolation_max_frac is None:
            if mrp is None or spread_tol_deg is None:
                raise ValueError(
                    "gamma_pick='spread' needs both mrp and spread_tol_deg "
                    "(the orientation-spread check has nothing to measure otherwise)."
                )
        if mrp is not None and spread_tol_deg is not None:
            for r in sweep:
                chk = FragmentationAnalyzer.cluster_leak_check(
                    mrp, r["labels"], spread_tol_deg, symmetry=symmetry, q=spread_q
                )
                r["n_flagged"] = int(sum(v["flagged"] for v in chk.values()))
                r["max_spread_deg"] = (
                    float(max(v["spread_deg"] for v in chk.values())) if chk else 0.0
                )
        if percolation_max_frac is not None:  # legacy opt-in
            return FragmentationAnalyzer.pick_gamma_by_percolation(
                sweep, percolation_max_frac
            )
        ordered = sorted(sweep, key=lambda r: r["gamma"])
        if mode == "lowest":
            return ordered[0]
        for r in ordered:
            if r["n_flagged"] == 0:
                return r
        return min(ordered, key=lambda r: r["n_flagged"])

    def segment(
        self,
        coords: np.ndarray,
        mrp: np.ndarray,
        misori_tol_deg: float = 5.0,
        graph_mode: str = "grid",
        gamma: Optional[float] = None,
        gamma_sweep: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
        rbf_sigma_deg: Optional[float] = None,
        manhattan_radius: int = 2,
        k: int = 16,
        reduce_topk: Optional[int] = 12,
        grain_threshold_final: int = 30,
        remerge_miso_deg: float = 3.0,
        gamma_pick: str = "lowest",
        spread_tol_deg: Optional[float] = None,
        remerge_spread_check: Union[bool, str] = "auto",
        percolation_max_frac: Optional[float] = None,
        absorb_max_miso_deg: Union[float, str, None] = "auto",
        n_jobs: int = 1,
        n_threads: int = 1,
        seed: int = 42,
        workdir: Optional[str] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """Segment a point set by crystal orientation (graph / Leiden).

        Preferred over flood fill, which over-merges into percolating grains. Wraps
        :class:`~graintrace.graph_spatial_cluster.GraphSpatialCluster` with the
        symmetry-aware misorientation metric on neml2 MRP orientations and the tuned
        recipe: fixed broad RBF sigma, hard misorientation cutoff, a gamma pick,
        then a misorientation-gated absorb + an agglomerative adjacency re-merge.

        **No grain-size cap by default.** Grain size is not a percolation signal, so
        nothing here forces a cluster below a fraction of the point set; a real
        grain that occupies 10-30% of a small field of view survives whole. Mask the
        data first (confidence/void) so there is no bridge to percolate along, then
        leave the cap off. ``percolation_max_frac`` restores the legacy cap for
        unmasked data, and ``gamma_pick="spread"`` is the physical alternative: it
        flags a cluster whose intragranular orientation spread exceeds the
        segmentation cutoff (:meth:`cluster_leak_check`). Per-gamma diagnostics
        (``n_grains``, ``max_frac``, ``n_flagged``, ``max_spread_deg`` for every
        swept gamma) land in :attr:`last_gamma_diagnostics`.

        **Clean-up stages (no-cap defaults).** Sub-``grain_threshold_final``
        fragments are absorbed only into a touching neighbour within
        ``absorb_max_miso_deg`` (default ``misori_tol_deg``) -- otherwise a small,
        differently oriented grain is swallowed and drags its neighbour's mean.
        The re-merge is agglomerative (closest pair first, means recomputed, so it
        cannot chain) and rejects a union that fails the percolation check.
        Passing ``percolation_max_frac`` restores the legacy pipeline exactly:
        size-capped pick, orientation-blind absorb, fixed-mean union-find re-merge
        with no spread check (unless ``remerge_spread_check=True`` or a numeric
        ``absorb_max_miso_deg`` is passed explicitly).

        Args:
            coords: ``(N, 3)`` point coordinates (voxel centres or element centroids).
            mrp: ``(N, 3)`` neml2 MRP orientations.
            misori_tol_deg: hard misorientation edge cutoff (degrees).
            graph_mode: ``"grid"`` for regular voxel grids (NF/EBSD), ``"knn"`` for
                scattered mesh element centroids.
            gamma: fixed Leiden resolution; if ``None``, a swept-and-picked gamma.
            gamma_sweep: resolutions to sweep when ``gamma`` is ``None``.
            rbf_sigma_deg: fixed RBF sigma (degrees); defaults to half ``misori_tol_deg``.
            manhattan_radius: grid connectivity radius (``graph_mode="grid"``).
            k: neighbours per node (``graph_mode="knn"``).
            reduce_topk: keep top-k edges per node by weight before clustering.
            grain_threshold_final: minimum cluster size; smaller ones are absorbed
                (into an orientation-compatible neighbour, see
                ``absorb_max_miso_deg``).
            remerge_miso_deg: adjacency re-merge misorientation threshold (degrees).
            gamma_pick: ``"lowest"`` (least over-resolved swept gamma) or
                ``"spread"`` (lowest gamma flagging no cluster by orientation
                spread). Ignored when ``percolation_max_frac`` is set.
            spread_tol_deg: tolerance of the percolation check
                (:meth:`cluster_leak_check`) used by the ``"spread"`` pick, the
                diagnostics and the re-merge guard; defaults to
                ``misori_tol_deg``, the segmentation cutoff.
            remerge_spread_check: reject an adjacency re-merge whose union fails
                :meth:`cluster_leak_check` at ``spread_tol_deg``. ``"auto"`` (the
                default) means on, except on the legacy ``percolation_max_frac``
                path where it is off.
            percolation_max_frac: legacy size cap for the gamma pick and the
                re-merge; ``None`` (the default) applies no cap. Setting it also
                restores the legacy absorb and re-merge (see above).
            absorb_max_miso_deg: orientation gate for the fragment absorption
                (degrees; see :meth:`absorb_fragments`). ``"auto"`` (the default)
                means ``misori_tol_deg``, except on the legacy
                ``percolation_max_frac`` path where it is off; ``None`` disables
                the gate (orientation-blind largest-contact absorption).
            n_jobs: parallel workers for edge-distance build.
            n_threads: threads for Leiden.
            seed: RNG seed.
            workdir: scratch directory for the graph input CSV + checkpoint (a temp
                directory is used and cleaned up if ``None``).
            verbose: print the per-gamma diagnostics table.

        Returns:
            ``(N,)`` contiguous 0-based cluster labels aligned to ``coords`` rows.
        """
        # pylint: disable=import-outside-toplevel
        import pandas as pd

        from .similarity_metric_library import SimilarityMetricLibrary
        from .user_data_class import WeightConfig

        coords = np.asarray(coords, dtype=float)
        mrp = np.asarray(mrp, dtype=float)
        n_points = coords.shape[0]
        if rbf_sigma_deg is None:
            rbf_sigma_deg = misori_tol_deg / 2.0

        spec = SimilarityMetricLibrary().misorientation(
            feature_cols=["mrp_x", "mrp_y", "mrp_z"],
            symmetry=self.symmetry,
            angle_convention="mrp",
            output_unit="radians",
        )
        weight_cfg = WeightConfig(
            mode="rbf",
            power=2.0,
            sigma=float(np.deg2rad(rbf_sigma_deg)),
            sigma_auto=None,
        )

        tmp = workdir or tempfile.mkdtemp(prefix="segori_")
        os.makedirs(tmp, exist_ok=True)
        csv_path = os.path.join(tmp, "seg_input.csv")
        ckpt_base = os.path.join(tmp, "seg_ckpt")
        pd.DataFrame(
            {
                "id": np.arange(n_points),
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "mrp_x": mrp[:, 0],
                "mrp_y": mrp[:, 1],
                "mrp_z": mrp[:, 2],
            }
        ).to_csv(csv_path, index=False)

        gammas = [gamma] if gamma is not None else list(gamma_sweep)
        sweep = self.gamma_sweep_leiden(
            csv_path=csv_path,
            ckpt_base=ckpt_base,
            gammas=gammas,
            n_points=n_points,
            spec=spec,
            weight_cfg=weight_cfg,
            max_edge_distance=float(np.deg2rad(misori_tol_deg)),
            graph_mode=graph_mode,
            manhattan_radius=manhattan_radius,
            k=k,
            reduce_topk=reduce_topk,
            n_jobs=n_jobs,
            n_threads=n_threads,
            seed=seed,
        )
        if spread_tol_deg is None:
            spread_tol_deg = misori_tol_deg
        legacy = percolation_max_frac is not None
        if isinstance(absorb_max_miso_deg, str):
            if absorb_max_miso_deg != "auto":
                raise ValueError(
                    "absorb_max_miso_deg must be a number, None or 'auto'."
                )
            absorb_max_miso_deg = None if legacy else misori_tol_deg
        if isinstance(remerge_spread_check, str):
            if remerge_spread_check != "auto":
                raise ValueError("remerge_spread_check must be a bool or 'auto'.")
            remerge_spread_check = not legacy
        best = self.pick_gamma(
            sweep,
            mode=gamma_pick,
            mrp=mrp,
            spread_tol_deg=spread_tol_deg,
            symmetry=self.symmetry,
            percolation_max_frac=percolation_max_frac,
        )
        labels = best["labels"]
        self.last_gamma_diagnostics = [
            {k: v for k, v in r.items() if k != "labels"} for r in sweep
        ]
        if verbose:
            print(f"gamma pick: gamma={best['gamma']:g} (mode={gamma_pick}, ", end="")
            print(
                "no size cap)"
                if percolation_max_frac is None
                else f"legacy size cap={percolation_max_frac:g})"
            )
            for r in self.last_gamma_diagnostics:
                flagged = r.get("n_flagged")
                print(
                    f"  gamma={r['gamma']:<8g} n_clusters={r['n_grains']:<8d} "
                    f"max_frac={r['max_frac']:<8.4f} "
                    f"n_flagged={'-' if flagged is None else flagged} "
                    f"max_spread_deg={r.get('max_spread_deg', float('nan')):.2f}"
                )

        if grain_threshold_final and grain_threshold_final > 1:
            labels = self.absorb_fragments(
                coords,
                labels,
                grain_threshold_final,
                mrp=mrp if absorb_max_miso_deg is not None else None,
                max_miso_deg=absorb_max_miso_deg,
                symmetry=self.symmetry,
            )
        if remerge_miso_deg and remerge_miso_deg > 0:
            labels = self.adjacency_remerge(
                coords,
                mrp,
                labels,
                remerge_miso_deg,
                symmetry=self.symmetry,
                percolation_max_frac=percolation_max_frac,
                spread_tol_deg=spread_tol_deg if remerge_spread_check else None,
            )
        _, inv = np.unique(labels, return_inverse=True)
        return inv.astype(np.int64)

    # ----------------------------------------------------------------------- #
    # Segmentation -> grain table bridge (FF / NF / EBSD converge here)
    # ----------------------------------------------------------------------- #
    def seg_to_grain_table(
        self,
        labels: np.ndarray,
        coords: np.ndarray,
        mrp: Optional[np.ndarray] = None,
        euler: Optional[np.ndarray] = None,
        angle_convention: str = "bunge",
        angle_type: str = "radians",
        out_csv: Optional[str] = None,
        background_label: Union[int, Iterable[int]] = -1,
    ) -> "object":
        """Reduce a labelled point set to a per-grain table (FF/NF/EBSD connector).

        Produces the ``grain_id, X, Y, Z, GrainRadius, Eul0, Eul1, Eul2`` schema
        shared by the FF reconstruction and synthetic-load experiment CSVs, so
        segmented NF/EBSD voxel grids and FF grain tables converge on one grain
        table that :meth:`detect_splits` consumes.

        Orientation is a proper quaternion mean per grain (not an arithmetic mean of
        Euler angles). Provide orientations as either ``mrp`` (neml2 MRP) or
        ``euler`` (with ``angle_convention``/``angle_type``).

        Args:
            labels: ``(N,)`` integer grain labels.
            coords: ``(N, 3)`` point coordinates.
            mrp: ``(N, 3)`` neml2 MRP orientations (mutually exclusive with ``euler``).
            euler: ``(N, 3)`` Euler angles (mutually exclusive with ``mrp``).
            angle_convention: Euler convention (used when ``euler`` is given).
            angle_type: ``"degrees"`` or ``"radians"`` for the input ``euler``.
            out_csv: if given, also write the table to this CSV path.
            background_label: label, or iterable of labels, treated as
                non-grain and skipped. Defaults to ``-1``. See the note on
                label domains below before changing it.

        .. note::

           **Two different label domains reach this method, and the right
           ``background_label`` differs between them.**

           *Output of* :meth:`segment` *(the documented NF/EBSD bridge).*
           Contiguous **0-based grain labels** over a point set the caller has
           already restricted to solid material. ``0`` is a real grain here, and
           there is no void row at all, so the default ``-1`` excludes nothing
           and is correct. Passing ``background_label=0`` on this input
           silently deletes a grain.

           *A raw segmented voxel grid* (``nf.segment.flood``, or anything
           written through :func:`graintrace.nf.mesh.write_spn`). The domain is
           ``0`` void, ``>= 1`` grain, ``-1`` material-but-unsegmented. Pass
           ``background_label=(0, -1)`` to drop both sentinels. Left at the
           default, the whole void region is emitted as one enormous "grain"
           with a meaningless mean orientation, which then matches against the
           next step in :meth:`detect_splits`.

        Returns:
            A ``pandas.DataFrame`` with the grain-table schema (Euler in degrees,
            Bunge convention).
        """
        # pylint: disable=import-outside-toplevel
        import pandas as pd
        import torch

        from .orientation_helper import euler_to_mrp, matrix_to_euler, mrp_to_matrix

        labels = np.asarray(labels, dtype=np.int64)
        coords = np.asarray(coords, dtype=float)

        if (mrp is None) == (euler is None):
            raise ValueError("Provide exactly one of mrp or euler.")
        if euler is not None:
            mrp = (
                euler_to_mrp(
                    torch.as_tensor(np.asarray(euler), dtype=torch.float64),
                    angle_convention,
                    angle_type,
                )
                .cpu()
                .numpy()
            )
        mrp = np.asarray(mrp, dtype=float)

        background_labels = _as_label_set(background_label)

        voxvol = _voxel_volume(coords)
        rows = []
        for gid in np.unique(labels):
            if int(gid) in background_labels:
                continue
            mask = labels == gid
            cnt = int(mask.sum())
            cen = coords[mask].mean(axis=0)
            mean_mrp = self.mean_orientation_mrp(mrp[mask], self.symmetry)
            eul = (
                matrix_to_euler(
                    mrp_to_matrix(
                        torch.as_tensor(mean_mrp.reshape(1, 3), dtype=torch.float64)
                    ),
                    "bunge",
                    "degrees",
                )
                .cpu()
                .numpy()
                .reshape(3)
            )
            radius = (3.0 * cnt * voxvol / (4.0 * np.pi)) ** (1.0 / 3.0)
            rows.append(
                {
                    "grain_id": int(gid),
                    "X": cen[0],
                    "Y": cen[1],
                    "Z": cen[2],
                    "GrainRadius": radius,
                    "Eul0": eul[0],
                    "Eul1": eul[1],
                    "Eul2": eul[2],
                }
            )

        df = pd.DataFrame(
            rows,
            columns=["grain_id", "X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"],
        )
        if out_csv is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
            df.to_csv(out_csv, index=False)
        return df

    # ----------------------------------------------------------------------- #
    # Simulation intragranular fragmentation analysis
    # ----------------------------------------------------------------------- #
    def grain_fragments(
        self,
        results,
        step: int,
        tol_deg: float = 5.0,
        block_id_col: str = "block_id",
        grain_ids: Optional[Sequence[int]] = None,
        min_fragment_elems: int = 5,
        min_subgrain_miso_deg: float = 1.0,
        seg_kwargs: Optional[Dict] = None,
    ) -> Tuple["object", Dict[int, str]]:
        """Re-segment each original grain's elements by orientation at a loaded step.

        Groups the per-element field at ``step`` by ``block_id`` (the original grain
        subdomain), runs :meth:`segment` (``graph_mode="knn"`` for scattered mesh
        elements) within each grain, and composes a ``fragment_label`` of the form
        ``"{grain}.{k}"`` (k the 0-based sub-grain index). This measures
        intragranular fragmentation -- how a single grain breaks into orientation
        sub-domains under load -- keeping the parent grain id.

        Args:
            results: a :class:`~graintrace.simulation_postprocessing.SimulationResults`.
            step: field-step index to analyse.
            tol_deg: misorientation tolerance (degrees) for the sub-grain segmenter.
            block_id_col: the grain/block-id column in the field CSV.
            grain_ids: grains (block ids) to analyse; defaults to all blocks present.
            min_fragment_elems: grains with fewer elements than this are left whole
                (one fragment); also passed as the segmenter absorb threshold.
            min_subgrain_miso_deg: minimum misorientation (degrees) between two
                sub-grains for them to count as *distinct* fragments. After Leiden,
                fragments whose mean orientations are closer than this are merged
                (agglomeratively, closest-pair first, means recomputed each step), so
                a smooth intragranular gradient chopped into near-identical Leiden
                communities collapses back to one fragment while genuine sub-grains
                (a real low-angle sub-boundary) survive. ``0`` disables the merge.
            seg_kwargs: extra keyword overrides forwarded to :meth:`segment`.
                Without a ``"gamma"`` key the documented sub-grain resolution
                ``gamma=0.25`` is used (not a sweep: the sweep's lowest-gamma pick
                would under-split, and ``0.1`` leaves grains whole); pass
                ``{"gamma": None}`` explicitly to sweep. Absorption here is
                orientation-blind (``absorb_max_miso_deg=None``): every fragment
                belongs to one parent grain and ``min_subgrain_miso_deg`` sets the
                physical threshold.

        Returns:
            ``(fragments_df, per_element_labels)`` where ``fragments_df`` has one row
            per grain (``grain_id``, ``n_fragments``, ``fragment_sizes``,
            ``n_elements``) and ``per_element_labels`` maps each element ``id`` to
            its ``"{grain}.{k}"`` fragment label.
        """
        # pylint: disable=import-outside-toplevel
        import pandas as pd

        df = results.load_field_data(step)
        if block_id_col not in df.columns:
            raise KeyError(
                f"Field CSV has no '{block_id_col}' column; re-run CPFE with the "
                f"block-id output."
            )
        blocks = np.rint(df[block_id_col].to_numpy()).astype(np.int64)
        ids = df["id"].to_numpy(dtype=np.int64)
        coords = df[list(self.coord_cols)].to_numpy(dtype=float)
        mrp = df[list(self.ori_cols)].to_numpy(dtype=float)

        if grain_ids is None:
            grain_ids = [int(b) for b in np.unique(blocks)]

        seg_kwargs = dict(seg_kwargs or {})
        per_element_labels: Dict[int, str] = {}
        rows = []
        for gid in grain_ids:
            mask = blocks == int(gid)
            n_elem = int(mask.sum())
            if n_elem == 0:
                continue
            if n_elem < max(2 * min_fragment_elems, 4):
                # too small to split meaningfully -> one fragment
                labels = np.zeros(n_elem, dtype=np.int64)
            else:
                # remerge_miso_deg=0 (OFF): the adjacency re-merge is an NF-grid tool
                # that on a single grain's ~100 elements over-merges by single-linkage
                # chaining (its means are computed once). gamma is the lever for
                # sub-grain resolution here, with min_subgrain_miso_deg below as the
                # physical threshold. verbose=False: one sweep table per grain is noise.
                seg_call = {
                    "misori_tol_deg": tol_deg,
                    "graph_mode": "knn",
                    "gamma": 0.25,
                    "grain_threshold_final": min_fragment_elems,
                    "remerge_miso_deg": 0.0,
                    "absorb_max_miso_deg": None,
                    "verbose": False,
                }
                seg_call.update(seg_kwargs)
                labels = self.segment(coords[mask], mrp[mask], **seg_call)
                if min_subgrain_miso_deg and min_subgrain_miso_deg > 0:
                    labels = self.merge_fragments_by_misorientation(
                        mrp[mask], labels, min_subgrain_miso_deg, self.symmetry
                    )
            _, sizes = np.unique(labels, return_counts=True)
            for eid, lab in zip(ids[mask], labels):
                per_element_labels[int(eid)] = f"{int(gid)}.{int(lab)}"
            rows.append(
                {
                    "grain_id": int(gid),
                    "n_fragments": int(len(sizes)),
                    "fragment_sizes": ";".join(map(str, sorted(sizes, reverse=True))),
                    "n_elements": n_elem,
                }
            )

        fragments_df = pd.DataFrame(
            rows, columns=["grain_id", "n_fragments", "fragment_sizes", "n_elements"]
        )
        return fragments_df, per_element_labels

    # ----------------------------------------------------------------------- #
    # Experiment grain-split detection across two load steps
    # ----------------------------------------------------------------------- #
    def detect_splits(
        self,
        nodes_a,
        nodes_b,
        d_tol: float,
        theta_tol_deg: float,
        angle_convention: str = "bunge",
        angle_type: str = "degrees",
        ori_cols: Sequence[str] = ("Eul0", "Eul1", "Eul2"),
        coord_cols: Sequence[str] = ("X", "Y", "Z"),
        id_col: str = "grain_id",
        top_k: Optional[int] = 8,
        adjacency_b: Optional[Iterable[Tuple[int, int]]] = None,
        out_csv: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect grain splits/merges/births/deaths between two load steps.

        Goes beyond the strictly one-to-one grain matcher: candidate cross edges
        from a centroid ``cKDTree`` (top-k) are kept when the centroid distance is
        within ``d_tol`` and the misorientation within ``theta_tol_deg``; connected
        components of the step-A/step-B union graph are grain lineages, classified by
        their ``(nA, nB)`` shape. FF supplies the node tables directly; NF/EBSD reach
        the same detector via :meth:`segment` + :meth:`seg_to_grain_table`.

        The grain tables use their own ``ori_cols``/``coord_cols`` (``Eul0/1/2`` and
        ``X/Y/Z`` by default), independent of the simulation column conventions on
        the instance.

        Args:
            nodes_a, nodes_b: per-grain node tables (DataFrame or CSV path) for the
                two steps, with an id column, three coordinate columns, and three
                orientation columns.
            d_tol: maximum centroid distance for a candidate cross edge.
            theta_tol_deg: maximum misorientation (degrees) for a candidate cross edge.
            angle_convention: ``"bunge"``/``"kocks"``/``"roe"`` Euler, or ``"mrp"``.
            angle_type: ``"degrees"`` or ``"radians"`` (Euler conventions only).
            ori_cols: the three orientation columns.
            coord_cols: the three coordinate columns.
            id_col: the grain-id column.
            top_k: cap on the number of B candidates probed per A grain
                (default 8), or ``None`` for no cap. With a cap, ``top_k`` --
                not ``d_tol`` -- is the binding limit on how many children a
                split can have whenever more than ``top_k`` B grains sit within
                ``d_tol``, so a parent that fragments into many pieces, or a
                dense fine-grained step B, needs ``top_k`` raised as well as
                ``d_tol``. Pass ``top_k=None`` to take every B grain within
                ``d_tol`` and let ``d_tol`` alone decide the search, at the
                cost of a radius query that is slower on dense data.

                Note that the order of the two filters does not matter: the
                ``top_k`` nearest B grains that survive ``d_tol`` are exactly
                the ``top_k`` nearest among those within ``d_tol``, because
                every grain inside the radius is closer than every grain
                outside it. Only raising or removing the cap changes the
                candidate set.
            adjacency_b: optional iterable of ``(grain_id, grain_id)`` pairs giving
                intra-step-B grain adjacency; used to reject a spurious split child
                that is not adjacent to any other child of the same parent. If no two
                of a parent's candidates are mutually adjacent there is no split to
                reject, so the single best candidate (lowest misorientation, then
                shortest centroid distance) is kept and the parent matches it rather
                than being dropped entirely.
            out_csv: if given, write the split correspondences to this CSV path.

        Returns:
            dict with ``components`` (list of ``{a_ids, b_ids, kind}``),
            ``split_correspondences`` (DataFrame), and the counts ``n_splits``,
            ``n_merges``, ``n_matches``, ``n_births``, ``n_deaths``.
        """
        # pylint: disable=import-outside-toplevel
        import pandas as pd
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        from .orientation_helper import misorientation_matrix

        da = _load_nodes(nodes_a)
        db = _load_nodes(nodes_b)
        n_a, n_b = len(da), len(db)
        ids_a = da[id_col].to_numpy()
        ids_b = db[id_col].to_numpy()
        coords_a = da[list(coord_cols)].to_numpy(dtype=float)
        coords_b = db[list(coord_cols)].to_numpy(dtype=float)
        r_a = _orientation_matrices(da, ori_cols, angle_convention, angle_type)
        r_b = _orientation_matrices(db, ori_cols, angle_convention, angle_type)

        # Candidate cross edges: the B centroids within d_tol of each A grain,
        # capped at top_k nearest. Capping before or after the d_tol gate gives
        # the same set (every grain inside the radius is nearer than every grain
        # outside it), so top_k=None is the only way to widen the search.
        tree_b = cKDTree(coords_b)
        ai, bj, pair_dist = [], [], []
        if top_k is None:
            for i in range(n_a):
                for j in tree_b.query_ball_point(coords_a[i], r=d_tol):
                    ai.append(i)
                    bj.append(int(j))
                    pair_dist.append(float(np.linalg.norm(coords_b[j] - coords_a[i])))
        else:
            kq = min(top_k, n_b)
            dist, jdx = tree_b.query(coords_a, k=kq)
            dist = np.asarray(dist).reshape(n_a, kq)
            jdx = np.asarray(jdx).reshape(n_a, kq)

            for i in range(n_a):
                for col in range(kq):
                    j = int(jdx[i, col])
                    if dist[i, col] <= d_tol:
                        ai.append(i)
                        bj.append(j)
                        pair_dist.append(float(dist[i, col]))
        ai = np.asarray(ai, dtype=np.int64)
        bj = np.asarray(bj, dtype=np.int64)
        pair_dist = np.asarray(pair_dist, dtype=float)
        pair_miso = np.zeros(len(ai), dtype=float)

        # Filter by symmetry-aware misorientation.
        if len(ai) > 0:
            miso = (
                misorientation_matrix(r_a[ai], r_b[bj], self.symmetry, "degrees")
                .cpu()
                .numpy()
            )
            keep = miso <= theta_tol_deg
            ai, bj = ai[keep], bj[keep]
            pair_dist, pair_miso = pair_dist[keep], np.asarray(miso)[keep]

        # Optional adjacency consistency: a candidate child that is not adjacent (in B)
        # to any sibling under the same parent is not evidence of a split, so it is
        # dropped -- but only when the parent still keeps at least one adjacent pair.
        # When NO two candidates are mutually adjacent there is no split to reject, and
        # dropping every candidate would turn a legitimate correspondence into a death
        # plus N spurious births; keep the single best candidate (lowest misorientation,
        # then shortest centroid distance) so the parent still matches.
        if adjacency_b is not None and len(ai) > 0:
            idmap_b = {int(g): k for k, g in enumerate(ids_b)}
            adj: Dict[int, set] = {}
            for g1, g2 in adjacency_b:
                u, v = idmap_b.get(int(g1)), idmap_b.get(int(g2))
                if u is None or v is None:
                    continue
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            keep = np.ones(len(ai), dtype=bool)
            for a in np.unique(ai):
                sel = np.where(ai == a)[0]
                children = bj[sel]
                if len(children) <= 1:
                    continue
                cset = set(children.tolist())
                adjacent = np.array(
                    [bool(adj.get(int(c), set()) & (cset - {int(c)})) for c in children]
                )
                if adjacent.any():
                    keep[sel[~adjacent]] = False
                else:
                    best = sel[np.lexsort((bj[sel], pair_dist[sel], pair_miso[sel]))[0]]
                    keep[sel[sel != best]] = False
            ai, bj = ai[keep], bj[keep]

        # Union graph A (0..n_a-1) + B (n_a..n_a+n_b-1); components = grain lineages.
        n = n_a + n_b
        rows = (
            np.concatenate([ai, bj + n_a]) if len(ai) else np.array([], dtype=np.int64)
        )
        csum = (
            np.concatenate([bj + n_a, ai]) if len(ai) else np.array([], dtype=np.int64)
        )
        data = np.ones(len(rows), dtype=np.int8)
        graph = coo_matrix((data, (rows, csum)), shape=(n, n))
        _, comp = connected_components(graph, directed=False)

        components = []
        counts = {
            "n_matches": 0,
            "n_splits": 0,
            "n_merges": 0,
            "n_births": 0,
            "n_deaths": 0,
        }
        split_rows = []
        for c in np.unique(comp):
            members = np.where(comp == c)[0]
            a_local = members[members < n_a]
            b_local = members[members >= n_a] - n_a
            a_grains = [int(ids_a[i]) for i in a_local]
            b_grains = [int(ids_b[j]) for j in b_local]
            n_ai, n_bj = len(a_grains), len(b_grains)
            if n_ai == 1 and n_bj == 1:
                kind = "match"
                counts["n_matches"] += 1
            elif n_ai == 1 and n_bj > 1:
                kind = "split"
                counts["n_splits"] += 1
                sizes = [
                    (
                        float(db.loc[db[id_col] == g, "GrainRadius"].iloc[0])
                        if "GrainRadius" in db.columns
                        else np.nan
                    )
                    for g in b_grains
                ]
                split_rows.append(
                    {
                        "parent_a": a_grains[0],
                        "children_b": ";".join(map(str, b_grains)),
                        "n_children": n_bj,
                        "child_radii": ";".join(f"{s:.4g}" for s in sizes),
                    }
                )
            elif n_ai > 1 and n_bj == 1:
                kind = "merge"
                counts["n_merges"] += 1
            elif n_ai == 1 and n_bj == 0:
                kind = "death"
                counts["n_deaths"] += 1
            elif n_ai == 0 and n_bj == 1:
                kind = "birth"
                counts["n_births"] += 1
            else:
                kind = "complex"
            components.append({"a_ids": a_grains, "b_ids": b_grains, "kind": kind})

        split_df = pd.DataFrame(
            split_rows,
            columns=["parent_a", "children_b", "n_children", "child_radii"],
        )
        if out_csv is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
            split_df.to_csv(out_csv, index=False)

        return {"components": components, "split_correspondences": split_df, **counts}
