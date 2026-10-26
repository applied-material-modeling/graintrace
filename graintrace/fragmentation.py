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
cutoff, gamma chosen by percolation, then fragment absorption and an
adjacency-only re-merge.

Heavy dependencies (torch, neml2, scipy, networkit via ``GraphSpatialCluster``)
are imported lazily inside the methods.
"""

from __future__ import annotations

import os
import tempfile
import warnings
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


def _knn_cluster_adjacency(
    coords: np.ndarray, labels: np.ndarray, k_neighbors: int
) -> Dict[Tuple[int, int], int]:
    """Adjacent (touching) cluster-label pairs and their contact counts, via kNN.

    Two clusters are adjacent when a point of one is among the ``k_neighbors``
    nearest neighbours of a point of the other. Works for both voxel grids and
    scattered mesh element centroids (no regular-grid assumption).
    """
    # pylint: disable=import-outside-toplevel
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    kq = min(k_neighbors + 1, len(coords))
    _, idx = tree.query(coords, k=kq)
    idx = np.atleast_2d(idx)

    contacts: Dict[Tuple[int, int], int] = {}
    for i in range(coords.shape[0]):
        li = labels[i]
        for j in idx[i, 1:]:
            lj = labels[j]
            if li == lj:
                continue
            key = (int(min(li, lj)), int(max(li, lj)))
            contacts[key] = contacts.get(key, 0) + 1
    return contacts


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


# Cost assigned to a non-candidate cell of a one-to-one assignment block. Any real
# cost is <= 2 (two normalized terms), so the Hungarian solver fills a sentinel cell
# only when the block admits no smaller complete assignment; those picks are dropped.
_INFEASIBLE_COST = 1e6


def _candidate_pairs(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    rad_a: np.ndarray,
    rad_b: np.ndarray,
    d_tol: float,
    d_tol_mode: str,
    alpha_a: float,
    alpha_b: float,
    alpha: float,
    top_k: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Candidate cross-step pairs passing the centroid-distance gate.

    The gate is one absolute distance in ``"absolute"`` mode and size-aware in the
    radius modes, where a pair's own gate depends on the two grain radii. Because the
    gate then varies per pair, the radius modes find candidates with a ball query at
    each A grain's *largest possible* gate and apply the exact per-pair gate
    afterwards, so the search provably covers the gate and ``top_k`` is not consulted.

    Args:
        coords_a: ``(n_a, 3)`` step-A centroids.
        coords_b: ``(n_b, 3)`` step-B centroids.
        rad_a: ``(n_a,)`` step-A grain radii (ignored in ``"absolute"`` mode).
        rad_b: ``(n_b,)`` step-B grain radii (ignored in ``"absolute"`` mode).
        d_tol: absolute distance gate; a floor on the gate in ``"radius"`` mode.
        d_tol_mode: ``"absolute"``, ``"radius"`` or ``"radius_sum"``.
        alpha_a: multiplier on the A radius in ``"radius"`` mode.
        alpha_b: multiplier on the B radius in ``"radius"`` mode.
        alpha: multiplier on the radius sum in ``"radius_sum"`` mode.
        top_k: cap on B candidates probed per A grain (``"absolute"`` mode only).

    Returns:
        ``(ai, bj, dist, gate)``: the A index, B index, centroid distance and the
        gate that pair was tested against, one entry per surviving candidate.
    """
    # pylint: disable=import-outside-toplevel
    from scipy.spatial import cKDTree

    n_a, n_b = len(coords_a), len(coords_b)
    if n_a == 0 or n_b == 0:
        zeros_i, zeros_f = np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
        return zeros_i, zeros_i.copy(), zeros_f, zeros_f.copy()

    tree_b = cKDTree(coords_b)

    if d_tol_mode == "absolute":
        # Unchanged legacy path: top_k nearest B centroids within a single d_tol.
        # Capping before or after the d_tol gate gives the same set (every grain
        # inside the radius is nearer than every grain outside it), so top_k=None
        # is the only way to widen the search.
        ai_l: List[int] = []
        bj_l: List[int] = []
        dist_l: List[float] = []
        if top_k is None:
            for i in range(n_a):
                for j in tree_b.query_ball_point(coords_a[i], r=d_tol):
                    ai_l.append(i)
                    bj_l.append(int(j))
                    dist_l.append(float(np.linalg.norm(coords_b[j] - coords_a[i])))
        else:
            kq = min(top_k, n_b)
            dist, jdx = tree_b.query(coords_a, k=kq)
            dist = np.asarray(dist).reshape(n_a, kq)
            jdx = np.asarray(jdx).reshape(n_a, kq)
            for i in range(n_a):
                for col in range(kq):
                    if dist[i, col] <= d_tol:
                        ai_l.append(i)
                        bj_l.append(int(jdx[i, col]))
                        dist_l.append(float(dist[i, col]))
        ai = np.asarray(ai_l, dtype=np.int64)
        bj = np.asarray(bj_l, dtype=np.int64)
        pair_dist = np.asarray(dist_l, dtype=float)
        return ai, bj, pair_dist, np.full(ai.size, float(d_tol))

    # Size-aware gates: query at the loosest gate the A grain could possibly use
    # (its own radius term against the LARGEST B radius), then apply the exact gate.
    # nanmax: one NaN radius must not turn every search radius into NaN (an empty
    # ball query). detect_splits already zeroes NaN radii; this guards direct callers.
    rad_b_max = float(np.nanmax(rad_b)) if np.any(np.isfinite(rad_b)) else 0.0
    if d_tol_mode == "radius":
        r_search = np.maximum(
            np.maximum(float(d_tol), alpha_a * rad_a), alpha_b * rad_b_max
        )
    else:  # "radius_sum"
        r_search = alpha * (rad_a + rad_b_max)

    cand = tree_b.query_ball_point(coords_a, np.asarray(r_search, dtype=float))
    counts = [len(c) for c in cand]
    ai = np.repeat(np.arange(n_a, dtype=np.int64), counts)
    if ai.size == 0:
        zeros_f = np.empty(0, dtype=float)
        return ai, np.empty(0, dtype=np.int64), zeros_f, zeros_f.copy()
    # Sort each A grain's hits: query_ball_point does not define an order, and the
    # downstream tie-breaks must not depend on it.
    bj = np.concatenate([np.sort(np.asarray(c, dtype=np.int64)) for c in cand if c])

    pair_dist = np.linalg.norm(coords_a[ai] - coords_b[bj], axis=1)
    if d_tol_mode == "radius":
        gate = np.maximum.reduce(
            [
                np.full(ai.size, float(d_tol)),
                alpha_a * rad_a[ai],
                alpha_b * rad_b[bj],
            ]
        )
    else:  # "radius_sum"
        gate = alpha * (rad_a[ai] + rad_b[bj])

    keep = pair_dist <= gate
    return ai[keep], bj[keep], pair_dist[keep], gate[keep]


def _sanitize_radii(rad: np.ndarray, name: str) -> np.ndarray:
    """Replace NaN or negative grain radii by 0, warning once with the count.

    A radius of 0 drops that grain's radius term from the size-aware gate, so the
    gate falls back to ``d_tol`` (``"radius"``) or to the other grain's radius
    (``"radius_sum"``) instead of a NaN that silently removes every link.

    Args:
        rad: ``(n,)`` per-grain radii.
        name: table name, used in the warning.

    Returns:
        ``(n,)`` radii with every NaN or negative entry set to 0.
    """
    bad = ~(rad >= 0)  # True for NaN and for negatives
    n_bad = int(bad.sum())
    if n_bad:
        warnings.warn(
            f"{n_bad} grain(s) in {name} have a NaN or negative radius; treating "
            "them as radius 0, so their distance gate falls back to d_tol.",
            UserWarning,
            stacklevel=3,
        )
        rad = np.where(bad, 0.0, rad)
    return rad


def _overlap_mask(spec, df, n: int, name: str) -> np.ndarray:
    """Boolean "inside the shared volume" mask for one side of the recall report.

    Args:
        spec: ``None`` (everything counts), a boolean array-like of length ``n``, or
            the name of a boolean column in ``df``.
        df: the grain table that side came from.
        n: number of grains on that side.
        name: parameter name, used in the error message.

    Returns:
        ``(n,)`` boolean mask.

    Raises:
        ValueError: if ``spec`` names a column the table does not have, or is an
            array of the wrong length.
    """
    if spec is None:
        return np.ones(n, dtype=bool)
    if isinstance(spec, str):
        # A bare pandas KeyError here is unhelpful next to the length check below.
        if spec not in df.columns:
            raise ValueError(
                f"{name}={spec!r} is not a column of the grain table; "
                f"available columns are {list(df.columns)}."
            )
        arr = df[spec].to_numpy()
    else:
        arr = np.asarray(spec)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have length {n}, got shape {arr.shape}.")
    return arr.astype(bool)


def _side_recall(
    linked: np.ndarray, mask: np.ndarray, volume: Optional[np.ndarray], side: str
) -> Dict[str, Any]:
    """Linked fraction for one side, by grain count and optionally by volume.

    Args:
        linked: ``(n,)`` boolean, True where the grain has at least one cross link.
        mask: ``(n,)`` boolean overlap mask; only these grains are counted.
        volume: ``(n,)`` per-grain volume, or ``None`` to skip the volume weighting.
        side: ``"a"`` or ``"b"``, used to name the returned keys.

    Returns:
        dict of recall entries for that side (``NaN`` fractions on an empty mask).
    """
    n_total = int(mask.sum())
    n_linked = int((linked & mask).sum())
    out: Dict[str, Any] = {
        f"{side}_fraction": (n_linked / n_total) if n_total else float("nan"),
        f"n_{side}_linked": n_linked,
        f"n_{side}_total": n_total,
    }
    if volume is not None:
        vol_total = float(volume[mask].sum())
        vol_linked = float(volume[linked & mask].sum())
        out[f"{side}_fraction_volume"] = (
            (vol_linked / vol_total) if vol_total > 0 else float("nan")
        )
    return out


def _one_to_one_blocks(
    comp: np.ndarray,
    ai: np.ndarray,
    bj: np.ndarray,
    cost: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Solve the one-to-one assignment, one connected component at a time.

    A pair that is not a surviving candidate is infeasible, so no optimal assignment
    ever crosses a component boundary: the global Hungarian problem decomposes
    exactly into the per-component blocks solved here, which avoids building the
    dense ``n_a x n_b`` cost matrix.

    Args:
        comp: ``(n_a + n_b,)`` connected-component label per union-graph node, whose
            first ``n_a`` entries are the A grains (so ``comp[ai]`` labels a pair).
        ai: ``(n_pairs,)`` A index of each surviving candidate pair.
        bj: ``(n_pairs,)`` B index of each surviving candidate pair.
        cost: ``(n_pairs,)`` assignment cost of each surviving candidate pair.

    Returns:
        list of ``(a_idx, b_idx, pair_idx)`` arrays, one per component that
        contributed at least one assigned pair.
    """
    # pylint: disable=import-outside-toplevel
    from scipy.optimize import linear_sum_assignment

    pair_comp = comp[ai]
    blocks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for c in np.unique(pair_comp):
        sel = np.where(pair_comp == c)[0]
        a_local = np.unique(ai[sel])
        b_local = np.unique(bj[sel])
        a_pos = {int(v): k for k, v in enumerate(a_local)}
        b_pos = {int(v): k for k, v in enumerate(b_local)}
        block = np.full((len(a_local), len(b_local)), _INFEASIBLE_COST, dtype=float)
        where = {}
        for p in sel:
            r, s = a_pos[int(ai[p])], b_pos[int(bj[p])]
            block[r, s] = cost[p]
            where[(r, s)] = int(p)
        rows, cols = linear_sum_assignment(block)
        ok = block[rows, cols] < _INFEASIBLE_COST
        if not ok.any():
            continue
        pair_idx = np.asarray(
            [where[(int(r), int(s))] for r, s in zip(rows[ok], cols[ok])],
            dtype=np.int64,
        )
        blocks.append((a_local[rows[ok]], b_local[cols[ok]], pair_idx))
    return blocks


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
    ) -> np.ndarray:
        """Absorb sub-``min_size`` clusters into their largest-contact neighbour.

        Uses kNN contact counts for adjacency, so it works for any point set (voxel
        grids or scattered mesh centroids). Iterates until no sub-``min_size``
        cluster remains (or no further merge is possible), then relabels
        contiguously from 0.

        Args:
            coords: ``(N, 3)`` point coordinates.
            labels: ``(N,)`` integer cluster labels.
            min_size: clusters with fewer points are absorbed.
            k_neighbors: neighbours per point used to measure cluster contact.
            max_passes: safety cap on absorption passes.

        Returns:
            ``(N,)`` contiguous 0-based labels.
        """
        labels = np.asarray(labels, dtype=np.int64).copy()

        for _ in range(max_passes):
            uniq, counts = np.unique(labels, return_counts=True)
            small = set(int(u) for u, c in zip(uniq, counts) if c < min_size)
            if not small or len(uniq) == 1:
                break

            contacts = _knn_cluster_adjacency(coords, labels, k_neighbors)
            # best (largest-contact) neighbour per label
            best: Dict[int, Tuple[int, int]] = {}
            for (a, b), n in contacts.items():
                for x, y in ((a, b), (b, a)):
                    if n > best.get(x, (0, -1))[0]:
                        best[x] = (n, y)

            remap = {}
            for s in small:
                if s in best:
                    remap[s] = best[s][1]
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
            labels = np.array(
                [remap.get(int(l), int(l)) for l in labels], dtype=np.int64
            )

        _, inv = np.unique(labels, return_inverse=True)
        return inv.astype(np.int64)

    @staticmethod
    def adjacency_remerge(
        coords: np.ndarray,
        mrp: np.ndarray,
        labels: np.ndarray,
        tol_deg: float,
        symmetry: str = "432",
        k_neighbors: int = 8,
        percolation_max_frac: float = 0.03,
    ) -> np.ndarray:
        """Merge spatially adjacent clusters within ``tol_deg`` misorientation.

        Tightest (lowest-misorientation) adjacent pairs merge first, and any union
        that would exceed ``percolation_max_frac`` of the point count is skipped so
        a chain of low-angle fragments cannot re-percolate into one giant grain (the
        failure mode a global orientation merge or naive single-linkage produces).
        Adjacency is kNN contact, so it works for both voxel grids and scattered
        mesh element centroids.

        Intended for **large, many-cluster** segmentations (NF/EBSD grids): the size
        cap is a fraction of the *whole* point set, so on a small single-grain point
        set (e.g. one grain's ~100 elements in :meth:`grain_fragments`) it is smaller
        than any fragment and blocks every merge (inert). Raising the cap there instead
        lets union-find chain low-angle fragments (the means are computed once), which
        over-merges. So sub-grain segmentation disables re-merge and relies on ``gamma``.

        Args:
            coords: ``(N, 3)`` point coordinates.
            mrp: ``(N, 3)`` neml2 MRP orientations.
            labels: ``(N,)`` integer cluster labels.
            tol_deg: merge adjacent clusters whose mean orientations are within this.
            symmetry: crystal symmetry (orbifold notation).
            k_neighbors: neighbours per point used to measure cluster contact.
            percolation_max_frac: hard cap on any merged cluster's size fraction.

        Returns:
            ``(N,)`` contiguous 0-based labels.
        """
        labels = np.asarray(labels, dtype=np.int64).copy()
        mrp = np.asarray(mrp, dtype=float)

        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) < 2:
            return labels
        id2idx = {int(u): k for k, u in enumerate(uniq)}
        means = np.array(
            [
                FragmentationAnalyzer.mean_orientation_mrp(mrp[labels == u], symmetry)
                for u in uniq
            ]
        )

        contacts = _knn_cluster_adjacency(coords, labels, k_neighbors)
        if not contacts:
            return labels

        pairs = list(contacts.keys())
        misos = np.array(
            [
                _pair_misorientation_deg(means[id2idx[a]], means[id2idx[b]], symmetry)
                for a, b in pairs
            ]
        )

        size_cap = percolation_max_frac * float(len(labels))
        comp_size = counts.astype(np.float64).copy()
        parent = list(range(len(uniq)))

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
            parent[rb] = ra
            comp_size[ra] += comp_size[rb]

        roots = np.array([find(k) for k in range(len(uniq))])
        remap = {r: i for i, r in enumerate(np.unique(roots))}
        new_of_idx = np.array([remap[r] for r in roots])
        lab_idx = np.array([id2idx[int(l)] for l in labels])
        return new_of_idx[lab_idx].astype(np.int64)

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
        sweep: Sequence[dict], percolation_max_frac: float = 0.03
    ) -> dict:
        """Pick the smallest gamma whose biggest cluster is below the percolation cap.

        The raw Leiden count is near gamma-invariant here (all gammas over-segment);
        gamma controls only the biggest ("percolating") cluster, and the final grain
        count is set by absorption. So choose the least over-resolved, still
        de-percolated gamma; if none qualifies, the overall minimum ``max_frac``.
        """
        ok = [r for r in sweep if r["max_frac"] < percolation_max_frac]
        if ok:
            return min(ok, key=lambda r: r["gamma"])
        return min(sweep, key=lambda r: r["max_frac"])

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
        percolation_max_frac: float = 0.03,
        n_jobs: int = 1,
        n_threads: int = 1,
        seed: int = 42,
        workdir: Optional[str] = None,
    ) -> np.ndarray:
        """Segment a point set by crystal orientation (graph / Leiden).

        Preferred over flood fill, which over-merges into percolating grains. Wraps
        :class:`~graintrace.graph_spatial_cluster.GraphSpatialCluster` with the
        symmetry-aware misorientation metric on neml2 MRP orientations and the tuned
        recipe: fixed broad RBF sigma, hard misorientation cutoff,
        gamma-by-percolation, then absorb + adjacency re-merge.

        Args:
            coords: ``(N, 3)`` point coordinates (voxel centres or element centroids).
            mrp: ``(N, 3)`` neml2 MRP orientations.
            misori_tol_deg: hard misorientation edge cutoff (degrees).
            graph_mode: ``"grid"`` for regular voxel grids (NF/EBSD), ``"knn"`` for
                scattered mesh element centroids.
            gamma: fixed Leiden resolution; if ``None``, a percolation-picked sweep.
            gamma_sweep: resolutions to sweep when ``gamma`` is ``None``.
            rbf_sigma_deg: fixed RBF sigma (degrees); defaults to half ``misori_tol_deg``.
            manhattan_radius: grid connectivity radius (``graph_mode="grid"``).
            k: neighbours per node (``graph_mode="knn"``).
            reduce_topk: keep top-k edges per node by weight before clustering.
            grain_threshold_final: minimum cluster size; smaller ones are absorbed.
            remerge_miso_deg: adjacency re-merge misorientation threshold (degrees).
            percolation_max_frac: percolation cap for gamma pick and re-merge.
            n_jobs: parallel workers for edge-distance build.
            n_threads: threads for Leiden.
            seed: RNG seed.
            workdir: scratch directory for the graph input CSV + checkpoint (a temp
                directory is used and cleaned up if ``None``).

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
        best = self.pick_gamma_by_percolation(sweep, percolation_max_frac)
        labels = best["labels"]

        if grain_threshold_final and grain_threshold_final > 1:
            labels = self.absorb_fragments(coords, labels, grain_threshold_final)
        if remerge_miso_deg and remerge_miso_deg > 0:
            labels = self.adjacency_remerge(
                coords,
                mrp,
                labels,
                remerge_miso_deg,
                symmetry=self.symmetry,
                percolation_max_frac=percolation_max_frac,
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
                # whose percolation size-cap (a fraction of the point count) is
                # meaningless on a single grain's ~100 elements -- it is either inert
                # (cap < any fragment) or, if forced on, over-merges by single-linkage
                # chaining. gamma is the lever for sub-grain resolution here.
                seg_call = {
                    "misori_tol_deg": tol_deg,
                    "graph_mode": "knn",
                    "grain_threshold_final": min_fragment_elems,
                    "remerge_miso_deg": 0.0,
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
        d_tol_mode: str = "absolute",
        alpha_a: float = 1.0,
        alpha_b: float = 1.0,
        alpha: float = 1.0,
        radius_col: str = "GrainRadius",
        assignment: Optional[str] = None,
        volume_col: str = "volume",
        in_overlap_a: Optional[Union[str, Sequence[bool], np.ndarray]] = None,
        in_overlap_b: Optional[Union[str, Sequence[bool], np.ndarray]] = None,
        n_null: int = 0,
        seed: Optional[int] = None,
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
                (default 8), or ``None`` for no cap. **Used by**
                ``d_tol_mode="absolute"`` **only** -- the radius modes size
                their own search to cover the gate and ignore it, warning if a
                non-default value is passed anyway. With a cap,
                ``top_k`` -- not ``d_tol`` -- is the binding limit on how many
                children a split can have whenever more than ``top_k`` B grains
                sit within ``d_tol``, so a parent that fragments into many
                pieces, or a dense fine-grained step B, needs ``top_k`` raised
                as well as ``d_tol``. Pass ``top_k=None`` to take every B grain
                within ``d_tol`` and let ``d_tol`` alone decide the search, at
                the cost of a radius query that is slower on dense data.

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
            d_tol_mode: how the centroid-distance gate is sized.

                * ``"absolute"`` (default, unchanged behavior): one ``d_tol`` for
                  every pair.
                * ``"radius"``: link when
                  ``d <= max(d_tol, alpha_a * R_a, alpha_b * R_b)``.
                * ``"radius_sum"``: link when ``d <= alpha * (R_a + R_b)``.

                One absolute gate fails whenever grain sizes span a wide range or
                several small B grains sit inside one large A grain (FF duplicates
                across scan layers, sub-grains an NF step resolves separately,
                deformation fragments of a coarse parent): those children lie
                anywhere up to about the parent's radius from its centroid, so a
                gate sized for the median grain misses them and a gate sized for the
                largest is too loose for the small ones. The radius modes read the
                ``radius_col`` column and also replace the ``top_k`` search with a
                ball query at each A grain's largest possible gate, so the candidate
                search provably covers the gate.
            alpha_a: multiplier on the A radius in ``d_tol_mode="radius"``.
            alpha_b: multiplier on the B radius in ``d_tol_mode="radius"``.
            alpha: multiplier on the radius sum in ``d_tol_mode="radius_sum"``.
            radius_col: per-grain radius column read by the radius modes (default
                ``"GrainRadius"``, the grain-table schema). Required in both tables
                for those modes; ``"absolute"`` never reads it. A NaN or negative
                radius counts as 0 (with a ``UserWarning``), so that grain's gate
                falls back to ``d_tol``.
            assignment: ``"one_to_one"`` to additionally run a Hungarian assignment
                over the gated candidates and return the strict pairs, or ``None``
                (default) for the lineage view only. The lineage components are
                unaffected either way; the assignment is reported alongside them in
                ``one_to_one`` for callers that need a clean matched count or a
                per-grain misorientation and centroid error without reaching for
                :class:`~graintrace.grain_graph_matching.GraphGrainMatcher` and its
                tessellation graphs. Cost is
                ``miso / theta_tol_deg + d / gate``, both terms in ``[0, 1]``.
            volume_col: per-grain volume column (default ``"volume"``). When present
                in a table, that side's recall is also reported volume-weighted.
            in_overlap_a: optional boolean mask, or the name of a boolean column in
                ``nodes_a``, flagging the A grains inside the volume both datasets
                see. Recall is then computed over those grains only, so material
                outside the shared volume does not count as a miss.
            in_overlap_b: the same for ``nodes_b``.
            n_null: number of chance trials (default 0 = skip). Each trial reruns
                the misorientation filter over the same distance-gated candidates
                with uniformly random B orientations, which gives the number of
                links the gate finds by chance -- a significance check for any gate
                setting.
            seed: seed for the ``n_null`` random orientations.

        Returns:
            dict with ``components`` (list of ``{a_ids, b_ids, kind}``),
            ``split_correspondences`` (DataFrame), the counts ``n_splits``,
            ``n_merges``, ``n_matches``, ``n_births``, ``n_deaths``, and
            ``recall`` (per-side linked fractions). ``one_to_one`` (a DataFrame of
            ``a_id, b_id, miso_deg, dist, cost``) is present when
            ``assignment="one_to_one"``, and ``null`` (observed against chance
            links) when ``n_null > 0``. ``null`` carries ``n_null``,
            ``mean_links``, ``std_links``, ``observed_links`` and
            ``observed_links_after_adjacency``. Compare ``observed_links``
            against ``mean_links``: both count distance-gated pairs surviving
            the misorientation filter, so they are measured at the same point.
            ``observed_links_after_adjacency`` is the same count once the
            optional ``adjacency_b`` prune has run, which the null never sees;
            the two are equal when ``adjacency_b`` is None.

        Raises:
            ValueError: on an unknown ``d_tol_mode`` or ``assignment``; on a
                negative (or NaN) ``d_tol``, ``theta_tol_deg``, ``alpha``,
                ``alpha_a`` or ``alpha_b``; or when a radius mode is requested and
                ``radius_col`` is missing from either table.

        Warns:
            UserWarning: in the radius modes, when any ``radius_col`` value is NaN
                or negative. Those radii are treated as 0, so that grain's gate falls
                back to ``d_tol`` (``"radius"``) rather than dropping its links.
        """
        # pylint: disable=import-outside-toplevel
        import pandas as pd
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        from .orientation_helper import misorientation_matrix

        if d_tol_mode not in ("absolute", "radius", "radius_sum"):
            raise ValueError(
                f"Unknown d_tol_mode: {d_tol_mode!r} "
                "(expected 'absolute', 'radius' or 'radius_sum')."
            )
        # Negative (or NaN) tolerances and multipliers have no meaning; the legacy
        # path silently returned no links for them, which hides the typo.
        for nm, val in (
            ("d_tol", d_tol),
            ("theta_tol_deg", theta_tol_deg),
            ("alpha", alpha),
            ("alpha_a", alpha_a),
            ("alpha_b", alpha_b),
        ):
            if np.isnan(float(val)) or float(val) < 0:
                raise ValueError(f"{nm} must be non-negative, got {val!r}.")
        if assignment not in (None, "one_to_one"):
            raise ValueError(
                f"Unknown assignment: {assignment!r} (expected None or 'one_to_one')."
            )
        # The radius modes search with a ball query at the loosest possible gate,
        # so top_k has no effect there. Silently ignoring a caller's explicit cap
        # would let them believe the candidate set is bounded when it is not; the
        # default (8) is not a choice they made, so it stays quiet. Checked with
        # the other argument validation, before the node tables are read.
        if d_tol_mode != "absolute" and top_k != 8:
            warnings.warn(
                f"top_k={top_k!r} is ignored when d_tol_mode={d_tol_mode!r}: the "
                "size-aware gate varies per pair, so candidates come from a radius "
                "query covering the largest possible gate rather than a k-nearest "
                "query. Use d_tol_mode='absolute' if you need the cap.",
                UserWarning,
                stacklevel=2,
            )

        da = _load_nodes(nodes_a)
        db = _load_nodes(nodes_b)
        n_a, n_b = len(da), len(db)
        ids_a = da[id_col].to_numpy()
        ids_b = db[id_col].to_numpy()
        coords_a = da[list(coord_cols)].to_numpy(dtype=float)
        coords_b = db[list(coord_cols)].to_numpy(dtype=float)
        r_a = _orientation_matrices(da, ori_cols, angle_convention, angle_type)
        r_b = _orientation_matrices(db, ori_cols, angle_convention, angle_type)

        # Per-grain radii, needed only by the size-aware gates.
        rad_a = np.zeros(n_a, dtype=float)
        rad_b = np.zeros(n_b, dtype=float)
        if d_tol_mode != "absolute":
            missing = [
                nm
                for nm, df in (("nodes_a", da), ("nodes_b", db))
                if radius_col not in df.columns
            ]
            if missing:
                raise ValueError(
                    f"d_tol_mode={d_tol_mode!r} needs the {radius_col!r} column, "
                    f"missing from {', '.join(missing)}."
                )
            rad_a = _sanitize_radii(da[radius_col].to_numpy(dtype=float), "nodes_a")
            rad_b = _sanitize_radii(db[radius_col].to_numpy(dtype=float), "nodes_b")

        # Candidate cross edges passing the centroid-distance gate (one absolute
        # d_tol, or a size-aware per-pair gate searched with a covering ball query).
        ai, bj, pair_dist, pair_gate = _candidate_pairs(
            coords_a,
            coords_b,
            rad_a,
            rad_b,
            d_tol,
            d_tol_mode,
            alpha_a,
            alpha_b,
            alpha,
            top_k,
        )
        pair_miso = np.zeros(len(ai), dtype=float)

        # Chance null: how many of these distance-gated pairs survive the
        # misorientation filter when step B's orientations carry no information.
        null_report = (
            self._chance_links(
                r_a, ai, bj, n_b, theta_tol_deg, n_null, seed, misorientation_matrix
            )
            if n_null > 0
            else None
        )

        # Filter by symmetry-aware misorientation.
        if len(ai) > 0:
            miso = (
                misorientation_matrix(r_a[ai], r_b[bj], self.symmetry, "degrees")
                .cpu()
                .numpy()
            )
            keep = miso <= theta_tol_deg
            ai, bj, pair_gate = ai[keep], bj[keep], pair_gate[keep]
            pair_dist, pair_miso = pair_dist[keep], np.asarray(miso)[keep]

        # Record the observed count HERE, where the null measures it: _chance_links
        # counts distance-gated pairs surviving the misorientation filter, and never
        # sees the adjacency prune below. Measured after the prune it would be
        # deflated against mean_links whenever adjacency_b is supplied, making the
        # significance read falsely weak. The post-prune count is reported too, under
        # its own key, since that is the number of links actually used downstream.
        if null_report is not None:
            null_report["observed_links"] = int(len(ai))

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
            # All four pair arrays stay aligned: the assignment and recall steps
            # below index them by the same surviving-pair position.
            ai, bj, pair_gate = ai[keep], bj[keep], pair_gate[keep]
            pair_dist, pair_miso = pair_dist[keep], pair_miso[keep]

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

        # Per-side recall over the surviving links, restricted to the shared volume
        # when the caller flags it and volume-weighted where a volume column exists.
        linked_a = np.zeros(n_a, dtype=bool)
        linked_b = np.zeros(n_b, dtype=bool)
        linked_a[ai] = True
        linked_b[bj] = True
        recall: Dict[str, Any] = {}
        for side, linked, df, spec, name in (
            ("a", linked_a, da, in_overlap_a, "in_overlap_a"),
            ("b", linked_b, db, in_overlap_b, "in_overlap_b"),
        ):
            vol = (
                df[volume_col].to_numpy(dtype=float)
                if volume_col in df.columns
                else None
            )
            mask = _overlap_mask(spec, df, len(df), name)
            recall.update(_side_recall(linked, mask, vol, side))

        result: Dict[str, Any] = {
            "components": components,
            "split_correspondences": split_df,
            "recall": recall,
            **counts,
        }

        if assignment == "one_to_one":
            # Normalise each term by its own tolerance so the two are comparable.
            # Guard the misorientation division the same way the distance one
            # below is guarded: theta_tol_deg == 0 is a degenerate but legal gate
            # (only exactly-equal orientations survive the filter, so every
            # surviving pair has miso == 0 and the term is 0). Without the guard
            # it produced NaN, and linear_sum_assignment then raised.
            miso_term = np.divide(
                pair_miso,
                float(theta_tol_deg),
                out=np.zeros_like(pair_miso),
                where=float(theta_tol_deg) > 0,
            )
            cost = miso_term + np.divide(
                pair_dist,
                pair_gate,
                out=np.zeros_like(pair_dist),
                where=pair_gate > 0,
            )
            rows_o: List[Dict[str, Any]] = []
            for a_idx, b_idx, pair_idx in _one_to_one_blocks(comp, ai, bj, cost):
                for a_i, b_j, p in zip(a_idx, b_idx, pair_idx):
                    rows_o.append(
                        {
                            "a_id": int(ids_a[a_i]),
                            "b_id": int(ids_b[b_j]),
                            "miso_deg": float(pair_miso[p]),
                            "dist": float(pair_dist[p]),
                            "cost": float(cost[p]),
                        }
                    )
            result["one_to_one"] = pd.DataFrame(
                rows_o, columns=["a_id", "b_id", "miso_deg", "dist", "cost"]
            ).sort_values("a_id", ignore_index=True)

        if null_report is not None:
            null_report["observed_links_after_adjacency"] = int(len(ai))
            result["null"] = null_report

        return result

    def _chance_links(
        self,
        r_a,
        ai: np.ndarray,
        bj: np.ndarray,
        n_b: int,
        theta_tol_deg: float,
        n_null: int,
        seed: Optional[int],
        misorientation_matrix,
    ) -> Dict[str, Any]:
        """Links the distance gate finds by chance, over ``n_null`` random trials.

        Each trial keeps the distance-gated candidate pairs fixed and redraws step
        B's orientations uniformly at random, so the count it returns is what the
        misorientation filter admits with no real orientation correspondence. Run
        before the adjacency prune, so it is comparable with the pair count the
        misorientation filter itself produces.

        Args:
            r_a: ``(n_a, 3, 3)`` step-A rotation matrices.
            ai: ``(n_pairs,)`` A index of each distance-gated pair.
            bj: ``(n_pairs,)`` B index of each distance-gated pair.
            n_b: number of step-B grains.
            theta_tol_deg: misorientation gate, in degrees.
            n_null: number of trials.
            seed: seed for the random orientations.
            misorientation_matrix: the symmetry-aware misorientation function.

        Returns:
            dict with ``n_null``, ``mean_links`` and ``std_links``.
        """
        # pylint: disable=import-outside-toplevel
        import torch
        from scipy.spatial.transform import Rotation

        if len(ai) == 0:
            return {"n_null": int(n_null), "mean_links": 0.0, "std_links": 0.0}

        rng = np.random.default_rng(seed)
        counts = []
        for _ in range(int(n_null)):
            # A normalized 4D Gaussian is a uniform random unit quaternion, hence a
            # uniform random rotation. Drawn here rather than with Rotation.random
            # so the seeding does not depend on the scipy version.
            quat = rng.standard_normal((n_b, 4))
            quat /= np.linalg.norm(quat, axis=1, keepdims=True)
            rand_b = Rotation.from_quat(quat).as_matrix()
            tens_b = torch.as_tensor(np.asarray(rand_b, dtype=float)[bj])
            miso = (
                misorientation_matrix(r_a[ai], tens_b, self.symmetry, "degrees")
                .cpu()
                .numpy()
            )
            counts.append(int((miso <= theta_tol_deg).sum()))
        return {
            "n_null": int(n_null),
            "mean_links": float(np.mean(counts)),
            "std_links": float(np.std(counts)),
        }
