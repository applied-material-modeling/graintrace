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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
        ckpt_exists = os.path.exists(ckpt_base + ".edges.npy") and os.path.exists(
            ckpt_base + ".weights.npy"
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
        background_label: int = -1,
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
            background_label: label treated as void/background and skipped.

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

        voxvol = _voxel_volume(coords)
        rows = []
        for gid in np.unique(labels):
            if gid == background_label:
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
        top_k: int = 8,
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
            top_k: nearest B centroids probed per A grain.
            adjacency_b: optional iterable of ``(grain_id, grain_id)`` pairs giving
                intra-step-B grain adjacency; used to reject a spurious split child
                that is not adjacent to any other child of the same parent.
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

        # Candidate cross edges: each A grain's top-k nearest B centroids.
        tree_b = cKDTree(coords_b)
        kq = min(top_k, n_b)
        dist, jdx = tree_b.query(coords_a, k=kq)
        dist = np.asarray(dist).reshape(n_a, kq)
        jdx = np.asarray(jdx).reshape(n_a, kq)

        ai, bj = [], []
        for i in range(n_a):
            for col in range(kq):
                j = int(jdx[i, col])
                if dist[i, col] <= d_tol:
                    ai.append(i)
                    bj.append(j)
        ai = np.asarray(ai, dtype=np.int64)
        bj = np.asarray(bj, dtype=np.int64)

        # Filter by symmetry-aware misorientation.
        if len(ai) > 0:
            miso = (
                misorientation_matrix(r_a[ai], r_b[bj], self.symmetry, "degrees")
                .cpu()
                .numpy()
            )
            keep = miso <= theta_tol_deg
            ai, bj = ai[keep], bj[keep]

        # Optional adjacency consistency: drop a candidate child not adjacent (in B)
        # to any sibling under the same parent.
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
                for pos, child in zip(sel, children):
                    if not adj.get(int(child), set()) & (cset - {int(child)}):
                        keep[pos] = False
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
