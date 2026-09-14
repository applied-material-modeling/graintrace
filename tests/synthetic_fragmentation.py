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

"""Test helper: synthetic grain-split datasets with known ground truth.

Not part of the graintrace package (no physics; purely for testing/examples). A
polycrystal of several grains fragments with **mixed multiplicity** across load
steps -- some grains stay whole (1->1), some split binary (1->2), some ternary
(1->3) -- like real fragmentation, not a single grain always splitting. A grain
with multiplicity ``m`` breaks into ``m`` sub-grains, each rotating by ``theta(t)``
about a distinct axis, so all sub-grains diverge (in different directions) as the
load grows -- early steps read as one grain, late steps as ``m``. Each emitter
writes a ``ground_truth.json`` (the
per-grain multiplicities + split step) so ``FragmentationAnalyzer.detect_splits``
can be scored exactly.

The default microstructure has 5 grains with multiplicities ``[1, 2, 3, 1, 2]``
(two whole, two binary, one ternary). The shipped
``mwe_data/synthetic_split_{ff,nf,ebsd}/`` were generated with these emitters;
re-run to regenerate. Orientations are Bunge Euler angles in degrees.

* :func:`generate_ebsd_split` -- per-step merged ``x,y,z,Eul0,Eul1,Eul2`` CSVs.
* :func:`generate_nf_split` -- per-step folders of ``.mic``-schema per-layer CSVs.
* :func:`generate_ff_split` -- per-step FF grain-table CSVs
  (``grain_id,X,Y,Z,GrainRadius,Eul0,Eul1,Eul2``).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_MULTIPLICITIES = (1, 2, 3, 1, 2)  # whole / binary / ternary / whole / binary


def _rotate_euler_deg(eul_deg: np.ndarray, theta_deg: float, axis=(1.0, 0.0, 0.0)):
    """Compose Bunge Euler angles (deg) with a rotation of ``theta`` about ``axis``."""
    # pylint: disable=import-outside-toplevel
    import torch

    from graintrace.orientation_helper import euler_to_matrix, matrix_to_euler

    rmat = euler_to_matrix(
        torch.as_tensor(np.atleast_2d(eul_deg), dtype=torch.float64), "bunge", "degrees"
    )
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    th = np.deg2rad(theta_deg)
    cos, sin = np.cos(th), np.sin(th)
    kx, ky, kz = ax
    # Rodrigues' rotation matrix
    rot = np.array(
        [
            [
                cos + kx * kx * (1 - cos),
                kx * ky * (1 - cos) - kz * sin,
                kx * kz * (1 - cos) + ky * sin,
            ],
            [
                ky * kx * (1 - cos) + kz * sin,
                cos + ky * ky * (1 - cos),
                ky * kz * (1 - cos) - kx * sin,
            ],
            [
                kz * kx * (1 - cos) - ky * sin,
                kz * ky * (1 - cos) + kx * sin,
                cos + kz * kz * (1 - cos),
            ],
        ]
    )
    rot_t = torch.as_tensor(rot, dtype=torch.float64)
    new_r = torch.matmul(rot_t.unsqueeze(0), rmat)
    return matrix_to_euler(new_r, "bunge", "degrees").cpu().numpy()


def _base_orientations(n: int, min_sep_deg: float = 25.0, seed: int = 0) -> np.ndarray:
    """``n`` Bunge-deg orientations pairwise cubic-misoriented by > ``min_sep_deg``.

    Greedy rejection sampling so distinct grains never cross-link in the detector.
    """
    # pylint: disable=import-outside-toplevel
    import torch

    from graintrace.orientation_helper import euler_to_matrix, misorientation_matrix

    rng = np.random.default_rng(seed)
    picked: List[np.ndarray] = []
    r_picked: List = []
    tries = 0
    while len(picked) < n and tries < 200000:
        tries += 1
        e = rng.uniform([0.0, 0.0, 0.0], [360.0, 90.0, 360.0])
        r = euler_to_matrix(
            torch.as_tensor(e[None, :], dtype=torch.float64), "bunge", "degrees"
        )
        if picked:
            all_r = torch.cat(r_picked, dim=0)
            d = misorientation_matrix(
                all_r, r.repeat(len(all_r), 1, 1), "432", "degrees"
            ).numpy()
            if d.min() <= min_sep_deg:
                continue
        picked.append(e)
        r_picked.append(r)
    if len(picked) < n:
        raise RuntimeError(
            f"could not sample {n} orientations > {min_sep_deg} deg apart"
        )
    return np.asarray(picked)


# Distinct rotation axes for the sub-grains of a fragmenting grain. Each child is
# the parent rotated by ``theta`` about its OWN axis, so all children move (a
# ternary shows three arrows, not two + a hidden static one), each stays within
# ``theta`` of the parent (they link to it in the detector), and they separate
# from each other. Axes are normalized inside ``_rotate_euler_deg``.
_CHILD_AXES = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
]


def _child_axes(m: int) -> List[Tuple[float, float, float]]:
    """The ``m`` distinct rotation axes for a grain fragmenting into ``m`` children."""
    if m > len(_CHILD_AXES):
        raise ValueError(
            f"multiplicity {m} exceeds the {len(_CHILD_AXES)} defined axes"
        )
    return _CHILD_AXES[:m]


def _ground_truth(
    multiplicities: Sequence[int], thetas: np.ndarray, misorientation_tol_deg: float
) -> Dict:
    over = np.where(thetas > misorientation_tol_deg)[0]
    return {
        # parent grain id (1-based) -> number of sub-grains it fragments into
        "multiplicities": {str(g + 1): int(m) for g, m in enumerate(multiplicities)},
        "n_splitting_grains": int(sum(1 for m in multiplicities if m > 1)),
        "split_step": int(over[0]) if len(over) else None,
        "thetas_deg": [float(t) for t in thetas],
        "misorientation_tol_deg": float(misorientation_tol_deg),
    }


# --------------------------------------------------------------------------- #
# Voxel microstructure (EBSD / NF): grains as x-slabs, sub-grains as y-bands
# --------------------------------------------------------------------------- #
def _voxel_series(
    nx: int,
    ny: int,
    nz: int,
    spacing: float,
    multiplicities: Sequence[int],
    thetas: np.ndarray,
    seed: int = 0,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return ``(coords, [euler_per_step])`` for the voxel block.

    Grain ``g`` occupies an x-slab; if its multiplicity is ``m`` the slab is split
    into ``m`` y-bands, band ``j`` rotated by ``theta(t)`` about the distinct axis
    ``_child_axes(m)[j]`` so all sub-grains diverge (in different directions).
    """
    xs, ys, zs = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    coords = (
        np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(float) * spacing
    )
    n_grains = len(multiplicities)
    slab = np.clip(
        (coords[:, 0] / (nx * spacing) * n_grains).astype(int), 0, n_grains - 1
    )
    base = _base_orientations(n_grains, seed=seed)
    base_eul = base[slab]

    # per-voxel (multiplicity, band index); band 0 for whole grains
    grain_mult = np.ones(len(coords), dtype=int)
    band_idx = np.zeros(len(coords), dtype=int)
    for g, m in enumerate(multiplicities):
        mask = slab == g
        grain_mult[mask] = m
        if m == 1:
            continue
        yg = coords[mask, 1]
        span = yg.max() - yg.min()
        band = np.floor((yg - yg.min()) / (span + 1e-9) * m).astype(int)
        band_idx[mask] = np.clip(band, 0, m - 1)

    steps = []
    for th in thetas:
        eul = base_eul.copy()
        if th > 0:
            for m in sorted({int(x) for x in multiplicities}):
                if m == 1:
                    continue
                axes = _child_axes(m)
                for j in range(m):
                    mm = (grain_mult == m) & (band_idx == j)
                    if mm.any():
                        eul[mm] = _rotate_euler_deg(eul[mm], float(th), axes[j])
        steps.append(eul)
    return coords, steps


def generate_ebsd_split(
    out_dir: str,
    nx: int = 30,
    ny: int = 24,
    nz: int = 3,
    n_steps: int = 5,
    theta_max_deg: float = 12.0,
    misorientation_tol_deg: float = 5.0,
    spacing: float = 2.0,
    seed: int = 0,
    multiplicities: Sequence[int] = DEFAULT_MULTIPLICITIES,
) -> Dict:
    """Write per-step EBSD ``x,y,z,Eul0,Eul1,Eul2`` CSVs with mixed-multiplicity splits.

    Returns the ground-truth dict (also written to ``ground_truth.json``).
    """
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    thetas = np.linspace(0.0, theta_max_deg, n_steps)
    coords, steps = _voxel_series(
        nx, ny, nz, spacing, multiplicities, thetas, seed=seed
    )

    paths = []
    for k, eul in enumerate(steps):
        df = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "Eul0": eul[:, 0],
                "Eul1": eul[:, 1],
                "Eul2": eul[:, 2],
            }
        )
        p = os.path.join(out_dir, f"ebsd_step{k:02d}.csv")
        df.to_csv(p, index=False)
        paths.append(p)

    gt = _ground_truth(multiplicities, thetas, misorientation_tol_deg)
    gt["step_csvs"] = paths
    with open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=2)
    return gt


def generate_nf_split(
    out_dir: str,
    nx: int = 30,
    ny: int = 24,
    nz: int = 3,
    n_steps: int = 5,
    theta_max_deg: float = 12.0,
    misorientation_tol_deg: float = 5.0,
    spacing: float = 2.0,
    seed: int = 0,
    exp_file_token: str = "layer",
    multiplicities: Sequence[int] = DEFAULT_MULTIPLICITIES,
) -> Dict:
    """Write per-step folders of ``.mic``-schema per-layer NF CSVs (mixed splits).

    Each step gets ``<out_dir>/step{k}/`` holding one ``<token>{L}.mic`` per z
    layer, ingestible by ``NearFieldMeshBuilder`` via ``exp_file_token``. Returns
    the ground-truth dict (also written to ``ground_truth.json``).
    """
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    thetas = np.linspace(0.0, theta_max_deg, n_steps)
    coords, steps = _voxel_series(
        nx, ny, nz, spacing, multiplicities, thetas, seed=seed
    )
    z_vals = np.unique(coords[:, 2])

    step_dirs = []
    for k, eul in enumerate(steps):
        sdir = os.path.join(out_dir, f"step{k:02d}")
        os.makedirs(sdir, exist_ok=True)
        for li, zval in enumerate(z_vals, start=1):
            m = coords[:, 2] == zval
            xy = coords[m][:, :2]
            e = eul[m]
            n = len(xy)
            df = pd.DataFrame(
                {
                    "%OrientationRowNr": np.arange(1, n + 1, dtype=float),
                    "NrMatches": np.ones(n),
                    "RunTime": np.full(n, 0.1),
                    "X": xy[:, 0],
                    "Y": xy[:, 1],
                    "TriEdgeSize": np.full(n, spacing),
                    "UpDown": np.full(n, -1.0),
                    "Eul1": e[:, 0],
                    "Eul2": e[:, 1],
                    "Eul3": e[:, 2],
                    "Confidence": np.full(n, 0.9),
                    "PhaseNr": np.ones(n, dtype=int),
                }
            )
            p = os.path.join(sdir, f"{exp_file_token}{li}.mic")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(f"%TriEdgeSize {spacing:.6f}\n")
                fh.write("%NumPhases 1\n")
                fh.write(f"%GlobalPosition {float(zval):.6f}\n")
                df.to_csv(fh, sep="\t", index=False)
        step_dirs.append(sdir)

    gt = _ground_truth(multiplicities, thetas, misorientation_tol_deg)
    gt["step_dirs"] = step_dirs
    gt["exp_file_token"] = exp_file_token
    with open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=2)
    return gt


# --------------------------------------------------------------------------- #
# FF grain table: one row per grain (pre-split) / per sub-grain (post-split)
# --------------------------------------------------------------------------- #
def generate_ff_split(
    out_dir: str,
    n_steps: int = 5,
    theta_max_deg: float = 12.0,
    misorientation_tol_deg: float = 5.0,
    seed: int = 0,
    box: float = 200.0,
    multiplicities: Sequence[int] = DEFAULT_MULTIPLICITIES,
) -> Dict:
    """Write per-step FF grain-table CSVs with mixed-multiplicity grain splits.

    A grain is one row while ``theta <= tol``; once past it, a grain of
    multiplicity ``m`` becomes ``m`` rows at nearby sub-centroids with orientations
    spread symmetrically about the parent. Returns the ground-truth dict (also
    written to ``ground_truth.json``); it adds ``parent_children`` mapping each
    splitting parent grain id to its post-split child ids. Columns:
    ``grain_id,X,Y,Z,GrainRadius,Eul0,Eul1,Eul2``.
    """
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    n_grains = len(multiplicities)
    base = _base_orientations(n_grains, seed=seed)
    centers = np.array(
        [[(g + 0.5) / n_grains * box, box / 2.0, 0.0] for g in range(n_grains)]
    )
    thetas = np.linspace(0.0, theta_max_deg, n_steps)

    # deterministic child ids: the first child keeps the parent id, the rest are new
    child_ids: Dict[int, List[int]] = {}
    nxt = n_grains + 1
    for g, m in enumerate(multiplicities):
        ids = [g + 1]
        for _ in range(m - 1):
            ids.append(nxt)
            nxt += 1
        child_ids[g] = ids

    paths = []
    for k, th in enumerate(thetas):
        rows = []
        for g, m in enumerate(multiplicities):
            c, e = centers[g], base[g]
            if m == 1 or th <= misorientation_tol_deg:
                rows.append(_ff_row(g + 1, c, e, 15.0))
            else:
                axes = _child_axes(m)
                for j in range(m):
                    ej = _rotate_euler_deg(e, float(th), axes[j])[0]
                    # fan the child sub-centroids in y around the parent centroid
                    cj = c + np.array([0.0, (j - (m - 1) / 2.0) * 12.0, 0.0])
                    rows.append(_ff_row(child_ids[g][j], cj, ej, 10.0))
        p = os.path.join(out_dir, f"ff_step{k:02d}.csv")
        pd.DataFrame(rows).to_csv(p, index=False)
        paths.append(p)

    gt = _ground_truth(multiplicities, thetas, misorientation_tol_deg)
    gt["step_csvs"] = paths
    gt["parent_children"] = {
        str(g + 1): child_ids[g] for g, m in enumerate(multiplicities) if m > 1
    }
    with open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=2)
    return gt


def _ff_row(gid: int, c: np.ndarray, eul: np.ndarray, radius: float) -> Dict:
    return {
        "grain_id": int(gid),
        "X": float(c[0]),
        "Y": float(c[1]),
        "Z": float(c[2]),
        "GrainRadius": radius,
        "Eul0": float(eul[0]),
        "Eul1": float(eul[1]),
        "Eul2": float(eul[2]),
    }
