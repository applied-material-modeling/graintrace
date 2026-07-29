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

"""Compare two rare-event-identification (REI) point clouds with REIComparison.

Generates two synthetic voxelized rare regions on regular grids (of possibly
different spacing) that share several spherical "hot" blobs, then compares them:
overlap metrics (IoU/Dice/containment), a 1-to-1 cluster correspondence, and a
classified point cloud (only-1 / only-2 / both) exported to VTK.

Synthetic data is generated on demand under ``mwe_data/rei_comparison/`` (not
committed to the repo). Set ``PROFILE = "large"`` for a performance-sized run.
"""

import os
import time

import numpy as np
import pandas as pd

from graintrace.rei_comparison import REIComparison

# INPUT
out_data_dir = "mwe_data/rei_comparison"
out_compare_dir = "rei_comparison_out"

PROFILE = "small"  # "small" (correctness) | "large" (performance)

# grid geometry
if PROFILE == "large":
    grid_n = 300  # base fine grid is grid_n^3
    n_blobs = 20
else:
    grid_n = 40
    n_blobs = 5

spacing_1 = 1.0  # REI 1 (fine) grid spacing
spacing_2 = 2.0  # REI 2 (coarse) grid spacing
blob_radius_range = (max(3, grid_n // 25), max(6, grid_n // 12))
region2_relabel = True  # give REI 2 different cluster ids (tests label-agnostic match)
seed = 42

generate_synthetic = True

os.makedirs(out_data_dir, exist_ok=True)
os.makedirs(out_compare_dir, exist_ok=True)

csv_1 = os.path.join(out_data_dir, f"rei_1_{PROFILE}.csv")
csv_2 = os.path.join(out_data_dir, f"rei_2_{PROFILE}.csv")


# Synthetic data generation
def generate_rei_pair(
    csv_1,
    csv_2,
    grid_n=40,
    n_blobs=5,
    spacing_1=1.0,
    spacing_2=2.0,
    radius_range=(3, 6),
    region2_relabel=True,
    seed=42,
):
    """Write two voxelized rare-region CSVs sharing the same spherical blobs.

    REI 1 samples the blobs on a fine grid (``spacing_1``); REI 2 samples the
    SAME physical blobs on a coarser grid (``spacing_2``, subsampled). Each blob
    is a distinct ``rare_cluster_id``. Columns: x, y, z, rare_cluster_id.
    """
    rng = np.random.default_rng(seed)
    grid = np.zeros((grid_n, grid_n, grid_n), dtype=np.int32)
    zz, yy, xx = np.ogrid[:grid_n, :grid_n, :grid_n]
    pad = radius_range[1] + 1
    for b in range(n_blobs):
        cx, cy, cz = rng.integers(pad, grid_n - pad, size=3)
        radius = int(rng.integers(radius_range[0], radius_range[1] + 1))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius**2
        grid[mask] = b + 1  # cluster id (1-based)

    idx = np.argwhere(grid > 0)
    cid = grid[grid > 0]

    # REI 1: fine grid, physical coords = index * spacing_1
    pd.DataFrame(
        {
            "x": idx[:, 0] * spacing_1,
            "y": idx[:, 1] * spacing_1,
            "z": idx[:, 2] * spacing_1,
            "rare_cluster_id": cid,
        }
    ).to_csv(csv_1, index=False)

    # REI 2: coarse grid = same physical region subsampled every (spacing_2/spacing_1)
    step = max(1, int(round(spacing_2 / spacing_1)))
    sub = (idx % step == 0).all(axis=1)
    idx2 = idx[sub] // step
    cid2 = cid[sub]
    if region2_relabel:  # different labels; matching must rely on overlap, not ids
        cid2 = (cid2 * 7) % max(2, n_blobs + 3) + 1
    pd.DataFrame(
        {
            "x": idx2[:, 0] * spacing_2,
            "y": idx2[:, 1] * spacing_2,
            "z": idx2[:, 2] * spacing_2,
            "rare_cluster_id": cid2,
        }
    ).to_csv(csv_2, index=False)

    print(
        f"Generated REI pair: {csv_1} ({(grid > 0).sum()} pts), "
        f"{csv_2} ({int(sub.sum())} pts)."
    )
    return csv_1, csv_2


# Main

if generate_synthetic:
    generate_rei_pair(
        csv_1,
        csv_2,
        grid_n=grid_n,
        n_blobs=n_blobs,
        spacing_1=spacing_1,
        spacing_2=spacing_2,
        radius_range=blob_radius_range,
        region2_relabel=region2_relabel,
        seed=seed,
    )

comparison = REIComparison(
    rei_csv_1=csv_1,
    rei_csv_2=csv_2,
    output_dir=out_compare_dir,
    spacing_1=spacing_1,  # set None to auto-detect from the CSV
    spacing_2=spacing_2,
    coord_cols=("x", "y", "z"),
    cluster_col="rare_cluster_id",  # set None to skip cluster-level matching
    supersample=1,
)

t0 = time.time()
result = comparison.run_comparison()
print(f"\nComparison finished in {time.time() - t0:.2f}s")

print("\nOutputs:")
print("  metrics JSON  :", result["metrics_path"])
print("  overlap VTK   :", result["overlap_vtk_path"])
print("  cluster match :", result["cluster_match_path"])
