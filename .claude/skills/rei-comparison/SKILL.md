---
name: rei-comparison
description: >
  Compare two rare-event-identification (REI) point clouds for spatial overlap — IoU/Dice/
  containment metrics, a 1-to-1 cluster correspondence (split/merge detection), and a classified
  point cloud (only-1 / only-2 / both) exported to VTK (REIComparison). Use to compare two REI
  results (two metrics, thresholds, methods, or prediction vs. reference), possibly on grids of
  different spacing.
---

# REI comparison

Uses `rei_comparison.REIComparison`. Env: `conda activate graintrace_env`.
Pure Python (numpy + scipy). No MOOSE/NEPER/CUBIT. Analogous to `ScanStitchingComparison`
but for rare-region point clouds instead of grain sets.

## Model (why not KD-tree / alpha-shape / marching cubes)
Each rare point is the center of its voxel cube, so a REI is a union of axis-aligned cubes and
overlap is a boolean **volume** intersection. Both regions are resampled onto a common **finer**
lattice (`s_ref = min(spacing_1, spacing_2)`); membership is then an O(1) integer-index hash
lookup (`ijk = round((p-origin)/s)` — voxels partition space, so "inside" ≡ "its cell is
occupied"). Non-contiguous regions are handled for free. No surface reconstruction needed:
region volume is just `voxel_count × voxel_volume`.

## Inputs
Two voxelized REI point-cloud CSVs, each with `x,y,z` columns and an optional integer
`rare_cluster_id`. Produce them from the REI pipeline: pass `rare_points_csv_path=...` to
`IdentifyRareClusters.run_get_rare_cluster` (writes `x,y,z,rare_cluster_id` for the rare
points). See the `/rare-event-identification` skill.

**Assumptions:** each grid is regular (constant per-axis spacing) and the two grids **share an
origin** (no rotation/translation is applied here — register the clouds first if they don't).

## Recipe
```python
from graintrace.rei_comparison import REIComparison

comp = REIComparison(
    rei_csv_1="out/rei_A.csv", rei_csv_2="out/rei_B.csv",
    output_dir="out/rei_comparison",
    spacing_1=1.0, spacing_2=2.0,   # pass true grid spacing; None = auto-detect (sparse-unsafe)
    coord_cols=("x", "y", "z"),
    cluster_col="rare_cluster_id",  # None -> skip cluster-level matching
    supersample=1,                  # >1 = sub-voxel boundary accuracy (rarely needed)
)
result = comp.run_comparison()
print(result["metrics"]["iou"], result["metrics"]["containment_1"])
```

## Outputs (in `output_dir`)
- `overlap_metrics.json` — IoU (Jaccard), Dice, `containment_1`/`containment_2` (asymmetric),
  voxel counts + volumes, and (with `cluster_col`) cluster/split/merge counts.
- `overlap_cloud.vtk` — classified polydata points, scalar `membership` (1=only-1, 2=only-2,
  3=both) + `cluster_id_1`/`cluster_id_2`. Color by `membership` in ParaView.
- `cluster_match.csv` — 1-to-1 cluster pairing (Hungarian by overlap volume, label-agnostic)
  with per-pair Jaccard/containment; unmatched clusters flagged with `-1`.

## Key parameters
- `spacing_1` / `spacing_2` — scalar or `[dx,dy,dz]`. **Pass the true spacing**; auto-detect
  uses the min positive coordinate step and can be wrong for sparse clouds.
- `cluster_col` — enables the cluster correspondence; `None` gives global overlap only.
- `split_merge_fraction` (default 0.2) — significance threshold for split/merge counting.

## Gotchas
- Different spacings are fine (the coarser region is upsampled to `s_ref`); different **origins**
  are not — align first.
- Boundary discretization error is ~one `s_ref` voxel; increase `supersample` only if that
  matters. Coarse-vs-fine IoU < 1 even for the "same" region is expected (cube extents differ at
  the boundary by half the coarse cell).

## See also
`examples/demonstrate_rei_comparison.py` (generates a synthetic pair on demand under
`mwe_data/rei_comparison/`); `/rare-event-identification`; CLAUDE.md §7 (REI comparison).
