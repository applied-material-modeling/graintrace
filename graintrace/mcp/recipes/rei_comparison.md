---
segment: rei_comparison
tool: compare_rei
applies_to: Compare two rare-event (REI) point clouds for spatial overlap
defaults:
  spacing_1: null
  spacing_2: null
  coord_cols: ["x", "y", "z"]
  cluster_col: rare_cluster_id
  supersample: 1
  split_merge_fraction: 0.2
---

# REI Comparison — recommended parameters

`compare_rei` wraps `REIComparison`: it measures the spatial overlap between two
rare-event point clouds (e.g. two field metrics, two thresholds, two methods, or
prediction vs. reference). **Pure Python** (numpy + scipy). No NEPER/CUBIT/MOOSE.

## Model
Each rare point is treated as its voxel cube, so a REI is a union of axis-aligned
cubes and overlap is a boolean volume intersection. Both regions are resampled
onto a common **finer** lattice (`s_ref = min(spacing_1, spacing_2)`); membership
is then an O(1) integer-index hash lookup. Non-contiguous regions are handled for
free. No KD-tree / alpha-shape / marching-cubes.

## Minimum you must supply
- `rei_csv_1`, `rei_csv_2` — voxelized REI point clouds. Columns `x,y,z` and an
  optional integer `rare_cluster_id`. Produce these from `identify_rare_events`
  via its `rare_points_csv_path` output.

## Choosing the key options

| Parameter | Recommended | Why |
|---|---|---|
| `spacing_1`, `spacing_2` | grid spacing (µm) | **Pass the true spacing.** Auto-detect (`null`) uses the min positive coordinate step, which can be wrong for sparse/scattered clouds. Scalar or `[dx,dy,dz]`. |
| `cluster_col` | `rare_cluster_id` | Enables the 1-to-1 (Hungarian) cluster correspondence + split/merge detection. Set `null` to skip and only compute global overlap. |
| `supersample` | `1` | Sub-divides `s_ref` for sub-voxel boundary accuracy at higher cost. Rarely needed. |
| `split_merge_fraction` | `0.2` | A cluster is counted as split/merged when it shares ≥ this fraction of its voxels with more than one partner. |

## Assumptions
- Both grids are **regular** (constant spacing per axis) and **share an origin**.
  Translation/rotation registration is out of scope — align the clouds first if
  they do not share a frame.

## Outputs (in `output_dir`)
- `overlap_metrics.json` — IoU (Jaccard), Dice, `containment_1`/`containment_2`
  (asymmetric), voxel counts and volumes.
- `overlap_cloud.vtk` — classified point cloud, scalar `membership`
  (1 = only-1, 2 = only-2, 3 = both) plus `cluster_id_1`/`cluster_id_2`.
- `cluster_match.csv` — 1-to-1 cluster pairing with overlap volume + per-pair
  Jaccard/containment; unmatched clusters flagged with `-1`.

## Reading the numbers
- `iou` — symmetric agreement of the two regions overall.
- `containment_1 = |A∩B|/|A|`, `containment_2 = |A∩B|/|B|` — how much of each
  region the other covers (use when one REI is a reference).
