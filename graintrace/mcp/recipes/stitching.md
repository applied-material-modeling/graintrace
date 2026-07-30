---
segment: stitching
tool: stitch_scans
applies_to: Merge overlapping FF-HEDM z-scan layers into one grain set
defaults:
  position_tolerance: 50
  orientation_tolerance: 5.0
  radius_tolerance: -1
  min_neighbors: 5
  orientation_convention: bunge
  orientation_units: degrees
  symmetry: "432"
  refine_extents: false
---

# HEDM Stitching: recommended parameters

`stitch_scans` wraps `RegionBaseStitching`: it merges duplicate grains across
overlapping z-scans into one consistent grain set. **Needs NEPER only if
`refine_extents=true`** (opt-in tessellation extent refinement); otherwise pure
Python.

## Minimum you must supply
- `scan_files`: list of pre-processed per-layer CSVs with Z already shifted by
  `Z + scan_idx * Zheight_per_file * (1 - overlap_fraction)`.
- `output_csv`: merged output path.
- `zlo`, `zhi`, `overlap_fraction`: passed to `.run(...)`.

**Must-ask (scan geometry not in the CSVs):** `zlo`, `zhi`, `overlap_fraction`,
and `orientation_units`. Pass them directly or via a `sample_json`; otherwise
`stitch_scans` returns `needs_input`. See the `experiment_metadata` recipe.

## Choosing the key options

| Parameter | Recommended | Why |
|---|---|---|
| `position_tolerance` | `50` (µm) | Max centroid distance for two detections to be the same grain. Scale to grain size. |
| `orientation_tolerance` | `5.0` | Degrees by default. **If `orientation_units="radians"`, convert first** (`np.deg2rad`). |
| `radius_tolerance` | `-1` | `-1` disables the radius gate. Set a positive µm value to also gate on size. |
| `weights` | `{"pos":0.1,"ori":1.0,"rad":0}` | Cost weighting for the match. `rad:0` drops radius from the cost. |
| `min_neighbors` | `5` | Neighborhood size for region classification. |
| `refine_extents` | `false` | `true` uses true per-cell `[zmin,zmax]` from a NEPER re-tessellation instead of `z ± GrainRadius`. Only helps elongated grains; slower; needs NEPER. Validate with the comparison tool before trusting on new data. |

## Units contract
`orientation_tolerance` **must be in the same unit as `orientation_units`**. If
radians, convert both the tolerance and any sample-rotate angle with
`np.deg2rad` before running.

## Validating a stitch
Follow with `compare_stitching` (`ScanStitchingComparison`) against a known/true
grain set to check recall/precision before using the result downstream.
