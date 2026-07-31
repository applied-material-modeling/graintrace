---
name: hedm-stitching
description: >
  Generate a synthetic crystal, simulate overlapping HEDM z-scans, stitch the scan
  layers into one grain set, and compare stitched vs. true. Use when the user wants
  to test/compare HEDM stitching (RegionBaseStitching / NaiveStitching), study scan
  overlap, or produce a stitched FF CSV from multiple scan layers.
---

# HEDM scan simulation + stitching

Mirrors `examples/demonstrate_hedm_study.py`. Env: `conda activate graintrace_env`.
External tool: **NEPER** (crystal generation + z-scan). No MOOSE/CUBIT.

## Inputs
Self-generating: `CrystalGenerator` writes `<out>/voronoi.tess`, `voronoi.csv`, and
`<out>/hedm_scan/scan_{i}.csv`. For real data, feed your own per-scan CSVs to the stitcher
(whitespace/comma FF grain tables with `X,Y,Z,GrainRadius,Eul0/1/2,eKen*`; z already shifted
per layer; see CLAUDE.md §2).

## Recipe
```python
from graintrace.generate_random_crystal import CrystalGenerator
from graintrace.hedm_stitching_techniques.region_base_stitching import RegionBaseStitching
from graintrace.scan_stitching_comparison import ScanStitchingComparison

bounding_box = [-500, 500, -500, 500, -1000, 500]
nscan, overlap_percentage = 4, 25

cg = CrystalGenerator(output_dir="hedm_out", bounding_box=bounding_box, seed=42)
cg.generate_tessellation(morpho_args={"type": "diameq", "distribution": "lognormal",
                                      "params": (130.0, 5.0)})
cg.hedm_zscan(tess_file="hedm_out/voronoi.tess", nstep=nscan,
              overlap_percentage=overlap_percentage,
              position_noise_std=0.0, orientation_noise_std=0.0, radius_noise=0.0,
              noise_seed=42)

scan_files = [f"hedm_out/hedm_scan/scan_{i}.csv" for i in range(nscan)]
stitcher = RegionBaseStitching(
    scan_files=scan_files, output_csv="hedm_out/stitched.csv",
    position_tolerance=20, orientation_tolerance=1, radius_tolerance=0,
    weights={"pos": 0.1, "ori": 1.0, "rad": 0}, min_neighbors=5,
)
stitched = stitcher.run(zlo=bounding_box[4], zhi=bounding_box[5],
                        overlap_fraction=overlap_percentage / 100.0)

ScanStitchingComparison(
    output_dir="hedm_out/comparison", true_csv="hedm_out/voronoi.csv",
    stitch_csv="hedm_out/stitched.csv", position_tolerance=20,
    orientation_tolerance=5.0, radius_tolerance=0,
    weights={"pos": 0.1, "ori": 1.0, "rad": 0},
).run_comparison()
```

## Key parameters
- `crystal_morpho_args`: grain morphology (`CrystalGenerator.show_morpho_options()` lists all).
- `nscan`, `overlap_percentage`: number of z-scan layers and their overlap.
- Noise (each applied only when > 0; `noise_seed` for reproducibility): `position_noise_std`
  (absolute centroid std, length units), `orientation_noise_std` (proper misorientation, degrees),
  `radius_noise` (relative fraction). `remove_minimum_volume`/`min_vol` drop small grains.
- Stitch: `position_tolerance` (length), `orientation_tolerance` (deg), `weights`, `min_neighbors`.
- Alternative stitchers: `NaiveStitching`, `RegionBaseStitching`.

## Gotchas
- `orientation_tolerance` is in **degrees** here; if a downstream step uses radians, convert
  (`np.deg2rad`); see CLAUDE.md §10.
- Radius: set `radius_tolerance=-1` to disable radius in the match cost.

## See also
`examples/demonstrate_hedm_study.py`; CLAUDE.md §2 (data formats), §3 Step 1 (stitching).
