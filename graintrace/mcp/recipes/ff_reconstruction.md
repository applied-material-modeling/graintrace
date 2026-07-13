---
segment: ff_reconstruction
tool: ff_reconstruct
applies_to: Far-field HEDM grain-centroid CSV -> 3D Voronoi microstructure
defaults:
  dim: 3
  option: centroid
  CVT_iter: 1000
  morphoalgo: subplex
  weighted: false
  auto_fix_bbox: true
  bbox_fix_mode: remove_points
  bbox_tolerance: 2.5
  mesh_quality_min: 0.7
  relative_el_size: 2.0
  generate_mesh: false
  strain_unit: microstrain
  unit: deg
---

# FF Reconstruction — recommended parameters

`ff_reconstruct` wraps `VoronoiMeshBuilder`: it turns an FF-HEDM grain-centroid
CSV into a NEPER Voronoi/CVT tessellation, per-grain orientations, and a
per-grain initial elastic-strain (`ee`) file for CPFE. **Needs NEPER** (and GMSH
only if `generate_mesh=true`).

## Minimum you must supply
- `input_csv` — stitched/rotated FF CSV with `X,Y,Z`, `Eul0/1/2`, and (for the
  `ee` file) `eKen11..eKen33` or `eFab11..eFab33`.
- `output_dir` — where to write `reconstruction*.{csv,dat}` (and `.msh`).
- `bounding_box` — `[xlo,xhi,ylo,yhi,zlo,zhi]` in micrometers.

## Choosing the key options

| Parameter | Recommended | Why |
|---|---|---|
| `option` | `centroid` | Seeds cells at the measured grain centroids. Use `voronoi` for plain Voronoi, `centroidsize` to also honor `GrainRadius`. |
| `CVT_iter` | `1000` | CVT relaxation steps. Lower (100–300) for a quick look; 1000 for production. |
| `morphoalgo` | `subplex` | Robust default. `lloyd` is faster but lower quality; `praxis` is an alternative optimizer. |
| `unit` | `deg` or `rad` | **Must match the actual Euler units in the CSV.** Detect: any value > 2π ⇒ degrees. |
| `strain_unit` | `microstrain` | `eKen*` columns are typically microstrain; use `strain` if already dimensionless. |
| `generate_mesh` | `false` | Leave off unless you need the GMSH `.msh` now — it is slow. For NF+FF you only need the `ee` file, so keep it `false`. |
| `weighted` | `false` | Set `true` for a Laguerre (radius-weighted) tessellation when grain sizes vary a lot. |

## Sample tilt correction
If the sample was tilted, set `auto_rotate=true` with
`rotate_angles=[0,0,<deg>]` and `rotate_convention="xyz"` (angles in the same
unit as `unit`). Otherwise leave `auto_rotate=false`.

## Bounding box hygiene
Keep `auto_fix_bbox=true`. Use `bbox_fix_mode="remove_points"` for production
(drops out-of-box grains) and `"extend_bounding_box"` only for debugging.

## Key outputs (in `output_dir`)
- `reconstruction_reformatted.csv` — per-voxel grain IDs + orientations
- `reconstruction_cpfe_ee.csv` — per-grain elastic strain (CPFE initial condition)
- `orientations.dat` — per-grain Euler angles (**always degrees after FF build**)
- `reconstruction.msh` — GMSH mesh (only if `generate_mesh=true`)

## Gotchas
- FF `orientations.dat` is always degrees regardless of input `unit`. When
  feeding a downstream `VoxelMeshBuilder`, set its `angle_type="degrees"`.
- For CPFE you convert `orientations.dat` (Euler-Bunge, deg) → NEML2 v3 MRP via
  `orientation_helper.euler_to_mrp` before passing as the sim `ori_file`.
- Anisotropic/elongated grains: FF cannot recover true grain shape; a per-scan
  tessellation does not reliably beat the equivalent sphere. Use NF-HEDM for
  morphology.
