---
name: ff-reconstruction
description: >
  Reconstruct a 3D microstructure from far-field (FF) HEDM grain centroids via NEPER
  Voronoi/CVT tessellation (VoronoiMeshBuilder). Use when the user wants a .tess/.msh
  mesh, per-grain orientations, or a per-grain initial elastic-strain (ee) file from an
  FF grain CSV, to feed CPFE or graph building.
---

# FF (far-field) Voronoi reconstruction

Uses `VoronoiMeshBuilder` (NEPER). Env: `conda activate moose-src`. External tool:
**NEPER** (+ GMSH if `generate_mesh=True`).

## Inputs
An FF grain CSV with `X,Y,Z` (+ optional `GrainRadius`), Euler columns (`Eul0/1/2`), and a
9-component elastic strain (`eKen11..eKen33`). Self-contained example data:
`mwe_data/ff_calibration/0.csv` (500 grains). See CLAUDE.md §2 for the raw-FF format.

## Recipe
```python
from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder

builder = VoronoiMeshBuilder(
    input_csv="mwe_data/ff_calibration/0.csv",
    output_dir="out/FF",
    bounding_box=[-477, 528, -487, 532, -1025, 625],   # xlo,xhi,ylo,yhi,zlo,zhi
    dim=3, weighted=False,
    auto_fix_bbox=True, bbox_fix_mode="remove_points",  # "extend_bounding_box" to debug
    auto_rotate=False, rotate_angles=(0, 0, 0),
    angle_identifier=["Eul0", "Eul1", "Eul2"],
    orientation_descriptor="euler-bunge", orientation_active_convention=True,
    elastic_strain_identifier=[f"eKen{i}{j}" for i in (1,2,3) for j in (1,2,3)],
    strain_unit="microstrain", unit="rad",              # unit of the Euler angles
)
builder.build_voronoi(
    generate_mesh=False,         # default. True -> NEPER/GMSH tet .msh (slow), a FALLBACK only
    option="centroid",           # "voronoi" | "centroid" | "centroidsize"
    CVT_iter=1000, morphoalgo="subplex",
    mesh_quality_min=0.7, relative_el_size=2.0,
)
```

## Key parameters
- `bounding_box`: [xlo,xhi,ylo,yhi,zlo,zhi] (µm). `auto_fix_bbox` + `bbox_fix_mode` handle
  out-of-box points (`remove_points` for production).
- `unit` (`"rad"`/`"deg"`) must match the actual Euler units; `strain_unit` for the ee columns.
- `option`/`CVT_iter`/`morphoalgo`: tessellation morphology + CVT optimization.
- `generate_mesh`: keep `False`. The default CPFE mesh is SCULPT hex from
  `reconstruction_reformatted.csv` via `VoxelMeshBuilder` (see `/meshing`); the NEPER/GMSH tet
  `.msh` (`generate_mesh=True`, `relative_el_size` ~ elements per grain) is a fallback only.

## Outputs (in `output_dir`)
- `reconstruction.tess` / `reconstruction.ori` (9-col rotmat) / `reconstruction.msh` (GMSH tet, fallback; only if `generate_mesh=True`)
- `orientations.dat`: per-grain Euler, **always degrees after FF build** (convert with
  `orientation_helper.euler_to_mrp` before CPFE)
- `reconstruction_cpfe_ee.csv`: per-grain initial elastic strain (12 cols: x,y,z + 9)
- `reconstruction_reformatted.csv`: per-voxel grain IDs + Euler (FF→voxel input)

## Gotchas
- FF `orientations.dat` is degrees regardless of input; downstream `VoxelMeshBuilder` needs
  `angle_type="degrees"`, and CPFE needs `euler_to_mrp(...)`.
- To only build a graph instead of a mesh, use `builder.build_graph(...)` (see `/grain-tracking`).

## See also
`examples/demonstrate_farfield.py` (build_voronoi branch); CLAUDE.md §3.
