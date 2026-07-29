---
segment: meshing
tool: voxel_mesh / nf_reconstruct (mesh) / ff_reconstruct (generate_mesh=true)
applies_to: SCULPT hex meshing of a voxel/tessellation grain grid
defaults:
  sculpt_options_adapt4: -A 4 -df 1 -S 2 -CS 4
  sculpt_options_df1: -df 1 -mvs 2 -S 2 -CS 5
  nprocs: <= physical cores
---

# Meshing (SCULPT) — recommended parameters

**SCULPT hex is the recommended mesh path for all of FF/NF/EBSD.** GMSH tets
(`ff_reconstruct(build_params.generate_mesh=true)`) are an **FF-only last resort**
— hex elements behave better for crystal plasticity, and the rest of graintrace
is built around SCULPT. The recommended FF route is:
`ff_reconstruct(generate_mesh=false)` → `voxel_mesh` (SCULPT).

The `sculpt_config` (CUBIT paths) comes from your **tools.json**
(`deploy/tools.example.json`); `voxel_mesh` / `nf_reconstruct` load it
automatically, so you usually don't pass `sculpt_config` by hand. Check
`dependency_status` → `cubit` first.

Based on 12 studied cases. In the MCP these `sculpt_options` feed `voxel_mesh`,
`nf_reconstruct` (when a `sculpt_config` is supplied/configured). The low-level
API is `VoxelMeshBuilder.mesh()` / `NearFieldMeshBuilder.mesh()`.

## Usage

```python
from graintrace.construct_voxel_mesh import VoxelMeshBuilder

builder = VoxelMeshBuilder(
    file_path="voronoi.csv", save_dir="out/mesh",
    euler_cols=["Eul0", "Eul1", "Eul2"],
    angle_convention="bunge", angle_type="degrees", symmetry="432",
)
builder.mesh(
    sculpt_config={
        "psculpt": "/path/to/cubit/bin/psculpt",
        "epu":     "/path/to/cubit/bin/epu",
        "nprocs":  4,                       # <= physical cores
    },
    sculpt_options=SCULPT_OPTS,             # one of the two configs below
    merged_grid=merged_grid_path,
)
```

## Only two configs are worth using — both keep `-df 1` (defeaturing) safe

### `adapt4`

```python
SCULPT_OPTS = ("-A", "4", "-df", "1", "-S", "2", "-CS", "4")
```
- **Preservation ≥ 98%, Scaled Jacobian positive.**
- SCULPT adapt-type 4 refines without triggering aggressive small-grain absorption.

### `df1`

```python
SCULPT_OPTS = ("-df", "1", "-mvs", "2", "-S", "2", "-CS", "5")
```
- **Preservation ≥ 95%, Scaled Jacobian positive.**
- `-mvs 2` (minimum voxel size for defeaturing) stops `-df 1` from swallowing
  relatively larger grains.
- Generally faster than `adapt4`.

## Known limitation (SCULPT 2024.8)

- **`-df 1` on its own eats grains.** Without `-A 4` (adapt4) or `-mvs 2` (df1),
  `-df 1` silently absorbs ~34% of equiaxed and ~57% of elongated grains at
  100³ voxels.

## Verification (mandatory before running CPFE)

1. **Grain preservation** — `N_mesh / N_tess` ≥ 98% for `adapt4`, ≥ 95% for
   `df1`. Lower ⇒ defeaturing has silently absorbed grains.
2. **Min Scaled Jacobian > 0** — not just the mean. A single negative-SJ element
   crashes the MOOSE simulation.
3. **Mesh-vs-tess grain-size distribution** — same histogram check as after
   microstructure generation.

**Sample size:** use the minimum number of grains that faithfully represents the
target distribution — fewer grains ⇒ fewer elements ⇒ tractable CPFE wall-clock.
Add grains only when the distribution histogram or the CPFE quantity of interest
doesn't converge.
