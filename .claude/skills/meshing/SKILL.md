---
name: meshing
description: >
  Vetted parameter recommendations for MESHING a voxel/tessellation grain grid into
  a hex mesh via the graintrace mesh builders: SCULPT sculpt_options (the two safe
  configs from a 12-case study) and the no-SCULPT mesher="voxel" direct-to-Exodus
  dump (one cube hex per voxel, zero inverted elements). Use when choosing SCULPT
  flags, picking between SCULPT and the voxel dump, or for the mandatory post-mesh
  grain-preservation / scaled-Jacobian checks. Generation lives in the separate
  `/microstructure-generation` skill. Same content ships as MCP recipe 'meshing'.
---

# Meshing (graintrace mesh builders)

Two hex-meshing paths, both via `VoxelMeshBuilder.mesh()`:
`mesher="sculpt"` (CUBIT/SCULPT conformal mesh) and `mesher="voxel"` (direct
voxel-to-Exodus, no external tools). Recommendations below are from 12 studied
microstructure cases. For generating the grid, see `/microstructure-generation`.

Env: `conda activate graintrace_env`. External tool for SCULPT path:
**CUBIT/SCULPT** (the voxel path needs none).
Example: `examples/demonstrate_synthetic_cpfe.py` (generates, meshes both ways,
then runs CPFE).

**API:** `VoxelMeshBuilder.mesh()` (also `NearFieldMeshBuilder.mesh()`,
`VoronoiMeshBuilder.build_voronoi(generate_mesh=True)`).

---

## 1. SCULPT conformal mesh (`mesher="sculpt"`, default)

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
        "nprocs":  4,                              # <= physical cores
    },
    sculpt_options=SCULPT_OPTS,                    # one of the two configs below
    merged_grid=merged_grid_path,
)
```

**Only two configs are worth using, both keep `-df 1` (defeaturing) safe:**

### `adapt4`

```python
SCULPT_OPTS = ("-A", "4", "-df", "1", "-S", "2", "-CS", "4")
```

- **Preservation ≥ 98%, Scaled Jacobian positive.**
- Uses SCULPT adapt-type 4, which refines without triggering aggressive
  small-grain absorption.

### `df1`

```python
SCULPT_OPTS = ("-df", "1", "-mvs", "2", "-S", "2", "-CS", "5")
```

- **Preservation ≥ 95%, Scaled Jacobian positive.**
- Uses `-mvs 2` (minimum voxel size for defeaturing) to prevent `-df 1` from
  swallowing relatively larger grains.
- Generally faster than `adapt4`.

### Known limitations (SCULPT 2024.8)

- **`-df 1` on its own eats grains.** Without `-A 4` (adapt4) or `-mvs 2` (df1),
  `-df 1` silently absorbs ~34% of equiaxed and ~57% of elongated grains at 100³
  voxels.
- **Do not override the smoothing method** (`-S`) on irregular voxel meshes:
  forcing e.g. `-S 2` after guaranteed-quality smoothing can diverge (min SJ → −1)
  and segfault SCULPT. Prefer the two configs above, or use `mesher="voxel"`.

---

## 2. Direct voxel dump (`mesher="voxel"`, no SCULPT)

```python
builder.mesh(mesher="voxel", merged_grid=merged_grid_path)
# -> mesh.e (HEX8, one block per grain, ids 1..N) + <mapped_orientations>.csv (MRP)
```

- **One axis-aligned cube hex per voxel.** Scaled Jacobian = 1 everywhere, so
  **zero inverted/sliver elements and 100% grain preservation** by construction.
  Needs no `sculpt_config` / CUBIT. Verified MOOSE-readable (`--mesh-only`).
- Trade-offs: **stair-stepped** grain boundaries (not smoothed) and a fixed one
  hex per filled voxel (element count = voxel count) — control resolution via the
  tesr/grid size.
- Use it when SCULPT smoothing produces bad elements (junction slivers) or when a
  guaranteed-clean mesh matters more than boundary smoothness.

---

## Verification (mandatory before running CPFE)

For the SCULPT path, all three; the voxel path guarantees (1) and (2) by
construction, so only (3) matters there:

1. **Grain preservation**: `N_mesh / N_tess` ≥ 98% for `adapt4`, ≥ 95% for
   `df1`. If lower, defeaturing has silently absorbed grains.
2. **Min Scaled Jacobian > 0**: not just the mean. A single negative SJ element
   will crash the MOOSE simulation.
3. **Mesh-vs-tess grain-size distribution** check, same as after generation.

**Sample size guidance:** always use the **minimum number of grains that
faithfully represents the target distribution**. Fewer grains → fewer elements →
tractable CPFE wall-clock. Add grains only when the distribution histogram or the
CPFE quantity of interest doesn't converge.
