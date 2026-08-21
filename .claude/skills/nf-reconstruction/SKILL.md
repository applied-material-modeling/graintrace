---
name: nf-reconstruction
description: >
  Reconstruct a high-resolution 3D mesh from near-field (NF) HEDM .mic layers via
  voxel segmentation + CUBIT/SCULPT hex meshing (NearFieldMeshBuilder). Use when the
  user wants an Exodus mesh.e + per-element MRP orientations from NF data to drive CPFE
  (typically paired with FF residual strain).
---

# NF (near-field) mesh reconstruction

Uses `NearFieldMeshBuilder`. Env: `conda activate moose-src`. External tools:
**CUBIT/SCULPT** via `sculpt_config` (psculpt/epu/mpiexec). **Requires an
`if __name__ == "__main__"` guard**: NF `pointcloud_to_fixed_grid` uses `multiprocess.Pool`.

## Inputs
A folder of per-layer `.mic` files (tab-delimited, `%`-headers; convert `.ang`→`.mic` first;
see `run_experiment_afrl.py`). `exp_file_token` is the filename prefix used to find them. For
the synthetic path, `SyntheticHEDMGenerator` writes the NF folder (see `/cpfe-nf-ff`).

## Recipe
```python
from graintrace.construct_nf_mesh import NearFieldMeshBuilder
import numpy as np

def main():
    builder_nf = NearFieldMeshBuilder(
        input_folder="experiment_data/NF", save_dir="out/NF",
        exp_file_token="layer", angle_convention="bunge", angle_type="radians",
        symmetry="432", prefix="reconstructed", write_intermediate=True, write_vtk=True,
    )
    merged_grid = builder_nf.reconstruct(
        dz=5.0, nx=200, ny=300,
        segmentation={   # legacy flat dict (radians) for NearFieldMeshBuilder
            "misorientation_tol": 5.0/180*np.pi, "connectivity": 6,
            "batch_norm": 200_000, "grain_threshold": 1000, "stop_count": 500,
            "grain_threshold_final": 10000,
        },
    )
    mesh_path = builder_nf.mesh(sculpt_config=sculpt_config, sculpt_options=sculpt_options,
                               merged_grid=merged_grid)
    # per-element MRP orientations -> builder_nf.mapped_orientations_path + ".csv"

if __name__ == "__main__":
    main()
```

## Key parameters
- `dz` (layer thickness µm) must match the data; `nx,ny` in-plane grid resolution.
- `segmentation`: **legacy flat dict** (no `method`/`params` nesting), `misorientation_tol`
  in radians. `connectivity` 6 or 26.
- `sculpt_config` (required: `psculpt`,`epu`,`nprocs`; +`launcher`/`environment` for MPI) and
  `sculpt_options` tuple (e.g. `("--adapt","-S","2","-CS","4","--void_mat","0")`); see CLAUDE.md §9.

## Outputs (`save_dir`)
- `merged_segmented_fixed_grid.npy` (segmented voxel grid / restart checkpoint)
- `mesh.e` (Exodus mesh for CPFE)
- `orientations.csv` (per-element **neml2 MRP**; use as CPFE `ori_file`)

## Gotchas
- Restart: if segmentation already ran, load `merged_segmented_fixed_grid.npy` directly and
  pass it as `merged_grid` to `.mesh(...)` instead of re-running `.reconstruct(...)`.
- Derive the NF bounding box from the saved grid coords (see CLAUDE.md §6) for the CPFE BCs.

## See also
`examples/demonstrate_cpfe_nfff.py`; CLAUDE.md §4, §6, §9.
