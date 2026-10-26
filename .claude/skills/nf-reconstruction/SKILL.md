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
A folder of per-layer `.mic` files (tab-delimited, `%`-headers; convert `.ang`→`.mic` outside
graintrace first). `exp_file_token` is the filename prefix used to find them; layers sort by the
number after it and must be contiguous. MIDAS per-layer subfolders (`Layer1/…layer1.mic`): pass
`recursive=True` to the constructor (an empty top folder falls back to subfolders anyway). For
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
        dz=5.0, nx=200, ny=300,     # or voxel_size_um=(2.0, 2.0); both -> ValueError
        min_confidence=0.4,          # RECOMMENDED: void low-confidence (air/boundary) voxels
        max_point_distance=None,     # um; replaces the hull test (~1.5 * TriEdgeSize)
        segmentation={               # RECOMMENDED graph/Leiden path (degrees!)
            "method": "graph", "misori_tol_deg": 5.0,
            "gamma": None,           # percolation sweep; ~0.1 fixed for few-grain data
            "grain_threshold_final": 30, "remerge_miso_deg": 3.0,
            "n_jobs": 1, "n_threads": 1,
            "downsample": (2, 2, 1), # large NF only
        },
        # legacy flood (default when segmentation=None or flat):
        # segmentation={"misorientation_tol": 5.0/180*np.pi, "connectivity": 6,
        #   "batch_norm": 200_000, "grain_threshold": 1000, "stop_count": 500,
        #   "grain_threshold_final": 10000},
    )
    mesh_path = builder_nf.mesh(sculpt_config=sculpt_config, sculpt_options=sculpt_options,
                               merged_grid=merged_grid)
    # per-element MRP orientations -> builder_nf.mapped_orientations_path + ".csv"

if __name__ == "__main__":
    main()
```

## Key parameters
- `dz` (layer thickness µm) must match the data; `nx,ny` in-plane grid resolution (default
  300 x 900) or `voxel_size_um=(hx, hy)` — not both.
- `min_confidence` / `max_point_distance`: void air, failed and low-confidence boundary voxels
  at gridding time (the percolation source). No `Confidence` column -> warning, no masking.
- `segmentation`: `{"method": "graph", ...}` (graph/Leiden on solid voxels via
  `FragmentationAnalyzer.segment`, `misori_tol_deg` in degrees; keys in
  `NearFieldMeshBuilder.DEFAULT_GRAPH_SEGMENTATION`) or the **legacy flat flood dict**
  (no `method`/`params` nesting, `misorientation_tol` in radians, `connectivity` 6 or 26).
  Either way, isolated islands left at `-1` are voided before meshing.
- `sculpt_config` (required: `psculpt`,`epu`,`nprocs`; +`launcher`/`environment` for MPI) and
  `sculpt_options` tuple (e.g. `("--adapt","-S","2","-CS","4")`); **no `--void_mat`**:
  `mesh_sculpt` owns it (SCULPT is 1-based, so a grid with void needs `--void_mat 1`, which it
  sets; the old `0` meshed the void as block 1). See CLAUDE.md §9.

## Outputs (`save_dir`)
- `merged_segmented_fixed_grid.npy` (segmented voxel grid / restart checkpoint)
- `mesh.e` (Exodus mesh for CPFE)
- `fixed_grid_confidence.npy` (per-voxel confidence sidecar, 0 where void)
- `orientations.csv` (one **neml2 MRP** row per block; row k = block id k+1; CPFE `ori_file`)
- `mesh_block_map.csv` (`block_id` 1..N, `sculpt_block_id`, `grain_id`, `n_elem`,
  `source_grain_id`): block ids are renumbered contiguous after SCULPT drops small grains

## Gotchas
- Restart: if segmentation already ran, load `merged_segmented_fixed_grid.npy` directly and
  pass it as `merged_grid` to `.mesh(...)` instead of re-running `.reconstruct(...)`.
- Derive the NF bounding box from the saved grid coords (see CLAUDE.md §6) for the CPFE BCs.
- The mesh spans the voxel-EDGE box (half a voxel beyond the grid centres); a single-layer
  grid gets z thickness `dz` (or `mesh(layer_spacing=...)`).

## See also
`examples/demonstrate_cpfe_nfff.py`; CLAUDE.md §4, §6, §9.
