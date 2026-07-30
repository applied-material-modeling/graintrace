---
name: voxel-segmentation-mesh
description: >
  Segment a voxel/grid orientation field (EBSD or gridded NF) into grains via graph
  (Leiden) or flood-fill, then build a conformal hex mesh with CUBIT/SCULPT
  (VoxelMeshBuilder). Use when the user has a gridded Euler-angle CSV (x,y,z,Eul0/1/2)
  or FF reconstruction_reformatted.csv and wants a segmented mesh.
---

# Voxel / grid segmentation + meshing

Uses `VoxelMeshBuilder`. Env: `conda activate graintrace_env`. External tool:
**CUBIT/SCULPT** via `sculpt_config`. Graph segmentation uses networkit (Leiden).
Wrap driver in `if __name__ == "__main__": main()`.

## Inputs
A merged CSV with `x,y,z,Eul0,Eul1,Eul2` (EBSD), or an FF `reconstruction_reformatted.csv`
(pass `cell_id_col` + `angle_type="degrees"`). See CLAUDE.md §5.

## Recipe
```python
from graintrace.construct_voxel_mesh import VoxelMeshBuilder

builder = VoxelMeshBuilder(
    file_path="out/ebsd/EBSD_merged.csv", save_dir="out/ebsd/mesh",
    euler_cols=["Eul0", "Eul1", "Eul2"], angle_convention="bunge",
    angle_type="radians", symmetry="432",
)
merged_grid = builder.reconstruct(
    apply_smoothing=True,
    segmentation={
        "method": "graph",                       # "graph" or "flood"
        "params": {"misorientation_tol": 5.0,    # deg if angle_type="degrees" else rad
                   "connectivity": 26, "grain_threshold_final": 100},
        "graph_params": {
            "segmenter": "leiden", "graph_mode": "grid", "manhattan_radius": 2,
            "grid_tol": 1e-6, "n_jobs": 10, "weight_chunk_size": 1_000_000,
            "reduce_edges_topweights_k": 8, "nodes_chunk": 500_000, "seed": 42,
            "networkit_kwargs": {"gamma": 0.001},   # lower gamma = fewer clusters
            "weight_cfg": {"mode": "rbf", "sigma": None,
                           "sigma_auto": {"sample_size": 20_000, "random_state": 42,
                                          "quantile": 0.5}, "power": 2.0},
            "plot": True,
        },
    },
)
mesh_path = builder.mesh(sculpt_config=sculpt_config, sculpt_options=sculpt_options,
                        merged_grid=merged_grid)
```

## Key parameters
- `segmentation.method`: `"graph"` (Leiden; better for complex textures) or `"flood"`
  (simpler/faster; adds `batch_norm`/`grain_threshold`/`stop_count`). See CLAUDE.md §9.
- `misorientation_tol` units follow `angle_type`; `connectivity` 6 or 26.
- Graph tuning: `networkit_kwargs["gamma"]` (lower → fewer/larger grains), `weight_cfg` RBF.
- `sculpt_config`/`sculpt_options`: CUBIT hex meshing (CLAUDE.md §9).

## Outputs (`save_dir`)
- segmented voxel grid `.npy`, `mesh.e`, per-element MRP `orientations.csv`, optional VTK/plots.

## Gotchas
- FF→voxel path: input `reconstruction_reformatted.csv` is Euler in **degrees** →
  `angle_type="degrees"`.
- Large grids: tune `n_jobs`, `weight_chunk_size`, `nodes_chunk` for memory.

## See also
`examples/demonstrate_grid_segmentation_mesh.py`; CLAUDE.md §5, §9.
