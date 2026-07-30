---
name: cpfe-nf-ff
description: >
  Run a combined near-field + far-field CPFE simulation — NF provides the high-res mesh
  geometry + orientations, FF provides the initial elastic-strain field. Use when the
  user wants to drive CPFE from an NF mesh with an FF residual-strain initial condition
  (optionally end-to-end from synthetic HEDM).
---

# CPFE (NF geometry + FF initial strain)

Mirrors `examples/demonstrate_cpfe_nfff.py`. Env: `conda activate graintrace_env`.
External tools: **NEPER** (FF/synthetic), **CUBIT/SCULPT** (`sculpt_config`, NF mesh),
**MOOSE `puma-opt` + `neml2-compile`**. Wrap driver in `if __name__ == "__main__":`.

## Pipeline
1. (optional) `SyntheticHEDMGenerator(...).run(ff_iterations=10)` → writes `<out>/FF/ff.csv`
   and `<out>/NF/...`.
2. **NF mesh** — `/nf-reconstruction`: `NearFieldMeshBuilder(...).reconstruct(...)` + `.mesh(...)`
   → `nf_reconstruction/mesh.e` + `orientations.csv` (per-element MRP).
3. **FF residual strain** — `/ff-reconstruction`: `VoronoiMeshBuilder(...).build_voronoi(
   generate_mesh=False, ...)` → `ff_reconstruction/reconstruction_cpfe_ee.csv`. Spatially
   shift its x,y,z to the NF frame if the two aren't co-registered.
4. **CPFE** with NF mesh + NF orientations + FF ee (`use_ff_initial_field=False`, since the ee
   comes from a different mesh):
```python
from graintrace.run_cpfe_simulation import CPFESimulation
sim = CPFESimulation(
    mesh_file=out + "/nf_reconstruction/mesh.e",
    save_simulation_folder=out + "/simulation",
    eeres_file=out + "/ff_reconstruction/reconstruction_cpfe_ee_shifted.csv",
    ori_file=out + "/nf_reconstruction/orientations.csv",
    dim=3, element_order="FIRST",           # NF meshes are typically FIRST order
    moose_run_file="external/puma/puma-opt",   # EDIT: your built puma-opt
    use_ff_initial_field=False,             # ee from a DIFFERENT mesh
)
sim.set_parameters("material", ...)         # see /material-calibration for values
sim.set_parameters("simulation_parameters", device="cuda:0", device_batch=1000,
                   dt=0.5, total_time=2.0, initialize_time=1.0, sync_times="2.0")
sim.set_parameters("boundary", bounding_box=nf_bbox, bc={...})
sim.set_parameters("grid_properties", number_of_elements=[10,10,10], bounding_box=grid_bb)
sim.run(ncore=8)
```

## Key parameters
- `use_ff_initial_field=False` when the ee file and mesh are different meshes (FF ee on NF mesh);
  `True` only when mesh + ee are co-registered FF.
- FF→NF shift: add `(dx,dy,dz)` to the first 3 columns of `reconstruction_cpfe_ee.csv` to align
  frames before passing as `eeres_file`.
- Derive `nf_bbox` from `merged_segmented_fixed_grid.npy` coords (CLAUDE.md §6).
- `sculpt_config` for the NF hex mesh (CLAUDE.md §9).

## Gotchas
- Same v3 AOTI notes as `/cpfe-simulation` (baked params, no schedulers, device list over MPI).
- NF reconstruction needs the `__main__` guard (multiprocess).
- `moose_run_file` must be your built v3 puma-opt.

## See also
`examples/demonstrate_cpfe_nfff.py`; CLAUDE.md §6 (combined NF+FF), §4, §3.
