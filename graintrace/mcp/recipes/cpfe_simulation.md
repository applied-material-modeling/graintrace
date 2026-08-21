---
segment: cpfe_simulation
tool: run_cpfe
applies_to: MOOSE/PUMA crystal-plasticity FE run on a reconstructed mesh
defaults:
  element_order: SECOND
  dim: 3
  dt: 0.2
  total_time: 5.0
  initialize_time: 1.0
  device: "cuda:0"
  device_batch: 20000
  ncore: 8
  use_ff_initial_field: true
---

# CPFE Simulation: recommended parameters

`run_cpfe` wraps `CPFESimulation` (NEML2 v3 AOTI-compiled crystal plasticity via
MOOSE/PUMA). **Needs `puma-opt` (MOOSE/PUMA) and a working NEML2 v3 build.** This
is the heaviest step: runs on GPU, minutes to hours. It always runs as a
background job.

## Minimum you must supply
- `mesh_file`: a SCULPT/voxel hex `.e` (the default for FF and NF) or a GMSH tet `.msh`
  (FF fallback). CPFE runs on either; hex is the recommended path (see the `meshing` recipe).
- `save_simulation_folder`: output dir.
- `ori_file`: per-grain/element orientations, **NEML2 v3 MRP** (convert FF
  `orientations.dat` with `orientation_helper.euler_to_mrp` first).
- `moose_run_file`: path to your built `puma-opt`.

**Must-ask (loading + sample dimensions are NOT in the mesh):** provide
`bounding_box` + `total_strain` (+ `loaded_axis`), or a `sample_json`, and
`run_cpfe` auto-builds the `boundary` bc and `grid_properties`. Without them it
returns `needs_input` (otherwise CPFE would silently use a unit-cube domain).

## Key parameter groups (via `set_parameters`)

**material**: `slip_constant_strength`, `voce_hardening_initial_slope`,
`voce_hardening_saturation`, `power_slip_n`, `power_slip_g0`, `elastic_E`,
`elastic_nu`, `elastic_G`, `burger_scale`. Use values from
`material-calibration` when you have them.

**simulation_parameters**: `dt`, `total_time`, `initialize_time` (load ramps
from `initialize_time`→`total_time`), `device` (`"cpu"`, `"cuda:0"`, or a
space-separated list for multi-GPU = MPI ranks), `device_batch`, `sync_times`.

> **GPU policy: if a GPU is available, always use it.** Set `device="cuda:0"`
> (or `"cuda:0 cuda:1"` for multi-GPU). CPFE is neml2-dominated and far slower on
> CPU. `run_cpfe` auto-fills `device` with the first GPU when one is present and
> you didn't specify it; only use `"cpu"` when no GPU exists. Check
> `dependency_status` → `gpu`.

**boundary**: `bounding_box` + `bc` dict. `displace_amount = total_strain *
(zhi - zlo)`. Each face is `"stress_free"`, `0` (fixed), or a float displacement.

**grid_properties**: `number_of_elements=[nx,ny,nz]`, `bounding_box=grid_bb`
where `grid_bb` is the BC box inset by `0.0001` on every face.

## Setup notes
- `use_ff_initial_field=true` only when the mesh and `ee` file are the same,
  co-registered FF reconstruction. For NF-mesh + FF-strain, set it `false` and
  spatially shift the `ee` CSV into the NF frame first.
- `eeres_file=None` writes a zeroed initial strain (no residual).
- Multi-GPU: fewer MPI ranks (~#GPUs) → bigger per-rank NEML2 batches → better
  GPU utilization for the neml2-dominated solve.

## Sync times from strains
`sync_times = np.asarray(sync_strain)/total_strain*(total_time-initialize_time)
+ initialize_time`; pass as a space-separated string.
