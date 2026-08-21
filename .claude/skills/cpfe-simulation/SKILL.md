---
name: cpfe-simulation
description: >
  Run a MOOSE/PUMA CPFE simulation on a far-field Voronoi reconstruction with a NEML2
  v3 AOTI-compiled crystal-plasticity model (CPFESimulation). Use when the user has a
  mesh + orientations (+ optional FF residual strain) and wants to bake material params,
  neml2-compile, and launch puma-opt.
---

# CPFE simulation (FF, NEML2 v3 / AOTI)

Uses `CPFESimulation`. Env: `conda activate moose-src`. External tools:
**MOOSE `puma-opt` + `neml2-compile`** (AOTI). CUDA recommended.

## Inputs
A GMSH/Exodus mesh + per-grain orientations. Self-contained: `mwe_data/cpfe_ff/`
(`reconstruction.msh` + `orientations.dat`, 10 grains). Orientations must be **neml2 MRP**;
convert FF Euler `orientations.dat` (degrees) with `orientation_helper.euler_to_mrp`.

## Recipe (mirrors demonstrate_cpfe.py)
```python
import os, numpy as np, torch, meshio
from graintrace.run_cpfe_simulation import CPFESimulation
from graintrace import orientation_helper as oh

ff = "mwe_data/cpfe_ff"; out = "cpfe_out"; os.makedirs(out, exist_ok=True)
euler = np.loadtxt(ff + "/orientations.dat")                       # Euler-bunge, degrees
mrp = oh.euler_to_mrp(torch.tensor(euler, dtype=torch.float64), "bunge", "degrees")
np.savetxt(out + "/orientations_MRP.dat", mrp.numpy(), fmt="%.12g")

m = meshio.read(ff + "/reconstruction.msh"); P = m.points
bbox = [float(P[:,0].min()), float(P[:,0].max()), float(P[:,1].min()),
        float(P[:,1].max()), float(P[:,2].min()), float(P[:,2].max())]
order = "SECOND" if any("10" in c.type for c in m.cells) else "FIRST"

sim = CPFESimulation(
    mesh_file=ff + "/reconstruction.msh", save_simulation_folder=out,
    moose_run_file="external/puma/puma-opt",   # EDIT: your built puma-opt
    element_order=order, eeres_file=None, ori_file=out + "/orientations_MRP.dat",
    dim=3, use_ff_initial_field=True,           # eeres_file=None -> 12-col zero ee
)
sim.set_parameters("material", slip_constant_strength=130.0,
    voce_hardening_initial_slope=1556.09, voce_hardening_saturation=100.0,
    power_slip_n=20, power_slip_g0=1e-4, elastic_E=209016.0, elastic_nu=0.307,
    elastic_G=60355.0, burger_scale=2.22)
sim.set_parameters("simulation_parameters", device="cuda:0", device_batch=20000,
    dt=0.5, total_time=2.0, initialize_time=1.0, sync_times="2.0")

disp = 0.002 * (bbox[5] - bbox[4])
sim.set_parameters("boundary", bounding_box=bbox, bc={
    "x": {"negative": "stress_free", "positive": "stress_free"},
    "y": {"negative": "stress_free", "positive": "stress_free"},
    "z": {"negative": 0, "positive": disp}})
grid_bb = list(bbox)
for i in range(0,6,2): grid_bb[i] += 1e-4
for i in range(1,6,2): grid_bb[i] -= 1e-4
sim.set_parameters("grid_properties", number_of_elements=[10,10,10], bounding_box=grid_bb)

sim.run(ncore=4)   # ncore == mpiexec -n; also spreads a device list over ranks
```

## Key parameters
- `device`: `"cpu"`, `"cuda:0"`, or space-sep list `"cuda:0 cuda:1"` (multi-GPU over MPI).
- `device_batch`: per-device NEML2 chunk (quad pts/call); a finite value caps GPU memory
  (0 = whole batch → OOM risk on large meshes).
- `initialize_time`: load ramps from `initialize_time`→`total_time`; `sync_times` = grid-output times.
- `use_ff_initial_field=True` + real `eeres_file` = FF residual strain (12-col x,y,z+9);
  `eeres_file=None` writes a 12-col zero ee.
- AOTI: material params are **baked** into the model .i and `neml2-compile`d on `run()`;
  `recompile=True` (default) rebuilds when params change.

## Outputs (`save_simulation_folder/simulation_out`)
`out.csv` (block time series), `sim_output.e`/`sim_output_grid.e` (Exodus), `grid_out/*.csv`
(per-grid fields: cauchy_stress, ee, nye_tensor, ori_rodrigues). Feed `/post-processing`.

## Gotchas
- v3 has NO runtime `[NEML2] cli_args`/`[Schedulers]`; do not pass `scheduler_name`.
- For neml2-dominated CPFE, use **fewer** MPI ranks (~#GPUs) for bigger per-rank batches.
- Stiff first steps may make MOOSE cut `dt` and recover; that is normal, not a failure.
- Env/`LD_LIBRARY_PATH`/`neml2_load_files` auto-derive from `moose_run_file`'s repo layout.

## See also
`examples/demonstrate_cpfe.py`; CLAUDE.md §3 Step 4.
