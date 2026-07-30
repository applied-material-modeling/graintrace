# graintrace live demo

A self-contained, end-to-end demonstration: from synthetic "experiment output"
to the grains/locations a materials scientist should look at.

```
generate_experiment.py  ->  experiment/         (the handed-over data)
run_demo.py             ->  stitch -> calibrate -> FF reconstruct + SCULPT hex mesh
                            -> CPFE (cuda:0) -> rare-event ID -> plots + report
```

External tool paths (puma-opt, CUBIT/SCULPT) are read from a **tools.json**
(`~/.config/graintrace/tools.json`, `./graintrace_tools.json`, or
`$GRAINTRACE_TOOLS_JSON`); copy `graintrace/mcp/tools.example.json` and fill in
your paths. Meshing uses **CUBIT/SCULPT hex** (the recommended path); GMSH tets
are an FF-only last resort.

## What's in `experiment/` (the "experiment output")

Only what a real FF-HEDM experiment would give you, nothing pre-solved:

- `hedm_scan/scan_0..3.csv`: 4 far-field z-scans at 25% overlap, each with
  `X,Y,Z,GrainRadius,Eul0-2` **and residual elastic strain `eKen11..33`** (microstrain).
- `strain-stress.csv`: a hypothetical macroscopic stress-strain curve (the
  material-calibration target).
- `sample.json`: everything a CSV can't carry: **sample dimensions**, **loading
  conditions**, **scan geometry**, and **units**. This is the file the MCP tools
  ask for when handed a raw CSV.

The ground-truth crystal (`_truth/voronoi.*`) is kept out of `experiment/` on
purpose; it's not something an experiment measures.

## Run it (script)

```bash
conda activate graintrace_env
python examples/demo/generate_experiment.py   # once: builds experiment/ (~200 grains, NEPER)
python examples/demo/run_demo.py              # the full pipeline, CPFE runs to completion
```

Outputs land in `examples/demo/out/` (gitignored):
- `stitched.csv`, `material_calibration/calibrated_material.json`,
  `FF/mesh/mesh.e` (SCULPT hex) + `FF/mesh/orientations.csv` (MRP),
  `simulation/simulation_out/grid_out/…`,
  `rei/rei_rare_cluster_stats.csv` + `rei/rei_rare_clusters.vtk`,
- `plots/`: grains/reconstruction (IPF), calibration stress-strain, CPFE nye
  field, and REI hotspots (PNG, rendered off-screen via pyvista; no ParaView).

The run ends by printing **which grains / locations to look at** (rare-cluster
centroids, extent, severity, nearest grain).

### Re-running individual stages

Set `DEMO_REUSE` to reuse existing outputs and only re-run later stages:

```bash
DEMO_REUSE="stitch,calibrate,reconstruct" python examples/demo/run_demo.py   # re-run only CPFE->REI
```

### Notes / tuning (verified run)

Verified end-to-end at ~200 grains on 2× RTX A5000 (32 cores). Timing is dominated
by **MOOSE CPU-side** work (per-grain postprocessors + Nye-tensor recovery; the
NEML2 GPU eval is ~0.5s), so:

- **SCULPT hex (FIRST-order) is faster and more robust than GMSH tets** here:
  ~6s per evaluation vs ~14s, and it converged with **0 timestep cutbacks** (the
  tet path diverged and thrashed). This is the reason the demo uses SCULPT.
- **`ncore`** is the biggest lever: the demo uses `ncore=16` (of 32); per-eval
  cost scales ~linearly with ranks ("use nodes").
- **`dt=0.05`** caps the adaptive stepper (`dtmax=10*dt=0.5`); a grown `dt≈1.0`
  is too large for the rate-dependent slip integration and diverges.
- **`total_strain=0.001`** (in `sample.json`) keeps each load increment gentle.
- **Grain count** and **`TESR_SIZE`** (voxel→hex resolution) trade speed for
  fidelity; shrink `BOUNDING_BOX` in `generate_experiment.py` (~60-80 grains) or
  lower `TESR_SIZE` for a faster demo.
- Calibration is bounded to physical ranges (`PARAM_RANGES`) so `nu` stays ~0.3
  (essential for FE convergence).
- If CPFE can't complete, REI falls back to `mwe_data/grid_out/`.
- Exodus/large fields are best explored in ParaView; the demo renders static PNGs
  of the essentials only.
