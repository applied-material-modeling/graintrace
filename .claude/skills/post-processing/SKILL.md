---
name: post-processing
description: >
  Load CPFE simulation (and experiment) field outputs and plot field distributions,
  macroscopic stress-strain, per-block time series, pole figures, and IPF-colored
  meshes (SimulationResults, plot_postprocessing, IPFProcessor). Use to analyze
  cpfe_out/ results after a run.
---

# Post-CPFE analysis & plotting

Uses `simulation_postprocessing.SimulationResults` + `plot_postprocessing` (+ optional
`ipf_postprocess.IPFProcessor`, `experiment_postprocessing.ExperimentResults`). Env:
`conda activate moose-src`. Pure Python (no MOOSE/NEPER).

## Inputs
A CPFE block CSV + a directory of per-time grid field CSVs. Self-contained:
`mwe_data/out.csv` + `mwe_data/grid_out/` (`out_element_centroid_0000.csv`…). Experiment
half: `mwe_data/synthetic_load_exp/` (`expsyn_<t>time.csv`).

## Recipe
```python
from graintrace.simulation_postprocessing import SimulationResults, FieldFileNaming
from graintrace import plot_postprocessing as postprocess

res = SimulationResults(
    block_csv="mwe_data/out.csv",
    field_dir="mwe_data/grid_out",
    field_naming=FieldFileNaming(prefix="out_element_centroid", index_width=4,
                                 sep="_", suffix=".csv"),
)
postprocess.plot_block_properties_distribution(res, time=1.0, tensor_prefix="ee",
                                               order=2, output_folder="out/post")
postprocess.plot_macroscopic_stress_strain(res, stress_tensor_prefix="cauchy_stress",
    strain_tensor_prefix="strain", volume_prefix="volume", output_folder="out/post")
postprocess.plot_block_properties_over_time(res, tensor_prefix="cauchy_stress", order=2,
                                            output_folder="out/post")
```

Optional, pole figure (needs neml2 v3 `neml2.texture`) and IPF coloring:
```python
postprocess.plot_pole_figure(res, tensor_prefix="ori_rodrigues", time=1.0,
    direction=[0,0,1], crystal_symmetry="432", device="cpu", output_folder="out/post")

from graintrace.ipf_postprocess import IPFProcessor
ipf = IPFProcessor(crystal_symmetry="432", sample_symmetry="432", save_dir="out/mesh")
ipf.add_block_rgb_to_exodus(mesh_file="out/mesh/mesh.e",
    orientations_csv="out/mesh/orientations.csv", output_file="mesh_rgb.e",
    direction=[0,0,1], angle_convention="mrp")
```

## Key parameters
- `FieldFileNaming(prefix, index_width, sep, suffix)` must match the grid CSV filenames.
- `tensor_prefix`: field to plot (`ee`, `cauchy_stress`, `nye_tensor`, `ori_rodrigues`,
  `strain`); `order` = tensor rank (2 for rank-2).
- Pole figure: `crystal_symmetry`, `direction`, `construct_odf`.

## Gotchas
- Pre-load steps (t ≤ `initialize_time`) have ~zero stress/ee; that's expected.
- Pole figures require v3 neml2 (`neml2.texture`); reinstall neml2 v3 if the import fails.
- Experiment half uses `experiment_postprocessing.ExperimentResults` with its own
  `FieldFileNaming` (e.g. prefix `expsyn_`, suffix `time.csv` for `mwe_data/synthetic_load_exp`).

## See also
`examples/demonstrate_postprocess.py`; CLAUDE.md §7.
