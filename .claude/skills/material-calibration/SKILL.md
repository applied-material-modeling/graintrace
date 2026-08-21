---
name: material-calibration
description: >
  Calibrate 6 crystal-plasticity parameters (elastic E/nu/G, slip strength, Voce
  hardening slope/saturation) to a macroscopic stress-strain curve + full-field elastic
  strains using a neml2 v3 + pyzag analytic-adjoint Taylor model with LBFGS
  (MaterialCalibration / TaylorModel). Use to fit material params before CPFE.
---

# Material calibration (pyzag-adjoint Taylor model)

Uses `MaterialCalibration` + `TaylorModel`. Env: `conda activate moose-src`.
External tool: **NEML2 v3** only (no MOOSE). CUDA optional.

## Inputs
A folder of per-stress-level CSVs with orientation matrix `O11..O33` + `eKen11..33` +
`GrainRadius`, plus a `strain-stress.csv` (2 cols: strain, stress). Self-contained:
`mwe_data/ff_calibration/` (9 load steps × 500 grains). For a physically registered fit, first
run `/experiment-rotation`.

## Recipe
```python
from pathlib import Path
import graintrace as _gt
from graintrace.material_calibration import MaterialCalibration
from graintrace.taylor import TaylorModel
_cpfe_base = str(Path(_gt.__file__).parent / "cpfe_base")

calib = MaterialCalibration(
    model_class=TaylorModel,
    model_args=dict(neml2_path=_cpfe_base + "/neml2_cpfe_calibration.i",
                    npoints=30, nchunk=2, device="cuda", compile=False),
    data_args=dict(data_dir="mwe_data/ff_calibration",
                   strain_stress_file="mwe_data/ff_calibration/strain-stress.csv",
                   npoints=30, full_field_strain_units="microstrain", straintype="eKen",
                   max_strain=0.006, n_grains=100, seed=42),
    save_dir="out/material_calibration", apply_elastic_correction=False,
    strain_window=(0.0, 0.0015),
)
calib.plot_texture(direction=[1, 1, 1])
calib.plot_stress_strain()
calib.calibrate(maxiter=15, lr=0.3, max_iter_per_step=6, line_search_fn="strong_wolfe",
                plateau_rtol=1e-3, plateau_window=2)   # guard stops early
calib.load("out/material_calibration/calibrated_material.json")
calib.plot_stress_strain(include_model=True)
calib.plot_strain_histogram(include_initial_strain=True)
```

## Key parameters
- model: `device="cuda"|"cpu"`, `npoints` (= pyzag time steps), `nchunk` (chunk size).
- data: `n_grains` (subsample per load step; None=all), `max_strain` (macro-curve cap),
  `straintype` (`"eKen"`/`"eFab"`).
- `calibrate`: `maxiter` is an **upper bound**; the plateau guard (`plateau_rtol`,
  `plateau_window`) stops early when relative loss improvement stalls.

## Outputs (`save_dir`)
`calibrated_material.json`, `autosave_material.json`, pole figures, stress-strain overlays,
elastic-strain histograms. Map `TaylorModel.opt_vars` → CPFE names (see CLAUDE.md §8) to feed
`/cpfe-simulation`'s `material` dict.

## Gotchas
- **cuda**: works because `taylor.py` moves the model with `nsys.to(device)`. A cuda/cpu
  mismatch (silent `loss=inf`) means the model stayed on CPU; do NOT reintroduce a
  `torch.set_default_device` hack.
- v3 mixed-control uses an **unweighted** grain mean; some params can hit reparametrization
  bounds on small/demo configs (few grains, narrow window); use more grains + wider window
  for production fits.

## See also
`examples/demonstrate_material_calibration.py`, `examples/demonstrate_farfield.py`; CLAUDE.md §8.
