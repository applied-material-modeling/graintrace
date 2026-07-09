---
name: experiment-rotation
description: >
  Rotate/register raw far-field experiment CSVs into the simulation frame before
  calibration — applies a sample tilt, builds a Voronoi per file, and appends the
  rotated orientation matrix O11..O33 (experiment_rotation_helper.update_experiments).
  Use when preparing per-stress-level FF CSVs for material calibration.
---

# Experiment rotation (register FF data to sim frame)

Uses `graintrace.experiment_rotation_helper`. Env: `conda activate graintrace_env`.
External tool: **NEPER** (a Voronoi build per file supplies the rotated `O`).

## Inputs
A folder of numeric-named per-stress-level FF CSVs (`0.csv, 50.csv, ...`) each with
`X,Y,Z,GrainRadius,Eul0/1/2,eKen11..33` (raw FF files may also already carry `O11..O33`).
Self-contained: `mwe_data/ff_calibration/`.

## Recipe
```python
import numpy as np
from graintrace.experiment_rotation_helper import update_experiments, collect_experiment_files

files, stress_levels = collect_experiment_files("mwe_data/ff_calibration")
update_experiments(
    input_files=files,
    output_root="out/rotated_experiments",
    bounding_box=[-477, 528, -487, 532, -1025, 625],
    auto_fix_bbox=True, bbox_fix_mode="remove_points",
    rotate_angles=(0, 0, -3.6/180*np.pi),      # sample tilt; unit must match `unit`
    unit="rad",
    angle_identifier=["Eul0", "Eul1", "Eul2"],
    orientation_descriptor="euler-bunge", orientation_active_convention=True,
    elastic_strain_identifier=[f"eKen{i}{j}" for i in (1,2,3) for j in (1,2,3)],
)
```

## Key parameters
- `rotate_angles` + `unit` — the sample→sim tilt (radians here). Applied to positions and
  orientations.
- `bounding_box` + `auto_fix_bbox`/`bbox_fix_mode` — drop out-of-box grains.
- `elastic_strain_identifier` — the 9 eKen columns (scaled by 1e6 if `strain_unit="microstrain"`).

## Outputs
`out/rotated_experiments/<name>.csv` per input, each with rotated `O11..O33` (from
`reconstruction.ori`) + coords + Euler + eKen. These feed `/material-calibration`
(`data_dir="out/rotated_experiments"`).

## Gotchas
- The helper **replaces** any pre-existing `O` columns with the freshly rotated ones (it drops
  the raw `O` before concatenating) — no duplicate `O11.1` columns.
- Each file triggers a full NEPER tessellation; ~500 grains is fast, thousands are slower.
- For a quick (non-registered) calibration you can skip this and point calibration straight at
  raw CSVs that already contain `O11..O33` (like `mwe_data/ff_calibration`).

## See also
`examples/demonstrate_farfield.py` (`update_experiments_data` branch); CLAUDE.md §8.
