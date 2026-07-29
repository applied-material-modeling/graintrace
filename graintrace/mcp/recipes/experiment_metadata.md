---
segment: experiment_metadata
tool: inspect_experiment / sample_json
applies_to: the non-inferrable metadata a raw HEDM CSV cannot carry
---

# Experiment metadata (`sample.json`) — what a CSV can't tell you

A raw FF-HEDM grain CSV carries per-grain `X,Y,Z`, `GrainRadius`, Euler angles,
and optionally residual elastic strain (`eKen*`/`eFab*`). It does **not** carry:

- **Sample dimensions** — the bounding box `[xlo,xhi,ylo,yhi,zlo,zhi]` (µm). CPFE
  otherwise silently defaults to a **unit cube**; FF reconstruction needs it too.
- **Loading conditions** — the applied `total_strain` (+ loaded axis), or explicit
  boundary conditions. Not in the CSV; there is no default that is physically right.
- **Scan geometry** — the z-range (`zlo`,`zhi`) and `overlap_fraction` used when
  stitching multiple z-scans.
- **Units & conventions** — orientation `degrees`/`radians`, Euler convention,
  strain unit (`microstrain` vs `strain`), crystal symmetry. graintrace does
  **NOT** auto-detect these (there is no "|angle|>2π ⇒ degrees" rule in the code).

So: when given a raw CSV, call `inspect_experiment(path)` (it returns a suggested
bounding box from the coordinate ranges, a unit *guess*, and a checklist), then
either pass a `sample.json` or confirm the values with the user. The tools return
`status: "needs_input"` until these are supplied.

## `sample.json` schema

```json
{
  "sample": {
    "name": "my_sample",
    "bounding_box_um": [-300, 300, -300, 300, -320, 320],
    "units": {
      "orientation": "degrees",
      "orientation_convention": "bunge",
      "strain": "microstrain",
      "symmetry": "432"
    }
  },
  "scan_geometry": { "n_scans": 4, "overlap_fraction": 0.25, "zlo": -320, "zhi": 320 },
  "loading": {
    "mode": "uniaxial_tension",
    "loaded_axis": "z",
    "total_strain": 0.002,
    "temperature_K": 298,
    "bc": {
      "x": {"negative": "stress_free", "positive": "stress_free"},
      "y": {"negative": "stress_free", "positive": "stress_free"},
      "z": {"negative": 0, "positive": "total_strain * (zhi - zlo)"}
    }
  },
  "elastic_strain_columns": { "prefix": "eKen", "unit": "microstrain" }
}
```

## How it maps to the tools

| sample.json field | consumed by |
|---|---|
| `sample.bounding_box_um` | `ff_reconstruct` (required), `run_cpfe` (boundary + grid) |
| `sample.units.orientation` | `ff_reconstruct` (unit deg/rad), `stitch_scans` |
| `scan_geometry.{zlo,zhi,overlap_fraction}` | `stitch_scans` |
| `loading.{total_strain,loaded_axis}` | `run_cpfe` (builds bc: displace = total_strain × axis extent) |
| `elastic_strain_columns.prefix` | FF `elastic_strain_identifier` (`eKen*`/`eFab*`) |

Pass it as the `sample_json` argument to `stitch_scans`, `ff_reconstruct`, and
`run_cpfe`. `demo/experiment/sample.json` is a complete working example.
