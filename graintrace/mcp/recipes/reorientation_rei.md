---
segment: reorientation_rei
tool: reorientation_rei
applies_to: Use reorientation (misorientation-from-initial) as a rare-event (REI) criterion
defaults:
  step: null
  ref_step: null
  symmetry: "432"
  k: 5
  gamma: 1.0
  graph_mode: knn
  extra_cols: []
  n_jobs: 12
  seed: 42
---

# Reorientation as an REI criterion: recommended parameters

`reorientation_rei` writes a per-element misorientation-from-initial field
(`reorientation_deg`) between a reference and a loaded step, then runs the standard
REI pipeline with the `abs_scalar_diff` metric. It flags the regions that have
**reoriented the most** under load. neml2 (orientation math) + networkit (clustering);
no external binaries.

## Model
Per element, `reorientation_deg` = symmetry-aware misorientation between its
`ori_rodrigues` (neml2 v3 MRP) at `ref_step` and at `step`
(`FragmentationAnalyzer.reorientation_from_reference`). That scalar feeds
`IdentifyRareClusters` exactly like nye/stress does.

## Minimum you must supply
- `block_csv`, `field_dir`: a CPFE result with per-element field CSVs (`mesh_out/`,
  `mesh_csv="sync"|"per_step"`) carrying `id,x,y,z,ori_rodrigues_x/y/z`.

## Standalone vs combined (the caller's choice)
- **Standalone** (default): rare = highest reorientation. `abs_scalar_diff` on
  `reorientation_deg`.
- **Combined**: set `extra_cols` (e.g. `["nye_tensor_11"]`) to carry other fields
  through; the criterion then uses a multi-feature spec — same multi-metric REI
  pattern, no new code.

## Parameters
- `step` / `ref_step`: loaded and reference field-step indices (default last / first).
  Indices are the CSV indices (may be non-contiguous with sync times).
- `graph_mode`: `knn` for scattered `mesh_out` element centroids (default), `grid`
  for a regular grid.
- `gamma`: Leiden resolution (default 1.0).
- `k`: number of rare clusters to keep.

## Gotchas
- `ori_rodrigues` is neml2 MRP (misnamed), mapped via `mrp_to_matrix` — not Gibbs.
- `abs_scalar_diff` requires exactly one feature column.
