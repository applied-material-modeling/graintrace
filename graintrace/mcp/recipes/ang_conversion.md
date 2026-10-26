---
segment: ang_conversion
tool: convert_ang
applies_to: Converting EDAX/TSL .ang EBSD or NF orientation maps into a schema graintrace reads
defaults:
  angle_type: radians
  output_angle_type: degrees
  file_token: layer
  confidence_min: null
---

# EDAX/TSL `.ang` conversion

No builder reads `.ang` directly; `convert_ang` (`graintrace.AngConverter`) runs first.

| `target` | Writes | Feed it to |
|---|---|---|
| `ebsd_csv` | one merged `x,y,z,Eul0,Eul1,Eul2` CSV | `voxel_mesh` |
| `mic` | one `.mic` per layer (`Confidence` = CI, `PhaseNr` = 1) | `nf_reconstruct` |

- TSL writes Bunge radians, the default `angle_type`. A file whose angles exceed
  4π (TSL's unindexed marker) is rejected as radians; pass `angle_type="degrees"`.
- **`output_angle_type` must equal the downstream builder's `angle_type`.**
- Layers are ordered **numerically** by the integer after `file_token`, and
  `z = (index - lowest) * zstep`; a missing layer leaves a gap (and `nf_reconstruct`
  then raises `Missing layers`). For `mic`, pass the same `zstep` as `dz`.
- `confidence_min=0.0` drops TSL's negative-CI (unindexed) points; `None` keeps all.
- `GRID: HexGrid` scans are refused for `ebsd_csv` (not a voxel grid); resample to
  SqrGrid first. `mic` accepts them (NF grids scattered points).
