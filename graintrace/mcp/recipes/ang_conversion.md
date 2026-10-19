---
segment: ang_conversion
tool: convert_ang
applies_to: Converting EDAX/TSL .ang EBSD or NF orientation maps into a schema graintrace reads
defaults:
  angle_convention: bunge
  angle_type: radians
  output_angle_type: degrees
  file_token: layer
  confidence_min: 0.0
---

# EDAX/TSL `.ang` conversion: recommended parameters

Nothing in graintrace reads `.ang` directly. `convert_ang` (library:
`graintrace.AngConverter`) is the step *before* `nf_reconstruct` or `voxel_mesh`
when the source data is `.ang`.

## Pick the target by the downstream tool

| `target` | Writes | Feed it to |
|---|---|---|
| `ebsd_csv` | one merged `x,y,z,Eul0,Eul1,Eul2` CSV | `voxel_mesh` (`VoxelMeshBuilder`) |
| `mic` | one `.mic` per layer, with the three `%` metadata lines | `nf_reconstruct` (`NearFieldMeshBuilder`) |

`ebsd_csv` is the usual choice for EBSD serial sections. Use `mic` only when the
rest of the workflow is already the NF path.

## The convention and the unit are never guessed

`convert_ang` reads them from the `.ang` `#` header when it pins them
(`ANGLE_UNITS`, `EULER_CONVENTION`), and otherwise **requires** them as
arguments. It raises instead of defaulting. That is deliberate: a degrees file
read as radians yields a microstructure that looks plausible, segments without
complaint and is wrong.

For an unmodified EDAX/TSL OIM export the right answer is
`angle_convention="bunge"`, `angle_type="radians"` — but confirm with the person
who produced the file rather than assuming, especially for data that has been
through MTEX or DREAM.3D.

## Key knobs

| Knob | Recommended | Why |
|---|---|---|
| `angle_type` | `radians` for raw OIM | TSL writes radians; pass it explicitly when the header is silent |
| `angle_convention` | `bunge` for raw OIM | TSL writes `(phi1, PHI, phi2)` |
| `output_angle_type` | `degrees` | matches the usual `angle_type="degrees"` on the downstream builder; **the two must agree** |
| `zstep` | the serial-section spacing (µm) | `z` of layer *i* is `i * zstep`; a single-file input is `z = 0` |
| `confidence_min` | `0.0` | TSL marks unindexed points with a negative CI; `0.0` drops exactly those. Raise it (0.1–0.2) only for noisy scans |
| `file_token` | `layer` | the filename token the layer index follows, e.g. `layer03.ang`. Layers are ordered **numerically**, so `layer10` sorts after `layer9` |

## Gotchas

- **`output_angle_type` and the builder's `angle_type` must match.** The
  converter always writes Bunge angles, but in whichever unit you asked for.
  Mismatching them is the same silent corruption the refusal above protects
  against.
- **Layer step in two places.** For the `mic` target, `zstep` is written into
  `%GlobalPosition`, but `NearFieldMeshBuilder` recomputes `z` from its own `dz`.
  Pass the same number to both.
- **A filtered layer can go empty.** `confidence_min` raises rather than writing
  an empty layer, so a too-high floor fails loudly.
- Column layout is driven by the header's `COLUMN_HEADERS` when present,
  otherwise by column count: the first eight are
  `phi1, PHI, phi2, x, y, IQ, CI, phase` and the next two are `SEM, fit`.
