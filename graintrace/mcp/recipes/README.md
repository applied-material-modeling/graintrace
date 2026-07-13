# Recommendation recipes

Each `<name>.md` here is a **parameter preset + guidance** for one workflow
setup. The MCP server serves them to the connecting model via the
`get_recommended_parameters` tool and the `recipe://<name>` resources, so the
model proposes good parameters *before* it previews a run.

These are meant to be **edited and extended by domain experts** — they are plain
Markdown, so improving guidance never touches server code. Drop in a new
`<name>.md` and it is picked up automatically (also add it to the package-data
glob in `pyproject.toml` if you want it shipped in the wheel).

## Format

Optional front-matter block, then free-form Markdown:

```markdown
---
segment: ff_reconstruction
tool: ff_reconstruct
applies_to: one-line description of when to use this
defaults:
  key: value      # informational recommended values, shown to the model
  another: 1000
---

# Human-readable title

...prose: what the tool wraps, what you must supply, a table of key
parameters with recommended values and *why*, gotchas...
```

Notes:
- `defaults:` is parsed into a `defaults` dict returned by
  `get_recommended_parameters`. Values are strings (guidance for the model). The
  *operative* typed defaults live in each tool module; keep the two consistent.
- Everything below the front matter is returned verbatim as the recipe body and
  as the `recipe://<name>` resource.
- `name` (used to look it up) is the filename without `.md`.

## Current recipes

- `ff_reconstruction` — FF Voronoi reconstruction (`ff_reconstruct`)
- `stitching` — FF z-scan stitching (`stitch_scans`)
- `cpfe_simulation` — MOOSE/PUMA CPFE run (`run_cpfe`)
- `microstructure_generation` — NEPER morpho recipes incl. `aspratio` (feeds
  `generate_synthetic_hedm` / `CrystalGenerator`), 12-case study
- `meshing` — SCULPT `sculpt_options` (`voxel_mesh` / `nf_reconstruct` /
  `ff_reconstruct(generate_mesh=true)`), 12-case study

Good next additions: `nf_reconstruction`, `material_calibration`,
`rare_event_identification`.
