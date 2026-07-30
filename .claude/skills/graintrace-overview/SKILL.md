---
name: graintrace-overview
description: >
  Index/router for the graintrace HEDM→CPFE→REI workflow. Use when the user asks
  "what can graintrace do", "how do I run <segment>", "which script/skill for X",
  or when a task spans several stages and you need to pick the right per-segment
  skill. Maps each workflow segment to its skill, example script, mwe_data, and
  external-tool needs.
---

# graintrace workflow — overview & router

`graintrace` links experimental grain data (FF/NF HEDM, EBSD) to MOOSE/PUMA CPFE
simulations with NEML2 v3 material models, and post-processes for rare-event ID (REI).

**Always run in the project env:** `conda activate graintrace_env` (neml2 v3 + pyzag +
torch/CUDA). Examples live in `examples/demonstrate_*.py` (flat top-level `## INPUT` style).
Full API reference: `.claude/CLAUDE.md`.

## Segments → skills → examples

| Stage | `/skill` | Example | Data |
|---|---|---|---|
| Simulate + stitch HEDM scans | `/hedm-stitching` | demonstrate_hedm_study.py | self-gen (NEPER) |
| FF Voronoi reconstruction | `/ff-reconstruction` | demonstrate_farfield.py | mwe_data/ff_calibration |
| NF mesh reconstruction | `/nf-reconstruction` | demonstrate_cpfe_nfff.py | synthetic (NEPER/CUBIT) |
| EBSD/NF voxel seg + mesh | `/voxel-segmentation-mesh` | demonstrate_grid_segmentation_mesh.py | synthetic gen |
| Rotate exp CSVs to sim frame | `/experiment-rotation` | demonstrate_farfield.py | mwe_data/ff_calibration |
| Material calibration (Taylor) | `/material-calibration` | demonstrate_material_calibration.py | mwe_data/ff_calibration |
| CPFE simulation (FF) | `/cpfe-simulation` | demonstrate_cpfe.py | mwe_data/cpfe_ff |
| CPFE (NF geom + FF strain) | `/cpfe-nf-ff` | demonstrate_cpfe_nfff.py | synthetic |
| Post-processing / plots / IPF | `/post-processing` | demonstrate_postprocess.py | mwe_data/out.csv+grid_out |
| Rare-event identification | `/rare-event-identification` | demonstrate_rei_pipeline.py | mwe_data/synthetic_vms.csv |
| Grain tracking across loads | `/grain-tracking` | demonstrate_graintracking.py | mwe_data/synthetic_load_exp |

## Typical pipelines
- **Experimental FF → CPFE:** stitch (`/hedm-stitching` techniques) → `/ff-reconstruction`
  → `/experiment-rotation` + `/material-calibration` → `/cpfe-simulation` → `/post-processing`
  → `/rare-event-identification`.
- **NF+FF:** `/nf-reconstruction` (geometry) + FF residual strain → `/cpfe-nf-ff`.

## External-tool matrix
- **NEPER** (Voronoi/tessellation): ff-reconstruction, nf/ff/hedm synthetic generation.
- **CUBIT/SCULPT** (`sculpt_config`: psculpt/epu/mpiexec): nf-reconstruction,
  voxel-segmentation-mesh, cpfe-nf-ff.
- **MOOSE `puma-opt` + `neml2-compile`** (AOTI): cpfe-simulation, cpfe-nf-ff.
- **NEML2 v3 only** (no MOOSE): material-calibration.
- **None** (pure Python): post-processing, rare-event-identification, grain-tracking.

## Orientation convention (important)
All orientations communicate as **neml2 v3 MRP** (`tan(θ/4)·axis`). Convert via
`graintrace.orientation_helper` (`euler_to_mrp`, `load_orientations_mrp`) — it delegates to
neml2; do not hand-roll. graintrace "mrp" historically meant Gibbs/Rodrigues (`tan θ/2`) —
not the same thing; use the helpers.
