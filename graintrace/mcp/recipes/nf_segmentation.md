---
segment: nf_segmentation
tool: nf_reconstruct
applies_to: Recommended graph (Leiden) segmentation settings for NF/EBSD voxel data
defaults:
  method: graph
  misorientation_tol_deg: 5.0
  manhattan_radius: 2
  rbf_sigma_deg: 2.5
  reduce_edges_topweights_k: 12
  downsample: [2, 2, 1]
  gamma_sweep: [0.5, 1.0, 2.0, 4.0, 8.0]
  percolation_max_frac: 0.03
  grain_threshold_final: 30
  remerge_miso_deg: 3.0
---

# NF/EBSD graph (Leiden) segmentation: recommended parameters

**Graph/Leiden is the default and better segmentation** for NF/EBSD voxel data;
**flood over-merges into percolating grains**. Use `method="graph"` in the
`nf_reconstruct` / `voxel_mesh` segmentation config. This is the vetted Fe9Cr recipe
(validated on Fe-9Cr NF reconstruction), also exposed in the
library as `FragmentationAnalyzer.segment` + its staticmethods
`gamma_sweep_leiden`, `pick_gamma_by_percolation`, `absorb_fragments`, `adjacency_remerge`.

## Recommended knobs
| Knob | Value | Why |
|---|---|---|
| misorientation cutoff | **5°** hard (`max_edge_distance`) | removes boundary edges → no percolation |
| `manhattan_radius` | **2** (18-neighbor) | denser intra-grain links |
| RBF sigma | **fixed ≈ ½ cutoff (~2.5°)**, `sigma_auto=None` | `sigma_auto` collapses to ~0.1° and shatters grains |
| `weight_cfg` | `mode="rbf", power=2.0` | bounded 0..1 weights (not `inverse`'s huge dynamic range) |
| `reduce_edges_topweights_k` | **12** | sparser graph, faster Leiden |
| `downsample` | `(2,2,1)` (large NF) | NF grid is finer than needed for ~40 µm grains |
| Leiden `gamma` | **sweep `[0.5,1,2,4,8]`, pick by PERCOLATION** | smallest gamma whose biggest grain < 3% of solid |
| absorb `grain_threshold_final` | **30 vox** | merge sub-threshold fragments into largest-contact neighbor |
| adjacency re-merge | **~3°**, adjacency-ONLY, size-capped | coalesce Leiden-split fragments without re-percolating |

## Scale caveat
Gamma-by-percolation (3%-of-solid cap) is for **large, many-grain** NF/EBSD. For
**small / few-grain** data or **intragranular sub-grain** segmentation (a legit grain
is a large fraction of solid), the percolation pick misfires → falls back to the
highest gamma and shatters. Pass a **fixed low `gamma` (~0.1)** there instead.

## Why not flood
Flood fill chains low-angle neighbors into one percolating grain (a single blob
spanning the whole solid). The hard misorientation cutoff + Leiden + percolation-picked
gamma + adjacency re-merge de-percolate while keeping distinct grains separate.
