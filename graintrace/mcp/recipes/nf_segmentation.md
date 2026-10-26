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
  downsample: null
  gamma_sweep: [0.5, 1.0, 2.0, 4.0, 8.0]
  gamma_pick: lowest
  percolation_max_frac: null
  grain_threshold_final: 30
  absorb_max_miso_deg: 5.0
  remerge_miso_deg: 3.0
---

# NF/EBSD graph (Leiden) segmentation: recommended parameters

**Graph/Leiden is the default and better segmentation** for NF/EBSD voxel data;
**flood over-merges into percolating grains**. Use `method="graph"` in the
`nf_reconstruct` / `voxel_mesh` segmentation config. This is the vetted Fe9Cr recipe
(validated on Fe-9Cr NF reconstruction), also exposed in the
library as `FragmentationAnalyzer.segment` + its staticmethods
`gamma_sweep_leiden`, `pick_gamma`, `absorb_fragments`, `adjacency_remerge`, `cluster_leak_check`.

## Recommended knobs
| Knob | Value | Why |
|---|---|---|
| misorientation cutoff | **5°** hard (`max_edge_distance`) | removes boundary edges → no percolation |
| `manhattan_radius` | **2** (24-neighbor) | denser intra-grain links |
| RBF sigma | **fixed ≈ ½ cutoff (~2.5°)**, `sigma_auto=None` | `sigma_auto` collapses to ~0.1° and shatters grains |
| `weight_cfg` | `mode="rbf", power=2.0` | bounded 0..1 weights (not `inverse`'s huge dynamic range) |
| `reduce_edges_topweights_k` | **12** | sparser graph, faster Leiden |
| `downsample` | off (opt in if the graph is too big) | coarsens boundaries |
| Leiden `gamma` | **sweep `[0.5,1,2,4,8]`, `gamma_pick="lowest"`** | no size cap; `"spread"` = lowest gamma `cluster_leak_check` does not flag |
| absorb `grain_threshold_final` | **30 vox**, `absorb_max_miso_deg` = cutoff | joins only the closest neighbor within tol |
| adjacency re-merge | **~3°**, agglomerative, leak-checked | no chaining |

Mask low-confidence/void voxels instead of capping grain size;
`percolation_max_frac=0.03` restores the legacy pipeline.

## Why not flood
Flood fill chains low-angle neighbors into one percolating grain (a single blob
spanning the whole solid). The hard misorientation cutoff + Leiden + adjacency re-merge de-percolate while keeping distinct grains separate.

## In `nf_reconstruct`
`segmentation={"method": "graph", "misori_tol_deg": 5.0, "gamma": None, "grain_threshold_final": 30}`
runs this recipe on the solid voxels. Pass `min_confidence` (~0.3-0.5) to void the
low-confidence boundary network at gridding time.
