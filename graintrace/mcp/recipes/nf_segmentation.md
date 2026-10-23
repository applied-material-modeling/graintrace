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
  remerge_miso_deg: 3.0
---

# NF/EBSD graph (Leiden) segmentation: recommended parameters

**Graph/Leiden is the default and better segmentation** for NF/EBSD voxel data;
**flood over-merges into percolating grains**. Use `method="graph"` in the
`nf_reconstruct` / `voxel_mesh` segmentation config. Also exposed in the
library as `FragmentationAnalyzer.segment` + its staticmethods
`gamma_sweep_leiden`, `pick_gamma`, `absorb_fragments`, `adjacency_remerge`,
`cluster_orientation_spread`.

## Recommended knobs
| Knob | Value | Why |
|---|---|---|
| misorientation cutoff | **5°** hard (`max_edge_distance`) | removes boundary edges → no percolation |
| `manhattan_radius` | **2** (24-neighbor) | denser intra-grain links |
| RBF sigma | **fixed ≈ ½ cutoff (~2.5°)**, `sigma_auto=None` | `sigma_auto` collapses to ~0.1° and shatters grains |
| `weight_cfg` | `mode="rbf", power=2.0` | bounded 0..1 weights (not `inverse`'s huge dynamic range) |
| `reduce_edges_topweights_k` | **12** | sparser graph, faster Leiden |
| `downsample` | **off** (opt in only if the graph is too big) | coarsens boundaries, drops grains thinner than the coarsened step; `(2,2,1)` keeps a quarter of the voxels |
| Leiden `gamma` | **sweep `[0.5,1,2,4,8]`, `gamma_pick="lowest"`** | least over-resolved; no size cap |
| `gamma_pick="spread"` | when a leak is suspected | lowest gamma where no cluster's orientation spread exceeds the cutoff |
| `percolation_max_frac` | **`None`** (no cap) | grain size is not a percolation signal |
| absorb `grain_threshold_final` | **30 vox** | merge sub-threshold fragments into largest-contact neighbor |
| adjacency re-merge | **~3°**, adjacency-ONLY | coalesce Leiden-split fragments; bounded by misorientation, not size |

## Mask first, do not cap grain size
The 3%-of-solid percolation cap is **legacy**. It is a symptom fix for
segmentation leaking across boundaries through air or low-confidence voxels, and
it forces any legitimately large grain below 3% and cuts it into arbitrary
pieces. On few-grain data, a small field of view, or intragranular sub-grain
segmentation a real grain is easily 10–30% of the solid, and the chosen gamma
then reflects the cap rather than the data. **Mask low-confidence and void
voxels first**, then leave `percolation_max_frac=None`. Pass a number only to
reproduce legacy behaviour on unmasked data.

If a leak is still suspected, use the physical check rather than a size limit:
`gamma_pick="spread"` flags a cluster whose 95th-percentile misorientation to its
own mean exceeds `spread_tol_deg` (default: the segmentation cutoff) and picks
the lowest gamma that flags nothing. Per-gamma diagnostics (clusters, largest
fraction, number flagged) are reported so the choice is visible.

## Why not flood
Flood fill chains low-angle neighbors into one percolating grain (a single blob
spanning the whole solid). The hard misorientation cutoff + Leiden + adjacency
re-merge de-percolate while keeping distinct grains separate. Flood itself has no
size cap: `grain_threshold` is a *minimum* segment size and `stop_count` an
iteration budget.
