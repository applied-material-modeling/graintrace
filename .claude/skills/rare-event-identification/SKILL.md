---
name: rare-event-identification
description: >
  Rare-event identification (REI): find spatially coherent rare regions in a CPFE
  field (e.g. high Nye-tensor / von-Mises) via graph spatial clustering + hierarchical
  merge + rare-cluster selection, exporting VTK (IdentifyRareClusters, GraphSpatialCluster,
  ClusterAnalysisIndicator). Use to locate hotspots in grid_out results.
---

# Rare-event identification (REI)

Uses `rare_cluster_indicator.IdentifyRareClusters` (+ `GraphSpatialCluster`,
`ClusterAnalysisIndicator`, `SimilarityMetricLibrary`). Env: `conda activate graintrace_env`.
Pure Python (networkit Leiden + scipy hierarchical + PyVista/VTK). No MOOSE/NEPER/CUBIT.
The clustering pipeline no longer needs an `if __name__ == "__main__"` guard.

## Inputs
A point-cloud CSV with an `id` column, `x,y,z` coords, and field columns (e.g. the last
`grid_out/out_element_centroid_*.csv` from a CPFE run). Demo data: `mwe_data/synthetic_vms.csv`
(the REI example scripts regenerate it with `generate_synthetic=True`).

## Recipe (full pipeline)
```python
import pandas as pd
from graintrace.rare_cluster_indicator import IdentifyRareClusters
from graintrace.similarity_metric_library import SimilarityMetricLibrary
from graintrace.user_data_class import SimilarityMetric, WeightConfig, RareCriteria
from graintrace import rare_criteria_selection_library as rcs

lib = SimilarityMetricLibrary()
spec = lib.nye_tensor_norm(cols=[f"nye_tensor_{i}{j}" for i in (1,2,3) for j in (1,2,3)])
spec_reduced = SimilarityMetric(name=spec.name + "_mean",
    feature_cols=[f"{c}_mean" for c in spec.feature_cols], func=spec.func)
weight_cfg = WeightConfig(mode="rbf", power=2.0, sigma=None,
    sigma_auto={"sample_size": 500_000, "random_state": 42, "quantile": 0.5})
rare = RareCriteria(selector=lambda df: rcs.select_highest_scalar(
    df, k=5, required_cols="nye_tensor_norm_mean_mean", min_size=1))

irc = IdentifyRareClusters(input_csv_path="out/last_grid.csv", id_col="id",
                           coord_cols=("x","y","z"))
gsc, indicator = irc.make_stage_objects(graph_cluster_out="out/rei_reduced.csv")
bundle = irc.run_clustering(gsc=gsc, indicator=indicator,
    reduced_csv_path="out/rei_reduced.csv",
    gsc_run_kwargs=dict(spec=spec, graph_mode="grid", manhattan_radius=4, grid_tol=1e-6,
        n_jobs=12, weight_chunk_size=500_000, segmenter="leiden", seed=42,
        weight_cfg=weight_cfg, reduce_edges_topweights_k=20,
        networkit_kwargs={"gamma": 10.0}, resume_from_checkpoint=False),
    indicator_run_kwargs=dict(method_type="scipy_hierarchical", spec=spec_reduced,
        threshold=0.0005, method="average", criterion="distance"))
out = irc.run_get_rare_cluster(bundle=bundle, criteria=rare,
    output_vtk_path="out/rare_clusters.vtk", export_control="auto",
    background_block_id=1, first_rare_block_id=2)
```

For simpler demos: `ClusterAnalysisIndicator` (single-stage hierarchical, `rei_example_2D`) or
`GraphSpatialCluster` + `ClusterAnalysisIndicator` (two-stage, `rei_example_3D`).

## Key parameters
- `spec`: the similarity feature (`SimilarityMetricLibrary`: `nye_tensor_norm`,
  `von_mises_stress`, …). Reduce to per-cluster means for stage 2 (`spec_reduced`).
- Graph: `graph_mode` (`grid`/`knn`/`auto`), `manhattan_radius`, `networkit_kwargs["gamma"]`
  (higher = more clusters), `reduce_edges_topweights_k`, `weight_cfg` (RBF).
- Hierarchical: `threshold`, `method`, `criterion`.
- `RareCriteria`: how rare clusters are picked (`select_highest_scalar`, size quantiles).

## Gotchas
- Three checkpoint levels (bundle pickle → reduced CSV+labels → graph edges); see CLAUDE.md §10
  REI restart pattern.
- The `demonstrate_rei_*` scripts regenerate `synthetic_vms.csv` (`generate_synthetic=True`);
  point `input_csv_path` at your own CPFE grid CSV for real analysis.

## See also
`examples/demonstrate_rei_pipeline.py`, `_example_3D.py`, `_example_2D.py`; CLAUDE.md §7 (REI), §10.
