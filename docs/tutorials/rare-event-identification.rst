Rare-event identification (REI)
===============================

Find spatially coherent rare regions in a CPFE field (for example high
Nye-tensor or von-Mises stress) via graph spatial clustering, a hierarchical
merge, and a rare-cluster selection, then export the result to VTK. Use this to
locate hotspots in CPFE field results, whether on the true mesh (``mesh_out/``)
or a resampled grid (``grid_out/``). The pipeline is pure Python (networkit
Leiden, scipy hierarchical clustering, PyVista/VTK).

**Needs:** none beyond graintrace — no MOOSE, NEPER, or CUBIT.

Inputs
------

A point-cloud CSV with an ``id`` column, ``x, y, z`` coordinates, and field
columns. REI is not grid-locked: a regular grid uses the fast grid graph, and
arbitrary points fall back to the kNN graph (``graph_mode="knn"`` or ``"auto"``).
Two CPFE sources share the same schema:

- ``mesh_out/out_element_centroid_*.csv`` — crisp per-element fields on the true
  mesh (``mesh_csv="sync"`` default); full fidelity, one row per element (kNN
  path). Preferred.
- ``grid_out/out_element_centroid_*.csv`` — a regular grid, from
  ``grid_transfer="per_step"`` or from an offline
  :class:`~graintrace.GridResampler` (the resampled grid is smoothed; see
  :doc:`post-processing`).

The example scripts regenerate ``mwe_data/synthetic_vms.csv`` when
``generate_synthetic=True``.

Recipe
------

.. code-block:: python

   import pandas as pd
   from graintrace.rare_cluster_indicator import IdentifyRareClusters
   from graintrace.similarity_metric_library import SimilarityMetricLibrary
   from graintrace.user_data_class import SimilarityMetric, WeightConfig, RareCriteria
   from graintrace import rare_criteria_selection_library as rcs

   lib = SimilarityMetricLibrary()
   spec = lib.nye_tensor_norm(cols=[f"nye_tensor_{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)])
   spec_reduced = SimilarityMetric(
       name=spec.name + "_mean",
       feature_cols=[f"{c}_mean" for c in spec.feature_cols], func=spec.func)
   weight_cfg = WeightConfig(mode="rbf", power=2.0, sigma=None,
       sigma_auto={"sample_size": 500_000, "random_state": 42, "quantile": 0.5})
   rare = RareCriteria(selector=lambda df: rcs.select_highest_scalar(
       df, k=5, required_cols="nye_tensor_norm_mean_mean", min_size=1))

   irc = IdentifyRareClusters(input_csv_path="out/last_grid.csv", id_col="id",
                              coord_cols=("x", "y", "z"))
   gsc, indicator = irc.make_stage_objects(graph_cluster_out="out/rei_reduced.csv")
   bundle = irc.run_clustering(
       gsc=gsc, indicator=indicator, reduced_csv_path="out/rei_reduced.csv",
       gsc_run_kwargs=dict(spec=spec, graph_mode="grid", manhattan_radius=4,
           grid_tol=1e-6, n_jobs=12, weight_chunk_size=500_000, segmenter="leiden",
           seed=42, weight_cfg=weight_cfg, reduce_edges_topweights_k=20,
           networkit_kwargs={"gamma": 10.0}, resume_from_checkpoint=False),
       indicator_run_kwargs=dict(method_type="scipy_hierarchical", spec=spec_reduced,
           threshold=0.0005, method="average", criterion="distance"))
   out = irc.run_get_rare_cluster(bundle=bundle, criteria=rare,
       output_vtk_path="out/rare_clusters.vtk", export_control="auto",
       background_block_id=1, first_rare_block_id=2)

For simpler demos, :class:`~graintrace.ClusterAnalysisIndicator` runs a
single-stage hierarchical clustering, and
:class:`~graintrace.GraphSpatialCluster` combined with
:class:`~graintrace.ClusterAnalysisIndicator` runs the two-stage form.

Key parameters
--------------

- ``spec`` — the similarity feature from
  :class:`~graintrace.SimilarityMetricLibrary` (``nye_tensor_norm``,
  ``von_mises_stress``, and others). Reduce it to per-cluster means for stage two
  (``spec_reduced``).
- Graph — ``graph_mode`` (``grid``/``knn``/``auto``), ``manhattan_radius``,
  ``networkit_kwargs["gamma"]`` (higher gives more clusters),
  ``reduce_edges_topweights_k``, and ``weight_cfg`` (RBF; see
  :doc:`../configuration`).
- Hierarchical — ``threshold``, ``method``, ``criterion``.
- :class:`~graintrace.RareCriteria` — how rare clusters are chosen
  (``select_highest_scalar``, size quantiles).

Outputs
-------

- ``rare_clusters.vtk`` — the rare regions as labeled blocks.
- The reduced per-cluster CSV and, optionally, a rare-cluster statistics CSV and
  a rare-points CSV (``x, y, z, rare_cluster_id``) for :doc:`rei-comparison`.

Gotchas
-------

- Three checkpoint levels are available (bundle pickle, reduced CSV plus labels,
  graph edges); see the REI restart pattern in :doc:`../configuration`.
- The demo scripts regenerate ``synthetic_vms.csv``; point ``input_csv_path`` at
  your own CPFE field CSV for real analysis.
- The clustering pipeline does not use multiprocessing, so no
  ``if __name__ == "__main__":`` guard is required.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_rei_pipeline.py
   :language: python
   :caption: examples/demonstrate_rei_pipeline.py
