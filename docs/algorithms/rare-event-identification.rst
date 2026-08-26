Rare-event identification
=========================

Overview
--------
Rare-event identification (REI) locates spatially coherent regions of a CPFE field that are
extreme in some quantity (e.g. high Nye-tensor norm, high von Mises stress). It is driven by
:class:`~graintrace.IdentifyRareClusters`, which runs a two-stage clustering: an over-segmenting
graph pass (:class:`~graintrace.GraphSpatialCluster`) followed by a hierarchical merge in feature
space (:class:`~graintrace.ClusterAnalysisIndicator`), then selects the rare merged clusters and
exports them to VTK.

Method
------
Points are compared through a **similarity metric** on their field features
(:class:`~graintrace.SimilarityMetricLibrary`). Built-in metrics include the Frobenius-norm
difference of a full :math:`3\times 3` tensor (e.g. the Nye tensor),

.. math::

    d(\mathbf{A}_i, \mathbf{A}_j) = \lVert \mathbf{A}_i - \mathbf{A}_j \rVert_F,

the relative von Mises stress difference,

.. math::

    d = \frac{\lvert \sigma^{vM}_i - \sigma^{vM}_j\rvert}
             {\lvert \sigma^{vM}_i\rvert + \lvert \sigma^{vM}_j\rvert + \varepsilon},
    \qquad
    \sigma^{vM} = \sqrt{\tfrac{1}{2}\!\left[(\sigma_{xx}-\sigma_{yy})^2 + (\sigma_{yy}-\sigma_{zz})^2
                 + (\sigma_{zz}-\sigma_{xx})^2\right] + 3(\sigma_{xy}^2+\sigma_{yz}^2+\sigma_{xz}^2)},

and symmetry-aware misorientation.

**Stage 1 (over-segment).** A spatial graph over the field points (grid connectivity at a
Manhattan radius, or mutual kNN) carries the metric distance on each edge, mapped to a weight by
the RBF kernel :math:`w_{ij} = \exp[-(d_{ij}/\sigma)^p]`, and is partitioned by Leiden into many
small, spatially compact clusters. Each cluster is reduced to its feature means.

**Stage 2 (merge).** The reduced per-cluster means are agglomerated by SciPy hierarchical linkage
(default average linkage) using the same metric, and cut at a distance ``threshold`` to form
merged super-clusters. The cophenetic correlation of the linkage is reported as a quality check.

**Stage 3 (select).** A :class:`~graintrace.RareCriteria` selector picks which merged clusters are
rare, e.g. the top-:math:`k` by mean Nye-norm via
:func:`graintrace.rare_criteria_selection_library.select_highest_scalar`, or the built-in bottom
size-quantile default. Selected clusters receive distinct block ids (background id first, then one
per rare cluster) and are written to a STRUCTURED_GRID or POLYDATA VTK, optionally with a rare
point-cloud CSV for downstream comparison.

Algorithm
---------
1. Load the field CSV (a ``mesh_out/`` true-mesh or ``grid_out/`` regular-grid file) with id and
   coordinate columns.
2. Stage 1: build the spatial graph, compute per-edge metric distances, RBF-weight them,
   optionally prune to top-:math:`k` edges per node, and Leiden-partition into fine clusters;
   write the reduced per-cluster CSV and per-point stage-1 labels.
3. Stage 2: run hierarchical linkage on the reduced cluster feature means and cut at ``threshold``
   (criterion ``distance``) to obtain merged super-labels; optionally save a dendrogram.
4. Map each stage-1 label to its merged super-label to get a per-point ``final_label``.
5. Stage 3: apply the rarity criteria to the merged clusters, assign block ids (smallest rare
   clusters first), and mark rare points.
6. Export the classified field to VTK (grid vs. points auto-detected), plus optional
   per-rare-cluster statistics and an ``(x, y, z, rare_cluster_id)`` point cloud.

Parameters that matter
----------------------
- ``spec`` (the ``SimilarityMetric``): which field quantity defines rarity (Nye norm, von Mises,
  misorientation); stage 2 uses the ``*_mean`` reduced version.
- Stage-1 graph knobs: ``graph_mode``, ``manhattan_radius``, ``weight_cfg``,
  ``reduce_edges_topweights_k``, and ``networkit_kwargs={"gamma": ...}`` (lower :math:`\gamma`
  = fewer, larger clusters).
- Stage-2 merge knobs: ``threshold`` (linkage cut distance), ``method`` (linkage, e.g.
  ``average``), ``criterion`` (``distance``).
- ``RareCriteria``: ``selector`` (custom top-:math:`k`) or the ``size_quantile`` / ``min_size`` /
  ``max_rare`` size-based default.
- ``export_control`` and block-id options control the VTK output form.

See :doc:`/configuration` for the metric, weight, and criteria dataclasses.

References
----------
- Leiden community detection (stage 1): Traag, Waltman, van Eck (2019),
  https://doi.org/10.1038/s41598-019-41695-z.
- Agglomerative hierarchical clustering with average linkage and cophenetic distance (stage 2),
  as implemented in ``scipy.cluster.hierarchy``.
- The RBF edge weighting is a standard Gaussian affinity of the feature-space distance.

See also
--------
- Tutorial: :doc:`/tutorials/rare-event-identification`
- API: :class:`~graintrace.IdentifyRareClusters`, :class:`~graintrace.GraphSpatialCluster`,
  :class:`~graintrace.ClusterAnalysisIndicator`
