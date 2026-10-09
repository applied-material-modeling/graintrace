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

Why two stages
--------------
A single clustering cannot answer the question REI asks. Partitioning the field purely in
*feature* space finds the extreme values but scatters them across the domain, so the result is
a histogram tail rather than a region. Partitioning purely in *space* finds compact blobs that
need not be extreme in anything. REI therefore runs one pass of each, in that order: an
over-segmenting graph pass constrained by the spatial graph, which guarantees every stage-1
cluster is a connected neighbourhood, then a feature-space merge over the *reduced* clusters,
which is what makes the second pass affordable -- :math:`10^6` field points become
:math:`10^3`--:math:`10^4` cluster means before any :math:`O(n^2)` work happens.

The cost asymmetry is the design constraint. Stage 2 evaluates a pairwise distance matrix with
a Python-callable metric, so its work grows as the square of the number of stage-1 clusters;
:math:`\gamma` in stage 1 is therefore as much a budget knob as a physical one.

Method
------
Points are compared through a **similarity metric** on their field features
(:class:`~graintrace.SimilarityMetricLibrary`). Every metric is a *distance*: smaller means more
alike, and the weight kernels of :doc:`segmentation` invert it. Built-in metrics are

.. list-table::
   :header-rows: 1
   :widths: 26 46 28

   * - Metric
     - :math:`d(\mathbf{u}, \mathbf{v})`
     - Feature columns
   * - ``nye_tensor_norm``
     - :math:`\lVert \mathbf{A}_i - \mathbf{A}_j \rVert_F`
     - the nine ``nye_tensor_ij``
   * - ``von_mises_stress``
     - relative difference, :eq:`rei-vm`
     - six stress components
   * - ``misorientation``
     - disorientation angle, :eq:`seg-disorientation`
     - three orientation components
   * - ``abs_scalar_diff``
     - :math:`\lvert u_1 - v_1 \rvert`
     - exactly one column

The tensor metric is the Frobenius norm of the *difference*, not the difference of the norms,
so it separates two points whose Nye tensors have equal magnitude but different structure. The
von Mises metric is **relative**, not absolute:

.. math::
   :label: rei-vm

    d = \frac{\lvert \sigma^{vM}_i - \sigma^{vM}_j\rvert}
             {\lvert \sigma^{vM}_i\rvert + \lvert \sigma^{vM}_j\rvert + \varepsilon},
    \qquad
    \sigma^{vM} = \sqrt{\tfrac{1}{2}\!\left[(\sigma_{xx}-\sigma_{yy})^2 + (\sigma_{yy}-\sigma_{zz})^2
                 + (\sigma_{zz}-\sigma_{xx})^2\right] + 3(\sigma_{xy}^2+\sigma_{yz}^2+\sigma_{xz}^2)},

with :math:`\varepsilon = 10^{-8}` fixed in the implementation. Normalising by the sum of the
two magnitudes makes :math:`d` dimensionless and bounded on :math:`[0, 1]`, so a stage-2
``threshold`` carries over between load steps and between materials. The cost is that it is
*scale-free*: a 1 MPa step beside a 2 MPa neighbour is as distant as 500 beside 1000. If the
absolute level is what matters, drive the run from ``abs_scalar_diff`` on a precomputed von
Mises column instead.

**Stage 1 (over-segment).** A spatial graph over the field points (grid connectivity at a
Manhattan radius, or mutual kNN) carries the metric distance on each edge, mapped to a weight
by one of the kernels in :doc:`segmentation`, and is partitioned by Leiden into many small,
spatially compact clusters. Each cluster :math:`C` is reduced to

.. math::
   :label: rei-reduce

   n_C = |C|, \qquad
   \bar{x}_C = \frac{1}{n_C}\sum_{i \in C} x_i, \qquad
   \overline{f}_C = \frac{1}{n_C}\sum_{i \in C} f_i,

one ``<col>_mean`` per feature column. When the feature columns contain a complete
:math:`3 \times 3` set ``<prefix>_11`` ... ``<prefix>_33``, an extra ``<prefix>_norm_mean``
column carries :math:`\tfrac{1}{n_C}\sum_{i \in C} \lVert \mathbf{A}_i \rVert_F` -- the mean of
the per-point norms, **not** the norm of the mean tensor. The distinction matters: the mean
tensor of a cluster straddling a dislocation wall partially cancels, while the mean norm does
not, so rarity survives the reduction.

**Stage 2 (merge).** The reduced per-cluster means are agglomerated by SciPy hierarchical
linkage. With the default ``method="average"`` (UPGMA) the distance between two merged clusters
is the mean over all cross pairs,

.. math::
   :label: rei-upgma

   D(\mathcal{A}, \mathcal{B})
     = \frac{1}{|\mathcal{A}|\,|\mathcal{B}|}
       \sum_{a \in \mathcal{A}} \sum_{b \in \mathcal{B}} d(a, b),

which is the right default here because it is insensitive to a single outlying cluster in a way
``single`` (chaining) and ``complete`` (fragmenting) are not. The dendrogram is cut at
``criterion="distance"`` and height ``threshold``, so ``threshold`` is **in the units of the
metric** -- dimensionless for the relative von Mises metric, degrees or radians for
misorientation, and in the field's own units for a Frobenius norm. There is no normalisation
step, so a threshold transplanted between metrics is meaningless.

Quality is reported as the **cophenetic correlation**: the Pearson correlation between the
original pairwise distances and the dendrogram heights at which each pair first merges,

.. math::
   :label: rei-cophenetic

   c = \frac{\sum_{i<j} \bigl(d_{ij} - \bar{d}\bigr)\bigl(t_{ij} - \bar{t}\bigr)}
            {\sqrt{\sum_{i<j} \bigl(d_{ij} - \bar{d}\bigr)^2
                   \sum_{i<j} \bigl(t_{ij} - \bar{t}\bigr)^2}},

with :math:`t_{ij}` the cophenetic (ultrametric) distance. A value near 1 means the tree is a
faithful summary of the distance matrix and a height cut is meaningful; a low value means the
data does not have a hierarchical structure and the ``threshold`` is close to arbitrary. Treat
it as a go/no-go on the cut, not as a score to optimise.

.. note::

   Stage 2 evaluates the metric on all pairs **three times** per run -- once for the reported
   ``cophenetic_correlation``, once inside ``linkage``, and once more inside ``fclusterdata``,
   which recomputes an identical tree internally to produce the labels -- and then runs a 1-D
   MDS embedding on the dense :math:`n \times n` ultrametric. All four steps are
   :math:`O(n^2)` in the number of stage-1 clusters, with a Python call per pair. Keeping
   stage 1 to :math:`\mathcal{O}(10^3)` clusters is what keeps stage 2 interactive.

**Stage 3 (select).** A :class:`~graintrace.RareCriteria` picks which merged clusters are rare.
With an explicit ``selector`` the choice is entirely the caller's -- e.g. the top-:math:`k` by
mean Nye-norm via
:func:`graintrace.rare_criteria_selection_library.select_highest_scalar`.

.. warning::

   The **default**, used when ``selector`` is ``None``, does not look at the field at all. It
   keeps the clusters whose point count :math:`n` falls at or below the ``size_quantile`` of
   the surviving size distribution, after dropping anything smaller than ``min_size``:

   .. math::
      :label: rei-size-default

      \mathcal{R} = \bigl\{\, C : \mathrm{min\_size} \le n_C \le Q_{q}(n) \,\bigr\},

   truncated to the ``max_rare`` smallest. Its notion of rare is **small**, not **extreme**.
   That is a reasonable proxy only when the merge has already isolated the extreme material
   into a few small clusters; it is wrong whenever a rare region is large. Pass a ``selector``
   for anything field-driven.

Selected clusters receive distinct block ids (background id first, then one per rare cluster,
smallest first) and are written to a STRUCTURED_GRID or POLYDATA VTK, optionally with a rare
point-cloud CSV for downstream comparison with :doc:`rei-comparison`.

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
