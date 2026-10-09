Grain segmentation
==================

Overview
--------
Grain segmentation groups the voxels of a gridded orientation field (EBSD or gridded NF-HEDM)
into grains, so that each connected region of near-common orientation gets one grain label.
graintrace offers two segmenters used by :class:`~graintrace.VoxelMeshBuilder` (and
:class:`~graintrace.NearFieldMeshBuilder`): a graph-based community detector,
:class:`~graintrace.GraphSpatialCluster`, and a flood-fill labeller,
:func:`graintrace.nf.segment.flood`. Both treat two voxels as belonging to the same grain when
their symmetry-aware misorientation is below a tolerance.

Method
------
Symbols and the rotation convention are defined once in :doc:`/notation`; this page uses them
without redefining them.

The pairwise dissimilarity between voxels :math:`i` and :math:`j` is the **disorientation**
angle: the misorientation matrix :math:`\Delta g_{ij} = g_i g_j^{\mathsf T}` minimised over the
symmetry orbits of *both* crystals,

.. math::
   :label: seg-disorientation

    d_{ij}
      = \min_{O_a,\, O_b \,\in\, G}\,
        \arccos\!\left(
          \mathrm{clamp}\!\left(
            \frac{\operatorname{tr}\!\bigl(O_a\, \Delta g_{ij}\, O_b^{\mathsf T}\bigr) - 1}{2},
            \,-1,\, 1
          \right)
        \right).

The minimisation is over the **ordered pair** :math:`(O_a, O_b)`, not over a single operator:
for the cubic group ``"432"`` that is :math:`|G|^2 = 24^2 = 576` candidates per edge, not 24.
This matters in practice as well as in principle -- a one-sided minimisation returns a larger
angle for most pairs, so it would silently split grains that :eq:`seg-disorientation` keeps
together. The grain-exchange symmetry does not need to be enumerated as well, because
:math:`\operatorname{tr}(M) = \operatorname{tr}(M^{\mathsf T})` makes the candidate *set* for
:math:`\Delta g_{ij}^{\mathsf T}` the transpose of the set for :math:`\Delta g_{ij}`, with the
same minimum.

.. note::

   The default ``symmetry`` of the low-level helpers in
   :mod:`graintrace.orientation_helper` is ``"1"``, the trivial group, which makes
   :eq:`seg-disorientation` a single term and returns the raw rotation angle -- up to
   :math:`180^\circ` where a cubic disorientation can never exceed :math:`62.8^\circ`. The
   builders pass their own ``symmetry`` (``"432"`` by default), so the segmentation path is
   correct without configuration; a direct call to the helper is not. See :doc:`/notation`.

**Graph segmentation.** A spatial graph is built over the grid: on a regular lattice each voxel
connects to every site within an :math:`\ell^1` ball of radius :math:`r`,
:math:`|\Delta i| + |\Delta j| + |\Delta k| \le r`, which is
:math:`\tfrac{1}{3}(4r^3 + 6r^2 + 8r) = 6/24/62/128` sites for :math:`r = 1/2/3/4`. Note
:math:`r = 2` is the 24-neighbour ball, not the 18-neighbour face-plus-edge stencil it is
sometimes called. Off-grid data falls back to mutual :math:`k`-nearest neighbours.

Each edge carries the distance :math:`d_{ij}`, converted to a similarity weight. Four kernels
are available; all are decreasing in :math:`d`, so a larger weight means "more alike":

.. list-table::
   :header-rows: 1
   :widths: 16 44 40

   * - ``mode``
     - :math:`w_{ij}`
     - Reads
   * - ``"rbf"``
     - :math:`\exp\!\left[-\left(d_{ij}/\sigma\right)^{p}\right]`
     - ``sigma``, ``power``
   * - ``"exp"``
     - :math:`\exp\!\left(-d_{ij}/\sigma\right)`
     - ``sigma``
   * - ``"inverse"``
     - :math:`1/(d_{ij} + \varepsilon)`
     - ``eps``
   * - ``"log_inv"``
     - :math:`-\log(d_{ij} + \varepsilon)`
     - ``eps``

``"rbf"`` is the recommended one: it is bounded on :math:`(0, 1]`, so the weight distribution
has a dynamic range Leiden can work with. ``"inverse"`` is unbounded as :math:`d \to 0`, and
with the default :math:`\varepsilon = 10^{-8}` two identically oriented voxels get a weight of
:math:`10^{8}` while a :math:`5^\circ` edge gets :math:`0.2` -- a range of nine orders of
magnitude in which every intra-grain edge dominates every boundary edge, whatever
:math:`\gamma` is set to.

.. important::

   :math:`\sigma` **is in the units of the distance metric**, which here is degrees or radians
   of misorientation, matching ``angle_type``. It is not a length and not a normalised
   quantity. The vetted value is half the misorientation cutoff, so
   :math:`\sigma \approx 2.5^\circ` against a :math:`5^\circ` cutoff.

When ``sigma`` is ``None``, it is estimated as a quantile of the surviving edge distances.
Only the ``"quantile"`` key of ``sigma_auto`` is read -- ``sample_size`` and ``random_state``
appear in several published recipes but are **not** consulted, and the quantile is taken over
all edges rather than a sample. On a grid where most edges are intra-grain the median distance
is near zero, so the estimate collapses to a fraction of a degree and shatters every grain;
pin :math:`\sigma` by hand for orientation data.

Optionally each node keeps only its top-:math:`k` highest-weight edges. The weighted graph is
then partitioned by Leiden, which maximises the resolution-scaled modularity

.. math::
   :label: seg-modularity

   Q(\gamma) = \frac{1}{2m} \sum_{i,j}
       \left( w_{ij} - \gamma\, \frac{k_i k_j}{2m} \right)
       \delta(c_i, c_j),

with :math:`k_i = \sum_j w_{ij}` the weighted degree, :math:`2m = \sum_i k_i`, and
:math:`c_i` the community of node :math:`i`. The :math:`\gamma` term is the weight the *null
model* -- a random graph with the same degree sequence -- expects on edge :math:`ij`, so
raising :math:`\gamma` makes that expectation harder to beat and communities break up.
Lower :math:`\gamma` therefore yields fewer, larger grains.

**Flood-fill segmentation.** Starting from a random unlabelled material voxel, a breadth-first
front grows outward, absorbing neighbours whose misorientation to the current voxel is below the
tolerance, until no more can be added. Segments smaller than ``grain_threshold`` are discarded
and their voxels permanently rejected: a flood is seed-independent, so re-queueing them would
only make the next iteration re-find the same segment. Rejected voxels are reported as
unlabelled, and the pass stops after ``stop_count`` discarded segments or when no voxels remain
(``stop_count`` must be at least 1). A cleanup pass infills unlabelled voxels from filled neighbours and merges
sub-threshold segments into the adjacent grain with the largest contact area.

.. note::

   The out-of-domain guard in the flood stencil is **inert**. ``flood`` builds a
   ``valid`` mask for neighbour offsets that fall outside the grid and then calls
   ``distances.masked_fill(~valid, inf)``, which is the *out-of-place* form: it
   returns a new tensor and the return value is discarded, so ``distances`` is
   never modified. This is harmless as written only because the neighbour indices
   are clamped to the grid first, which makes every out-of-domain lookup resolve
   to an in-domain voxel that the front would have reached anyway. It is not a
   guarantee: any change that stops clamping, or that reads ``distances`` for
   something other than the tolerance comparison, turns the dead mask into wrong
   labels at the domain boundary. Treat the clamp, not the mask, as the thing
   holding this together.

Algorithm
---------
Graph path (:class:`~graintrace.GraphSpatialCluster`):

1. Load the gridded orientation CSV; auto-detect whether coordinates form a full regular grid.
2. Build edges: grid connectivity at ``manhattan_radius`` on a full lattice, else mutual kNN.
3. Compute the symmetry-aware misorientation distance for every edge (batched; GPU-capable).
4. Estimate the RBF :math:`\sigma` (quantile of distances) if not supplied, then map distances
   to weights.
5. Optionally prune to the top-:math:`k` edges per node (``reduce_edges_topweights_k``) to
   sparsify the graph. Exact weight ties are common here -- two voxels with identical
   orientations are at distance 0, so their RBF weight is exactly 1.0 -- and the prune
   resolves them deterministically: among equal weights the lower half-edge index wins
   (edge :math:`e` is half-edge :math:`e` at its first endpoint and :math:`E + e` at its
   second). The result is therefore the same whether or not the optional numba
   acceleration is installed. One consequence is worth knowing: because both endpoints of
   an edge now apply the *same* rule, their selections overlap less than under an
   arbitrary tie-break, so more distinct edges survive and the pruned graph comes out
   roughly 15% denser on tie-heavy input, which costs the Leiden pass proportionally more.
   Measured on the vetted NF/EBSD recipe (see :doc:`/configuration`) -- a
   :math:`40^3` grid carrying 120 grain-constant
   orientations, ``manhattan_radius=2``, :math:`k = 12`, :math:`\sigma = 2.5^\circ`, 79% of
   its 734,640 edges tied -- the prune keeps 596,871 edges where an arbitrary tie-break
   keeps 517,139, i.e. +15.4%. The percentage is the stable part (15.1--15.4% over five
   grain layouts); the two counts themselves shift by a few tenths of a percent with the
   layout.
6. Partition the weighted graph with Leiden (or PLM/PLP) at resolution :math:`\gamma`.
7. Aggregate per-cluster properties (size, centroid, feature means) and emit per-voxel labels.

Flood path (:func:`graintrace.nf.segment.flood`):

1. Precompute neighbour misorientation distances over the connectivity stencil (6 or 26).
2. Mark all material voxels unsegmented; pick a random seed and grow a misorientation-bounded
   front until it stops.
3. Keep the segment if it reaches ``grain_threshold``, else discard and decrement the small-
   segment budget; repeat until done.
4. Infill leftover voxels and merge small segments into their largest-contact neighbour.

Parameters that matter
----------------------
- ``misorientation_tol``: the same-grain angular threshold (radians for
  :class:`~graintrace.VoxelMeshBuilder`/flood; degrees or radians per ``angle_type`` in the graph
  path).
- ``connectivity`` (6/26) or ``manhattan_radius``: neighbourhood used to build the graph or the
  flood stencil.
- ``networkit_kwargs={"gamma": ...}``: Leiden resolution; lower gives fewer clusters.
- ``weight_cfg`` (``mode``/``sigma``/``sigma_auto``/``power``): the RBF weighting of edges.
- ``reduce_edges_topweights_k``: per-node edge budget that sparsifies the graph before Leiden.
  Tie resolution is deterministic and numba-independent (see step 5 above).
- ``grain_threshold`` / ``grain_threshold_final`` / ``stop_count``: minimum grain size and the
  small-segment cleanup budget (flood path).

See :doc:`/configuration` for the full ``segmentation`` dict layout.

References
----------
- Leiden community detection: Traag, V. A., Waltman, L., van Eck, N. J. (2019), *From Louvain to
  Leiden: guaranteeing well-connected communities*,
  https://doi.org/10.1038/s41598-019-41695-z (as implemented in NetworKit ``ParallelLeiden``).
- The graph partition maximizes a resolution-:math:`\gamma` modularity objective; the RBF edge
  weighting is a standard Gaussian affinity of the misorientation distance.
- Flood-fill / connected-component labelling of the voxel grid under a misorientation tolerance.

See also
--------
- Tutorial: :doc:`/tutorials/voxel-segmentation-mesh`
- API: :class:`~graintrace.VoxelMeshBuilder`, :class:`~graintrace.GraphSpatialCluster`
