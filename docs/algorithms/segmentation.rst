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
The pairwise dissimilarity between voxels is the misorientation angle under the crystal point
group :math:`G`,

.. math::

    d(\mathbf{o}_i, \mathbf{o}_j)
      = \min_{S \in G}\, \arccos\!\left(\frac{\operatorname{tr}(S\,R_i R_j^{\top}) - 1}{2}\right).

**Graph segmentation.** A spatial graph is built over the grid: on a regular lattice each voxel
connects to its Manhattan-radius-:math:`r` neighbours (6/24/62/... sites for
:math:`r = 1/2/3/\dots`); off-grid data falls back to mutual :math:`k`-nearest neighbours. Each
edge carries the misorientation distance :math:`d_{ij}`, converted to a similarity weight by a
radial-basis kernel,

.. math::

    w_{ij} = \exp\!\left[-\left(\frac{d_{ij}}{\sigma}\right)^{p}\right],

where :math:`\sigma` may be set directly or auto-estimated as a quantile of the edge-distance
distribution. Optionally each node keeps only its top-:math:`k` highest-weight edges. The
weighted graph is then partitioned by the Leiden community-detection algorithm, whose resolution
parameter :math:`\gamma` in the modularity objective controls granularity (lower :math:`\gamma`
yields fewer, larger grains).

**Flood-fill segmentation.** Starting from a random unlabelled material voxel, a breadth-first
front grows outward, absorbing neighbours whose misorientation to the current voxel is below the
tolerance, until no more can be added. Segments smaller than ``grain_threshold`` are discarded
and their voxels re-queued; the pass stops after ``stop_count`` consecutive small segments or
when no voxels remain. A cleanup pass infills unlabelled voxels from filled neighbours and merges
sub-threshold segments into the adjacent grain with the largest contact area.

Algorithm
---------
Graph path (:class:`~graintrace.GraphSpatialCluster`):

1. Load the gridded orientation CSV; auto-detect whether coordinates form a full regular grid.
2. Build edges: grid connectivity at ``manhattan_radius`` on a full lattice, else mutual kNN.
3. Compute the symmetry-aware misorientation distance for every edge (batched; GPU-capable).
4. Estimate the RBF :math:`\sigma` (quantile of distances) if not supplied, then map distances
   to weights.
5. Optionally prune to the top-:math:`k` edges per node (``reduce_edges_topweights_k``) to
   sparsify the graph.
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
