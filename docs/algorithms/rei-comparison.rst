REI comparison
==============

Overview
--------
REI comparison quantifies the spatial overlap between two rare-event point clouds: two metrics,
two thresholds, two methods, or a prediction against a reference. It is implemented by
:class:`~graintrace.REIComparison`, which voxelizes both regions onto a common lattice, computes
volumetric overlap metrics (IoU, Dice, containment), pairs their clusters one-to-one, and writes
a classified point cloud. It is pure NumPy/SciPy: no KD-tree, alpha-shape, or marching cubes.

Method
------
Each input is a voxelized rare region: coordinate columns plus an optional integer cluster id.
Every rare point is treated as the centre of its voxel cube, so a region is a union of
axis-aligned cubes. The two grids may have **different** spacings but are assumed to share an
origin (no rotation or translation is applied). Both regions are resampled onto a common fine
lattice with spacing :math:`s_{\mathrm{ref}} = \min(s_1, s_2)` per axis, divided by an optional
integer ``supersample`` for sub-voxel boundary accuracy. Occupancy is a nearest-cell integer
index, so membership on the fine lattice is an :math:`O(1)` hash lookup.

.. warning::

   The two grids must **share an origin**. No registration is performed: no rotation, no
   translation, no scaling beyond the per-axis spacing. Two regions described in frames
   offset by half a voxel will report a low IoU that measures the offset, not a physical
   disagreement. Register the clouds before comparing them.

Let :math:`V_1, V_2` be the fine-lattice cell sets of the two regions. The reported metrics are

.. math::
   :label: reic-overlap

    \mathrm{IoU} = \frac{|V_1 \cap V_2|}{|V_1 \cup V_2|}, \qquad
    \mathrm{Dice} = \frac{2\,|V_1 \cap V_2|}{|V_1| + |V_2|}, \qquad
    \mathrm{cont}_i = \frac{|V_1 \cap V_2|}{|V_i|},

with physical volumes obtained by multiplying counts by the fine-cell volume
:math:`\prod_a s_{\mathrm{ref},a}`.

The three are not independent. IoU and Dice are monotone in each other,
:math:`\mathrm{Dice} = 2\,\mathrm{IoU} / (1 + \mathrm{IoU})`, so they rank a set of
comparisons identically and Dice is always the larger; reporting both is a convenience for
readers used to one or the other, not two pieces of evidence. The pair that *is* informative
is the two **containments**, because they are asymmetric. Equal containments mean the regions
disagree symmetrically; :math:`\mathrm{cont}_1 \gg \mathrm{cont}_2` means region 1 is nearly
inside region 2, i.e. the prediction is correct but too conservative. A single IoU cannot
distinguish those cases, which is exactly the distinction a prediction-versus-reference
comparison needs.

When cluster ids are present, an overlap matrix :math:`O` counts shared fine cells between every
cluster of region 1 and every cluster of region 2,

.. math::
   :label: reic-overlap-matrix

   O_{ab} = \bigl| V_1^{(a)} \cap V_2^{(b)} \bigr|.

A one-to-one correspondence is found by solving the linear assignment problem

.. math::
   :label: reic-hungarian

   \pi^{\star} = \arg\max_{\pi} \sum_{a} O_{a\,\pi(a)},

maximising total shared volume, via ``scipy.optimize.linear_sum_assignment`` on :math:`-O`.
This is a genuine global optimum over one-to-one pairings -- unlike the grain matcher of
:doc:`grain-tracking`, which is a local fixed point. Each matched pair reports its Jaccard
index and containments; clusters with no partner are flagged with id :math:`-1`.

Being one-to-one is also the limitation: a cluster that genuinely split into two cannot be
represented by the pairing, so the split and merge counts are reported separately. A cluster
:math:`a` counts toward the split total when more than one partner :math:`b` satisfies

.. math::
   :label: reic-split

   \frac{O_{ab}}{\bigl|V_1^{(a)}\bigr|} \ge f_{\mathrm{sm}},

with :math:`f_{\mathrm{sm}} =` ``split_merge_fraction`` (default :math:`0.2`); merges are the
same test transposed.
The denominator is the cluster's **own** size, so the criterion asks "how much of me went
there", not "how much of them came from me". Lowering :math:`f_{\mathrm{sm}}` makes boundary
bleed between adjacent clusters register as a split, so it should stay well above the fraction
of a cluster that lies within one voxel of its surface.

Algorithm
---------
1. Load both CSVs; resolve per-axis spacings (given or auto-detected from coordinate steps), the
   shared origin, and the fine reference spacing :math:`s_{\mathrm{ref}}`.
2. Map each region's points to nearest integer cells in its own grid and deduplicate.
3. Rasterize both cell sets onto the common fine lattice by nearest-neighbour occupancy
   upsampling, carrying cluster ids through.
4. Encode fine cells as integer keys over a shared bounding box; take the union and mark each key
   as only-1, only-2, or both.
5. Compute IoU, Dice, containment, and voxel/volume counts; write ``overlap_metrics.json``.
6. If cluster ids are present, build the overlap matrix, solve the Hungarian pairing on
   :math:`-O`, count splits/merges, and write ``cluster_match.csv``.
7. Export ``overlap_cloud.vtk`` with a ``membership`` scalar (1 = only-1, 2 = only-2, 3 = both)
   and both cluster-id fields.

Parameters that matter
----------------------
- ``spacing_1`` / ``spacing_2``: per-grid voxel size (scalar or ``[dx, dy, dz]``); ``None``
  auto-detects from coordinate steps (unsafe for sparse clouds; pass explicit spacing then).
- ``s_ref`` / ``supersample``: the common fine lattice; ``supersample > 1`` refines boundary
  accuracy at higher cost.
- ``cluster_col``: enables the per-cluster correspondence and split/merge detection; ``None``
  gives global overlap metrics only.
- ``split_merge_fraction``: the overlap fraction (of a cluster's own size) above which a shared
  partner counts as a split or merge.
- ``coord_cols`` / ``origin``: the coordinate columns and shared lattice origin.

See :doc:`/configuration` for the surrounding REI export options.

References
----------
- Jaccard/IoU overlap index and the Dice–Sørensen coefficient are standard set-overlap measures.
- One-to-one cluster pairing solves the linear assignment problem (Kuhn, H. W., 1955,
  https://doi.org/10.1002/nav.3800020109) on the negated overlap matrix via
  ``scipy.optimize.linear_sum_assignment``.

See also
--------
- Tutorial: :doc:`/tutorials/rei-comparison`
- API: :class:`~graintrace.REIComparison`
