HEDM scan stitching
===================

Overview
--------
HEDM scan stitching merges several overlapping far-field scan layers, each a table of grain
centroids, radii, and orientations, into a single non-redundant grain set. It is implemented by
:class:`~graintrace.RegionBaseStitching`, which folds the layers together one pair at a time with
:class:`~graintrace.hedm_stitching_techniques.pair_stitching_utils.PairwiseStitcher`. Duplicate
grains observed in the overlap between two consecutive scans are matched by position,
orientation, and size, then merged with a volume-weighted average.

Method
------
Scans are sorted bottom-to-top by their minimum :math:`z` and folded sequentially: the
accumulator :math:`S_0` is stitched with :math:`S_1`, the result with :math:`S_2`, and so on.
For a :math:`z`-window :math:`[z_{lo}, z_{hi}]` of :math:`n` scans with overlap fraction
:math:`f`, the per-scan height and step are

.. math::

    h = \frac{z_{hi} - z_{lo}}{\,n - (n-1)f\,}, \qquad \Delta z = h\,(1-f),

and the overlap band between scan :math:`k` and :math:`k+1` is
:math:`[\,z_{lo} + (k+1)\Delta z,\ z_{lo} + k\,\Delta z + h\,]`.

Within a pair, candidate duplicate edges are scored by three differences: Euclidean centroid
distance :math:`\Delta p`, symmetry-aware misorientation :math:`\Delta\theta`, and relative
radius difference :math:`\Delta r = |r_B - r_A| / r_A`. The misorientation respects the crystal
point group :math:`G`,

.. math::

    \Delta\theta = \min_{S \in G}\, \arccos\!\left(\frac{\operatorname{tr}(S\,R_A R_B^{\top}) - 1}{2}\right),

and the matching cost normalizes each term by its tolerance and weights it,

.. math::

    c_{ab} = w_{\mathrm{pos}}\frac{\Delta p_{ab}}{\tau_p}
           + w_{\mathrm{ori}}\frac{\Delta\theta_{ab}}{\tau_\theta}
           + w_{\mathrm{rad}}\frac{\Delta r_{ab}}{\tau_r}.

Gating (which candidates are feasible) is independent of the cost weights: a tolerance of
:math:`-1` disables that dimension's gate, while a weight of :math:`0` only drops it from the
cost. Each grain is also assigned a **region** from its :math:`z`-extent :math:`[z_l, z_h]`
relative to the overlap band :math:`[z_{ol}, z_{oh}]`: ``CORE``, ``HIGH``, ``LOW``,
``BND-HIGH``, ``BND-LOW``, or ``CROSS-BOTH``. The extent defaults to the equivalent-sphere
approximation :math:`z \pm r`; with ``refine_extents=True`` it uses the true per-cell
:math:`[z_{\min}, z_{\max}]` from a NEPER (Laguerre/Voronoi) tessellation of the pair.

A merged grain combines the two observations by grain volume :math:`v = \tfrac{4}{3}\pi r^3`:
centroid is the volume-weighted mean, and orientation is a symmetry-aware, volume-weighted
average (B rotated to the symmetry equivalent closest to A, the two rotation matrices blended
and re-projected onto :math:`SO(3)` via SVD).

Algorithm
---------
1. Load each scan CSV into a grain set, record its :math:`[z_{\min}, z_{\max}]`, and sort the
   scans bottom-to-top.
2. For each consecutive pair, compute the overlap band :math:`[z_{ol}, z_{oh}]` from the scan
   geometry above (with ``refine_extents``, re-tessellate the accumulator and next scan to get
   true per-cell :math:`z`-extents first).
3. Classify every grain of both scans into one of the six regions relative to the overlap band.
4. Build candidate duplicate edges with a k-nearest-neighbour query (``min_neighbors`` per
   grain, ``scipy.spatial.cKDTree``) and evaluate :math:`\Delta p`, :math:`\Delta\theta`,
   :math:`\Delta r` for each.
5. Drop candidates that fail any enabled tolerance gate; assign the surviving pairs one-to-one
   by solving the linear assignment problem on the weighted cost with an augmented cost matrix
   that permits leaving grains unmatched (the Hungarian algorithm,
   ``scipy.optimize.linear_sum_assignment``).
6. For each matched pair, look up the action in the region rule table: merge (core/core and
   compatible boundary pairs), keep-A, keep-B, or reject (defer to the unmatched stage).
7. For unmatched and rejected grains, apply the per-region keep/remove rules (e.g. keep an A
   grain that is ``LOW`` and below the band; drop a redundant ``CORE`` A grain).
8. Concatenate merged, kept-matched, and kept-unmatched grains into the new accumulator and
   continue to the next scan; write the final stitched CSV.

Parameters that matter
----------------------
- ``position_tolerance`` / ``orientation_tolerance`` / ``radius_tolerance``: the per-dimension
  gates; ``-1`` disables a gate. ``orientation_tolerance`` must be in the same units as the
  orientation data (convert to radians when ``orientation_units="radians"``).
- ``weights`` (``pos``/``ori``/``rad``): relative importance of each term in the match cost; a
  weight of ``0`` removes a term from the cost without disabling its gate.
- ``min_neighbors``: number of nearest candidates queried per grain before assignment.
- ``overlap_fraction`` (passed to ``run``): sets the overlap band geometry; ``0`` uses the
  non-overlap slab path instead.
- ``refine_extents`` / ``tess_weighted``: opt-in true tessellation :math:`z`-extents for region
  classification (helps elongated grains; needs NEPER and is slower).

See :doc:`/configuration` for the full knob reference.

References
----------
- Hungarian assignment: Kuhn, H. W. (1955), *The Hungarian method for the assignment problem*,
  https://doi.org/10.1002/nav.3800020109 (as solved by ``scipy.optimize.linear_sum_assignment``).
- Laguerre/Voronoi tessellation for the optional true-extent refinement (via NEPER), see
  :doc:`ff-tessellation`.
- Crystal misorientation under point-group symmetry is computed with the neml2-backed
  orientation helpers.

See also
--------
- Tutorial: :doc:`/tutorials/hedm-stitching`
- API: :class:`~graintrace.RegionBaseStitching`
