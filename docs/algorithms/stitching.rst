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
distance :math:`\Delta p`, the symmetry-aware disorientation :math:`\Delta\theta` of
:eq:`seg-disorientation`, and the relative radius difference
:math:`\Delta r = |r_B - r_A| / r_A`. The matching cost normalizes each term by its tolerance
and weights it,

.. math::
   :label: stitch-cost

    c_{ab} = w_{\mathrm{pos}}\frac{\Delta p_{ab}}{\tau_p}
           + w_{\mathrm{ori}}\frac{\Delta\theta_{ab}}{\tau_\theta}
           + w_{\mathrm{rad}}\frac{\Delta r_{ab}}{\tau_r}.

Dividing by the tolerance is what makes the three terms commensurable: each contributes
:math:`1` at exactly its own gate, so the weights compare *fractions of an allowance* rather
than micrometres against degrees. A weight is therefore a statement about which evidence to
trust, and it only means what you intend if the tolerances are each set to a genuine
allowance. Note :math:`\Delta r` is relative but :math:`\Delta p` is absolute, so a
position tolerance tuned on a fine-grained sample does not transfer to a coarse one.

**Gating is independent of the cost weights.** A tolerance of :math:`-1` disables that
dimension's gate, while a weight of :math:`0` only drops the term from :eq:`stitch-cost` and
leaves the gate enforcing. The vetted FF recipe uses both levers at once --
``radius_tolerance=-1`` with ``weights["rad"]=0`` -- because the FF equivalent-sphere radius
is the least reliable of the three observables; dropping only the weight would still have it
rejecting candidates silently.

Regions
~~~~~~~
Each grain is assigned a **region** from its :math:`z`-extent :math:`[z_l, z_h]` relative to
the overlap band :math:`[z_{ol}, z_{oh}]`:

.. list-table::
   :header-rows: 1
   :widths: 10 20 70

   * - Code
     - Region
     - Condition
   * - 1
     - ``CORE``
     - entirely inside the band (the fallback case)
   * - 2
     - ``HIGH``
     - :math:`z_l \ge z_{oh}` -- entirely above
   * - 3
     - ``LOW``
     - :math:`z_h \le z_{ol}` -- entirely below
   * - 4
     - ``BND-HIGH``
     - :math:`z_l < z_{oh} < z_h` -- crosses the upper edge
   * - 5
     - ``BND-LOW``
     - :math:`z_l < z_{ol} < z_h` -- crosses the lower edge
   * - 6
     - ``CROSS-BOTH``
     - :math:`z_l < z_{ol}` and :math:`z_h > z_{oh}` -- spans the whole band

The tests are applied in the order ``LOW``, ``HIGH``, ``CROSS-BOTH``, ``BND-LOW``,
``BND-HIGH``, so a grain taller than the band is ``CROSS-BOTH`` rather than either boundary
class. The extent defaults to the equivalent-sphere approximation :math:`z \pm r`; with
``refine_extents=True`` it uses the true per-cell :math:`[z_{\min}, z_{\max}]` from a NEPER
(Laguerre/Voronoi) tessellation of the pair.

The region is what encodes the physics of the scan geometry: a grain clipped by the edge of a
scan was only partially observed, so its centroid and radius are biased toward the scan
interior and the *other* scan's observation of the same grain is the better one. The rules
below are that statement, made explicit.

Rule tables
~~~~~~~~~~~
A matched pair is resolved by its ``(region A, region B)`` entry. ``MC`` merges the two
observations, ``KA`` keeps A and drops B, ``KB`` keeps B and drops A, and ``RJ`` rejects the
match and sends both grains to the unmatched stage.

.. list-table:: Matched-pair actions, ``MatchRuleTable``. Rows: region of the accumulator grain A. Columns: region of the new-scan grain B.
   :header-rows: 1
   :stub-columns: 1
   :widths: 22 13 13 13 13 13 13

   * - A \\ B
     - CORE
     - HIGH
     - LOW
     - BND-HIGH
     - BND-LOW
     - CROSS
   * - **CORE**
     - MC
     - KB
     - RJ
     - KA
     - KA
     - KA
   * - **HIGH**
     - RJ
     - RJ
     - RJ
     - RJ
     - RJ
     - RJ
   * - **LOW**
     - KA
     - MC
     - RJ
     - MC
     - KA
     - KA
   * - **BND-HIGH**
     - KB
     - KB
     - RJ
     - KB
     - MC
     - KB
   * - **BND-LOW**
     - KB
     - KB
     - RJ
     - MC
     - KA
     - KA
   * - **CROSS**
     - KB
     - KB
     - RJ
     - KB
     - MC
     - MC

Two structural facts are worth reading off it. The whole ``LOW`` column is ``RJ``: B is the
upper scan, so a B grain entirely below the band cannot be the same grain as anything in A and
the match is spurious. The whole ``HIGH`` row is likewise ``RJ``, for the mirror reason.

Grains left unmatched -- including everything the table rejected -- are then kept or dropped
by region alone:

.. list-table:: Unmatched-grain decisions, ``UnmatchedRules``
   :header-rows: 1
   :widths: 24 20 20 36

   * - Region
     - A (accumulator)
     - B (new scan)
     - Why
   * - ``CORE``
     - REMOVE
     - KEEP
     - fully inside the band, so B's observation supersedes A's
   * - ``HIGH``
     - **ERROR**
     - KEEP
     - an A grain above the band is geometrically impossible
   * - ``LOW``
     - KEEP
     - **ERROR**
     - a B grain below the band is geometrically impossible
   * - ``BND-HIGH``
     - REMOVE
     - KEEP
     - clipped by A's upper edge; B saw it whole
   * - ``BND-LOW``
     - KEEP
     - REMOVE
     - clipped by B's lower edge; A saw it whole
   * - ``CROSS-BOTH``
     - REMOVE
     - KEEP
     - spans the band; prefer the newer scan

The two ``ERROR`` cells are assertions, not actions: reaching one means the scan
:math:`z`-geometry passed to ``run`` disagrees with the data, usually a wrong
``overlap_fraction`` or a :math:`z`-shift that was applied twice.

.. note::

   ``MatchRuleTable._build_table`` carries a second, unreachable rule set behind an
   ``optiona=True`` default argument that nothing sets to ``False``. The alternate set merges
   any overlap-intersecting pair instead of preferring one scan. Only the table above runs.

Merging
~~~~~~~
A merged grain combines the two observations by grain volume
:math:`v = \tfrac{4}{3}\pi r^3`. The centroid is the volume-weighted mean,

.. math::
   :label: stitch-merge-centroid

   \mathbf{x} = \frac{v_A \mathbf{x}_A + v_B \mathbf{x}_B}{v_A + v_B},

and the merged radius comes from the **mean** of the two volumes, not their sum:

.. math::
   :label: stitch-merge-radius

   r = \left( \frac{3}{4\pi} \cdot \frac{v_A + v_B}{2} \right)^{1/3} .

That is the right choice here and an easy one to get wrong. The two observations are the
*same grain* seen twice, so their volumes should be averaged; summing them would double the
material at every merge and inflate the stitched volume fraction by the overlap.

The orientation is a symmetry-aware volume-weighted average: :math:`R_B` is first replaced
by the symmetry equivalent closest to :math:`R_A`, the two matrices are blended, and the result
is re-projected onto :math:`SO(3)`,

.. math::
   :label: stitch-merge-ori

   \tilde{R} = \frac{v_A R_A + v_B R_B^{\star}}{v_A + v_B},
   \qquad
   R = U V^{\mathsf T}, \quad \tilde{R} = U \Sigma V^{\mathsf T}.

The polar factor :math:`UV^{\mathsf T}` is the closest rotation to :math:`\tilde{R}` in the
Frobenius norm, which is what makes the blend well defined -- a weighted sum of two rotation
matrices is not itself a rotation. Folding :math:`R_B` onto the nearest variant first is not
optional: without it a cubic pair reported in different symmetry variants averages to
something close to neither, and the result is a grain whose orientation matches no
observation. See :doc:`/notation` for the variant-folding rule and for why a quaternion mean
is used elsewhere in the package for the same job.

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
