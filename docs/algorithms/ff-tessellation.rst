FF Voronoi/CVT tessellation
===========================

Overview
--------
Reconstruction of a 3D grain structure from far-field (FF) HEDM grain centroids by driving
NEPER to build a Voronoi (or Laguerre) tessellation, optionally CVT-relaxed. The stage is
driven by :class:`~graintrace.VoronoiMeshBuilder`, which produces a ``.tess`` tessellation,
per-grain orientation and initial elastic-strain files, and a voxelized reconstruction that
feeds meshing and CPFE.

Method
------
The problem this stage solves is an **inverse** one, and that framing explains every option
below. FF-HEDM does not measure grain shape. For each grain it reports a volume centroid, an
equivalent-sphere radius, an orientation and a lattice strain -- four observables, none of
which is a grain boundary. CPFE needs a space-filling partition of the domain. A tessellation
is the standard way to manufacture one, and the question is only which tessellation, seeded
how.

.. important::

   The resulting cell shapes are a **plausible reconstruction, not a measurement**. A
   tessellation cell is convex; real grains are not. Cell :math:`z`-extents recovered this
   way are roughly :math:`r/3` noisy against ground truth even for the generating seeds, and
   for strongly elongated grains the per-scan tessellation cannot recover the true extent at
   all. For grain morphology that is actually measured, use NF-HEDM
   (:doc:`/algorithms/segmentation`). Use this stage for the orientation and strain fields it
   carries, and for a geometry that is statistically reasonable.

Voronoi and Laguerre cells
~~~~~~~~~~~~~~~~~~~~~~~~~~
A standard Voronoi tessellation assigns every point of the domain to the nearest seed,

.. math::
   :label: ff-voronoi

    C_i = \{\, x \in \Omega \;:\; \lVert x - s_i \rVert \le \lVert x - s_j \rVert
    \ \forall j \,\},

producing space-filling convex cells :math:`C_i`. Every cell boundary is the perpendicular
bisector of a seed pair, so cell size is controlled only by seed *spacing*: a Voronoi
tessellation of measured centroids cannot make one grain bigger than its neighbours except by
being further from them.

A Laguerre (power) tessellation removes that restriction by replacing the Euclidean distance
with the power distance :math:`\lVert x - s_i \rVert^2 - w_i`, which moves each bisecting
plane toward the lower-weighted seed. With ``weighted=True`` graintrace derives the weight
from the measured ``GrainRadius`` as the grain volume (3D) or area (2D), normalised to sum to
one:

.. math::
   :label: ff-weight

   w_i = \frac{v_i}{\sum_j v_j}, \qquad
   v_i = \tfrac{4}{3}\pi r_i^{3} \ \ (\text{3D}), \qquad
   v_i = \pi r_i^{2} \ \ (\text{2D}).

Only the *ratios* between weights survive that normalisation; the absolute scale does not.
The weights are passed to NEPER as the morphology optimisation's starting point
(``-morphooptiini weight:``), not as a final specification, so the tessellation NEPER returns
is the one its optimiser converges to rather than the exact Laguerre diagram of
:eq:`ff-weight`.

Seeding modes
~~~~~~~~~~~~~
``option`` chooses what the measured centroids *are* to NEPER -- initial seed positions, or an
optimisation target. The distinction matters: a seed and a cell centroid are not the same
point, and FF measures the latter.

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - ``option``
     - NEPER morphology
     - What the measured data is used as
   * - ``"voronoi"``
     - ``-morpho voronoi``
     - **initial seed positions** (``-morphooptiini coo:``), optionally with
       :eq:`ff-weight` weights
   * - ``"centroidal"``
     - ``-morpho centroidal``
     - initial seed positions for a **CVT** relaxation
   * - ``"centroid"``
     - ``-morpho centroid:file(...)``
     - an **optimisation target**: NEPER moves seeds until cell centroids match. Seeds are
       initialised from NEPER's own packing (``coo:LLLFP2011``), *not* from the data
   * - ``"centroidsize"``
     - ``-morpho centroidsize:file(...)``
     - an optimisation target on centroid **and** size together; requires ``GrainRadius``

``"centroid"`` is the documented default for FF reconstruction because it targets the
observable FF actually reports. Seeding at the measured centroids instead (``"voronoi"``)
asserts that the centroid is the seed, which is true only for a grain whose cell happens to be
symmetric about it.

.. note::

   ``"centroid"`` and ``"centroidal"`` are **not** synonyms, despite looking like one. They
   select different NEPER morphologies and different initialisations, as the table shows.
   ``"centroid"`` fits centroids to the data; ``"centroidal"`` is a CVT relaxation seeded from
   the data, which moves cells away from the measured positions.

.. note::

   ``weighted=True`` applies only on the ``"voronoi"``/``"centroidal"`` path. The two
   ``centroid*`` targets ignore it -- ``"centroidsize"`` carries the radius in its own target
   file instead. On the weighted path ``morphoalgo`` is also overridden to ``"lloyd"``
   regardless of what the caller passed.

NEPER carries out the tessellation and the optimisation; graintrace assembles the seeds,
weights, orientations and bounding box, invokes ``neper -T``, and parses the results.
``CVT_iter`` sets the optimiser's iteration budget (``-morphooptistop iter=``) and
``morphoalgo`` its algorithm (``-morphooptialgo``); ``iter=0`` disables relaxation entirely,
which is what the stitching extent helper uses when it wants cells exactly at the measured
seeds.

Algorithm
---------
1. Read the FF grain CSV; parse centroids, Euler orientations, and the 9-component elastic
   strain tensor per grain.
2. Optionally apply a sample-tilt rotation (``rotate_angles``) or PCA alignment
   (``auto_rotate``) to the point cloud and orientations.
3. Reconcile the data against ``bounding_box`` (``auto_fix_bbox`` with ``remove_points`` or
   ``extend_bounding_box``).
4. Write the NEPER seed/weight inputs and invoke NEPER (``neper -T``) with the chosen
   ``option`` and CVT settings (``CVT_iter``, ``morphoalgo``).
5. Collect outputs: ``reconstruction_reformatted.csv`` (per-voxel grain IDs and orientations),
   ``reconstruction_cpfe_ee.csv`` (per-grain initial elastic strain), ``orientations.dat``
   (per-grain Euler angles, always degrees), and the ``.tess`` file. A GMSH tet ``.msh`` is
   written only when ``generate_mesh=True`` (a fallback; the default CPFE mesh is SCULPT hex).

Parameters that matter
----------------------
See :doc:`/configuration` for the full list.

- ``bounding_box`` and ``auto_fix_bbox`` / ``bbox_fix_mode``: reconstruction domain and how
  out-of-box points are handled.
- ``weighted``: Voronoi vs. Laguerre (size-weighted) tessellation.
- ``option`` (``voronoi`` | ``centroid`` | ``centroidsize``) and ``CVT_iter`` / ``morphoalgo``:
  how seeds are relaxed to match measured centroids/sizes.
- ``tesr_size``: voxel grid resolution of the reconstruction.
- ``rotate_angles`` / ``auto_rotate`` and ``unit``: sample-frame alignment; ``unit`` must
  match the data.

Further details
---------------
For the full tessellation and CVT-optimization algorithm, see the NEPER documentation:
https://neper.info/doc/ .

See also
--------
- Tutorial: :doc:`/tutorials/ff-reconstruction`
- API: :class:`~graintrace.VoronoiMeshBuilder`
