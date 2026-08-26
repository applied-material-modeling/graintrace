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
Each measured grain contributes a seed at its centroid. A standard Voronoi tessellation
assigns every point of the domain to the nearest seed,

.. math::

    C_i = \{\, x \in \Omega \;:\; \lVert x - s_i \rVert \le \lVert x - s_j \rVert
    \ \forall j \,\},

producing space-filling convex cells :math:`C_i`. With ``weighted=True`` graintrace requests a
Laguerre (power) tessellation instead, replacing the Euclidean distance with the power distance
:math:`\lVert x - s_i \rVert^2 - w_i`, where the weight :math:`w_i` is derived from each grain's
``GrainRadius``; this lets cell sizes track the measured grain sizes. The centroid-based
``option`` values (``centroid``/``centroidsize``) and CVT optimization move or relax the seeds so
that cell centroids match the measured positions (and sizes) rather than taking the raw seeds
directly. NEPER carries out the tessellation and the CVT optimization; graintrace only assembles
the seeds, weights, and bounding box and parses the results.

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

- ``bounding_box`` and ``auto_fix_bbox`` / ``bbox_fix_mode`` -- reconstruction domain and how
  out-of-box points are handled.
- ``weighted`` -- Voronoi vs. Laguerre (size-weighted) tessellation.
- ``option`` (``voronoi`` | ``centroid`` | ``centroidsize``) and ``CVT_iter`` / ``morphoalgo``
  -- how seeds are relaxed to match measured centroids/sizes.
- ``tesr_size`` -- voxel grid resolution of the reconstruction.
- ``rotate_angles`` / ``auto_rotate`` and ``unit`` -- sample-frame alignment; ``unit`` must
  match the data.

Further details
---------------
For the full tessellation and CVT-optimization algorithm, see the NEPER documentation:
https://neper.info/doc/ .

See also
--------
- Tutorial: :doc:`/tutorials/ff-reconstruction`
- API: :class:`~graintrace.VoronoiMeshBuilder`
