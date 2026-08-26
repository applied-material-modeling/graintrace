REI comparison
==============

Compare two rare-event-identification point clouds for spatial overlap. This
reports IoU, Dice, and containment metrics, builds a one-to-one cluster
correspondence (with split/merge detection), and exports a classified point
cloud (only-1, only-2, both) to VTK, using :class:`~graintrace.REIComparison`.
Use it to compare two REI results — two metrics, thresholds, or methods, or a
prediction against a reference — possibly on grids of different spacing. The
implementation is pure Python (numpy and scipy).

**Needs:** none beyond graintrace — no MOOSE, NEPER, or CUBIT.

Model
-----

Each rare point is the center of its voxel cube, so an REI is a union of
axis-aligned cubes and overlap is a boolean volume intersection. Both regions are
resampled onto a common finer lattice (``s_ref = min(spacing_1, spacing_2)``);
membership is then an integer-index hash lookup, so non-contiguous regions are
handled without surface reconstruction. Region volume is voxel count times voxel
volume. This avoids KD-trees, alpha shapes, and marching cubes.

Inputs
------

Two voxelized REI point-cloud CSVs, each with ``x, y, z`` columns and an optional
integer ``rare_cluster_id``. Produce them from the REI pipeline by passing
``rare_points_csv_path=...`` to
:meth:`~graintrace.IdentifyRareClusters.run_get_rare_cluster`, which writes
``x, y, z, rare_cluster_id`` for the rare points (see
:doc:`rare-event-identification`).

Each grid must be regular (constant per-axis spacing), and the two grids must
share an origin. No rotation or translation is applied here; register the clouds
first if they do not share an origin.

Recipe
------

.. code-block:: python

   from graintrace.rei_comparison import REIComparison

   comp = REIComparison(
       rei_csv_1="out/rei_A.csv", rei_csv_2="out/rei_B.csv",
       output_dir="out/rei_comparison",
       spacing_1=1.0, spacing_2=2.0,   # true grid spacing; None auto-detects (sparse-unsafe)
       coord_cols=("x", "y", "z"),
       cluster_col="rare_cluster_id",  # None skips cluster-level matching
       supersample=1,                  # >1 for sub-voxel boundary accuracy (rarely needed)
   )
   result = comp.run_comparison()
   print(result["metrics"]["iou"], result["metrics"]["containment_1"])

Key parameters
--------------

- ``spacing_1`` / ``spacing_2`` — scalar or ``[dx, dy, dz]``. Pass the true
  spacing; auto-detect uses the smallest positive coordinate step and can be wrong
  for sparse clouds.
- ``cluster_col`` — enables the cluster correspondence; ``None`` gives global
  overlap only.
- ``split_merge_fraction`` (default ``0.2``) — the significance threshold for
  split/merge counting.

Outputs (in ``output_dir``)
---------------------------

- ``overlap_metrics.json`` — IoU (Jaccard), Dice, ``containment_1`` /
  ``containment_2`` (asymmetric), voxel counts and volumes, and, with
  ``cluster_col``, cluster/split/merge counts.
- ``overlap_cloud.vtk`` — classified point cloud with scalar ``membership``
  (1 = only-1, 2 = only-2, 3 = both) plus ``cluster_id_1`` / ``cluster_id_2``.
  Color by ``membership`` in ParaView.
- ``cluster_match.csv`` — one-to-one cluster pairing (Hungarian by overlap
  volume, label-agnostic) with per-pair Jaccard and containment; unmatched
  clusters are flagged ``-1``.

Gotchas
-------

- Different spacings are fine (the coarser region is upsampled to ``s_ref``);
  different origins are not — align first.
- Boundary discretization error is about one ``s_ref`` voxel; raise
  ``supersample`` only if that matters. A coarse-versus-fine IoU below 1 for the
  same region is expected, since cube extents differ at the boundary by half the
  coarse cell.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_rei_comparison.py
   :language: python
   :caption: examples/demonstrate_rei_comparison.py
