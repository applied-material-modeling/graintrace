HEDM scan simulation and stitching
===================================

Generate a synthetic crystal, simulate overlapping HEDM z-scans, stitch the scan
layers into one grain set, and compare the stitched result against the ground
truth. Use this to test or study HEDM stitching, examine the effect of scan
overlap, or produce a stitched FF CSV from multiple scan layers.

**Needs:** NEPER (crystal generation and z-scan). No MOOSE or CUBIT.

Inputs
------

The example is self-generating: :class:`~graintrace.CrystalGenerator` writes a
``voronoi.tess`` / ``voronoi.csv`` and a set of ``hedm_scan/scan_{i}.csv`` layers.
For real data, feed your own per-scan CSVs to the stitcher: whitespace- or
comma-delimited FF grain tables with ``X,Y,Z,GrainRadius,Eul0/1/2,eKen*``, each
layer already Z-shifted. See :doc:`../concepts` for the raw FF format.

Recipe
------

.. code-block:: python

   from graintrace.generate_random_crystal import CrystalGenerator
   from graintrace.hedm_stitching_techniques.region_base_stitching import RegionBaseStitching
   from graintrace.scan_stitching_comparison import ScanStitchingComparison

   bounding_box = [-500, 500, -500, 500, -1000, 500]
   nscan, overlap_percentage = 4, 25

   cg = CrystalGenerator(output_dir="hedm_out", bounding_box=bounding_box, seed=42)
   cg.generate_tessellation(morpho_args={"type": "diameq", "distribution": "lognormal",
                                         "params": (130.0, 5.0)})
   cg.hedm_zscan(tess_file="hedm_out/voronoi.tess", nstep=nscan,
                 overlap_percentage=overlap_percentage,
                 position_noise_std=0.0, orientation_noise_std=0.0, radius_noise=0.0,
                 noise_seed=42)

   scan_files = [f"hedm_out/hedm_scan/scan_{i}.csv" for i in range(nscan)]
   stitcher = RegionBaseStitching(
       scan_files=scan_files, output_csv="hedm_out/stitched.csv",
       position_tolerance=20, orientation_tolerance=1, radius_tolerance=0,
       weights={"pos": 0.1, "ori": 1.0, "rad": 0}, min_neighbors=5,
   )
   stitched = stitcher.run(zlo=bounding_box[4], zhi=bounding_box[5],
                           overlap_fraction=overlap_percentage / 100.0)

   ScanStitchingComparison(
       output_dir="hedm_out/comparison", true_csv="hedm_out/voronoi.csv",
       stitch_csv="hedm_out/stitched.csv", position_tolerance=20,
       orientation_tolerance=5.0, radius_tolerance=0,
       weights={"pos": 0.1, "ori": 1.0, "rad": 0},
   ).run_comparison()

Key parameters
--------------

- ``morpho_args`` — grain morphology passed to
  :class:`~graintrace.CrystalGenerator` (see :doc:`microstructure-generation`).
- ``nscan``, ``overlap_percentage`` — number of z-scan layers and their overlap.
- Noise, each applied only when greater than zero, with ``noise_seed`` for
  reproducibility: ``position_noise_std`` (absolute centroid std, length units),
  ``orientation_noise_std`` (proper misorientation, degrees), ``radius_noise``
  (relative fraction). ``remove_minimum_volume`` / ``min_vol`` drop small grains.
- Stitch: ``position_tolerance`` (length), ``orientation_tolerance`` (degrees),
  ``weights``, ``min_neighbors``.
- Alternative stitchers: ``NaiveStitching`` and ``RegionBaseStitching``.

Region classification (deciding which duplicate grain to trust or merge) uses each
grain's z-extent. By default this is the equivalent-sphere estimate ``z ± GrainRadius``.
For elongated grains, ``refine_extents=True`` uses a NEPER tessellation to recover
the true per-cell ``[zmin, zmax]``; this re-tessellates at each fold step and is
slower. See :doc:`../pitfalls` for the limits of extent recovery from FF observables.

Outputs
-------

- ``hedm_out/stitched.csv`` — the merged single grain set.
- ``hedm_out/comparison/`` — stitched-vs-true comparison from
  :class:`~graintrace.ScanStitchingComparison`.

Gotchas
-------

- ``orientation_tolerance`` is in degrees here. If a downstream step uses radians,
  convert with ``np.deg2rad``; see :doc:`../pitfalls`.
- Set ``radius_tolerance=-1`` to disable the radius gate in the match cost.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_hedm_study.py
   :language: python
   :caption: examples/demonstrate_hedm_study.py
