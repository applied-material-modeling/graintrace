Examples
========

``examples/demonstrate_*.py`` are runnable, self-contained scripts (flat
top-level ``## INPUT`` style), mostly backed by the small datasets under
``mwe_data/``. Each maps to a tutorial in :doc:`tutorials/tutorials`. The
**Needs** column: *pip only* runs from ``pip install graintrace`` alone; the rest
need the external tool(s) shown.

.. list-table::
   :header-rows: 1
   :widths: 34 46 20

   * - Example
     - What it shows
     - Needs
   * - ``demonstrate_hedm_study.py``
     - synthetic crystal → overlapping HEDM z-scans → stitch → compare to truth
     - NEPER
   * - ``demonstrate_hedm_anisotropic.py``
     - anisotropic (``aspratio``) microstructure generation benchmark
     - NEPER
   * - ``demonstrate_farfield.py``
     - FF Voronoi reconstruction: orientations, initial elastic strain, ``.tess``
     - NEPER
   * - ``demonstrate_grid_segmentation_mesh.py``
     - EBSD/gridded-NF voxel graph-segmentation + SCULPT hex mesh
     - CUBIT/SCULPT
   * - ``demonstrate_synthetic_cpfe.py``
     - meshing options — SCULPT flags and the no-CUBIT voxel-hex dump
     - NEPER, CUBIT/SCULPT
   * - ``demonstrate_material_calibration.py``
     - pyzag-adjoint Taylor calibration of 6 crystal-plasticity params
     - NEML2
   * - ``demonstrate_cpfe.py``
     - FF CPFE via NEML2 AOTI + ``puma-opt`` (ships a 10-grain mesh)
     - NEML2, puma-opt
   * - ``demonstrate_cpfe_nfff.py``
     - end-to-end: NF geometry + FF initial strain CPFE
     - full stack
   * - ``demonstrate_postprocess.py``
     - field distributions / macroscopic stress-strain / IPF
     - pip only
   * - ``demonstrate_rei_pipeline.py``
     - rare-event ID: graph cluster → hierarchical merge → rare VTK
     - pip only
   * - ``demonstrate_rei_example_2D.py``, ``..._3D.py``
     - REI on 2D / 3D synthetic fields
     - pip only
   * - ``demonstrate_rei_comparison.py``
     - compare two REI point clouds → overlap metrics + classified VTK
     - pip only
   * - ``demonstrate_graintracking.py``
     - match grains across load steps via a grain graph
     - NEPER

*full stack* = NEML2 + ``puma-opt`` + NEPER + CUBIT/SCULPT. The minimum
end-to-end check is ``demonstrate_cpfe_nfff.py``; before running, edit the
``sculpt_config``, ``moose_run_file``, ``ncore``, and ``device`` at the top of
the script to match your machine.

``examples/run_experiment_*.py`` are real-experiment driver templates (FF-only,
NF+FF, the AFRL dataset, stitching comparison, crystal reconstruction) that read
your own scan data — copy and adapt one rather than running it as-is.
