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
     - meshing options: SCULPT flags and the no-CUBIT voxel-hex dump
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

``examples/data_generation/make_cpfe_mwe.py`` regenerates the shipped CPFE
datasets (``mwe_data/out.csv`` + ``grid_out/``, ``synthetic_load_exp/``,
``cpfe_ff_fragmentation/``); it needs NEML2 and ``puma-opt`` and is not part of
the example sweep.

*full stack* = NEML2 + ``puma-opt`` + NEPER + CUBIT/SCULPT. The minimum
end-to-end check is ``demonstrate_cpfe_nfff.py``. The examples read
``puma-opt`` and the CUBIT/SCULPT paths from a ``tools.json``
(``$GRAINTRACE_TOOLS_JSON``, ``./graintrace_tools.json`` or
``~/.config/graintrace/tools.json``; see ``graintrace/mcp/tools.example.json``);
without one, edit the placeholder paths. Set ``ncore`` and ``device`` at the top
of the script for your machine.

To drive your own scan data, copy the ``demonstrate_*.py`` script closest to
your workflow and repoint its ``## INPUT`` block at your files. There are no
separate real-experiment driver templates in the repository.
