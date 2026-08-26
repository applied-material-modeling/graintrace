Tutorials
=========

One notebook per workflow stage. Each has a short walkthrough with the code and,
where the stage is pure Python, executed output (those also carry an "Open in
Colab" badge). Stages that need the native stack (NEPER, CUBIT/SCULPT,
MOOSE/PUMA, NEML2) state so at the top and are not runnable in Colab or CI —
their outputs are generated in a configured environment.

Each tutorial links to the matching :doc:`algorithm page </algorithms/index>`
and API reference.

.. toctree::
   :maxdepth: 1
   :caption: Data & microstructure

   hedm-stitching
   microstructure-generation
   ff-reconstruction
   nf-reconstruction
   voxel-segmentation-mesh
   meshing
   experiment-rotation

.. toctree::
   :maxdepth: 1
   :caption: Calibration & simulation

   material-calibration
   cpfe-simulation
   cpfe-nf-ff

.. toctree::
   :maxdepth: 1
   :caption: Analysis

   post-processing
   rare-event-identification
   rei-example-2d
   rei-example-3d
   rei-comparison
   grain-tracking

.. toctree::
   :maxdepth: 1
   :caption: Reference

   /examples
