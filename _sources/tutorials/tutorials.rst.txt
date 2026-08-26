Tutorials
=========

One guide per workflow stage. Each distills the recipe and embeds the matching
runnable script from ``examples/``. The **Needs** in each guide states which
external tools it requires; several run from a plain ``pip install`` alone.

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
   rei-comparison
   grain-tracking
