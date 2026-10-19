Algorithms & theory
===================

The methods behind each workflow stage: the model, the equations, and the
algorithm graintrace implements. Each page links to the matching tutorial and API.

graintrace-native methods (stitching, segmentation, rare-event identification,
grain tracking, REI comparison) are described in full. For methods
carried out by an external engine (NEPER tessellation, the NEML2 crystal-
plasticity constitutive model, the MOOSE/PUMA finite-element solve), the pages
give the graintrace-level view and point to the upstream documentation for the
full algorithm.

Start with :doc:`../notation` if you are reading these pages in order: it fixes
the orientation conventions, symbols and units that every other page here uses
without redefining them.

.. toctree::
   :maxdepth: 1

   ../notation
   ff-tessellation
   tess-to-graph
   stitching
   segmentation
   meshing
   microstructure-generation
   calibration
   cpfe
   rare-event-identification
   grain-tracking
   rei-comparison
