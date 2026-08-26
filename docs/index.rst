graintrace
==========

graintrace is a Python toolkit that links grain-scale characterization data
(far-field and near-field HEDM, EBSD) to crystal-plasticity finite element
(CPFE) simulations. It reconstructs 3D microstructure meshes from experimental
data, sets up and runs MOOSE/PUMA CPFE simulations with NEML2 material models,
and post-processes the results to identify rare events for targeted measurement.

graintrace is the Python orchestration layer over a compiled scientific stack:

- **MOOSE / PUMA** — finite element framework and solver.
- **NEML2 v3** — crystal-plasticity constitutive models, AOTI compiled, GPU capable.
- **pyzag** — analytic adjoint gradients used for material calibration.
- **NEPER** — Voronoi/CVT tessellation for far-field reconstruction.
- **Coreform CUBIT/SCULPT** — conformal hex meshing for FF/NF/EBSD (a NEPER/gmsh
  tet mesh is available as a fallback).

The pipeline runs in four stages: reconstruct a microstructure, calibrate the
material model, run the CPFE simulation, then analyze the fields and identify
rare events.

Some features are pure Python and run from a plain ``pip install`` (stitching,
post-processing, rare-event identification). Others need parts of the compiled
stack (calibration needs NEML2 + pyzag; CPFE needs MOOSE/PUMA; meshing needs
CUBIT/SCULPT). ``import graintrace`` lazy-imports the heavy dependencies, so the
pure-Python subset works without the native build.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      Install (three tiers), a pure-Python quickstart, and the pipeline and
      data formats.

   .. grid-item-card:: Tutorials
      :link: tutorials/tutorials
      :link-type: doc

      One runnable notebook per workflow stage — reconstruction, calibration,
      CPFE, and analysis.

   .. grid-item-card:: Algorithms & theory
      :link: algorithms/index
      :link-type: doc

      The model and algorithm behind each stage, with equations and references.

   .. grid-item-card:: API reference
      :link: api/api
      :link-type: doc

      Every public module, grouped by workflow stage.

.. toctree::
   :hidden:

   getting_started
   tutorials/tutorials
   algorithms/index
   guides
   api/api
