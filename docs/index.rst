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
pure-Python subset works without the native build. See :doc:`install` for the
three installation tiers.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   install
   quickstart
   concepts

.. toctree::
   :maxdepth: 1
   :caption: Guides

   configuration
   mcp
   benchmarks
   pitfalls

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & examples

   tutorials/tutorials
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/api
   development
