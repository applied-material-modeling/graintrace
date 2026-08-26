CPFE (NF geometry + FF initial strain)
======================================

Run a combined near-field + far-field CPFE simulation: the NF reconstruction
supplies the high-resolution hex mesh and per-element orientations, while the FF
reconstruction supplies the initial elastic-strain (residual-strain) field. Use
this when you want to drive CPFE from an NF mesh with an FF residual-strain
initial condition, optionally end to end from synthetic HEDM.

**Needs:** the native stack — NEPER (FF/synthetic), CUBIT/SCULPT (NF mesh),
MOOSE ``puma-opt`` + ``neml2-compile``.

Inputs
------

- An NF mesh and orientations, from :doc:`nf-reconstruction`
  (:class:`~graintrace.NearFieldMeshBuilder`): ``mesh.e`` plus a per-element MRP
  ``orientations.csv``.
- An FF per-grain elastic-strain file, from :doc:`ff-reconstruction`
  (:class:`~graintrace.VoronoiMeshBuilder` with ``generate_mesh=False``):
  ``reconstruction_cpfe_ee.csv``.

If the NF and FF data are not co-registered, shift the FF ``x, y, z`` into the NF
frame first (see below). A synthetic HEDM generator can produce both stages when
you do not have experimental data.

Pipeline
--------

#. (optional) Generate synthetic HEDM, writing an FF CSV and NF layers.
#. Build the NF mesh: :meth:`~graintrace.NearFieldMeshBuilder.reconstruct` then
   :meth:`~graintrace.NearFieldMeshBuilder.mesh` produce ``mesh.e`` and
   ``orientations.csv``.
#. Build the FF residual strain: ``build_voronoi(generate_mesh=False, ...)``
   produces ``reconstruction_cpfe_ee.csv``.
#. Run CPFE with the NF mesh, NF orientations, and the FF strain field.

Recipe
------

Shift the FF elastic-strain file into the NF frame when the two are not
co-registered:

.. code-block:: python

   import pandas as pd

   ff_translation = (dx, dy, dz)   # from experiment geometry
   ee = pd.read_csv("out/ff_reconstruction/reconstruction_cpfe_ee.csv",
                    header=None, index_col=False)
   ee.iloc[:, 0] += ff_translation[0]
   ee.iloc[:, 1] += ff_translation[1]
   ee.iloc[:, 2] += ff_translation[2]
   ee.to_csv("out/ff_reconstruction/reconstruction_cpfe_ee_shifted.csv",
             index=False, header=False)

Run the simulation:

.. code-block:: python

   from graintrace.run_cpfe_simulation import CPFESimulation

   sim = CPFESimulation(
       mesh_file="out/nf_reconstruction/mesh.e",
       save_simulation_folder="out/simulation",
       eeres_file="out/ff_reconstruction/reconstruction_cpfe_ee_shifted.csv",
       ori_file="out/nf_reconstruction/orientations.csv",
       dim=3, element_order="FIRST",           # NF meshes are typically FIRST order
       moose_run_file="external/puma/puma-opt",   # your built puma-opt
       use_ff_initial_field=False,             # ee comes from a DIFFERENT mesh
   )
   sim.set_parameters("material", ...)         # see :doc:`material-calibration`
   sim.set_parameters("simulation_parameters",
                      device="cuda:0", device_batch=1000,
                      dt=0.5, total_time=2.0, initialize_time=1.0, sync_times="2.0")
   sim.set_parameters("boundary", bounding_box=nf_bbox, bc={...})
   sim.set_parameters("grid_properties",
                      number_of_elements=[10, 10, 10], bounding_box=grid_bb)
   sim.run(ncore=8)

Key parameters
--------------

- ``use_ff_initial_field`` — ``False`` when the elastic-strain file and the mesh
  are different meshes (FF strain on an NF mesh); ``True`` only when mesh and
  strain file are co-registered FF.
- FF-to-NF shift — add ``(dx, dy, dz)`` to the first three columns of
  ``reconstruction_cpfe_ee.csv`` to align frames before passing it as
  ``eeres_file``.
- ``nf_bbox`` — derive from the NF ``merged_segmented_fixed_grid.npy`` coordinates
  (see :doc:`../configuration`).
- ``sculpt_config`` — configures the NF hex mesh (see :doc:`../configuration`).
- CPFE material, simulation, boundary, and grid parameters follow
  :doc:`cpfe-simulation`.

Gotchas
-------

- The same NEML2 v3 AOTI notes as :doc:`cpfe-simulation` apply: material
  parameters are baked into the model and compiled on ``run()``; there are no
  runtime schedulers; a device list maps to MPI ranks.
- NF reconstruction uses multiprocessing, so the driver must sit under an
  ``if __name__ == "__main__":`` guard.
- ``moose_run_file`` must point at your built v3 ``puma-opt``.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_cpfe_nfff.py
   :language: python
   :caption: examples/demonstrate_cpfe_nfff.py
