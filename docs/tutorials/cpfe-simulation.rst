CPFE simulation
===============

Run a MOOSE/PUMA CPFE simulation on a far-field Voronoi reconstruction with a
NEML2 v3 AOTI-compiled crystal-plasticity model, using
:class:`~graintrace.CPFESimulation`. Given a mesh and per-grain orientations
(and an optional FF residual-strain field), it bakes the material parameters into
the model, runs ``neml2-compile``, and launches ``puma-opt``.

**Needs:** MOOSE ``puma-opt`` and ``neml2-compile`` (AOTI); CUDA recommended.
See :doc:`../install`.

Inputs
------

A hex (``.e``) or tet (``.msh``) mesh plus per-grain orientations; CPFE runs on
either. The repo ships ``mwe_data/cpfe_ff/`` (``reconstruction.msh`` +
``orientations.dat``, 10 grains). This sample is a GMSH tet mesh, but the
recommended FF route is a SCULPT hex mesh (see :doc:`meshing`). Orientations must
be neml2 MRP; convert the FF Euler ``orientations.dat`` (degrees) with
``orientation_helper.euler_to_mrp``.

Recipe
------

.. code-block:: python

   import os, numpy as np, torch, meshio
   from graintrace.run_cpfe_simulation import CPFESimulation
   from graintrace import orientation_helper as oh

   ff = "mwe_data/cpfe_ff"; out = "cpfe_out"; os.makedirs(out, exist_ok=True)
   euler = np.loadtxt(ff + "/orientations.dat")                # Euler-bunge, degrees
   mrp = oh.euler_to_mrp(torch.tensor(euler, dtype=torch.float64), "bunge", "degrees")
   np.savetxt(out + "/orientations_MRP.dat", mrp.numpy(), fmt="%.12g")

   m = meshio.read(ff + "/reconstruction.msh"); P = m.points
   bbox = [float(P[:, 0].min()), float(P[:, 0].max()),
           float(P[:, 1].min()), float(P[:, 1].max()),
           float(P[:, 2].min()), float(P[:, 2].max())]
   order = "SECOND" if any("10" in c.type for c in m.cells) else "FIRST"

   sim = CPFESimulation(
       mesh_file=ff + "/reconstruction.msh", save_simulation_folder=out,
       moose_run_file="external/puma/puma-opt",   # your built puma-opt
       element_order=order, eeres_file=None, ori_file=out + "/orientations_MRP.dat",
       dim=3, use_ff_initial_field=True,           # eeres_file=None -> 12-col zero ee
   )
   sim.set_parameters("material", slip_constant_strength=130.0,
       voce_hardening_initial_slope=1556.09, voce_hardening_saturation=100.0,
       power_slip_n=20, power_slip_g0=1e-4, elastic_E=209016.0, elastic_nu=0.307,
       elastic_G=60355.0, burger_scale=2.22)
   sim.set_parameters("simulation_parameters", device="cuda:0", device_batch=20000,
       dt=0.5, total_time=2.0, initialize_time=1.0, sync_times="2.0",
       grid_transfer="final", exodus_output="sync")

   disp = 0.002 * (bbox[5] - bbox[4])
   sim.set_parameters("boundary", bounding_box=bbox, bc={
       "x": {"negative": "stress_free", "positive": "stress_free"},
       "y": {"negative": "stress_free", "positive": "stress_free"},
       "z": {"negative": 0, "positive": disp}})
   grid_bb = list(bbox)
   for i in range(0, 6, 2): grid_bb[i] += 1e-4
   for i in range(1, 6, 2): grid_bb[i] -= 1e-4
   sim.set_parameters("grid_properties", number_of_elements=[10, 10, 10],
                      bounding_box=grid_bb)

   sim.run(ncore=4)   # ncore == mpiexec -n; also spreads a device list over ranks

Key parameters
--------------

- ``device`` — ``"cpu"``, ``"cuda:0"``, or a space-separated list
  ``"cuda:0 cuda:1"`` (multi-GPU over MPI ranks).
- ``device_batch`` — per-device NEML2 chunk (quad points per call); a finite
  value caps GPU memory (0 means the whole batch, risking OOM on large meshes).
- ``initialize_time`` — the load ramps from ``initialize_time`` to
  ``total_time``; ``sync_times`` are the grid-output times.
- ``use_ff_initial_field=True`` with a real ``eeres_file`` applies an FF residual
  strain (12 columns: x, y, z plus 9); ``eeres_file=None`` writes a 12-column
  zero ee.
- ``grid_transfer`` and ``exodus_output`` default to the cheap settings (transfer
  only at the last step, Exodus only at ``sync_times``). Use ``"per_step"`` for
  every-step grid REI output or Exodus writes; with ``grid_transfer`` other than
  ``"per_step"``, regenerate grid CSVs offline with
  :class:`~graintrace.GridResampler`.
- AOTI: material parameters are baked into the model ``.i`` and ``neml2-compile``d
  on ``run()``; ``recompile=True`` (default) rebuilds when parameters change.

See :doc:`../configuration` for the ``bc`` dict and inset ``grid_properties``
bounding box.

Outputs (in ``save_simulation_folder/simulation_out``)
------------------------------------------------------

- ``out.csv`` — per-block time series.
- ``sim_output.e`` / ``sim_output_grid.e`` — Exodus output.
- ``grid_out/*.csv`` — per-grid fields (cauchy_stress, ee, nye_tensor,
  ori_rodrigues), for post-processing and REI.

Gotchas
-------

- v3 has no runtime ``[NEML2] cli_args`` or ``[Schedulers]``; do not pass
  ``scheduler_name``.
- For neml2-dominated CPFE, use fewer MPI ranks (about one per GPU) for bigger
  per-rank batches.
- Stiff first steps may make MOOSE cut ``dt`` and recover; that is normal, not a
  failure.
- The environment, ``LD_LIBRARY_PATH``, and ``neml2_load_files`` auto-derive from
  the ``moose_run_file`` repo layout.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_cpfe.py
   :language: python
   :caption: examples/demonstrate_cpfe.py
