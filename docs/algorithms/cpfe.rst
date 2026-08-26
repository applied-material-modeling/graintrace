Crystal-plasticity FE simulation
================================

Overview
--------
Crystal-plasticity finite-element (CPFE) solution of a reconstructed microstructure under
prescribed boundary conditions, producing per-element field histories (stress, strain, Nye
tensor, orientation). The stage is driven by :class:`~graintrace.CPFESimulation`, which writes
the MOOSE input decks, bakes the material parameters into a NEML2 v3 model, AOTI-compiles it with
``neml2-compile``, and launches the MOOSE/PUMA solver (``puma-opt``).

Method
------
The material response at each quadrature point is a NEML2 v3 single-crystal plasticity model.
graintrace configures its standard pieces:

- **Elasticity**: a cubic (or isotropic) linear elastic law relating elastic strain to stress,
  set by ``elastic_E``, ``elastic_nu``, ``elastic_G``.
- **Slip-rate power law**: the shear rate on slip system :math:`\alpha` follows

  .. math::

      \dot{\gamma}^{\alpha} = \dot{\gamma}_0 \,
      \left| \frac{\tau^{\alpha}}{\tau_c^{\alpha}} \right|^{n}
      \operatorname{sgn}(\tau^{\alpha}),

  with resolved shear stress :math:`\tau^{\alpha}`, reference rate :math:`\dot{\gamma}_0`
  (``power_slip_g0``), rate exponent :math:`n` (``power_slip_n``), and slip resistance
  :math:`\tau_c^{\alpha}`.
- **Voce hardening**: the slip resistance evolves from an initial strength
  (``slip_constant_strength``) toward a saturation value (``voce_hardening_saturation``) with
  initial slope ``voce_hardening_initial_slope``, driven by accumulated slip.
- **Orientation rate**: the crystal lattice reorients with the plastic spin; orientations are
  carried and output as NEML2 v3 MRP / Rodrigues.

MOOSE/PUMA assembles these point-wise responses into the global finite-element equilibrium solve
over the mesh, advancing in time as the load ramps between ``initialize_time`` and
``total_time``. graintrace sets up the model, parameters, mesh, and boundary conditions; NEML2
evaluates the constitutive model and MOOSE/PUMA performs the FE solve.

Algorithm
---------
1. Load the mesh and per-element MRP orientations (and optional initial elastic-strain field).
2. Write the MOOSE input decks (main run, initial conditions, transfer, grid output) from the
   templates in ``cpfe_base/``.
3. Bake the material parameters into the NEML2 model and AOTI-compile it with ``neml2-compile``
   for the target device(s) (``recompile`` rebuilds the ``.pt2`` when parameters change).
4. Launch ``puma-opt`` under the configured launcher (``mpiexec``/``srun``); multi-GPU runs map
   to MPI ranks over a device list.
5. Collect outputs: the native-mesh Exodus, per-element CSVs (``mesh_out/``), and, if requested,
   regular-grid CSVs (``grid_out/``) at the configured sync/step frequency.

Parameters that matter
----------------------
See :doc:`/configuration` for the full list.

- ``material``: ``slip_constant_strength``, ``voce_hardening_initial_slope``,
  ``voce_hardening_saturation``, ``power_slip_n``, ``power_slip_g0``, ``elastic_E`` /
  ``elastic_nu`` / ``elastic_G``, ``burger_scale``.
- ``simulation_parameters``: ``dt``, ``total_time``, ``initialize_time``, ``sync_times``,
  ``device`` / ``device_batch``, and the output-frequency knobs ``grid_transfer`` /
  ``exodus_output`` / ``mesh_csv``.
- ``boundary``: ``bounding_box`` and the ``bc`` dict (displacement or ``stress_free`` per face).
- ``grid_properties``: ``number_of_elements`` and the inset ``bounding_box`` for regular-grid
  output.

Further details
---------------
For the full constitutive-model formulation, see the NEML2 documentation:
https://applied-material-modeling.github.io/neml2/ . For the finite-element framework and the
nonlinear/transient solve, see the MOOSE documentation: https://mooseframework.inl.gov/ .

See also
--------
- Tutorial: :doc:`/tutorials/cpfe-simulation`
- API: :class:`~graintrace.CPFESimulation`
