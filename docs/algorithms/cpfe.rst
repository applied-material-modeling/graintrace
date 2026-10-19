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
Plastic flow is carried entirely by crystallographic slip: the lattice itself only stretches
elastically and rotates, and every bit of permanent shape change comes from dislocations
gliding on specific planes in specific directions. For a cubic metal that is the
:math:`\{111\}\langle 110 \rangle` family, supplied by the ``CubicCrystal`` geometry.

Symbols follow :doc:`/notation`.

Resolved shear stress
~~~~~~~~~~~~~~~~~~~~~
Elasticity is a cubic (or isotropic) linear law on the elastic strain,
:math:`\boldsymbol{\sigma} = \mathbb{C} : \boldsymbol{\varepsilon}^{e}`, with
:math:`\mathbb{C}` built from ``elastic_E``, ``elastic_nu`` and ``elastic_G``. Three constants
rather than two is what makes it *cubic*: an isotropic material satisfies
:math:`G = E / 2(1+\nu)`, and the departure from that identity is the elastic anisotropy
ratio, which for most metals is far from 1 and matters at grain boundaries.

Each slip system :math:`\alpha` has a unit slip direction :math:`\mathbf{d}^{\alpha}` and a
unit plane normal :math:`\mathbf{n}^{\alpha}`, both fixed **in the crystal frame**. The stress
the system feels is the Cauchy stress projected onto the Schmid tensor, after that tensor has
been rotated into the sample frame by the orientation matrix :math:`Q` (``ResolvedShear``):

.. math::
   :label: cpfe-resolved-shear

   \tau^{\alpha} = \boldsymbol{\sigma} :
     \Bigl[\, Q \operatorname{sym}\!\bigl(\mathbf{d}^{\alpha} \otimes \mathbf{n}^{\alpha}\bigr)
       Q^{\mathsf T} \Bigr].

The :math:`Q \,\cdot\, Q^{\mathsf T}` sandwich is the whole coupling between microstructure
and mechanics. Two grains under identical macroscopic stress see different
:math:`\tau^{\alpha}` purely because :math:`Q` differs, so which systems yield, how much they
slip, and how the grain hardens are all decided by orientation. Everything the reconstruction
contributes to the simulation enters through :eq:`cpfe-resolved-shear`, and every orientation
convention error in the pipeline shows up here as a grain deforming the wrong way.

Flow rule
~~~~~~~~~
The shear rate on system :math:`\alpha` follows a power law
(``PowerLawSlipRule``):

.. math::
   :label: cpfe-slip-rule

      \dot{\gamma}^{\alpha} = \dot{\gamma}_0 \,
      \left| \frac{\tau^{\alpha}}{\hat{\tau}^{\alpha}} \right|^{\,n-1}
      \frac{\tau^{\alpha}}{\hat{\tau}^{\alpha}},

with reference rate :math:`\dot{\gamma}_0` (``power_slip_g0``), rate exponent :math:`n`
(``power_slip_n``), and slip resistance :math:`\hat{\tau}^{\alpha}`. The
:math:`|\cdot|^{n-1}(\cdot)` form is how NEML2 writes it and is what the code evaluates; for
positive :math:`\hat{\tau}^{\alpha}` it equals the more familiar
:math:`\dot{\gamma}_0 |\tau^{\alpha}/\hat{\tau}^{\alpha}|^{n}
\operatorname{sgn}(\tau^{\alpha})`, but it is written this way so the derivative stays
continuous through :math:`\tau^{\alpha} = 0` instead of needing a sign function.

:eq:`cpfe-slip-rule` is a **viscoplastic regularisation of a yield criterion**, and that is
the point of it. Rate-independent crystal plasticity has to decide which subset of slip
systems is active, a combinatorial problem with a non-unique answer for a cubic crystal
(the Taylor ambiguity). The power law sidesteps it: every system always slips a little, and
as :math:`n \to \infty` the response approaches the rate-independent limit. So :math:`n` is
not a fitted material property so much as a numerical choice.

.. important::

   :math:`n` controls the stiffness of the integration directly. The Jacobian of
   :eq:`cpfe-slip-rule` scales as :math:`n/\hat{\tau}`, so raising :math:`n` for a sharper
   yield tightens the Newton solve at every quadrature point. ``power_slip_n = 20`` in the
   shipped deck and ``25`` in the documented recipe are already firm; going much above that
   trades convergence for a yield corner you probably cannot measure.

Hardening
~~~~~~~~~
The deck uses ``SingleSlipStrengthMap``, so **all systems share one scalar** resistance:

.. math::
   :label: cpfe-strength

   \hat{\tau}^{\alpha} = \bar{\tau} + \tau_0 \quad \text{for every } \alpha ,

with :math:`\tau_0` = ``slip_constant_strength`` a fixed offset and :math:`\bar{\tau}` the
single evolving hardening variable. There is no latent hardening and no system-to-system
interaction matrix; slip on one system hardens all of them equally. That is a real modelling
simplification, and it is why this model reproduces a macroscopic stress-strain curve well
while under-predicting the texture-dependent spread between grains.

:math:`\bar{\tau}` follows a Voce law (``VoceSingleSlipHardeningRule``) driven by the total
slip rate :math:`\dot{\gamma}_{\mathrm{tot}} = \sum_{\alpha} |\dot{\gamma}^{\alpha}|`:

.. math::
   :label: cpfe-voce

   \dot{\bar{\tau}} = \theta_0 \left( 1 - \frac{\bar{\tau}}{\tau_f} \right)
     \dot{\gamma}_{\mathrm{tot}},

with initial slope :math:`\theta_0` = ``voce_hardening_initial_slope`` and saturation
:math:`\tau_f` = ``voce_hardening_saturation``.

.. warning::

   :math:`\tau_f` is the saturated value of :math:`\bar{\tau}` alone, and :math:`\tau_0` is
   added on top of it by :eq:`cpfe-strength`. The slip resistance therefore saturates at
   :math:`\tau_0 + \tau_f`, not at :math:`\tau_f`. With the documented recipe's
   :math:`\tau_0 = 100` and :math:`\tau_f = 220` MPa the resistance runs from 100 to 320 MPa.
   Setting ``voce_hardening_saturation`` to a measured saturation stress overshoots by
   :math:`\tau_0`. (NEML2's own parameter description for ``saturated_hardening`` calls it
   "the final, saturated value of the slip system strength", which is loose: it is the
   saturated value of the evolving variable, before the constant offset.)

Kinematics and reorientation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Slip produces both a stretching and a spin, obtained from the symmetric and skew parts of the
same rotated Schmid tensor (``PlasticDeformationRate`` and ``PlasticVorticity``):

.. math::
   :label: cpfe-plastic-kinematics

   d^{p} = \sum_{\alpha} \dot{\gamma}^{\alpha}\,
       Q \operatorname{sym}\!\bigl(\mathbf{d}^{\alpha} \otimes \mathbf{n}^{\alpha}\bigr)
       Q^{\mathsf T},
   \qquad
   w^{p} = \sum_{\alpha} \dot{\gamma}^{\alpha}\,
       Q \operatorname{skew}\!\bigl(\mathbf{d}^{\alpha} \otimes \mathbf{n}^{\alpha}\bigr)
       Q^{\mathsf T}.

The elastic strain and the lattice orientation then evolve as (``ElasticStrainRate``,
``OrientationRate``)

.. math::
   :label: cpfe-state-rates

   \dot{\varepsilon} = d - d^{p} - \varepsilon w + w \varepsilon,
   \qquad
   \Omega^{e} \equiv \dot{Q} Q^{\mathsf T}
     = w - w^{p} - \varepsilon d^{p} + d^{p} \varepsilon,

where :math:`d` and :math:`w` are the total deformation rate and vorticity and
:math:`\varepsilon` is the elastic strain. The leading :math:`w - w^{p}` is the familiar
statement that the lattice takes up whatever spin the slip systems do not; the
:math:`\varepsilon d^{p}` pair is the finite-elastic-strain correction, small for metals but
not dropped.

That first term is the mechanism behind texture evolution and intragranular fragmentation. A
grain whose active systems produce a plastic spin that does not match the imposed one must
rotate its lattice to compensate, and neighbouring regions of the same grain with different
active sets rotate differently. :doc:`/algorithms/segmentation` and the reorientation REI
criterion both measure the result.

The state variables are integrated implicitly: backward Euler for the elastic strain and the
slip hardening, and an implicit **exponential** map for the orientation. The exponential
update is not a refinement -- integrating :math:`\dot{Q} = \Omega^{e} Q` additively would
take :math:`Q` off :math:`SO(3)` and accumulate a spurious stretch over a long load history.

.. note::

   The orientation state is stored and output as **NEML2 v3 MRP**,
   :math:`\tan(\theta/4)\,\hat{\mathbf{e}}`. The output columns are named
   ``ori_rodrigues_{x,y,z}``, which is a misnomer: they do **not** hold classical Rodrigues
   (Gibbs) vectors, :math:`\tan(\theta/2)\,\hat{\mathbf{e}}`. Every in-repository consumer
   treats them correctly as MRP; a third-party tool expecting Rodrigues will silently
   misinterpret them, and the error grows with rotation angle. See
   :ref:`notation-ori-rodrigues`.

MOOSE/PUMA assembles these point-wise responses into the global finite-element equilibrium solve
over the mesh, advancing in time as the load ramps between ``initialize_time`` and
``total_time``. graintrace sets up the model, parameters, mesh, and boundary conditions; NEML2
evaluates the constitutive model and MOOSE/PUMA performs the FE solve.

Algorithm
---------
1. Load the mesh and per-block MRP orientations (and optional initial elastic-strain field).
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
