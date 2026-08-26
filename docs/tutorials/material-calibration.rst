Material calibration
=====================

Calibrate six crystal-plasticity parameters (elastic E, nu, G, slip strength,
Voce hardening slope and saturation) to a macroscopic stress-strain curve plus
full-field elastic strains. Fitting uses a neml2 v3 + pyzag analytic-adjoint
Taylor model driven by LBFGS, through :class:`~graintrace.MaterialCalibration`
and :class:`~graintrace.TaylorModel`. Run this to fit material parameters before
CPFE.

**Needs:** NEML2 v3 only (no MOOSE); CUDA optional. See :doc:`../install`.

Inputs
------

A folder of per-stress-level CSVs with the orientation matrix ``O11..O33``,
``eKen11..eKen33``, and ``GrainRadius``, plus a ``strain-stress.csv`` (two
columns: strain, stress). The repo ships ``mwe_data/ff_calibration/`` (9 load
steps of 500 grains). For a physically registered fit, first run
:doc:`experiment-rotation`.

Recipe
------

.. code-block:: python

   from pathlib import Path
   import graintrace as _gt
   from graintrace.material_calibration import MaterialCalibration
   from graintrace.taylor import TaylorModel

   _cpfe_base = str(Path(_gt.__file__).parent / "cpfe_base")

   calib = MaterialCalibration(
       model_class=TaylorModel,
       model_args=dict(
           neml2_path=_cpfe_base + "/neml2_cpfe_calibration.i",
           npoints=30, nchunk=2, device="cuda", compile=False,
       ),
       data_args=dict(
           data_dir="mwe_data/ff_calibration",
           strain_stress_file="mwe_data/ff_calibration/strain-stress.csv",
           npoints=30, full_field_strain_units="microstrain", straintype="eKen",
           max_strain=0.006, n_grains=100, seed=42,
       ),
       save_dir="out/material_calibration",
       apply_elastic_correction=False,
       strain_window=(0.0, 0.0015),
   )
   calib.plot_texture(direction=[1, 1, 1])
   calib.plot_stress_strain()
   calib.calibrate(maxiter=15, lr=0.3, max_iter_per_step=6,
                   line_search_fn="strong_wolfe",
                   plateau_rtol=1e-3, plateau_window=2)   # guard stops early
   calib.load("out/material_calibration/calibrated_material.json")
   calib.plot_stress_strain(include_model=True)
   calib.plot_strain_histogram(include_initial_strain=True)

Key parameters
--------------

- model: ``device`` (``"cuda"`` or ``"cpu"``), ``npoints`` (pyzag time steps),
  ``nchunk`` (chunk size for the bidiagonal-in-time solve).
- data: ``n_grains`` (subsample per load step; ``None`` for all), ``max_strain``
  (macro-curve cap), ``straintype`` (``"eKen"`` or ``"eFab"``).
- ``calibrate``: ``maxiter`` is an upper bound; the plateau guard
  (``plateau_rtol``, ``plateau_window``) stops early when the relative loss
  improvement stalls.

Outputs (in ``save_dir``)
-------------------------

- ``calibrated_material.json`` and ``autosave_material.json``.
- Pole figures, stress-strain overlays, and elastic-strain histograms.

Map the ``TaylorModel.opt_vars`` to CPFE material names to feed the ``material``
dict in :doc:`cpfe-simulation`: ``elastic_tensor_E`` to ``elastic_E``,
``elastic_tensor_G`` to ``elastic_G``, ``elastic_tensor_nu`` to ``elastic_nu``,
``slip_strength_constant_strength`` to ``slip_constant_strength``,
``voce_hardening_initial_slope``, and ``voce_hardening_saturated_hardening`` to
``voce_hardening_saturation``.

Gotchas
-------

- CUDA works because ``taylor.py`` moves the whole nonlinear system with
  ``nsys.to(device)``. A cuda/cpu mismatch surfaces as a silent ``loss=inf`` and
  means the model stayed on CPU; do not reintroduce a ``torch.set_default_device``
  hack.
- v3 mixed-control uses an unweighted grain mean. Some parameters can hit
  reparametrization bounds on small demo configs (few grains, narrow window); use
  more grains and a wider window for production fits.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_material_calibration.py
   :language: python
   :caption: examples/demonstrate_material_calibration.py
