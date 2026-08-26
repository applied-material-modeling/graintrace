Material calibration
====================

Overview
--------
Calibration of the six crystal-plasticity parameters (elastic :math:`E`, :math:`G`,
:math:`\nu`; slip strength; Voce hardening slope and saturation) to a measured macroscopic
stress-strain curve and, optionally, full-field per-grain elastic strains. The stage is driven
by :class:`~graintrace.MaterialCalibration`, which wraps a :class:`~graintrace.TaylorModel`
forward model (NEML2 v3 + pyzag) and runs LBFGS with analytic-adjoint gradients.

Method
------
The forward model is a differentiable uniaxial Taylor aggregate: a NEML2 mixed-control
``NonlinearSystem`` integrated over the strain history by ``pyzag.nonlinear.solve_adjoint`` to
give the macroscopic Cauchy stress trajectory :math:`\sigma_{\mathrm{model}}(p)` for parameters
:math:`p`. Calibration minimizes the mean-squared error against the experimental macroscopic
stress, with an optional distribution-matched full-field elastic-strain term:

.. math::

    \mathcal{L}(p) = \frac{1}{N}\sum_{k=1}^{N}
    \bigl(\sigma_{\mathrm{model},k}(p) - \sigma_{\mathrm{exp},k}\bigr)^2
    \; + \; w_{\mathrm{ff}}\, \mathcal{L}_{\mathrm{ff}}(p),

where :math:`w_{\mathrm{ff}}` (``full_field_weight``, default 0) enables the per-grain
elastic-strain term and the macroscopic term is normalized before combining. pyzag supplies
**analytic-adjoint gradients** :math:`\partial \mathcal{L} / \partial p` through the time
integration (no finite differencing), which drive a torch **LBFGS** optimizer (strong-Wolfe
line search). Each parameter is reparametrized with pyzag ``RangeRescale`` onto its physical
range (``DEFAULT_PARAM_RANGES``), so the optimizer works on a bounded, well-scaled variable
while the model sees the true NEML2-unit value.

Algorithm
---------
1. Instantiate the forward model (``TaylorModel``) from the NEML2 calibration input; load the
   experimental macroscopic curve and per-grain elastic strains, subsampling grains/points.
2. Optionally apply the elastic-slope correction over ``strain_window``.
3. Register ``RangeRescale`` reparametrization on each of the six parameters.
4. Run LBFGS: each step runs the adjoint forward/backward to get the loss and its analytic
   gradient, then takes a line-search step; a plateau guard stops early once the relative loss
   improvement over ``plateau_window`` steps drops below ``plateau_rtol``.
5. Remove the reparametrization, save the calibrated parameters to JSON, and (optionally) plot
   the fitted stress-strain curve, strain histograms, and texture.

Parameters that matter
----------------------
See :doc:`/configuration` for the full list.

- ``model_args``: ``neml2_path``, ``npoints`` (pyzag time steps), ``nchunk`` (chunk size for the
  bidiagonal-in-time solve), ``device`` (``cuda`` recommended; the whole system is moved to the
  device).
- ``data_args``: ``data_dir`` / ``strain_stress_file``, ``straintype`` (``eKen``/``eFab``),
  ``full_field_strain_units``, ``max_strain``, ``n_grains``, ``seed``.
- ``calibrate`` knobs: ``maxiter``, ``lr``, ``max_iter_per_step``, ``line_search_fn``,
  ``plateau_rtol`` / ``plateau_window``, and ``full_field_weight`` / ``full_field_components``.
- ``apply_elastic_correction`` + ``strain_window``.

Further details
---------------
For the full constitutive model see the NEML2 documentation:
https://applied-material-modeling.github.io/neml2/ . For the analytic-adjoint time integration
and reparametrization, see the applied-material-modeling pyzag documentation:
https://applied-material-modeling.github.io/pyzag/ .

See also
--------
- Tutorial: :doc:`/tutorials/material-calibration`
- API: :class:`~graintrace.MaterialCalibration`, :class:`~graintrace.TaylorModel`
