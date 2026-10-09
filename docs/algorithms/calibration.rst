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
:math:`p`.

The Taylor assumption is that every grain sees the **same deformation gradient** as the
aggregate, and the macroscopic stress is the volume average of the grain stresses. It is the
stiff bound: it satisfies compatibility exactly and equilibrium not at all, so it
systematically over-predicts the flow stress of a real polycrystal. That is a deliberate trade.
A full CPFE forward model inside an optimisation loop is unaffordable, and the Taylor model is
cheap, differentiable, and wrong in a consistent direction -- so the parameters it returns are
a sound *starting point* for a CPFE run, not a finished calibration. Expect to see the CPFE
macroscopic curve sit below the Taylor fit at the same parameters.

Two losses
~~~~~~~~~~
Which loss runs depends entirely on whether ``full_field_weight`` is positive, and the two are
not on the same scale.

With ``full_field_weight = 0`` (the default) the loss is the **raw** mean-squared stress error,

.. math::
   :label: calib-macro-raw

    \mathcal{L}(p) = \frac{1}{N}\sum_{k=1}^{N}
    \bigl(\sigma_{\mathrm{model},k}(p) - \sigma_{\mathrm{exp},k}\bigr)^2 ,

in squared stress units. For a fit in MPa that puts :math:`\mathcal{L}` around
:math:`10^1`--:math:`10^4`.

With ``full_field_weight > 0`` the macroscopic term is divided by
:math:`\langle \sigma_{\mathrm{exp}}^2 \rangle` to make it dimensionless, and the full-field
term is added:

.. math::
   :label: calib-combined

    \mathcal{L}(p) = \underbrace{\frac{\sum_k (\sigma_{\mathrm{model},k} -
    \sigma_{\mathrm{exp},k})^2}{\sum_k \sigma_{\mathrm{exp},k}^2}}_{\text{relative macro}}
    \; + \; w_{\mathrm{ff}}\, \mathcal{L}_{\mathrm{ff}}(p).

.. important::

   Both terms of :eq:`calib-combined` are :math:`O(1)` and dimensionless, whereas
   :eq:`calib-macro-raw` is neither. Switching ``full_field_weight`` from :math:`0` to
   anything positive therefore rescales the loss by three to four orders of magnitude.
   ``lr`` and ``plateau_rtol`` do **not** transfer between the two modes -- ``plateau_rtol``
   is a *relative* improvement so it survives, but any absolute loss target does not, and a
   learning rate tuned on one will be badly wrong on the other.

The full-field term is **distribution-matched**, not grain-matched. For each experimental
stress level :math:`\ell` the model state is taken at the nearest simulated step, and each
strain component's sorted quantiles are compared:

.. math::
   :label: calib-fullfield

   \mathcal{L}_{\mathrm{ff}}(p) = \frac{1}{|L|\,|C|} \sum_{\ell \in L} \sum_{c \in C}
     \frac{\bigl\langle \bigl(q^{\mathrm{model}}_{\ell c}(\tau)
                           - q^{\mathrm{exp}}_{\ell c}(\tau)\bigr)^2 \bigr\rangle_\tau}
          {\bigl\langle q^{\mathrm{exp}}_{\ell c}(\tau)^2 \bigr\rangle_\tau + 10^{-30}},

where :math:`q(\tau)` is the :math:`\tau`-quantile over grains, evaluated at
``n_quantiles`` (default 64) points on :math:`[0, 1]`.

Comparing quantiles rather than grains is what makes this usable: **no grain correspondence is
required** between the model aggregate and the experiment, and the grain counts need not even
match. It reduces to a per-grain :math:`L^2` when the two sets are identically ordered and
equally sized. The cost is that it constrains only the *statistics* of the elastic strain
field -- a model that puts the right spread of strain on the wrong grains is scored perfectly.
Each (level, component) pair is normalised by its own experimental magnitude, so components of
very different size contribute equally; raw microstrain squared would otherwise be
:math:`O(10^{-6})` and inert.

Optimisation
~~~~~~~~~~~~
pyzag supplies **analytic-adjoint gradients** :math:`\partial \mathcal{L} / \partial p` through
the time integration -- no finite differencing, so the gradient cost is independent of the
number of parameters and does not inherit a step-size choice. These drive a torch **LBFGS**
optimizer with a strong-Wolfe line search.

Each parameter is reparametrized with pyzag ``RangeRescale`` onto its physical range, so the
optimizer works on a bounded, well-scaled variable while the model sees the true NEML2-unit
value. The default ranges are

.. list-table:: ``MaterialCalibration.DEFAULT_PARAM_RANGES``
   :header-rows: 1
   :widths: 48 26 26

   * - pyzag parameter
     - Range
     - Unit
   * - ``elastic_tensor_E``
     - :math:`[10^{4},\, 5\times10^{5}]`
     - MPa
   * - ``elastic_tensor_G``
     - :math:`[10^{3},\, 3\times10^{5}]`
     - MPa
   * - ``elastic_tensor_nu``
     - :math:`[-0.5,\, 0.5]`
     - --
   * - ``slip_strength_constant_strength``
     - :math:`[1,\, 2000]`
     - MPa
   * - ``voce_hardening_initial_slope``
     - :math:`[10^{-3},\, 5\times10^{4}]`
     - MPa
   * - ``voce_hardening_saturated_hardening``
     - :math:`[1,\, 2000]`
     - MPa

The rescaling matters more than it looks. These ranges span eight orders of magnitude, and
LBFGS builds a *single* inverse-Hessian approximation across all six directions at once, so
without the rescaling the problem is numerically ill-conditioned before any physics is
considered. The bounds also act as the only constraint in the fit: nothing else stops the
optimizer from proposing a negative hardening slope. Widen them only with a reason, and note
that ``elastic_tensor_nu`` admits negative values by design.

Note the names differ from the CPFE material names the result feeds; see §8 of the repository
``CLAUDE.md`` or the tutorial for the mapping.

A linear-algebra failure inside the forward solve -- a non-convergent Newton iteration at an
unphysical parameter set -- is caught, the gradient is zeroed, and the loss is returned as
``inf``, so the strong-Wolfe line search rejects that trial point and backs off rather than
aborting the run. A run whose reported loss is ``inf`` on the first iteration means the
*initial* parameters do not integrate, which the line search cannot fix; change the starting
point.

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
