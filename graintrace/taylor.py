# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Uniaxial Taylor crystal-plasticity forward + calibration model (NEML2 v3).

A mixed-control NEML2 ``NonlinearSystem`` (``cpfe_base/neml2_cpfe_calibration.i``)
is wrapped by :class:`neml2.pyzag.NEML2PyzagFactory` and driven through
:func:`pyzag.nonlinear.solve_adjoint` for analytic parameter gradients.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import scipy.interpolate as inter
import torch
from torch import nn

import neml2
from neml2.pyzag import NEML2PyzagFactory
from pyzag import chunktime, nonlinear

from .base_material_approximation import BaseMaterialApproximationModel
from . import orientation_helper


# Flat mixed-control state layout of cpfe_base/neml2_cpfe_calibration.i:
#   BLOCK group (per grain, 10): elastic_strain(6) + orientation MRP(3) + slip_hardening(1)
#   DENSE group (12):            deformation_rate(6) + target_cauchy_stress(6)
# Flat trajectory size per step = n_grains * _PER_GRAIN_BASE + _DENSE_BASE.
_PER_GRAIN_BASE = 10
_DENSE_BASE = 12
_N_ELASTIC_STRAIN = 6


class UniaxialTaylorModel(nn.Module):
    """Differentiable uniaxial Taylor forward via a NEML2 mixed-control eq_sys.

    Wraps a :class:`neml2.pyzag.NEML2PyzagFactory` and integrates the strain
    history with ``pyzag.nonlinear.solve_adjoint``. Uniaxial loading is expressed
    with NEML2's ``MixedControlSetup``: ``control`` selects the axial component as
    strain-controlled (``prescribed = rate``); the remaining components are
    stress-free, and a global constraint forces ``target_cauchy_stress`` to equal
    the grain-mean per-crystal stress.
    """

    def __init__(
        self,
        factory: NEML2PyzagFactory,
        axial_index: int = 2,
        nchunk: int = 5,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        linesearch_iter: int = 5,
    ):
        super().__init__()
        self.factory = factory
        self.axial_index = axial_index
        self.nchunk = nchunk
        self.rtol = rtol
        self.atol = atol
        self.linesearch_iter = linesearch_iter

    def _make_solver(self):
        return nonlinear.RecursiveNonlinearEquationSolver(
            self.factory,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
            direct_solve_operator=chunktime.BidiagonalThomasFactorization,
            nonlinear_solver=chunktime.ChunkNewtonRaphsonLineSearch(
                rtol=self.rtol, atol=self.atol, linesearch_iter=self.linesearch_iter
            ),
        )

    def forward(
        self,
        orientations: torch.Tensor,
        strain: torch.Tensor,
        rate: float,
        initial_strains: torch.Tensor | None = None,
        return_state: bool = False,
    ):
        """Integrate the uniaxial history and return the macro stress trajectory.

        Args:
            orientations: (n_grains, 3) NEML2 MRP orientations.
            strain: (ntime,) axial strain grid (mm/mm), typically starting at 0.
            rate: assumed strain rate (1/s); ``t = strain / rate``.
            initial_strains: optional (n_grains, 6) initial elastic strain.
            return_state: also return per-grain elastic strain history.

        Returns:
            macro cauchy stress ``(ntime, 6)`` (SR2 Voigt: xx, yy, zz, yz, xz, xy);
            if ``return_state``, also ``(ntime, n_grains, 6)`` elastic strain.
        """
        orientations = torch.as_tensor(orientations, dtype=torch.float64)
        device = orientations.device
        dtype = orientations.dtype
        strain = torch.as_tensor(strain, dtype=dtype, device=device)

        n_grains = orientations.shape[0]
        ntime = strain.shape[0]
        nbatch = 1

        if initial_strains is None:
            es0 = torch.zeros(nbatch, n_grains, 6, device=device, dtype=dtype)
        else:
            es0 = torch.as_tensor(initial_strains, device=device, dtype=dtype).reshape(
                nbatch, n_grains, 6
            )

        ic_dict = {
            "elastic_strain": es0,
            "orientation": orientations.unsqueeze(0),
            "slip_hardening": torch.zeros(nbatch, n_grains, device=device, dtype=dtype),
            "deformation_rate": torch.zeros(nbatch, 6, device=device, dtype=dtype),
            "target_cauchy_stress": torch.zeros(nbatch, 6, device=device, dtype=dtype),
        }
        y0 = self.factory.assemble_state(ic_dict, dynamic_dim=1)

        control_vec = torch.zeros(6, device=device, dtype=dtype)
        control_vec[self.axial_index] = 1.0
        control = control_vec.reshape(1, 1, 6).expand(ntime, nbatch, 6).contiguous()
        prescribed = torch.zeros(ntime, nbatch, 6, device=device, dtype=dtype)
        prescribed[..., self.axial_index] = rate
        times = (strain / rate).reshape(ntime, 1).expand(ntime, nbatch).contiguous()
        forces_dict = {
            "control": control,
            "prescribed": prescribed,
            "t": times,
            "vorticity": torch.zeros(ntime, nbatch, 3, device=device, dtype=dtype),
        }
        forces = self.factory.assemble_forces(forces_dict, dynamic_dim=2)

        solver = self._make_solver()
        result = nonlinear.solve_adjoint(solver, y0, ntime, forces)
        # result: (ntime, nbatch, nstate_flat)

        expected = n_grains * _PER_GRAIN_BASE + _DENSE_BASE
        if result.shape[-1] != expected:
            raise ValueError(
                f"Unexpected flat state size {result.shape[-1]}; expected {expected} "
                f"for n_grains={n_grains}."
            )

        # target_cauchy_stress = last 6 (the macro/aggregate stress).
        macro_stress = result[..., -_DENSE_BASE + 6 :].squeeze(1)  # (ntime, 6)

        if not return_state:
            return macro_stress

        # per-grain elastic strain = first 6 of each 10-wide grain block.
        block = result[..., : n_grains * _PER_GRAIN_BASE].reshape(
            ntime, nbatch, n_grains, _PER_GRAIN_BASE
        )
        elastic_strain = block[..., :_N_ELASTIC_STRAIN].squeeze(
            1
        )  # (ntime, n_grains, 6)
        return macro_stress, elastic_strain


class TaylorModel(BaseMaterialApproximationModel):
    """Taylor calibration model backed by the NEML2 v3 + pyzag adjoint engine.

    - Loads and preprocesses experimental data (per-stress-level CSVs).
    - Manages the six calibration parameters as NEML2/torch parameters.
    - Runs differentiable uniaxial stress-strain simulations for optimization.
    """

    # NEML2 parameters present on the model but held fixed during calibration.
    DEFAULT_EXCLUDE_PARAMETERS = ["slip_rule_gamma0", "slip_rule_n"]

    def __init__(
        self,
        neml2_path: str,
        neml2_model_name: str = "model_with_stress",
        axial_index: int = 2,
        assumed_rate: float = 1.0e-4,
        npoints: int = 500,
        nchunk: int = 5,
        equation_system: str = "eq_sys",
        exclude_parameters: list[str] | None = None,
        compile: bool = False,  # pylint: disable=redefined-builtin  # NEML2 factory kwarg name
        device: str | torch.device = "cpu",
    ):
        # pylint: disable=super-init-not-called  # v3 path wraps a NonlinearSystem, see below
        # Deliberately does NOT call BaseMaterialApproximationModel.__init__;
        # the v3 path loads a NonlinearSystem and wraps it in a pyzag factory.
        self.neml2_path = neml2_path
        self.neml2_model_name = neml2_model_name
        self.equation_system = equation_system
        self.axial_index = axial_index
        self.assumed_rate = assumed_rate
        self.npoints = npoints
        self.device = torch.device(device)

        if exclude_parameters is None:
            exclude_parameters = list(self.DEFAULT_EXCLUDE_PARAMETERS)

        # Move the whole nonlinear system to device first: factory.to() only
        # relocates the factory's own parameters, leaving the model's internal
        # crystal-geometry buffers (Schmid tensors, etc.) on CPU -> device
        # mismatch on cuda. nsys.to() moves all model buffers.
        nsys = neml2.load_nonlinear_system(neml2_path, equation_system)
        nsys.to(self.device)
        self.factory = NEML2PyzagFactory(
            nsys, exclude_parameters=exclude_parameters, compile=compile
        ).to(self.device)

        self.tmodel = UniaxialTaylorModel(
            self.factory, axial_index=axial_index, nchunk=nchunk
        )

        # Calibration parameters in fixed order (JSON save/load and p0 depend on it).
        self.opt_vars = [
            "elastic_tensor_E",
            "elastic_tensor_G",
            "elastic_tensor_nu",
            "slip_strength_constant_strength",
            "voce_hardening_initial_slope",
            "voce_hardening_saturated_hardening",
        ]

    def get_params(self) -> torch.Tensor:
        """Current values of the six calibration parameters (detached)."""
        return torch.stack(
            [getattr(self.factory, v).detach().reshape(()) for v in self.opt_vars]
        )

    def set_params(self, values) -> None:
        """Assign the six calibration parameters from a 1-D iterable."""
        values = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for v, pv in zip(self.opt_vars, values):
                getattr(self.factory, v).copy_(pv)
        # pylint: disable-next=protected-access  # required to push new values into NEML2
        self.factory._update_parameter_values()

    def load_experiment_data(  # pylint: disable=arguments-renamed  # richer v3 signature
        self,
        data_dir: str,
        strain_stress_file: str,
        full_field_strain_units: str | None = None,
        strain_stress_file_units: str | None = None,
        straintype: str = "eFab",
        npoints: int | None = None,
        max_strain: float | None = None,
        max_stress: float | None = None,
        n_grains: int | None = None,
        seed: int = 0,
    ):
        if npoints is None:
            npoints = self.npoints

        allowed_units = (None, "microstrain")
        if full_field_strain_units not in allowed_units:
            raise ValueError(
                f"Invalid full_field_strain_units: {full_field_strain_units!r}"
            )
        if strain_stress_file_units not in allowed_units:
            raise ValueError(
                f"Invalid strain_stress_file_units: {strain_stress_file_units!r}"
            )

        def try_parse_float(name):
            try:
                return float(name)
            except ValueError:
                return None

        # collect only numeric-named CSV files
        all_csvs = glob.glob(os.path.join(data_dir, "*.csv"))
        valid_files = []
        for f in all_csvs:
            stem = os.path.basename(f).split(".")[0]
            if try_parse_float(stem) is not None:
                valid_files.append(f)

        # sort by numeric stress value
        files = sorted(
            valid_files, key=lambda s: float(os.path.basename(s).split(".")[0])
        )
        stress_levels = [float(os.path.basename(f).split(".")[0]) for f in files]

        data = [pd.read_csv(f) for f in files]

        # Optionally subsample grains per file (deterministic via seed).
        if n_grains is not None:
            rng = np.random.default_rng(seed)
            subsampled = []
            for df in data:
                if n_grains < len(df):
                    idx = rng.choice(len(df), size=n_grains, replace=False)
                    subsampled.append(df.iloc[np.sort(idx)].reset_index(drop=True))
                else:
                    subsampled.append(df)
            data = subsampled

        strain_stress = np.loadtxt(strain_stress_file, delimiter=",")

        if (strain_stress_file_units or "").lower() == "microstrain":
            strain_stress[:, 0] *= 1e-6  # convert to mm/mm

        # Limit macro curve only
        if max_strain is not None and max_strain <= 0:
            raise ValueError("max_strain must be > 0")
        if max_stress is not None and max_stress <= 0:
            raise ValueError("max_stress must be > 0")

        if max_strain is not None:
            cutoff_strain = min(max_strain, float(strain_stress[-1, 0]))
        elif max_stress is not None:
            cutoff_stress = min(max_stress, float(strain_stress[-1, 1]))
            cutoff_strain = float(
                np.interp(cutoff_stress, strain_stress[:, 1], strain_stress[:, 0])
            )
        else:
            cutoff_strain = float(strain_stress[-1, 0])

        # Truncate and resample the stress-strain curve
        ifn = inter.interp1d(strain_stress[:, 0], strain_stress[:, 1], kind="linear")
        strain = np.linspace(0, cutoff_strain, npoints)
        stress = ifn(strain)
        strain_stress = np.stack((strain, stress), axis=1)

        # Conversion factor for full-field data
        factor = 1.0
        if (full_field_strain_units or "").lower() == "microstrain":
            factor = 1e-6

        # Full-field data: not filtered; orientations loaded as neml2 v3 MRPs.
        exp_strain = [
            orientation_helper.load_strains(d, factor=factor, field=straintype)
            for d in data
        ]
        exp_texture = [orientation_helper.load_orientations_mrp(d) for d in data]
        exp_weights = [orientation_helper.load_weights(d) for d in data]

        avg_exp_strain = [torch.mean(ds, dim=0) for ds in exp_strain]
        avg_axial_strain = [
            s[self.axial_index] - avg_exp_strain[0][self.axial_index]
            for s in avg_exp_strain
        ]
        use_weights = exp_weights[0]

        return {
            "files": files,
            "stress_levels": stress_levels,
            "strain_stress": strain_stress,
            "exp_strain": exp_strain,
            "exp_texture": exp_texture,
            "exp_weights": exp_weights,
            "avg_exp_strain": avg_exp_strain,
            "avg_axial_strain": avg_axial_strain,
            "use_weights": use_weights,
            "cutoff_strain": cutoff_strain,
        }

    def simulate(
        self,
        params,
        d: torch.Tensor | None = None,
        assumed_rate: float | None = None,
        experiment_data=None,
        return_state: bool = False,
        initial_strains=None,
    ):
        """Run a differentiable uniaxial stress-strain simulation.

        Args:
            params: optional 1-D tensor of the six calibration values. If given,
                they are assigned to the factory (no grad) before the solve;
                use this for plotting a specific parameter set. Pass ``None`` to
                run with the factory's current (possibly optimizer-owned)
                parameters so gradients flow (used during calibration).
            d: ignored (kept for API compatibility; the axial direction is set by
                ``self.axial_index``).
            assumed_rate: strain rate; defaults to ``self.assumed_rate``.
            experiment_data: dict from :meth:`load_experiment_data`.
            return_state: also return the per-grain elastic strain history.
            initial_strains: optional (n_grains, 6) initial elastic strain.

        Returns:
            ``stress_hist`` (npoints, 6), or ``(stress_hist, state_hist)`` where
            ``state_hist`` is (npoints, n_grains, 6) elastic strain.
        """
        if assumed_rate is None:
            assumed_rate = self.assumed_rate

        orientations = torch.as_tensor(
            experiment_data["exp_texture"][0], dtype=torch.float64
        ).to(self.device)
        strain = experiment_data["strain_stress"][:, 0]

        if initial_strains is not None:
            initial_strains = torch.as_tensor(initial_strains, dtype=torch.float64).to(
                self.device
            )

        if params is not None:
            self.set_params(params)

        return self.tmodel(
            orientations,
            strain,
            assumed_rate,
            initial_strains=initial_strains,
            return_state=return_state,
        )
