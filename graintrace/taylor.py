from __future__ import annotations

import time

import neml2
from neml2.reserved import *

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import os
import scipy.interpolate as inter
from neml2.postprocessing import polefigure

from .base_material_approximation import BaseMaterialApproximationModel
from . import orientation_helper


# -------------------------------------------------------------------------
# PHYSICS
# -------------------------------------------------------------------------
class UniaxialTaylorModel(nn.Module):
    """Runs a NEML2 Taylor model in uniaxial strain control

    Args:
        model (neml.Model): A NEML2 material model

    Keyword args:
        spin (lambda, default zero): A function of time returning a tensor of size (3,) giving the rotational spin
    """

    def __init__(
        self,
        model,
        spin=lambda t: torch.zeros(
            3,
        ),
    ):
        super().__init__()
        self.model = model
        self.spin = spin

        self._setup_assemblers()

    def _setup_assemblers(self):
        """Setup the assemblers for the state and forces"""

        self.input_axis = self.model.input_axis()
        self.output_axis = self.model.output_axis()

        self.input_asm = neml2.VectorAssembler(self.input_axis)
        self.output_asm = neml2.VectorAssembler(self.output_axis)
        self.deriv_asm = neml2.MatrixAssembler(
            self.output_axis, self.input_axis.subaxis(STATE)
        )

        self.state_asm = neml2.VectorAssembler(self.input_axis.subaxis(STATE))
        self.old_state_asm = neml2.VectorAssembler(self.input_axis.subaxis(OLD_STATE))
        self.forces_asm = neml2.VectorAssembler(self.input_axis.subaxis(FORCES))

    @property
    def nstate(self):
        return self.model.input_axis().subaxis(STATE).size()

    @property
    def nforce(self):
        return self.model.input_axis().subaxis(FORCES).size()

    def initial_state(self, orientations, elastic_strain=None):
        """Assemble the initial state vector

        Args:
            orientations (torch.tensor): (n,3) tensor with initial orientations
        """
        if elastic_strain is None:
            elastic_strain = torch.zeros(
                (orientations.shape[0], 6), device=orientations.device
            )
        state_dict = {
            "old_state/elastic_strain": elastic_strain,
            "old_state/orientation": orientations,
        }
        for var, size in zip(
            self.input_axis.subaxis(STATE).variable_names(),
            self.input_axis.subaxis(STATE).variable_sizes(),
        ):
            if var not in ["elastic_strain", "orientation"]:
                state_dict["old_state/" + str(var)] = torch.zeros(
                    (orientations.shape[0], size), device=orientations.device
                )

        return self.old_state_asm.assemble_by_variable(state_dict).torch()

    def forward(
        self,
        de,
        dt,
        d,
        old_state,
        old_time,
        old_stress,
        weights=None,
        stress_inc_guess=None,
        e_inc_guess=None,
    ):
        """
        Args:
            de (float): The strain increment
            dt (float): The time increment
            d (torch.tensor): Direction of the stress
            old_state (torch.tensor): Model state
            old_time (float): Previous time
            old_stress (torch.tensor): Collection of previous stresses

        Keyword args:
            weights (torch.tensor, default None): Weights for averaging the stress
            stress_inc_guess (float, default None): Initial guess for the stress increment
            e_inc_guess (torch.tensor, default None): Initial guess for the strain increment

        Returns:
            avg_stress (torch.tensor): the average macroscale stress
            stress (torch.tensor): the collection of microscale stresses
            state (torch.tensor): the updated model state
            stress_inc (torch.tensor): the computed stress increment (if you want to use it as a guess for the next step)
            e_inc (torch.tensor): the computed strain increment (if you want to use it as a guess for the next step)
        """
        if stress_inc_guess is None:
            stress_inc_guess = 10.0
        if e_inc_guess is None:
            e_inc_guess = d / torch.norm(d) * de

        if weights is None:
            weights = torch.ones(old_stress.shape[0], device=old_stress.device)

        # Just in case...
        weights = weights / torch.sum(weights)

        x0 = torch.cat([torch.tensor([stress_inc_guess]), e_inc_guess], dim=-1)

        def eval(x):
            e_inc = x[..., 1:]

            deformation_rate = e_inc / dt
            time = old_time + dt
            spin = self.spin(time)

            forces = {
                "forces/deformation_rate": deformation_rate,
                "forces/vorticity": spin,
                "forces/t": torch.tensor(time).unsqueeze(0),
            }
            old_state_dict = self.old_state_asm.split_by_variable(
                neml2.Tensor(old_state, 1)
            )
            state = {str(k)[4:]: v for k, v in old_state_dict.items()} | {
                "state/internal/cauchy_stress": old_stress
            }

            return self.model.value_and_dvalue(forces | state | old_state_dict)

        def RJ(x):
            stress_inc = x[..., 0:1]
            e_inc = x[..., 1:]
            prev_avg_stress = torch.sum(old_stress * weights.unsqueeze(-1), dim=0)

            output, J = eval(x)

            stress = output["state/internal/cauchy_stress"]
            avg_stress = torch.sum(stress.torch() * weights.unsqueeze(-1), dim=0)

            R1 = (avg_stress - prev_avg_stress) - (stress_inc * d)
            R2 = torch.dot(e_inc, d) - de

            R = torch.cat([R1, R2.unsqueeze(0)], dim=0)

            J11 = -d.unsqueeze(-1)
            J12 = (
                torch.sum(
                    J["state/internal/cauchy_stress"]["forces/deformation_rate"].torch()
                    * weights.unsqueeze(-1).unsqueeze(-1),
                    dim=0,
                )
                / dt
            )
            J21 = torch.zeros((1, 1), device=R.device)
            J22 = d.unsqueeze(0)

            J = torch.cat(
                [torch.cat([J11, J12], dim=1), torch.cat([J21, J22], dim=1)], dim=0
            )

            return R, J

        x = newton(RJ, x0)
        res, _ = eval(x)

        stress = res["state/internal/cauchy_stress"].torch()

        return (
            torch.sum(stress * weights.unsqueeze(-1), dim=0),
            stress,
            self.state_asm.assemble_by_variable(
                {
                    k: v
                    for k, v in res.items()
                    if str(k) != "state/internal/cauchy_stress"
                }
            ).torch(),
            x[..., 0],
            x[..., 1:],
        )
    
def finite_difference(f, x, eps=1e-8):
    """Finite difference the Jacobian of a function

    Args:
        f (function): A function that takes a tensor x and returns a tensor y
        x (torch.tensor): The point to evaluate the Jacobian at

    Keyword args:
        eps (float, default 1e-8): The finite difference step size

    Returns:
        J (torch.tensor): The Jacobian of f at x
    """
    y = f(x)
    J = torch.zeros((y.numel(), x.numel()), device=x.device)

    for i in range(x.numel()):
        dx = torch.zeros_like(x)
        dx.view(-1)[i] = eps
        y_pert = f(x + dx)
        J[:, i] = ((y_pert - y) / eps).view(-1)

    return J

def newton(RJ, x0, max_iter=50, rtol=1e-5, atol=1e-6):
    """Solve a nonlinear system using Newton's method

    Args:
        RJ (function): A function that takes a tensor x and returns the residual R and Jacobian J
        x0 (torch.tensor): Initial guess for the solution

    Keyword args:
        max_iter (int, default 50): Maximum number of iterations
        rtol (float, default 1e-6): Relative tolerance for convergence
        atol (float, default 1e-8): Absolute tolerance for convergence

    Returns:
        x (torch.tensor): The solution
    """
    x = x0.clone()
    R, J = RJ(x)

    nR = torch.norm(R)
    nR0 = nR.clone()

    for i in range(max_iter):
        if (nR < atol) or (nR / nR0 < rtol):
            return x

        dx = torch.linalg.solve(J, -R)
        x = x + dx
        R, J = RJ(x)
        nR = torch.norm(R)

    raise RuntimeError("Newton's method did not converge")

# -------------------------------------------------------------------------
# WRAPPER TO BaseMaterialApproximationModel
# -------------------------------------------------------------------------
class TaylorModel(BaseMaterialApproximationModel):
    """
    Wrapper for UniaxialTaylorModel.

    - Load and preprocess experimental data (assume .csv)
    - Manage model parameters
    - Run stress-strain simulations for optimization
    """

    def __init__(self,
                 neml2_path: str,
                 neml2_model_name: str = "model_with_stress",
                 axial_index: int = 2,
                 assumed_rate: float = 1.0e-4,
                 npoints: int = 500):
        super().__init__(neml2_path, neml2_model_name)

        self.axial_index = axial_index
        self.assumed_rate = assumed_rate
        self.npoints = npoints

        # Construct Taylor model
        self.tmodel = UniaxialTaylorModel(self.model)

        # Variables to optimize (could later be configurable)
        self.opt_vars = [
            "elastic_tensor_E",
            "elastic_tensor_G",
            "elastic_tensor_nu",
            "slip_strength_constant_strength",
            "voce_hardening_initial_slope",
            "voce_hardening_saturated_hardening",
        ]

    def load_experiment_data(
        self,
        data_dir: str,
        strain_stress_file: str,
        full_field_strain_units: str | None = None,
        strain_stress_file_units: str | None = None,
        straintype: str = "eFab",
        npoints: int | None = None,
        max_strain: float | None = None,
        max_stress: float | None = None,
    ):
        if npoints is None:
            npoints = self.npoints

        allowed_units = (None, "microstrain")
        if full_field_strain_units not in allowed_units:
            raise ValueError(f"Invalid full_field_strain_units: {full_field_strain_units!r}")
        if strain_stress_file_units not in allowed_units:
            raise ValueError(f"Invalid strain_stress_file_units: {strain_stress_file_units!r}")

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
        files = sorted(valid_files, key=lambda s: float(os.path.basename(s).split(".")[0]))
        stress_levels = [float(os.path.basename(f).split(".")[0]) for f in files]

        data = [pd.read_csv(f) for f in files]
        strain_stress = np.loadtxt(strain_stress_file, delimiter=",")

        if (strain_stress_file_units or "").lower() == "microstrain":
            strain_stress[:, 0] *= 1e-6  # convert to mm/mm

        # === Limit macro curve only ===
        if max_strain is not None and max_strain <= 0:
            raise ValueError("max_strain must be > 0")
        if max_stress is not None and max_stress <= 0:
            raise ValueError("max_stress must be > 0")

        if max_strain is not None:
            cutoff_strain = min(max_strain, float(strain_stress[-1, 0]))
        elif max_stress is not None:
            cutoff_stress = min(max_stress, float(strain_stress[-1, 1]))
            cutoff_strain = float(np.interp(cutoff_stress, strain_stress[:, 1], strain_stress[:, 0]))
        else:
            cutoff_strain = float(strain_stress[-1, 0])

        # Truncate and resample the stress–strain curve
        ifn = inter.interp1d(strain_stress[:, 0], strain_stress[:, 1], kind="linear")
        strain = np.linspace(0, cutoff_strain, npoints)
        stress = ifn(strain)
        strain_stress = np.stack((strain, stress), axis=1)

        # Conversion factor for full-field data
        factor = 1.0
        if (full_field_strain_units or "").lower() == "microstrain":
            factor = 1e-6

        # Full-field data: do NOT filter
        exp_strain = [orientation_helper.load_strains(d, factor=factor, field=straintype) for d in data]
        exp_texture = [orientation_helper.load_orientations(d) for d in data]
        exp_weights = [orientation_helper.load_weights(d) for d in data]

        # Compute averages
        avg_exp_strain = [torch.mean(ds, dim=0) for ds in exp_strain]
        avg_axial_strain = [s[self.axial_index] - avg_exp_strain[0][self.axial_index] for s in avg_exp_strain]
        use_weights = exp_weights[0]

        return dict(
            files=files,
            stress_levels=stress_levels,
            strain_stress=strain_stress,
            exp_strain=exp_strain,
            exp_texture=exp_texture,
            exp_weights=exp_weights,
            avg_exp_strain=avg_exp_strain,
            avg_axial_strain=avg_axial_strain,
            use_weights=use_weights,
            cutoff_strain=cutoff_strain,
        )

    def simulate(self,
                 params,
                 d: torch.Tensor,
                 assumed_rate: float,
                 experiment_data=None,
                 return_state=False,
                 initial_strains=None):

        # Unpack experimental data
        exp_texture = experiment_data["exp_texture"][0]
        use_weights = experiment_data["use_weights"]
        strain_stress = experiment_data["strain_stress"]

        # Update model parameters
        with torch.no_grad():
            for v, pv in zip(self.opt_vars, params):
                self.tmodel.model.set_parameter(v, neml2.Tensor(pv, 0))

            # Initialize state
            old_state = self.tmodel.initial_state(exp_texture, elastic_strain=initial_strains)
            old_stress = torch.zeros(old_state.shape[0], 6)
            old_time = 0.0

            stress_hist = [torch.zeros((6,))]
            state_hist = [old_state.clone()]
            ds_guess = None
            de_guess = None

            # Strain-controlled loading
            for e0, e1 in zip(strain_stress[:-1, 0], strain_stress[1:, 0]):
                
                # start timer
                start_time = time.perf_counter()

                de = e1 - e0
                dt = de / assumed_rate

                avg_stress, old_stress, old_state, ds_guess, de_guess = self.tmodel(
                    de, dt, d, old_state, old_time, old_stress,
                    weights=use_weights,
                    stress_inc_guess=ds_guess,
                    e_inc_guess=de_guess,
                )

                old_time += dt
                stress_hist.append(avg_stress)
                state_hist.append(old_state.clone())

                # end timer
                end_time = time.perf_counter()
                # print time taken for this step
                # print(end_time - start_time)

        stress_hist = torch.stack(stress_hist, dim=0)

        if return_state:
            return stress_hist, torch.stack(state_hist, dim=0)
        return stress_hist
