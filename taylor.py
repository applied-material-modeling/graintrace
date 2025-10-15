import torch
import torch.nn as nn

import neml2
from neml2.reserved import *


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

    def initial_state(self, orientations):
        """Assemble the initial state vector

        Args:
            orientations (torch.tensor): (n,3) tensor with initial orientations
        """
        state_dict = {
            "old_state/elastic_strain": torch.zeros(
                (orientations.shape[0], 6), device=orientations.device
            ),
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

        Returns:
            stress (torch.tensor): The updated stress
            state (torch.tensor): The updated model state
            forces (torch.tensor): The updated model forces
        """
        if stress_inc_guess is None:
            stress_inc_guess = 10.0
        if e_inc_guess is None:
            e_inc_guess = d / torch.norm(d) * de

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
            prev_avg_stress = torch.mean(old_stress, dim=0)

            output, J = eval(x)

            stress = output["state/internal/cauchy_stress"]
            avg_stress = torch.mean(stress.torch(), dim=0)

            R1 = (avg_stress - prev_avg_stress) - (stress_inc * d)
            R2 = torch.dot(e_inc, d) - de

            R = torch.cat([R1, R2.unsqueeze(0)], dim=0)

            J11 = -d.unsqueeze(-1)
            J12 = torch.mean(
                J["state/internal/cauchy_stress"]["forces/deformation_rate"].torch(),
                dim=0,
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
            stress.mean(0),
            stress,
            self.state_asm.assemble_by_variable(res).torch(),
        )


def newton(RJ, x0, max_iter=50, rtol=1e-6, atol=1e-8):
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
