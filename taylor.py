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
