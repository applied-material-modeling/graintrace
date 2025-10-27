import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from tqdm import tqdm
import neml2
from neml2.postprocessing import polefigure

class MaterialCalibration:
    """
    Perform parameter calibration.
    - Optimization (parameter fitting)
    - Error computation
    - Optional visualization (stress-strain curves, texture, strain histograms)
    """

    def __init__(self, model_class, model_args, data_args, save_dir="calibration_results"):
        """
        Parameters
        ----------
        model_class : class
            A subclass of BaseMaterialApproximationModel.
        model_args : dict
            NEML2 Arguments for constructing the model
        data_args : dict
            Arguments for loading experimental data (e.g. data_dir, strain_stress_file, npoints)
        """
        self.model = model_class(**model_args)
        self.experiment_data = self.model.load_experiment_data(**data_args)

        # Extract key data for objective function
        self.strain_stress = self.experiment_data["strain_stress"]
        self.axial_index = self.model.axial_index
        self.assumed_rate = self.model.assumed_rate

        # deformation tensor
        self.d = torch.zeros((6,))
        self.d[self.axial_index] = 1.0

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def objective(self, params, default_err=1e6):
        p = torch.tensor(params)
        try:
            sim_stress = self.model.simulate(
                p,
                d=self.d,
                assumed_rate=self.assumed_rate,
                experiment_data=self.experiment_data,
            )[:, self.axial_index]
        except Exception:
            return default_err

        err = sim_stress - torch.tensor(self.strain_stress[:, 1])
        return torch.norm(err).item()

    def calibrate(self, method="Nelder-Mead", maxiter=50):
        # Initial parameter guess
        p0 = torch.tensor([
            self.model.tmodel.model.get_parameter(v).torch().detach().clone()
            for v in self.model.opt_vars
        ])

        # Set up progress bar
        progress = tqdm(total=maxiter, desc="Optimization progress", ncols=80)

        def callback(xk):
            progress.update(1)
            progress.set_postfix({"obj": f"{self.objective(xk):.3e}"})

        result = opt.minimize(
            self.objective,
            p0.numpy(),
            method=method,
            options={"maxiter": maxiter, "disp": False},
            callback=callback,
        )

        progress.close()
        self.opt_params = torch.tensor(result.x)
        print(f"Optimization finished after {progress.n} iterations.")
        return self.opt_params

    def _save_figure(self, filename: str):
        path = os.path.join(self.save_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    def plot_stress_strain(self, optimized=True):
        """Plot experimental vs. simulated stress-strain curve."""
        params = self.opt_params if optimized else None

        if params is None:
            p0 = torch.tensor([
                self.model.tmodel.model.get_parameter(v).torch().detach().clone()
                for v in self.model.opt_vars
            ])
            params = p0

        sim_stress = self.model.simulate(
            params,
            d=self.d,
            assumed_rate=self.assumed_rate,
            experiment_data=self.experiment_data,
        )[:, self.axial_index].numpy()

        plt.figure()
        plt.plot(self.strain_stress[:, 0], self.strain_stress[:, 1], label="Experiment")
        plt.plot(self.strain_stress[:, 0], sim_stress, label="Model")
        plt.xlabel("Strain (mm/mm)")
        plt.ylabel("Stress (MPa)")
        plt.legend()
        self._save_figure("stress_strain_calibration")

    def plot_texture(self, direction=None, crystal_symmetry="432"):
        """
        Plot experimental pole figures.
        """
        if direction is None:
            direction = torch.tensor([1.0, 1.0, 1.0])
        else:
            direction = torch.tensor(direction, dtype=torch.double)

        for stress, tex in zip(
            self.experiment_data["stress_levels"], self.experiment_data["exp_texture"]
        ):
            polefigure.pretty_plot_pole_figure_points(
                tex,
                direction,
                crystal_symmetry=crystal_symmetry,
            )
            self._save_figure(f"pole_figure_{int(stress)}MPa")

    def plot_strain_histogram(self, state_hist, strain_index=None, label=None):
        """Plot elastic strain histogram for a given load step."""
        if strain_index is None:
            strain_index = self.axial_index

        vars = self.model.tmodel.state_asm.split_by_variable(torch.tensor(state_hist))
        strain_vals = vars["state/elastic_strain"].torch()[:, :, strain_index].numpy()

        plt.figure()
        plt.hist(strain_vals.flatten(), bins=30, alpha=0.6, label=label or "Model")
        plt.xlabel("Elastic strain (mm/mm)")
        plt.ylabel("Counts")
        if label:
            plt.legend(loc="best")

        self._save_figure(f"elastic_strain_distribution_idx{strain_index}")
