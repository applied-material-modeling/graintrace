from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union
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

    def __init__(
        self,
        model_class,
        model_args,
        data_args,
        apply_elastic_correction=False,
        correction_method="with_experiment_average",
        strain_window=None,
        save_dir="calibration_results",
    ):
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

        # elastic correction between model and experiment

        correction_method_allowed = ["with_experiment_average"]

        self.apply_elastic_correction = apply_elastic_correction
        self.correction_method = correction_method
        self.strain_window = strain_window

        if self.apply_elastic_correction:
            if self.strain_window is None:
                raise ValueError(
                    "'strain_window' must be provided when "
                    "apply_elastic_correction=True."
                )
            if self.correction_method not in correction_method_allowed:
                raise ValueError(f"Unknown correction_method: {self.correction_method}")

            self._apply_elastic_slope_correction()

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _apply_elastic_slope_correction(self) -> None:
        """
        Scale model stresses so that the elastic slope matches the experimental average strain.
        Only applied if experimental average strain exists and has > 2 data points.

        Algorithm: fit the slope of stress-strain in the specified small strain window
        for both full stress-strain curve and experimental average, then scale the stress-strain
        stresses accordingly.
        """

        exp = self.experiment_data
        strain_window = self.strain_window

        if self.correction_method == "with_experiment_average":

            print(
                "\nApplying elastic slope correction using experimental average strain.\n"
            )

            if "avg_axial_strain" not in exp or len(exp["avg_axial_strain"]) <= 2:
                print(
                    "\nNot enough average experimental data points. Skipping correction.\n"
                )
                return

            # Extract experimental grain-average response
            strain_avg = np.array(exp["avg_axial_strain"])
            stress_levels = np.array(exp["stress_levels"])
            mask = (strain_avg >= strain_window[0]) & (strain_avg <= strain_window[1])
            if np.sum(mask) < 2:
                print("\nInsufficient points in strain window. Skipping correction.\n")
                return

            # Fit slope of grain-averaged strain vs stress
            E_exp, _ = np.polyfit(strain_avg[mask], stress_levels[mask], 1)

            # Fit slope of provided macro stress–strain curve
            strain_macro = exp["strain_stress"][:, 0]
            stress_macro = exp["strain_stress"][:, 1]
            mask_macro = (strain_macro >= strain_window[0]) & (
                strain_macro <= strain_window[1]
            )
            if np.sum(mask_macro) < 2:
                raise ValueError("\nInsufficient macro points in window.\n")

            E_macro, _ = np.polyfit(
                strain_macro[mask_macro], stress_macro[mask_macro], 1
            )

            if E_macro <= 0 or E_exp <= 0:
                print("Invalid slope detected. Skipping correction.")
                return

            scale_factor = E_macro / E_exp
            exp["strain_stress"][:, 0] *= scale_factor

            print(
                f"\nScaled macro strain by {scale_factor:.3f} "
                f"to match average experimental slope.\n"
            )

    def objective(self, params: Any, default_err: float = 1e6) -> float:
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

    def calibrate(self, method: str = "Nelder-Mead", maxiter: int = 50, autosave: bool = True) -> None:
        # Initial parameter guess
        p0 = torch.tensor(
            [
                self.model.tmodel.model.get_parameter(v).torch().detach().clone()
                for v in self.model.opt_vars
            ]
        )

        # Set up progress bar
        progress = tqdm(total=maxiter, desc="Optimization progress", ncols=80)
        autosave_path = os.path.join(self.save_dir, "autosave_material.json")

        def callback(xk):
            current_loss = self.objective(xk)
            progress.update(1)
            progress.set_postfix({"obj": f"{current_loss:.3e}"})

            if autosave:
                # Temporarily assign current params to self.opt_params for save()
                self.opt_params = torch.tensor(xk)
                self.save(autosave_path)

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

        save_path = os.path.join(self.save_dir, "calibrated_material.json")
        self.save(save_path)

        return self.opt_params

    def _save_figure(self, filename: str):
        path = os.path.join(self.save_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    def plot_stress_strain(
        self, include_model=False, optimized=True, include_experiment_overlay=True
    ):
        """
        Plot stress–strain data.
        """

        exp = self.experiment_data
        exp_macro_strain = exp["strain_stress"][:, 0]
        exp_macro_stress = exp["strain_stress"][:, 1]
        exp_avg_strain = np.array(exp["avg_axial_strain"])
        exp_stress_levels = np.array(exp["stress_levels"])

        plt.figure()
        plt.plot(exp_macro_strain, exp_macro_stress, lw=2, label="Experiment (macro)")
        if include_experiment_overlay:
            plt.plot(
                exp_avg_strain, exp_stress_levels, "o", label="Experiment (avg grains)"
            )

        if include_model:
            if optimized and hasattr(self, "opt_params"):
                if not hasattr(self, "last_sim_stress"):
                    self.last_sim_stress = self.model.simulate(
                        self.opt_params,
                        d=self.d,
                        assumed_rate=self.assumed_rate,
                        experiment_data=exp,
                    )[:, self.axial_index].numpy()
                sim_stress = self.last_sim_stress
                plt.plot(exp_macro_strain, sim_stress, lw=2, label="Model (optimized)")
            else:
                # try plotting an initial curve if desired
                p0 = torch.tensor(
                    [
                        self.model.tmodel.model.get_parameter(v)
                        .torch()
                        .detach()
                        .clone()
                        for v in self.model.opt_vars
                    ]
                )
                sim_stress = self.model.simulate(
                    p0,
                    d=self.d,
                    assumed_rate=self.assumed_rate,
                    experiment_data=exp,
                )[:, self.axial_index].numpy()
                plt.plot(exp_macro_strain, sim_stress, "--", label="Model (initial)")

        plt.xlabel("Axial strain (mm/mm)")
        plt.ylabel("Stress (MPa)")
        plt.legend(loc="best")
        plt.tight_layout()

        suffix = "with_model" if include_model else "experiment_only"
        self._save_figure(f"stress_strain_{suffix}")

    def plot_texture(self, direction: Optional[List[float]] = None, crystal_symmetry: str = "432") -> None:
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

    def plot_strain_histogram(self, strain_index: Optional[int] = None, include_initial_strain: bool = False) -> None:
        """
        Plot elastic strain histograms (model vs experiment) at matching stress levels.
        """

        if strain_index is None:
            strain_index = self.axial_index

        init_strain = (
            self.experiment_data["exp_strain"][0] if include_initial_strain else None
        )

        stress_hist, state_hist = self.model.simulate(
            self.opt_params,
            d=self.d,
            assumed_rate=self.assumed_rate,
            experiment_data=self.experiment_data,
            initial_strains=init_strain,
            return_state=True,
        )

        vars = self.model.tmodel.state_asm.split_by_variable(
            neml2.Tensor(state_hist, 1)
        )
        model_strain = vars["state/elastic_strain"].torch()[:, :, strain_index].numpy()
        sim_stress = stress_hist[:, self.axial_index].numpy()

        # Match experiment stress levels to nearest simulation step
        exp_stress_levels = np.array(self.experiment_data["stress_levels"])
        exp_strains = self.experiment_data["exp_strain"]

        indices = [np.argmin(np.abs(sim_stress - s)) for s in exp_stress_levels]

        # Plot histogram comparison for each matched stress level
        for ii, (exp_s, stress, idx) in enumerate(
            zip(exp_strains, exp_stress_levels, indices)
        ):
            plt.figure()
            plt.hist(model_strain[idx, :], bins=30, alpha=0.6, label="Model")
            plt.hist(
                exp_s[:, strain_index].numpy(), bins=30, alpha=0.5, label="Experiment"
            )
            plt.xlabel("Elastic strain (mm/mm)")
            plt.ylabel("Counts")
            plt.legend(loc="best")
            plt.tight_layout()

            suffix = "with_initial" if include_initial_strain else "without_initial"
            self._save_figure(f"elastic_strain_distribution_{stress:.0f}MPa_{suffix}")

    def save(self, filepath: str):
        """
        Save the optimized material parameters to a JSON file.
        """
        import json

        if not hasattr(self, "opt_params"):
            raise RuntimeError("No optimized parameters found. Run calibrate() first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        params_dict = {
            k: float(v) for k, v in zip(self.model.opt_vars, self.opt_params)
        }

        with open(filepath, "w") as f:
            json.dump(params_dict, f, indent=2)

        print(f"[MaterialCalibration] Saved calibrated parameters to {filepath}")

        return filepath

    def load(self, filepath: str):
        """
        Load previously saved optimized material parameters from a JSON file.
        """
        import json

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find saved parameter file: {filepath}")

        with open(filepath, "r") as f:
            params_dict = json.load(f)

        # Restore parameter order and tensor
        opt_params = [params_dict[k] for k in self.model.opt_vars if k in params_dict]
        self.opt_params = torch.tensor(opt_params, dtype=torch.double)

        print(f"[MaterialCalibration] Loaded calibrated parameters from {filepath}")

        return self.opt_params
