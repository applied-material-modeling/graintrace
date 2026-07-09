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

from __future__ import annotations

from typing import Any, List, Optional
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.utils.parametrize as parametrize
from tqdm import tqdm

from pyzag import reparametrization


class MaterialCalibration:
    """Parameter calibration on the NEML2 v3 + pyzag adjoint engine (LBFGS with
    analytic gradients, RangeRescale reparametrization, optional plotting)."""

    # Physical ranges (NEML2-native units) for RangeRescale, keyed by NEML2 param name.
    DEFAULT_PARAM_RANGES = {
        "elastic_tensor_E": (10_000.0, 500_000.0),  # MPa
        "elastic_tensor_G": (1_000.0, 300_000.0),  # MPa
        "elastic_tensor_nu": (-0.5, 0.5),  # Poisson's ratio
        "slip_strength_constant_strength": (1.0, 2_000.0),  # MPa
        "voce_hardening_initial_slope": (1e-3, 50_000.0),  # MPa
        "voce_hardening_saturated_hardening": (1.0, 2_000.0),  # MPa
    }

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
        """Build the model from ``model_class``/``model_args`` and load
        experimental data via ``data_args``."""
        self.model = model_class(**model_args)
        self.experiment_data = self.model.load_experiment_data(**data_args)

        self.strain_stress = self.experiment_data["strain_stress"]
        self.axial_index = self.model.axial_index
        self.assumed_rate = self.model.assumed_rate

        # deformation direction (kept for API compatibility)
        self.d = torch.zeros((6,))
        self.d[self.axial_index] = 1.0

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
        """Scale macro strain so the elastic slope matches the experimental
        average strain (needs > 2 average data points)."""

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

            strain_avg = np.array([float(s) for s in exp["avg_axial_strain"]])
            stress_levels = np.array(exp["stress_levels"])
            mask = (strain_avg >= strain_window[0]) & (strain_avg <= strain_window[1])
            if np.sum(mask) < 2:
                print("\nInsufficient points in strain window. Skipping correction.\n")
                return

            E_exp, _ = np.polyfit(strain_avg[mask], stress_levels[mask], 1)

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

    def _apply_reparametrization(self, param_ranges=None):
        """Register RangeRescale on each calibration parameter (on the factory
        submodule) so the optimizer works in scaled [0, 1] space within bounds."""
        if param_ranges is None:
            param_ranges = self.DEFAULT_PARAM_RANGES

        factory = self.model.factory
        a_param = getattr(factory, self.model.opt_vars[0])
        device, dtype = a_param.device, a_param.dtype

        map_dict = {
            f"factory.{name}": reparametrization.RangeRescale(
                torch.tensor(lo, device=device, dtype=dtype),
                torch.tensor(hi, device=device, dtype=dtype),
                clamp=True,
            )
            for name, (lo, hi) in param_ranges.items()
            if name in self.model.opt_vars
        }
        reparam = reparametrization.Reparameterizer(map_dict, error_not_provided=False)
        reparam(self.model.tmodel)
        return reparam

    def _calibration_parameters(self) -> List[torch.nn.Parameter]:
        """The optimizer variables — the RangeRescale ``.original`` leaves after
        reparametrization (or the raw parameters before it)."""
        factory = self.model.factory
        out = []
        for name in self.model.opt_vars:
            parametrizations = getattr(factory, "parametrizations", None)
            if parametrizations is not None and name in parametrizations:
                out.append(parametrizations[name].original)
            else:
                out.append(getattr(factory, name))
        return out

    def _remove_reparametrization(self) -> None:
        """Bake the current (physical) values back into plain parameters."""
        factory = self.model.factory
        for name in self.model.opt_vars:
            if parametrize.is_parametrized(factory, name):
                parametrize.remove_parametrizations(
                    factory, name, leave_parametrized=True
                )
        factory._update_parameter_values()

    def _predict_axial(self) -> torch.Tensor:
        """Predicted axial stress trajectory (grad-enabled) at current params."""
        pred = self.model.simulate(
            None,
            d=self.d,
            assumed_rate=self.assumed_rate,
            experiment_data=self.experiment_data,
        )
        return pred[:, self.axial_index]

    def objective(self, params: Any = None, default_err: float = 1e6) -> float:
        """L2 norm of (model axial stress - experimental stress). If ``params``
        is given, evaluate at those values; otherwise at the current factory
        parameters. Kept for API compatibility / diagnostics."""
        target = torch.tensor(
            self.strain_stress[:, 1], dtype=torch.float64, device=self.model.device
        )
        try:
            with torch.no_grad():
                if params is not None:
                    sim = self.model.simulate(
                        torch.as_tensor(params, dtype=torch.float64),
                        d=self.d,
                        assumed_rate=self.assumed_rate,
                        experiment_data=self.experiment_data,
                    )[:, self.axial_index]
                else:
                    sim = self._predict_axial()
        except Exception:
            return default_err
        return torch.norm(sim - target).item()

    def calibrate(
        self,
        method: str = "lbfgs",
        maxiter: int = 50,
        lr: float = 0.1,
        max_iter_per_step: int = 20,
        line_search_fn: Optional[str] = None,
        param_ranges=None,
        plateau_rtol: float = 1e-3,
        plateau_window: int = 3,
        autosave: bool = True,
    ) -> torch.Tensor:
        """Calibrate the six parameters with analytic-adjoint gradients + LBFGS.
        ``method`` is accepted for backward compatibility only."""
        model = self.model

        self._apply_reparametrization(param_ranges)
        params = self._calibration_parameters()

        target = torch.tensor(
            self.strain_stress[:, 1], dtype=torch.float64, device=model.device
        )

        optimizer = torch.optim.LBFGS(
            params, lr=lr, max_iter=max_iter_per_step, line_search_fn=line_search_fn
        )

        progress = tqdm(total=maxiter, desc="Optimization progress", ncols=80)
        autosave_path = os.path.join(self.save_dir, "autosave_material.json")

        def closure():
            optimizer.zero_grad()
            try:
                pred = self._predict_axial()
                loss = ((pred - target) ** 2).mean()
                loss.backward()
            except (torch._C._LinAlgError, RuntimeError):
                optimizer.zero_grad()
                return torch.tensor(
                    float("inf"), device=target.device, dtype=target.dtype
                )
            return loss

        loss_history: List[float] = []
        for _ in range(maxiter):
            loss = optimizer.step(closure)
            loss_v = float(loss.detach()) if hasattr(loss, "detach") else float(loss)
            loss_history.append(loss_v)
            progress.update(1)
            progress.set_postfix({"loss": f"{loss_v:.3e}"})

            if autosave:
                self.opt_params = model.get_params()
                self.save(autosave_path)

            # Early stop on plateau
            if plateau_rtol > 0 and len(loss_history) > plateau_window:
                prev = loss_history[-plateau_window - 1]
                cur = loss_history[-1]
                if math.isfinite(prev) and math.isfinite(cur):
                    if (prev - cur) / max(abs(prev), 1e-12) < plateau_rtol:
                        break

        progress.close()

        # Bake final physical values back into plain parameters
        self._remove_reparametrization()
        self.opt_params = model.get_params()
        self.loss_history = loss_history

        print(f"Optimization finished after {len(loss_history)} iterations.")

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
        """Plot stress-strain data."""

        exp = self.experiment_data
        exp_macro_strain = exp["strain_stress"][:, 0]
        exp_macro_stress = exp["strain_stress"][:, 1]
        exp_avg_strain = np.array([float(s) for s in exp["avg_axial_strain"]])
        exp_stress_levels = np.array(exp["stress_levels"])

        plt.figure()
        plt.plot(exp_macro_strain, exp_macro_stress, lw=2, label="Experiment (macro)")
        if include_experiment_overlay:
            plt.plot(
                exp_avg_strain, exp_stress_levels, "o", label="Experiment (avg grains)"
            )

        if include_model:
            with torch.no_grad():
                if optimized and hasattr(self, "opt_params"):
                    sim_stress = (
                        self.model.simulate(
                            self.opt_params,
                            d=self.d,
                            assumed_rate=self.assumed_rate,
                            experiment_data=exp,
                        )[:, self.axial_index]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    plt.plot(
                        exp_macro_strain, sim_stress, lw=2, label="Model (optimized)"
                    )
                else:
                    p0 = self.model.get_params()
                    sim_stress = (
                        self.model.simulate(
                            p0,
                            d=self.d,
                            assumed_rate=self.assumed_rate,
                            experiment_data=exp,
                        )[:, self.axial_index]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    plt.plot(
                        exp_macro_strain, sim_stress, "--", label="Model (initial)"
                    )

        plt.xlabel("Axial strain (mm/mm)")
        plt.ylabel("Stress (MPa)")
        plt.legend(loc="best")
        plt.tight_layout()

        suffix = "with_model" if include_model else "experiment_only"
        self._save_figure(f"stress_strain_{suffix}")

    def plot_texture(
        self, direction: Optional[List[float]] = None, crystal_symmetry: str = "432"
    ) -> None:
        """Plot experimental pole figures (neml2.texture)."""
        from neml2 import texture

        if direction is None:
            direction = torch.tensor([1.0, 1.0, 1.0], dtype=torch.double)
        else:
            direction = torch.tensor(direction, dtype=torch.double)

        for stress, tex in zip(
            self.experiment_data["stress_levels"], self.experiment_data["exp_texture"]
        ):
            texture.pretty_plot_pole_figure_points(
                tex,
                direction,
                crystal_symmetry=crystal_symmetry,
            )
            self._save_figure(f"pole_figure_{int(stress)}MPa")

    def plot_strain_histogram(
        self, strain_index: Optional[int] = None, include_initial_strain: bool = False
    ) -> None:
        """Plot elastic strain histograms (model vs experiment) at matching
        stress levels."""

        if strain_index is None:
            strain_index = self.axial_index

        init_strain = (
            self.experiment_data["exp_strain"][0] if include_initial_strain else None
        )

        with torch.no_grad():
            stress_hist, state_hist = self.model.simulate(
                self.opt_params,
                d=self.d,
                assumed_rate=self.assumed_rate,
                experiment_data=self.experiment_data,
                initial_strains=init_strain,
                return_state=True,
            )

        # state_hist: (ntime, n_grains, 6) elastic strain
        model_strain = state_hist[:, :, strain_index].detach().cpu().numpy()
        sim_stress = stress_hist[:, self.axial_index].detach().cpu().numpy()

        # Match experiment stress levels to nearest sim step
        exp_stress_levels = np.array(self.experiment_data["stress_levels"])
        exp_strains = self.experiment_data["exp_strain"]

        indices = [int(np.argmin(np.abs(sim_stress - s))) for s in exp_stress_levels]

        for exp_s, stress, idx in zip(exp_strains, exp_stress_levels, indices):
            plt.figure()
            plt.hist(model_strain[idx, :], bins=30, alpha=0.6, label="Model")
            plt.hist(
                exp_s[:, strain_index].cpu().numpy(),
                bins=30,
                alpha=0.5,
                label="Experiment",
            )
            plt.xlabel("Elastic strain (mm/mm)")
            plt.ylabel("Counts")
            plt.legend(loc="best")
            plt.tight_layout()

            suffix = "with_initial" if include_initial_strain else "without_initial"
            self._save_figure(f"elastic_strain_distribution_{stress:.0f}MPa_{suffix}")

    def save(self, filepath: str):
        """Save the optimized material parameters to a JSON file."""
        import json

        if not hasattr(self, "opt_params"):
            raise RuntimeError("No optimized parameters found. Run calibrate() first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        params_dict = {
            k: float(v) for k, v in zip(self.model.opt_vars, self.opt_params)
        }

        with open(filepath, "w") as f:
            json.dump(params_dict, f, indent=2)

        return filepath

    def load(self, filepath: str):
        """Load previously saved optimized material parameters from a JSON file."""
        import json

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find saved parameter file: {filepath}")

        with open(filepath, "r") as f:
            params_dict = json.load(f)

        opt_params = [params_dict[k] for k in self.model.opt_vars if k in params_dict]
        self.opt_params = torch.tensor(opt_params, dtype=torch.float64)

        # Push the loaded values into the model so simulate()/plots use them.
        # push loaded values into the model for simulate()/plots
        self.model.set_params(self.opt_params)

        print(f"[MaterialCalibration] Loaded calibrated parameters from {filepath}")

        return self.opt_params
