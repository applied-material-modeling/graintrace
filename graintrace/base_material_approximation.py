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

from typing import Any, Optional
import torch
import neml2


class BaseMaterialApproximationModel:
    """
    Abstract interface for any constitutive or homogenized material model
    used for parameter fitting. This always use NEML2 as the constitutive
    model evaluation.
    """

    def __init__(self, neml2_path: str, neml2_model_name: str = "model"):
        """
        Initialize a material model approximation interface.

        Parameters
        ----------
        neml2_path : str
            Path to the NEML2 input file (.i) containing the model definition.
        neml2_model_name : str, default = 'model'
            Name of the specific material model block within the NEML2 input.
        """
        self.neml2_path = neml2_path
        self.neml2_model_name = neml2_model_name
        self.model = self._load_model()

    # ------------------------------------------------------------------
    # Must be implemented by subclasses
    # ------------------------------------------------------------------
    def _load_model(self) -> Any:
        return neml2.load_model(self.neml2_path, self.neml2_model_name)

    def load_experiment_data(
        self, data_dir: str, strain_stress_file: str, npoints: int
    ) -> None:
        """
        Expected inputs and outputs depend on model type but generally, should always include:
        ----------
        data_dir : str
            Directory containing experimental data files (e.g., CSVs).
        strain_stress_file : str
            Path to macroscopic stress-strain data file.
        npoints : int
            Number of points for resampling or interpolation.
        """
        return None

    def simulate(
        self,
        params: Any,
        d: torch.Tensor,
        assumed_rate: float,
        experiment_data: Optional[dict] = None,
        return_state: bool = False,
        initial_strains: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run a stress–strain simulation for given model parameters.

        Parameters
        ----------
        params : array-like or dict
            Model parameters to use in the simulation.
        d : torch.Tensor
            Strain direction tensor (6-component vector).
        assumed_rate : float
            Applied strain rate.
        experiment_data : dict, optional
            Experimental data dictionary (orientations, weights, etc.).
        return_state : bool
            If True, also return full internal state history.
        initial_strains : torch.Tensor, optional
            Optional initial elastic strain per grain.

        Returns
        -------
        torch.Tensor or (torch.Tensor, torch.Tensor)
            If return_state=False:
                stress_history  (N, 6)
            If return_state=True:
                (stress_history, state_history)
        """
        raise NotImplementedError("Subclasses must implement simulate().")
