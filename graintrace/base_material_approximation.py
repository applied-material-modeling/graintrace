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

"""Abstract base interface for NEML2-backed material approximation models."""

from __future__ import annotations

from typing import Any, Optional

import torch

import neml2


class BaseMaterialApproximationModel:
    """Abstract interface for a NEML2-backed material model used for parameter fitting."""

    def __init__(self, neml2_path: str, neml2_model_name: str = "model"):
        """Initialize the interface from a NEML2 input file and model block name."""
        self.neml2_path = neml2_path
        self.neml2_model_name = neml2_model_name
        self.model = self._load_model()

    def _load_model(self) -> Any:
        return neml2.load_model(self.neml2_path, self.neml2_model_name)

    def load_experiment_data(
        self, data_dir: str, strain_stress_file: str, npoints: int
    ) -> None:
        # pylint: disable=unused-argument  # abstract stub; args define the interface
        """Load experiment data (macroscopic stress-strain plus per-file CSVs)."""
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
        """Run a stress-strain simulation for given model parameters.

        Returns stress_history (N, 6), or (stress_history, state_history) if return_state.
        """
        raise NotImplementedError("Subclasses must implement simulate().")
