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

# End-to-end material calibration on the NEML2 v3 + pyzag adjoint stack.
# Calibrates six crystal-plasticity parameters of a mixed-control Taylor
# aggregate to a uniaxial stress-strain curve (analytic adjoint gradients +
# LBFGS). Writes pole figures, stress-strain overlays, histograms, and
# calibrated_material.json.

import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt

import graintrace as _gt
from graintrace.material_calibration import MaterialCalibration
from graintrace.taylor import TaylorModel

_cpfe_base = str(Path(_gt.__file__).parent / "cpfe_base")

fsize = 14
plt.rcParams.update(
    {
        "font.size": fsize,
        "axes.labelsize": fsize,
        "axes.titlesize": fsize,
        "xtick.labelsize": fsize,
        "ytick.labelsize": fsize,
        "legend.fontsize": fsize,
    }
)

## INPUT
## ------------------------------------
outputdir = "material_calibration_out"

# Folder with numeric-named per-stress-level CSVs (e.g. 0.csv, 100.csv, ...)
# plus strain-stress.csv. Self-contained FF dataset shipped in mwe_data (9 load
# steps, 500 grains each: orientation matrix O, coords, Euler, eKen strain). For
# a physically registered calibration the CSVs should first be rotated into the
# simulation frame (see demonstrate_farfield.py / experiment_rotation_helper);
# here we calibrate on the raw data directly.
exp_data_dir = "mwe_data/ff_calibration"

device = "cuda"  # "cpu" or "cuda"
n_grains = 100  # subsample this many grains per stress level; None for all
npoints = 30  # resampled stress-strain points (= number of pyzag time steps)
max_strain = 0.006  # cap the calibration window (mm/mm) to a convergent regime
straintype = "eKen"  # full-field strain column prefix ("eKen" or "eFab")
seed = 42

nchunk = 2  # pyzag chunk size for the bidiagonal-in-time solve
maxiter = 15  # LBFGS outer-iteration cap (plateau guard stops earlier)
lr = 0.3  # LBFGS learning rate
max_iter_per_step = 6  # LBFGS inner iterations per closure
line_search = "strong_wolfe"  # None or "strong_wolfe"

run_calibration = True

# NEML2 opt-var name -> CPFESimulation material-parameter name.
material_name_map = {
    "elastic_tensor_E": "elastic_E",
    "elastic_tensor_G": "elastic_G",
    "elastic_tensor_nu": "elastic_nu",
    "slip_strength_constant_strength": "slip_constant_strength",
    "voce_hardening_initial_slope": "voce_hardening_initial_slope",
    "voce_hardening_saturated_hardening": "voce_hardening_saturation",
}

## ------------------------------------

if run_calibration:
    print("\n=== Material Calibration (NEML2 v3 + pyzag adjoint) ===\n")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)

    save_dir = outputdir + "/figures/material_calibration"
    os.makedirs(save_dir, exist_ok=True)

    calib = MaterialCalibration(
        model_class=TaylorModel,
        model_args=dict(
            neml2_path=_cpfe_base + "/neml2_cpfe_calibration.i",
            npoints=npoints,
            nchunk=nchunk,
            device=device,
            compile=False,
        ),
        data_args=dict(
            data_dir=exp_data_dir,
            strain_stress_file=exp_data_dir + "/strain-stress.csv",
            npoints=npoints,
            full_field_strain_units="microstrain",
            straintype=straintype,
            max_strain=max_strain,
            n_grains=n_grains,
            seed=seed,
        ),
        save_dir=save_dir,
        apply_elastic_correction=False,
    )

    n_used = calib.experiment_data["exp_texture"][0].shape[0]
    print(f"Loaded {n_used} grains, {calib.strain_stress.shape[0]} stress-strain points.")
    print(f"Initial objective (L2): {calib.objective():.4f}\n")

    # Experimental pole figures + initial stress-strain curve.
    calib.plot_texture(direction=[1, 1, 1])
    calib.plot_stress_strain()

    # Calibrate (analytic adjoint gradients + LBFGS).
    # maxiter is only an upper bound: the plateau guard stops early once the
    # relative loss improvement over `plateau_window` steps falls below rtol.
    calib.calibrate(
        maxiter=maxiter,
        lr=lr,
        max_iter_per_step=max_iter_per_step,
        line_search_fn=line_search,
        plateau_rtol=1e-3,
        plateau_window=2,
    )
    calib.load(save_dir + "/calibrated_material.json")

    # Model overlays.
    calib.plot_stress_strain(include_model=True)
    calib.plot_strain_histogram(include_initial_strain=True)

    print(f"\nFinal objective (L2): {calib.objective():.4f}")

    optimized_material = {
        material_name_map[k]: float(v)
        for k, v in zip(calib.model.opt_vars, calib.opt_params)
    }

    print("\nCalibrated material parameters:")
    for k, v in optimized_material.items():
        print(f"  {k} = {v:.6g}")
    print(f"\nFigures + calibrated_material.json written to: {save_dir}")
