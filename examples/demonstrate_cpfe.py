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

# Run a CPFE simulation on a far-field (Voronoi) reconstruction with NEML2 v3.
# CPFESimulation.run() bakes material params, runs neml2-compile (env/paths
# auto-derived from moose_run_file), then launches puma-opt. FF orientations.dat
# is Euler and converted to neml2 MRP (the canonical interchange) before use.

import numpy as np
import torch
import meshio

from graintrace.run_cpfe_simulation import CPFESimulation
from graintrace import orientation_helper as oh

## INPUT
## ------------------------------------
# A far-field reconstruction folder containing the co-registered mesh + initial
# elastic strain, and the per-grain Euler orientations (see demonstrate_farfield).
# Minimal FF dataset shipped in mwe_data: a 10-grain reconstruction mesh + its
# per-grain Euler orientations. (To use a real FF residual strain, point
# eeres_file at a reconstruction_cpfe_ee.csv; here we use zero initial strain.)
ff_dir = "mwe_data/cpfe_ff"
mesh_file = ff_dir + "/reconstruction.msh"
eeres_file = None  # None -> zero initial elastic strain (no FF ee field in this MWE)
euler_file = ff_dir + "/orientations.dat"  # Euler bunge, degrees (FF output)

outputdir = "cpfe_out"
# EDIT: path to your built PUMA binary (e.g. external/puma/puma-opt after building it).
moose_run_file = "external/puma/puma-opt"

angle_convention = "bunge"
angle_type = "degrees"

# NEML2 device(s): "cpu", "cuda:0", or a space-separated list for multi-GPU.
device = "cuda:0"
ncore = 4
# Per-device NEML2 batch chunk (quad points per call). 0 = whole batch; a finite
# value caps GPU memory by chunking.
device_batch = 20000

total_strain = 0.002  # applied axial (z) engineering strain
dt = 0.5
total_time = 2.0
initialize_time = 1.0
sync_times = "2.0"
grid_elements = [10, 10, 10]

material = dict(
    slip_constant_strength=130.0,
    voce_hardening_initial_slope=1556.09,
    voce_hardening_saturation=100.0,
    power_slip_n=20,
    power_slip_g0=0.0001,
    elastic_E=209016.0,
    elastic_nu=0.307,
    elastic_G=60355.0,
    burger_scale=2.22,
)
## ------------------------------------

print("\n=== CPFE simulation (NEML2 v3 / AOTI) ===\n")

# FF Euler orientations -> neml2 MRP (canonical interchange)
euler = np.loadtxt(euler_file)
mrp = oh.euler_to_mrp(
    torch.tensor(euler, dtype=torch.float64), angle_convention, angle_type
)
ori_file = outputdir + "/orientations_MRP.dat"
import os

os.makedirs(outputdir, exist_ok=True)
np.savetxt(ori_file, mrp.numpy(), fmt="%.12g")
print(f"Converted {euler.shape[0]} Euler orientations -> neml2 MRP")

# Mesh bounding box + element order
m = meshio.read(mesh_file)
P = m.points
bbox = [
    float(P[:, 0].min()),
    float(P[:, 0].max()),
    float(P[:, 1].min()),
    float(P[:, 1].max()),
    float(P[:, 2].min()),
    float(P[:, 2].max()),
]
celltypes = {c.type for c in m.cells}
element_order = "SECOND" if any("10" in t for t in celltypes) else "FIRST"
print(f"element_order={element_order}, bbox={[round(x, 2) for x in bbox]}")

sim = CPFESimulation(
    mesh_file=mesh_file,
    save_simulation_folder=outputdir,
    moose_run_file=moose_run_file,
    element_order=element_order,
    eeres_file=eeres_file,
    ori_file=ori_file,
    dim=3,
    use_ff_initial_field=True,  # eeres_file=None -> zero initial elastic strain
)

sim.set_parameters("material", **material)
sim.set_parameters(
    "simulation_parameters",
    device=device,
    device_batch=device_batch,
    dt=dt,
    total_time=total_time,
    initialize_time=initialize_time,
    sync_times=sync_times,
    # Output-frequency knobs (defaults shown). mesh_csv -> crisp per-element fields in
    # mesh_out/ for full-mesh REI; grid_transfer/exodus_output default to cheap (final/sync).
    # For a regular grid without per-step transfers, keep grid_transfer="final" and resample
    # sim_output.e offline with graintrace.grid_resampling.GridResampler.
    mesh_csv="sync",
    grid_transfer="final",
    exodus_output="sync",
)

displace = total_strain * (bbox[5] - bbox[4])
sim.set_parameters(
    "boundary",
    bounding_box=bbox,
    bc={
        "x": {"negative": "stress_free", "positive": "stress_free"},
        "y": {"negative": "stress_free", "positive": "stress_free"},
        "z": {"negative": 0, "positive": displace},
    },
)

grid_bb = list(bbox)
for i in range(0, 6, 2):
    grid_bb[i] += 1e-4
for i in range(1, 6, 2):
    grid_bb[i] -= 1e-4
sim.set_parameters(
    "grid_properties", number_of_elements=grid_elements, bounding_box=grid_bb
)

# Bakes material params -> neml2-compile -> AOTI package -> launches puma-opt.
sim.run(ncore=ncore)
print(
    f"\nCPFE simulation launched; outputs will appear under {outputdir}/simulation_out"
)
