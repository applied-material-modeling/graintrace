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

"""MOOSE/PUMA crystal-plasticity FE simulation runner (NEML2 v3 / AOTI)."""

from __future__ import annotations

import copy
import importlib
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .orientation_helper import load_orientations


class CPFESimulation:
    """Configure and run a MOOSE/PUMA crystal-plasticity FE simulation (NEML2 v3 / AOTI).

    Given a mesh, per-grain MRP orientations, and optional initial elastic strain, this
    writes the MOOSE input decks, bakes the material parameters into the NEML2 model,
    AOTI-compiles it with ``neml2-compile``, and launches ``puma-opt``. Configure via
    ``set_parameters(section, **kwargs)`` for the ``material``, ``simulation_parameters``,
    ``boundary``, and ``grid_properties`` sections, then call ``run(ncore=...)``. See
    ``examples/demonstrate_cpfe.py`` and the ``/cpfe-simulation`` skill.
    """

    # Default parameters for CPFE simulation
    DEFAULT_PARAMS = {
        "simulation_parameters": {
            "base_folder": "simulation_out",
            "sync_times": "0.1 1.0 2.0 3.0 4.0 5.0",
            "dt": 0.1,
            "total_time": 5.0,
            "strain_unit_conversion": 1.0,
            "initialize_time": 1.0,
            # Runtime device for the MOOSE [NEML2] action: "cpu", "cuda:0", etc.
            "device": "cpu",
            # Per-device chunk size for the NEML2 scheduler; 0 = whole batch at once.
            "device_batch": 0,
            # Devices to build AOTI packages for. None -> {cpu|cuda} from `device`.
            "compile_devices": None,
            # neml2-compile `--load` extensions. None -> auto-locate R2IncrementToRate.py.
            "neml2_load_files": None,
            # Recompile the AOTI package for this run (material params are baked in).
            "recompile": True,
            # Extra dirs for LD_LIBRARY_PATH. None -> auto-derive from moose_run_file.
            "extra_ld_library_paths": None,
            # MPI launcher for puma-opt: "mpiexec" (default) or "srun" (Cray/Slurm).
            "launcher": "mpiexec",
            # Regular-grid MultiApp transfer frequency:
            #   "final" (default) -> transfer/grid-output only at the last step;
            #   "per_step" -> every step (needed for on-the-fly REI at each step);
            #   "off" -> never (pure solve; resample offline with GridResampler).
            "grid_transfer": "final",
            # Native-mesh Exodus write frequency:
            #   "sync" (default) -> only at sync_times; "per_step" -> every step.
            "exodus_output": "sync",
            # CPFE-mesh element-centroid CSV (mesh_out/out_element_centroid_*.csv):
            #   "sync" (default) -> only at sync_times; "per_step" -> every step;
            #   "off" -> not written. Crisp true-mesh fields (no transfer/resampling)
            #   for full-mesh REI.
            "mesh_csv": "sync",
            # Distributed mesh for large problems. False (default) -> replicated mesh
            #   (every rank holds the full mesh). True -> pre-split the mesh once
            #   (`--split-mesh ncore`) and run with `--use-split` so each rank reads
            #   only its partition (low per-rank memory). Distributed mesh is pre-split
            #   ONLY (no in-situ option) and requires ncore >= 2. Needs a puma-opt build
            #   with the EqualValueBoundaryConstraint distributed-mesh fix.
            "distributed_mesh": False,
        },
        "material": {
            "slip_constant_strength": 130.0,
            "voce_hardening_initial_slope": 1556.09,
            "voce_hardening_saturation": 100.0,
            "power_slip_n": 20,
            "power_slip_g0": 0.0001,
            "elastic_E": 209016,
            "elastic_nu": 0.307,
            "elastic_G": 60355.0,
            "burger_scale": 2.22,
        },
        "boundary": {
            "bounding_box": [
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
            ],  # xlo, xhi, ylo, yhi, zlo, zhi
            "fix_tolerance": 1e-8,
            "bc": {
                "x": {"negative": "stress_free", "positive": "stress_free"},
                "y": {"negative": "stress_free", "positive": "stress_free"},
                "z": {"negative": 0, "positive": 0.001},
            },
            "bounding_box_buffer": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "grid_properties": {
            "number_of_elements": [20, 20, 20],
            "bounding_box": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        },
    }

    def __init__(
        self,
        mesh_file,
        save_simulation_folder,
        moose_run_file,
        element_order="SECOND",
        eeres_file=None,
        ori_file=None,
        use_ff_initial_field=False,
        dim=3,
    ):
        """Configure a CPFE run from a mesh, orientations, and optional initial strain.

        Args:
            mesh_file: Path to the FE mesh. Default is a SCULPT/voxel hex Exodus (.e) from VoxelMeshBuilder; a GMSH tet .msh is the fallback. Must exist.
            save_simulation_folder: Output directory for the generated input decks, AOTI package, and results. Created if needed; may not be named 'cpfe_base'.
            moose_run_file: Path to the built PUMA binary (puma-opt) used to launch the simulation. Must exist.
            element_order: FE shape-function order, "FIRST" or "SECOND". Use "FIRST" for SCULPT/voxel hex meshes and "SECOND" for a GMSH tet .msh. Default "SECOND".
            eeres_file: Optional per-grain initial elastic-strain CSV (residual strain). If None, a zero 12-column strain file is written (no residual strain).
            ori_file: Path to per-grain orientations, either 3-column neml2 MRP or 9-column rotation matrix. Required for a real run.
            use_ff_initial_field: True when the mesh and eeres_file are co-registered FF data (uses initial_conditions_ff.i); False when eeres_file comes from a different mesh.
            dim: Spatial dimension, 2 or 3. Only dim=3 is currently supported by run(). Default 3.
        """
        self.mesh_file = Path(mesh_file)
        if not self.mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_file}")

        self.eeres_file = Path(eeres_file) if eeres_file else None
        if eeres_file and not self.eeres_file.exists():
            raise FileNotFoundError(f"Elastic strain file not found: {self.eeres_file}")

        self.ori_file = Path(ori_file) if ori_file else None
        if ori_file and not self.ori_file.exists():
            raise FileNotFoundError(f"Orientation file not found: {self.ori_file}")

        self.save_simulation_folder = Path(save_simulation_folder).resolve()
        if self.save_simulation_folder.name == "cpfe_base":
            raise ValueError("save_simulation_folder cannot be 'cpfe_base'")
        self.save_simulation_folder.mkdir(parents=True, exist_ok=True)

        self.moose_run_file = Path(moose_run_file).resolve()
        if not self.moose_run_file.exists():
            raise FileNotFoundError(f"MOOSE run file not found: {self.moose_run_file}")

        if dim not in (2, 3):
            raise ValueError(f"Invalid dimension {dim}. Must be 2 or 3.")
        self.dim = dim

        if element_order not in ("FIRST", "SECOND"):
            raise ValueError(
                f"Invalid element_order {element_order}. Must be 'FIRST' or 'SECOND'."
            )

        self.element_order = element_order
        self.params = copy.deepcopy(self.DEFAULT_PARAMS)
        self.ncell_ff = None
        self.use_ff_initial_field = use_ff_initial_field

    def set_parameters(self, section, **kwargs):
        """Update one parameter section in place with keyword overrides.

        Recognized sections and their common keys:

        "material": slip_constant_strength, voce_hardening_initial_slope, voce_hardening_saturation, power_slip_n, power_slip_g0, elastic_E, elastic_nu, elastic_G, burger_scale (MPa/micrometer conventions per the model).

        "simulation_parameters": dt, total_time, initialize_time (load ramps from initialize_time to total_time),
        device ("cpu"/"cuda:N"/space-separated list), device_batch (per-device NEML2 chunk size; 0 = whole batch),
        sync_times (space-separated string of MOOSE grid-output times), grid_transfer ("final"|"per_step"|"off"),
        exodus_output ("sync"|"per_step"), mesh_csv ("sync"|"per_step"|"off"), launcher ("mpiexec"|"srun"),
        distributed_mesh (bool; True pre-splits the mesh to ncore and runs --use-split, requires ncore>=2),
        recompile, compile_devices, neml2_load_files, extra_ld_library_paths, base_folder, strain_unit_conversion.

        "boundary": bounding_box ([xlo,xhi,ylo,yhi,zlo,zhi]), bc (per-axis dict of "negative"/"positive" values, each "stress_free" or a numeric displacement), fix_tolerance, bounding_box_buffer.

        "grid_properties": number_of_elements ([nx,ny,nz]) and bounding_box (the inset regular-grid box, typically the BC box shrunk by 0.0001 per face).

        Args:
            section: One of "material", "simulation_parameters", "boundary", "grid_properties".
            **kwargs: Keys to override within that section; unknown keys are added as-is.
        """
        if section not in self.params:
            raise KeyError(f"Unknown parameter section: {section}")
        self.params[section].update(kwargs)

    def get_section(self, section):
        """Return the parameter dict for the given section."""
        if section not in self.params:
            raise KeyError(f"Unknown parameter section: {section}")
        return self.params[section]

    def validate_geometry_and_bcs(self):
        """Validate bounding box and boundary conditions based on dimension."""
        b = self.params["boundary"]
        bb = b["bounding_box"]

        if not isinstance(bb, (list, tuple)):
            raise ValueError("bounding_box must be a list or tuple.")
        if self.dim == 3 and len(bb) != 6:
            raise ValueError(
                "For dim=3, bounding_box must have 6 entries: [xlo, xhi, ylo, yhi, zlo, zhi]."
            )
        if self.dim == 2 and len(bb) != 4:
            raise ValueError(
                "For dim=2, bounding_box must have 4 entries: [xlo, xhi, ylo, yhi]."
            )

        if not all(isinstance(v, (int, float)) for v in bb):
            raise ValueError("All bounding_box entries must be numeric.")

        if self.dim == 3:
            xlo, xhi, ylo, yhi, zlo, zhi = bb
            if not (xhi > xlo and yhi > ylo and zhi > zlo):
                raise ValueError(
                    f"Invalid 3D bounding_box: {bb}. Upper bounds must exceed lower bounds."
                )
        else:  # dim == 2
            xlo, xhi, ylo, yhi = bb
            if not (xhi > xlo and yhi > ylo):
                raise ValueError(
                    f"Invalid 2D bounding_box: {bb}. Upper bounds must exceed lower bounds."
                )

        for axis in ("x", "y") + (("z",) if self.dim == 3 else ()):
            for side in ("negative", "positive"):
                val = b["bc"][axis][side]
                if not (val == "stress_free" or isinstance(val, (int, float))):
                    raise ValueError(
                        f"Invalid BC value for {axis}/{side}: {val!r}. "
                        "Must be either 'stress_free' or a numeric value."
                    )

    def write_bc_file(self):
        """Write boundary_conditions.i and return the fix/roll node coordinates."""
        self.validate_geometry_and_bcs()
        b = self.params["boundary"]
        sim = self.params["simulation_parameters"]

        if self.dim == 3:
            xlo, _xhi, ylo, yhi, zlo, _zhi = b["bounding_box"]
            xbufflo, _xbuffhi, ybufflo, ybuffhi, zbufflo, _zbuffhi = b[
                "bounding_box_buffer"
            ]
        else:
            xlo, _xhi, ylo, yhi = b["bounding_box"]
            xbufflo, _xbuffhi, ybufflo, ybuffhi = b["bounding_box_buffer"]
            zlo, _zhi = 0.0, 0.0
            zbufflo, _zbuffhi = 0.0, 0.0

        fixnode_x, fixnode_y, fixnode_z = xlo + xbufflo, ylo + ybufflo, zlo + zbufflo
        yroll_x, yroll_y, yroll_z = xlo + xbufflo, yhi + ybuffhi, zlo + zbufflo

        coupled_axes = set()

        out = self.save_simulation_folder / "boundary_conditions.i"
        with open(out, "w", encoding="utf-8") as f:
            f.write("[BCs]\n")
            f.write("    ## BCs for all stages\n")

            # Shared fixed BCs
            shared_blocks = [
                ("fixnode_x", "disp_x", "fixnode"),
                ("fixnode_y", "disp_y", "fixnode"),
                ("fixnode_z", "disp_z", "fixnode"),
                ("yrollnode_x", "disp_x", "yrollnode"),
                ("yrollnode_z", "disp_z", "yrollnode"),
            ]
            for label, var, boundary in shared_blocks:
                f.write(f"    [{label}]\n")
                f.write("        type = DirichletBC\n")
                f.write(f"        variable = {var}\n")
                f.write(f"        boundary = {boundary}\n")
                f.write("        value = 0\n")
                f.write("    []\n")

            # User-defined BCs
            coupled_bcs = []

            for axis in ("x", "y", "z") if self.dim == 3 else ("x", "y"):
                for side in ("negative", "positive"):
                    val = b["bc"][axis][side]
                    if val == "stress_free":
                        continue
                    boundary_name = {
                        ("x", "negative"): "left",
                        ("x", "positive"): "right",
                        ("y", "negative"): "bottom",
                        ("y", "positive"): "top",
                        ("z", "negative"): "back",
                        ("z", "positive"): "front",
                    }[(axis, side)]

                    if abs(val) < 1e-15:
                        f.write(f"    [{boundary_name}_boundary]\n")
                        f.write("        type = DirichletBC\n")
                        f.write(f"        variable = disp_{axis}\n")
                        f.write(f"        boundary = {boundary_name}\n")
                        f.write("        value = 0\n")
                        f.write("    []\n")
                        continue

                    # Nonzero numeric BC (ramped), per boundary
                    func_name = f"ramping_load_{axis}_{boundary_name}"
                    coupled_axes.add(axis)
                    coupled_bcs.append((axis, boundary_name, float(val)))

                    f.write("    # only turn on during the loading state\n")
                    f.write(f"    [{boundary_name}_boundary]\n")
                    f.write("        type = CoupledDirichletBC\n")
                    f.write(f"        variable = disp_{axis}\n")
                    f.write(f"        boundary = {boundary_name}\n")
                    f.write(f"        coupled_variable = disp_{axis}_residual\n")
                    f.write(f"        function = {func_name}\n")
                    f.write("        enable = false\n")
                    f.write("    []\n")
                    f.write("    #\n")

            f.write("[]\n\n")

            # [Functions]: per-axis ramping functions
            f.write("[Functions]\n")
            for axis, boundary_name, val in coupled_bcs:
                func_name = f"ramping_load_{axis}_{boundary_name}"
                f.write(f"    [{func_name}]\n")
                f.write("        type = ParsedFunction\n")
                f.write(
                    "        expression = "
                    f"'if(t <= {sim['initialize_time']:.12g}, 0, "
                    f"if(t < {sim['total_time']:.12g}, "
                    f"{val:.12g} * (t - {sim['initialize_time']:.12g}) / "
                    f"({sim['total_time']:.12g} - {sim['initialize_time']:.12g}), "
                    f"{val:.12g}))'\n"
                )
                f.write("    []\n")
            f.write("[]\n\n")

            f.write("[AuxVariables]\n")
            for axis in sorted(coupled_axes):
                f.write(f"    [disp_{axis}_residual]\n")
                f.write(f"        order = {self.element_order}\n")
                f.write("        family = LAGRANGE\n")
                f.write("    []\n")
            f.write("[]\n\n")

            f.write("[AuxKernels]\n")
            for axis in sorted(coupled_axes):
                f.write(f"    [disp_{axis}_residual]\n")
                f.write("        type = CopyValueAux\n")
                f.write(f"        variable = disp_{axis}_residual\n")
                f.write(f"        source = disp_{axis}\n")
                f.write("        enable = true\n")
                f.write("    []\n")
            f.write("[]\n\n")

            # Front face flat constraint
            f.write("[Constraints]\n")
            f.write("    [zface_flat]\n")
            f.write("        type = EqualValueBoundaryConstraint\n")
            f.write("        secondary = 'front'\n")
            f.write("        variable = disp_z\n")
            f.write("        penalty = 1e6\n")
            f.write("        enable = true\n")
            f.write("    []\n")
            f.write("[]\n\n")

            f.write("[Controls]\n")
            f.write("    [switch_loading]\n")
            f.write("        type = TimePeriod\n")
            f.write(f"        start_time = {sim['initialize_time']:.12g}\n")
            f.write(f"        end_time = {(sim['total_time']+1):.12g}\n")

            enable_objs = [f"BCs::{bname}_boundary" for _, bname, _ in coupled_bcs]
            disable_objs = [
                f"AuxKernels::disp_{axis}_residual" for axis, _, _ in coupled_bcs
            ]

            f.write(f"        enable_objects = '{' '.join(enable_objs)}'\n")
            f.write(f"        disable_objects = '{' '.join(disable_objs)}'\n")
            f.write("        execute_on = 'TIMESTEP_BEGIN'\n")
            f.write("    []\n")
            f.write("[]\n")

        return fixnode_x, fixnode_y, fixnode_z, yroll_x, yroll_y, yroll_z

    def write_postprocess_file(self, ncell=None):
        """Generate grain_average_postprocessor.i."""

        out = self.save_simulation_folder / "grain_average_postprocessor.i"

        if self.eeres_file is None:
            if ncell is None or not isinstance(ncell, int):
                raise ValueError(
                    "ncell must be provided as an integer when eeres_file is None"
                )

            # 12 columns (x, y, z + 9 strain), all zero: matches the nprop=12
            # PropertyReadFile in initial_conditions[_ff].i (block read -> row
            # maps to grain, coords unused), giving zero residual strain.
            ee_file = self.save_simulation_folder / "zero_initial_strain.ee"
            strain_data = np.zeros((ncell, 12), dtype=float)
            np.savetxt(ee_file, strain_data, fmt="%.12g")

            self.eeres_file = ee_file
            self.ncell_ff = ncell
        else:
            df = pd.read_csv(
                self.eeres_file, sep=r"[,\s]+", engine="python", header=None
            )
            self.ncell_ff = df.shape[0]
            shutil.copy(
                self.eeres_file,
                self.save_simulation_folder / Path(self.eeres_file).name,
            )

        with open(out, "w", encoding="utf-8") as f:
            f.write("[Postprocessors]\n")
            f.write("    # Automatically generated strain postprocessors\n")

            components = ["xx", "yy", "zz", "xy", "yz", "xz"]
            for comp in components:
                f.write(f"    # --- strain_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [strain_{comp}_{i}]\n")
                    f.write("        type = ElementAverageValue\n")
                    f.write(f"        variable = strain_{comp}\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")
                f.write(f"    # --- ee_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [ee_{comp}_{i}]\n")
                    f.write("        type = ElementAverageValue\n")
                    f.write(f"        variable = ee_{comp}\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")
                f.write(f"    # --- cauchy_stress_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [cauchy_stress_{comp}_{i}]\n")
                    f.write("        type = ElementAverageValue\n")
                    f.write(f"        variable = cauchy_stress_{comp}\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")

            components = ["11", "12", "13", "21", "22", "23", "31", "32", "33"]
            for comp in components:
                f.write(f"    # --- nye_tensor_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [nye_tensor_{comp}_{i}]\n")
                    f.write("        type = ElementAverageValue\n")
                    f.write(f"        variable = nye_tensor_{comp}\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")

            components = ["x", "y", "z"]
            for comp in components:
                f.write(f"    # --- centroid_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [centroid_{comp}_{i}]\n")
                    f.write("        type = FunctionElementAverage\n")
                    f.write(f"        function = coord_{comp}\n")
                    f.write("        use_displaced_mesh = True\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")
                f.write(f"    # --- ori_rodrigues_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [ori_rodrigues_{comp}_{i}]\n")
                    f.write("        type = ElementAverageValue\n")
                    f.write(f"        variable = ori_rodrigues_{comp}\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")

            f.write("    # --- volume ---\n")
            for i in range(1, ncell + 1):
                f.write(f"    [volume_{i}]\n")
                f.write("        type = ElementAverageValue\n")
                f.write("        variable = volume\n")
                f.write("        use_displaced_mesh = True\n")
                f.write(f"        block = {i}\n")
                f.write("    []\n")
            f.write("\n")

            f.write("[]\n")

    def write_orientation_file(self):
        """Write mrps_orientation.csv from the orientation file; return grain count."""
        df = pd.read_csv(self.ori_file, sep=r"[,\s]+", engine="python", header=None)

        # 9 columns = rotation matrix
        if df.shape[1] == 9:

            mrps = load_orientations(df, field=None)

            np.savetxt(
                self.save_simulation_folder / "mrps_orientation.csv",
                mrps.numpy(),
                delimiter=",",
                comments="",
                fmt="%.12g",
            )
        # 3 columns = MRPs, copy as-is
        elif df.shape[1] == 3:
            # pylint: disable=import-outside-toplevel  # torch is a heavy optional dep
            import torch

            shutil.copy(
                self.ori_file, self.save_simulation_folder / "mrps_orientation.csv"
            )
            mrps = torch.tensor(df.values, dtype=torch.float32)
        else:
            raise ValueError(
                "Orientation file must have either 3 (MRPs) or 9 (rotation matrix) columns."
            )

        return mrps.shape[0]

    def _compile_device_set(self):
        """Devices to build AOTI packages for. None -> {cpu|cuda} from `device`."""
        sim = self.params["simulation_parameters"]
        if sim.get("compile_devices"):
            return [str(d) for d in sim["compile_devices"]]
        dev = str(sim["device"]).lower()
        return ["cuda"] if "cuda" in dev else ["cpu"]

    def _resolve_neml2_load_files(self):
        """Locate the neml2-compile `--load` extensions (R2IncrementToRate.py)."""
        sim = self.params["simulation_parameters"]
        loads = sim.get("neml2_load_files")
        if loads:
            files = [Path(f) for f in loads]
        else:
            rel = "modules/solid_mechanics/data/neml2/R2IncrementToRate.py"
            candidates = []
            moose_dir = os.environ.get("MOOSE_DIR")
            if moose_dir:
                candidates.append(Path(moose_dir) / rel)
            # sibling moose checkout
            candidates.append(self.moose_run_file.parent.parent / "moose" / rel)
            files = [c for c in candidates if c.exists()][:1]
            if not files:
                raise FileNotFoundError(
                    "Could not locate R2IncrementToRate.py for `neml2-compile --load`. "
                    "Set simulation_parameters['neml2_load_files'] or export MOOSE_DIR."
                )
            need_puma_ext = False
            try:
                import neml2  # pylint: disable=import-outside-toplevel

                registry = neml2.factory._registry  # pylint: disable=protected-access
                need_puma_ext = "R2LinearCombination" not in registry
            except ImportError:
                pass
            if need_puma_ext:
                pkg = self.moose_run_file.parent / "neml2_models" / "python"
                if pkg.exists():
                    files.append(pkg)
        missing = [f for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"neml2_load_files not found: {missing}")
        return [str(f.resolve()) for f in files]

    def _bake_neml2_model(self, src, dst):
        """Substitute @BAKE-marked material params into the model .i for AOTI compile."""
        text = Path(src).read_text(encoding="utf-8")
        m = self.params["material"]
        enu_g = f"{m['elastic_E']:.12g} {m['elastic_nu']:.12g} {m['elastic_G']:.12g}"
        text, n = re.subn(
            r"(coefficients = ')[^']*(')(\s*#\s*@BAKE E nu G)",
            rf"\g<1>{enu_g}\g<2>\g<3>",
            text,
        )
        if n != 1:
            raise RuntimeError("Failed to bake 'E nu G' @BAKE marker in neml2_cpfe.i")
        singles = {
            "power_slip_n": m["power_slip_n"],
            "power_slip_g0": m["power_slip_g0"],
            "slip_constant_strength": m["slip_constant_strength"],
            "voce_hardening_initial_slope": m["voce_hardening_initial_slope"],
            "voce_hardening_saturation": m["voce_hardening_saturation"],
        }
        for marker, val in singles.items():
            text, n = re.subn(
                rf"(=\s*)\S+(\s*#\s*@BAKE {re.escape(marker)}\b)",
                rf"\g<1>{float(val):.12g}\g<2>",
                text,
            )
            if n != 1:
                raise RuntimeError(f"Failed to bake '@BAKE {marker}' in neml2_cpfe.i")
        Path(dst).write_text(text, encoding="utf-8")

    def _compile_neml2_model(self, model_i, out_dir, devices, load_files):
        """Run `neml2-compile` to produce the AOTI package + stub; return the stub."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "neml2-compile",
            str(model_i),
            "--model",
            "model",
            "--dtype",
            "float64",
            "--device",
            *devices,
            "--output-dir",
            str(out_dir),
            "-d",
            "neml2_stress:spatial_deformation_gradient_increment",
        ]
        for lf in load_files:
            cmd += ["--load", lf]

        env = os.environ.copy()
        env.setdefault("CC", "x86_64-conda-linux-gnu-gcc")
        env.setdefault("CXX", "x86_64-conda-linux-gnu-g++")
        # AOTI cuda packages link against libcuda; use the CUDA driver stub.
        if any("cuda" in str(d) for d in devices):
            stub = "/usr/local/cuda/lib64/stubs"
            if os.path.isdir(stub):
                env["LIBRARY_PATH"] = stub + os.pathsep + env.get("LIBRARY_PATH", "")

        log = out_dir / "neml2_compile.log"
        print(f"\n==> neml2-compile ({' '.join(devices)}) -> {out_dir}", flush=True)
        with open(log, "w", encoding="utf-8") as lf:
            subprocess.run(
                cmd,
                cwd=self.save_simulation_folder,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=True,
            )
        stub = out_dir / "model_aoti.i"
        if not stub.exists():
            raise RuntimeError(f"neml2-compile produced no stub at {stub}; see {log}")
        return stub

    @staticmethod
    def _package_lib_dirs():
        """Lib dirs of the installed neml2 + torch packages.

        These hold ``libneml2_eager.so`` / ``libtorch`` that ``puma-opt`` links, so
        adding them lets the binary resolve them regardless of the build layout.
        """
        dirs = []
        for mod in ("neml2", "torch"):
            try:
                pkg = importlib.import_module(mod)
            except ImportError:
                continue
            lib = Path(pkg.__file__).parent / "lib"
            if lib.is_dir():
                dirs.append(str(lib))
        return dirs

    def _runtime_env(self):
        """Env for the puma-opt run: add libtorch + PETSc lib dirs to LD_LIBRARY_PATH.

        Auto-derives from the moose_run_file repo layout, or override via
        simulation_parameters['extra_ld_library_paths']. The installed neml2 + torch
        package lib dirs are always added so puma-opt resolves libneml2_eager.so /
        libtorch regardless of the build layout.
        """
        env = os.environ.copy()
        sim = self.params["simulation_parameters"]
        paths = []
        extra = sim.get("extra_ld_library_paths")
        if extra:
            paths.extend(str(p) for p in extra)
        else:
            root = self.moose_run_file.parent.parent
            for cand in (
                root / "libtorch" / "lib",
                root / "moose" / "petsc" / "arch-moose" / "lib",
            ):
                if cand.is_dir():
                    paths.append(str(cand))
        paths.extend(self._package_lib_dirs())
        if paths:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                paths + ([existing] if existing else [])
            )
        return env

    @staticmethod
    def _split_command(
        moose_run_file, deck_and_args, ncore, split_file="mesh_split.cpr"
    ):
        """Build the one-time ``--split-mesh`` pre-split command for a distributed run.

        Partitions the mesh into ``ncore`` pieces (written to ``split_file``) so the solve
        can then run with ``--use-split`` and each rank reads only its partition (low
        per-rank memory). ``deck_and_args`` is the ``-i <decks> <key=value args>`` tail of
        the solve argv, reused verbatim so the split builds the same mesh (including the
        BC-node coordinates the mesh generators need). Distributed mesh is pre-split only
        and requires at least two ranks.

        Args:
            moose_run_file: path to the puma-opt binary.
            deck_and_args: the ``["-i", <decks...>, <key=value args...>]`` argv tail.
            ncore: number of MPI ranks / split partitions (must be >= 2).
            split_file: name of the split file/dir (relative to the run folder).

        Returns:
            list[str]: the ``puma-opt ... --split-mesh <ncore> --split-file <name>`` command.

        Raises:
            ValueError: if ``ncore < 2``.
        """
        if int(ncore) < 2:
            raise ValueError(
                "distributed_mesh requires ncore >= 2; use a replicated mesh for a single rank."
            )
        return [
            str(moose_run_file),
            *deck_and_args,
            "--split-mesh",
            str(int(ncore)),
            "--split-file",
            split_file,
        ]

    def run(self, ncore=8):
        """Prepare the input decks, AOTI-compile the NEML2 model, and launch puma-opt.

        Copies the cpfe_base templates into save_simulation_folder, writes the run-specific
        boundary/orientation/postprocessor decks, bakes the material parameters into the NEML2
        model and neml2-compiles it (unless a cached AOTI package exists and recompile is False),
        then starts puma-opt as a detached background process. Only dim=3 is supported.

        When simulation_parameters['distributed_mesh'] is True, the mesh is first pre-split
        into ncore partitions (a one-time synchronous `--split-mesh` step) and the solve runs
        with `--use-split` so each rank reads only its partition (low per-rank memory for large
        meshes). This requires ncore >= 2 and a puma-opt build with the
        EqualValueBoundaryConstraint distributed-mesh fix.

        Args:
            ncore: Number of MPI ranks to launch (passed as -n to the launcher). For multi-GPU runs set this to the number of GPUs in the device list. With distributed_mesh=True this must be >= 2. Default 8.
        """

        self.validate_geometry_and_bcs()

        if self.dim == 2:
            raise NotImplementedError(
                "Input file generation for dim=2 is not yet implemented. "
                "Currently only dim=3 simulations are supported."
            )

        # cpfe_base is a reserved folder containing base files
        cpfe_base = Path(__file__).parent / "cpfe_base"
        if not cpfe_base.exists():
            raise FileNotFoundError("cpfe_base folder not found.")

        # shared base files
        if self.use_ff_initial_field:
            base_files = [
                "initial_conditions_ff.i",
                "run_cpfe.i",
                "grid_file.i",
                "transfer.i",
            ]
            initial_conditions_file = "initial_conditions_ff.i"
        else:
            base_files = [
                "initial_conditions.i",
                "run_cpfe.i",
                "grid_file.i",
                "transfer.i",
            ]
            initial_conditions_file = "initial_conditions.i"

        for fname in base_files:
            src = cpfe_base / fname
            dst = self.save_simulation_folder / fname
            if not src.exists():
                raise FileNotFoundError(f"Required base file missing: {src}")
            shutil.copy(src, dst)

        shutil.copy(self.mesh_file, self.save_simulation_folder / self.mesh_file.name)

        # generate the run-specific input files
        fixnode_x, fixnode_y, fixnode_z, yroll_x, yroll_y, yroll_z = (
            self.write_bc_file()
        )
        ncells = self.write_orientation_file()
        self.write_postprocess_file(ncell=ncells)

        # Bake material params from the cpfe_base template directly into a single
        # neml2_cpfe.i in the run folder, then AOTI-compile it; produces
        # aoti/model_aoti.i, referenced by run_cpfe.i via ${neml2_stub}.
        sim = self.params["simulation_parameters"]
        baked_model = self.save_simulation_folder / "neml2_cpfe.i"
        self._bake_neml2_model(cpfe_base / "neml2_cpfe.i", baked_model)
        aoti_dir = self.save_simulation_folder / "aoti"
        neml2_stub = "aoti/model_aoti.i"
        if sim.get("recompile", True) or not (aoti_dir / "model_aoti.i").exists():
            self._compile_neml2_model(
                baked_model,
                aoti_dir,
                self._compile_device_set(),
                self._resolve_neml2_load_files(),
            )

        if self.use_ff_initial_field:
            ncell_args = [f"ncell={ncells}"]
        else:
            ncell_args = [f"ncell={ncells:.12g}", f"ncell_ff={self.ncell_ff:.12g}"]

        grid_info = self.params["grid_properties"]
        ncell_x, ncell_y, ncell_z = grid_info["number_of_elements"]
        grid_bbox = grid_info["bounding_box"]

        vol_correction_cond = "true" if self.element_order == "FIRST" else "false"

        sim_p = self.params["simulation_parameters"]
        grid_transfer = str(sim_p.get("grid_transfer", "final")).lower()
        grid_exec = {
            "final": "FINAL",
            "per_step": "INITIAL TIMESTEP_END",
            "off": "NONE",
        }.get(grid_transfer)
        if grid_exec is None:
            raise ValueError(
                f"Invalid grid_transfer {grid_transfer!r}. Must be 'final', 'per_step', or 'off'."
            )
        exodus_output = str(sim_p.get("exodus_output", "sync")).lower()
        if exodus_output not in ("sync", "per_step"):
            raise ValueError(
                f"Invalid exodus_output {exodus_output!r}. Must be 'sync' or 'per_step'."
            )
        exodus_sync_only = "true" if exodus_output == "sync" else "false"
        mesh_csv = str(sim_p.get("mesh_csv", "sync")).lower()
        if mesh_csv not in ("sync", "per_step", "off"):
            raise ValueError(
                f"Invalid mesh_csv {mesh_csv!r}. Must be 'sync', 'per_step', or 'off'."
            )
        mesh_sampler_execute_on = "NONE" if mesh_csv == "off" else "TIMESTEP_END"
        mesh_csv_sync_only = "true" if mesh_csv == "sync" else "false"
        distributed_mesh = bool(sim_p.get("distributed_mesh", False))

        log_path = self.save_simulation_folder / "cpfe_run.log"
        launcher = shlex.split(
            str(self.params["simulation_parameters"].get("launcher", "mpiexec"))
        )
        argv = [
            "nohup",
            *launcher,
            "-n",
            str(ncore),
            str(self.moose_run_file),
            "-i",
            "run_cpfe.i",
            "boundary_conditions.i",
            initial_conditions_file,
            "grain_average_postprocessor.i",
            "transfer.i",
            "orientation_file=mrps_orientation.csv",
            f"neml2_stub={neml2_stub}",
            f"sync_times={self.params['simulation_parameters']['sync_times']}",
            f"device_neml2={self.params['simulation_parameters']['device']}",
            f"device_neml2_batch={self.params['simulation_parameters']['device_batch']:.12g}",
            f"mesh_file={self.mesh_file.name}",
            f"residual_strain_file={self.eeres_file.name}",
            f"base_folder={self.params['simulation_parameters']['base_folder']}",
            f"dt={self.params['simulation_parameters']['dt']:.12g}",
            f"total_time={self.params['simulation_parameters']['total_time']:.12g}",
            f"strain_unit_conversion={self.params['simulation_parameters']['strain_unit_conversion']:.12g}",
            # burger_scale is the only material param still passed at runtime.
            f"burger_scale={self.params['material']['burger_scale']:.12g}",
            f"fixnode_x={fixnode_x:.12g}",
            f"fixnode_y={fixnode_y:.12g}",
            f"fixnode_z={fixnode_z:.12g}",
            f"yroll_x={yroll_x:.12g}",
            f"yroll_y={yroll_y:.12g}",
            f"yroll_z={yroll_z:.12g}",
            f"vol_lock_correction_cond={vol_correction_cond}",
            f"grid_transfer_execute_on={grid_exec}",
            f"exodus_sync_only={exodus_sync_only}",
            f"mesh_sampler_execute_on={mesh_sampler_execute_on}",
            f"mesh_csv_sync_only={mesh_csv_sync_only}",
            f"grid_nx={ncell_x:.12g}",
            f"grid_ny={ncell_y:.12g}",
            f"grid_nz={ncell_z:.12g}",
            f"grid_xmin={grid_bbox[0]:.12g}",
            f"grid_xmax={grid_bbox[1]:.12g}",
            f"grid_ymin={grid_bbox[2]:.12g}",
            f"grid_ymax={grid_bbox[3]:.12g}",
            f"grid_zmin={grid_bbox[4]:.12g}",
            f"grid_zmax={grid_bbox[5]:.12g}",
        ] + ncell_args

        # Distributed mesh: pre-split the mesh once (--split-mesh), then run with
        # --use-split so each rank reads only its partition (low per-rank memory).
        # Pre-split is the only distributed path (no in-situ --distributed-mesh); the
        # split reuses the exact -i decks + args so it builds the same mesh + BC nodesets.
        if distributed_mesh:
            split_file = "mesh_split.cpr"
            deck_and_args = argv[argv.index("-i") :]
            split_cmd = self._split_command(
                self.moose_run_file, deck_and_args, ncore, split_file
            )
            print(f"\n==> Pre-splitting mesh {ncore} ways -> {split_file}", flush=True)
            split_log = self.save_simulation_folder / "cpfe_split.log"
            with open(split_log, "w", encoding="utf-8") as sf:
                subprocess.run(
                    split_cmd,
                    cwd=self.save_simulation_folder,
                    env=self._runtime_env(),
                    stdout=sf,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            argv += ["--use-split", "--split-file", split_file]

        # Run as a persistent background process
        print(
            f"\n==> Running CPFE simulation in {self.save_simulation_folder}",
            flush=True,
        )
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w", buffering=1, encoding="utf-8") as lf:
            # pylint: disable=consider-using-with  # persistent detached background run
            proc = subprocess.Popen(
                argv,
                cwd=self.save_simulation_folder,
                env=self._runtime_env(),
                stdin=subprocess.DEVNULL,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                close_fds=True,
            )

        print(f"CPFE simulation started, PID={proc.pid}")
