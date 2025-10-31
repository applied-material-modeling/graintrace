import subprocess, sys
import shutil
from pathlib import Path
import numpy as np
import copy
from orientation_helper import load_orientations
import pandas as pd

class CPFESimulation:
    
    # Default required informations for CPFE simulation
    DEFAULT_PARAMS = {
        "simulation_parameters": {
            "base_folder": "simulation_out",
            "dt": 0.1,
            "total_time": 5.0,
            "strain_unit_conversion": 1.0,
            "initialize_time": 1.0,
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
        },
        "boundary": {
            "bounding_box": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], #xlo, xhi, ylo, yhi, zlo, zhi
            "fix_tolerance": 1e-8,    
            "bc": {
                "x": {"negative": "stress_free", "positive": "stress_free"},
                "y": {"negative": "stress_free", "positive": "stress_free"},
                "z": {"negative": 0, "positive": 0.001},
                },
        }
    }

    def __init__(self, 
                mesh_file,
                save_simulation_folder,
                moose_run_file,
                eeres_file=None, 
                ori_file=None,
                dim=3):
        
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

        self.params = copy.deepcopy(self.DEFAULT_PARAMS)

    def set_parameters(self, section, **kwargs):
        if section not in self.params:
            raise KeyError(f"Unknown parameter section: {section}")
        self.params[section].update(kwargs)
    
    def get_section(self, section):
        if section not in self.params:
            raise KeyError(f"Unknown parameter section: {section}")
        return self.params[section]
    
    def validate_geometry_and_bcs(self):
        """Validate bounding box and boundary conditions based on dimension."""
        b = self.params["boundary"]
        bb = b["bounding_box"]

        # --- Bounding box check ---
        if not isinstance(bb, (list, tuple)):
            raise ValueError("bounding_box must be a list or tuple.")
        if self.dim == 3 and len(bb) != 6:
            raise ValueError("For dim=3, bounding_box must have 6 entries: [xlo, xhi, ylo, yhi, zlo, zhi].")
        if self.dim == 2 and len(bb) != 4:
            raise ValueError("For dim=2, bounding_box must have 4 entries: [xlo, xhi, ylo, yhi].")

        if not all(isinstance(v, (int, float)) for v in bb):
            raise ValueError("All bounding_box entries must be numeric.")

        # --- Physical ordering check ---
        if self.dim == 3:
            xlo, xhi, ylo, yhi, zlo, zhi = bb
            if not (xhi > xlo and yhi > ylo and zhi > zlo):
                raise ValueError(f"Invalid 3D bounding_box: {bb}. Upper bounds must exceed lower bounds.")
        else:  # dim == 2
            xlo, xhi, ylo, yhi = bb
            if not (xhi > xlo and yhi > ylo):
                raise ValueError(f"Invalid 2D bounding_box: {bb}. Upper bounds must exceed lower bounds.")

        # --- Boundary conditions check ---
        for axis in ("x", "y") + (("z",) if self.dim == 3 else ()):
            for side in ("negative", "positive"):
                val = b["bc"][axis][side]
                if not (val == "stress_free" or isinstance(val, (int, float))):
                    raise ValueError(
                        f"Invalid BC value for {axis}/{side}: {val!r}. "
                        "Must be either 'stress_free' or a numeric value."
                    )

    def write_bc_file(self):
        self.validate_geometry_and_bcs()
        b = self.params["boundary"]
        sim = self.params["simulation_parameters"]

        if self.dim == 3:
            xlo, xhi, ylo, yhi, zlo, zhi = b["bounding_box"]
        else:
            xlo, xhi, ylo, yhi = b["bounding_box"]
            zlo, zhi = 0.0, 0.0

        fixnode_x, fixnode_y, fixnode_z = xlo, ylo, zlo
        yroll_x, yroll_y, yroll_z = xlo, yhi, zlo

        coupled_axes = set()
        coupled_boundaries = []  # (axis, boundary_name)

        out = self.save_simulation_folder / "boundary_conditions.i"
        with open(out, "w") as f:
            # [BCs]
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
            for axis in ("x", "y", "z") if self.dim == 3 else ("x", "y"):
                for side in ("negative", "positive"):
                    val = b["bc"][axis][side]
                    if val == "stress_free":
                        continue

                    label = f"{axis}_{side}"
                    boundary_name = {
                        ("x", "negative"): "left",
                        ("x", "positive"): "right",
                        ("y", "negative"): "bottom",
                        ("y", "positive"): "top",
                        ("z", "negative"): "back",
                        ("z", "positive"): "front",
                    }.get((axis, side), label)

                    # Case 1: zero BC
                    if val == "0" or (isinstance(val, (int, float)) and abs(val) < 1e-15):
                        f.write(f"    [{boundary_name}_boundary]\n")
                        f.write("        type = DirichletBC\n")
                        f.write(f"        variable = disp_{axis}\n")
                        f.write(f"        boundary = {boundary_name}\n")
                        f.write("        value = 0\n")
                        f.write("    []\n")

                    # Case 2: nonzero numeric BC (ramped)
                    elif isinstance(val, (int, float)):
                        coupled_axes.add(axis)
                        coupled_boundaries.append((axis, boundary_name))
                        f.write("    # only turn on during the loading state\n")
                        f.write(f"    [{boundary_name}_boundary]\n")
                        f.write("        type = CoupledDirichletBC\n")
                        f.write(f"        variable = disp_{axis}\n")
                        f.write(f"        boundary = {boundary_name}\n")
                        f.write(f"        coupled_variable = disp_{axis}_residual\n")
                        f.write(f"        function = ramping_load_{axis}\n")
                        f.write("        enable = false\n")
                        f.write("    []\n")
                        f.write("    #\n")

            f.write("[]\n\n")

            # [Functions] — per-axis ramping functions
            f.write("[Functions]\n")
            for axis in sorted(coupled_axes):
                total_disp_key = f"total_disp_{axis}"
                total_disp_val = sim.get(total_disp_key, 0.0)
                f.write(f"    [ramping_load_{axis}]\n")
                f.write("        type = ParsedFunction\n")
                f.write(
                    f"        expression = 'if(t <= {sim['initialize_time']:.12g}, 0, "
                    f"if(t < {sim['total_time']:.12g}, "
                    f"{total_disp_val:.12g} * (t - {sim['initialize_time']:.12g}) / "
                    f"({sim['total_time']:.12g} - {sim['initialize_time']:.12g}), "
                    f"{total_disp_val:.12g}))'\n"
                )
                f.write("    []\n")
            f.write("[]\n\n")

            # [AuxVariables]
            f.write("[AuxVariables]\n")
            for axis in sorted(coupled_axes):
                f.write(f"    [disp_{axis}_residual]\n")
                f.write("        order = SECOND\n")
                f.write("        family = LAGRANGE\n")
                f.write("    []\n")
            f.write("[]\n\n")

            # [AuxKernels]
            f.write("[AuxKernels]\n")
            for axis in sorted(coupled_axes):
                f.write(f"    [disp_{axis}_residual]\n")
                f.write("        type = CopyValueAux\n")
                f.write(f"        variable = disp_{axis}_residual\n")
                f.write(f"        source = disp_{axis}\n")
                f.write("        enable = true\n")
                f.write("    []\n")
            f.write("[]\n\n")

            # [Controls] — dynamically determined
            f.write("[Controls]\n")
            f.write("    [switch_loading]\n")
            f.write("        type = TimePeriod\n")
            f.write(f"        start_time = {sim['initialize_time']:.12g}\n")
            f.write(f"        end_time = {sim['total_time']:.12g}\n")

            enable_objs = [f"BCs::{bname}_boundary" for _, bname in coupled_boundaries]
            disable_objs = [f"AuxKernels::disp_{axis}_residual" for axis, _ in coupled_boundaries]

            f.write(f"        enable_objects = '{' '.join(enable_objs)}'\n")
            f.write(f"        disable_objects = '{' '.join(disable_objs)}'\n")
            f.write("        execute_on = 'TIMESTEP_BEGIN'\n")
            f.write("    []\n")
            f.write("[]\n")

        return fixnode_x, fixnode_y, fixnode_z, yroll_x, yroll_y, yroll_z

    def write_strain_postprocess_file(self, ncell=None):
        """
        Generate strain_postprocessor.i.
        """

        out = self.save_simulation_folder / "strain_postprocessor.i"
   
        if self.eeres_file is None:
            if ncell is None or not isinstance(ncell, int):
                raise ValueError("ncell must be provided as an integer when eeres_file is None")

            ee_file = self.save_simulation_folder / "zero_initial_strain.ee"
            strain_data = np.zeros((ncell, 9), dtype=float)
            np.savetxt(ee_file, strain_data, fmt="%.12g")

            self.eeres_file = ee_file
        else:
            shutil.copy(self.eeres_file, self.save_simulation_folder / Path(self.eeres_file).name)

        # --- Generate strain_postprocessor.i ---
        with open(out, "w") as f:
            f.write("[Postprocessors]\n")
            f.write("    # Automatically generated strain postprocessors\n")

            components = ["xx", "yy", "zz", "xy", "yz", "xz"]
            for comp in components:
                # strain_*
                f.write(f"    # --- strain_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [strain_{comp}_{i}]\n")
                    f.write("        type = ElementAverageValue\n")
                    f.write(f"        variable = strain_{comp}\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")
                # ee_* (from neml2)
                f.write(f"    # --- ee_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [ee_{comp}_{i}]\n")
                    f.write("        type = ElementAverageValue\n")
                    f.write(f"        variable = ee_{comp}\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")

            f.write("[]\n")
    
    def write_orientation_file(self):

        df = pd.read_csv(self.ori_file, sep=r"\s+", header=None, engine="python")
        # df = df.apply(pd.to_numeric, errors="coerce")

        mrps = load_orientations(df, field=None)  # existing helper untouched

        np.savetxt(
            self.save_simulation_folder / "mrps_orientation.csv",
            mrps.numpy(),
            delimiter=",",
            comments="",
            fmt="%.12g",
        )
        return mrps.shape[0]

    def run(self, ncore=8):
        """
        Prepare and run CPFE simulation.
        """

        self.validate_geometry_and_bcs()

        if self.dim == 2:
            raise NotImplementedError(
                "Input file generation for dim=2 is not yet implemented. "
                "Currently only dim=3 simulations are supported."
            )
        
        # cpfe_base is a reserved folder containing base files
        cpfe_base = Path("cpfe_base").resolve()
        if not cpfe_base.exists():
            raise FileNotFoundError("cpfe_base folder not found.")

        # list of shared base files
        base_files = ["initial_conditions.i", "neml2_cpfe.i", "run_cpfe.i"]

        for fname in base_files:
            src = cpfe_base / fname
            dst = self.save_simulation_folder / fname
            if not src.exists():
                raise FileNotFoundError(f"Required base file missing: {src}")
            shutil.copy(src, dst)
        
        # copy the mesh file
        shutil.copy(self.mesh_file, self.save_simulation_folder / self.mesh_file.name)

        # generate the new specific file
        fixnode_x, fixnode_y, fixnode_z, yroll_x, yroll_y, yroll_z = self.write_bc_file()
        ncells = self.write_orientation_file()
        self.write_strain_postprocess_file(ncell=ncells)

        # Build command for subprocess
        log_path = self.save_simulation_folder / "cpfe_run.log"
        argv = [
            "nohup",
            "mpiexec",
            "-n", str(ncore),
            str(self.moose_run_file),
            "-i",
            "run_cpfe.i",
            "boundary_conditions.i",
            "initial_conditions.i",
            "strain_postprocessor.i",
            "orientation_file=mrps_orientation.csv",
            f"mesh_file={self.mesh_file.name}",
            f"residual_strain_file={self.eeres_file.name}",
            f"ncell={ncells:.12g}",
            f"base_folder={self.params['simulation_parameters']['base_folder']}",
            f"dt={self.params['simulation_parameters']['dt']:.12g}",
            f"total_time={self.params['simulation_parameters']['total_time']:.12g}",
            f"strain_unit_conversion={self.params['simulation_parameters']['strain_unit_conversion']:.12g}",
            f"slip_constant_strength={self.params['material']['slip_constant_strength']:.12g}",
            f"voce_hardening_initial_slope={self.params['material']['voce_hardening_initial_slope']:.12g}",
            f"voce_hardening_saturation={self.params['material']['voce_hardening_saturation']:.12g}",
            f"power_slip_n={self.params['material']['power_slip_n']:.12g}",
            f"power_slip_g0={self.params['material']['power_slip_g0']:.12g}",
            f"elastic_E={self.params['material']['elastic_E']:.12g}",
            f"elastic_nu={self.params['material']['elastic_nu']:.12g}",
            f"elastic_G={self.params['material']['elastic_G']:.12g}",
            f"fixnode_x={fixnode_x:.12g}",
            f"fixnode_y={fixnode_y:.12g}",
            f"fixnode_z={fixnode_z:.12g}",
            f"yroll_x={yroll_x:.12g}",
            f"yroll_y={yroll_y:.12g}",
            f"yroll_z={yroll_z:.12g}",
        ]

        # Run the simulation with persistent background process
        print(f"\n==> Running CPFE simulation in {self.save_simulation_folder}", flush=True)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w", buffering=1) as lf:
            proc = subprocess.Popen(
                argv,
                cwd=self.save_simulation_folder,
                stdin=subprocess.DEVNULL,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True
            )

        if proc.returncode != 0:
            print(f"ERROR: CPFE simulation failed with exit code {proc.returncode}", file=sys.stderr)
            raise RuntimeError(f"CPFE simulation failed with exit code {proc.returncode}")

