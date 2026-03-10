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
            "sync_times": "0.1 1.0 2.0 3.0 4.0 5.0",
            "dt": 0.1,
            "total_time": 5.0,
            "strain_unit_conversion": 1.0,
            "initialize_time": 1.0,
            "device": "cpu",
            "device_batch": 100,
            "scheduler_name": "hybrid",
            "hybrid_batch_sizes": (2000, 2000),
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
            "bounding_box": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], #xlo, xhi, ylo, yhi, zlo, zhi
            "fix_tolerance": 1e-8,    
            "bc": {
                "x": {"negative": "stress_free", "positive": "stress_free"},
                "y": {"negative": "stress_free", "positive": "stress_free"},
                "z": {"negative": 0, "positive": 0.001},
                },
        },
        "grid_properties": {
            "number_of_elements": [20, 20, 20],
            "bounding_box": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "bounding_box_buffer": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    def __init__(self, 
                mesh_file,
                save_simulation_folder,
                moose_run_file,
                element_order="SECOND",
                eeres_file=None, 
                ori_file=None,
                use_ff_initial_field=False,
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

        if element_order not in ("FIRST", "SECOND"):
            raise ValueError(f"Invalid element_order {element_order}. Must be 'FIRST' or 'SECOND'.")

        self.element_order = element_order
        self.params = copy.deepcopy(self.DEFAULT_PARAMS)
        self.ncell_ff = None
        self.use_ff_initial_field = use_ff_initial_field

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
            xbufflo, xbuffhi, ybufflo, ybuffhi, zbufflo, zbuffhi = b["bounding_box_buffer"]
        else:
            xlo, xhi, ylo, yhi = b["bounding_box"]
            xbufflo, xbuffhi, ybufflo, ybuffhi = b["bounding_box_buffer"]
            zlo, zhi = 0.0, 0.0
            zbufflo, zbuffhi = 0.0, 0.0

        fixnode_x, fixnode_y, fixnode_z = xlo+xbufflo, ylo+ybufflo, zlo+zbufflo
        yroll_x, yroll_y, yroll_z = xlo+xbufflo, yhi+ybuffhi, zlo+zbufflo

        coupled_axes = set()

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

                    # Zero BC
                    if abs(val) < 1e-15:
                        f.write(f"    [{boundary_name}_boundary]\n")
                        f.write("        type = DirichletBC\n")
                        f.write(f"        variable = disp_{axis}\n")
                        f.write(f"        boundary = {boundary_name}\n")
                        f.write("        value = 0\n")
                        f.write("    []\n")
                        continue

                    # Nonzero numeric BC (ramped) — PER BOUNDARY
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

                    # TO BE ADDED: for the face that are moving,
                    # the other two directions of that face should be zero

            f.write("[]\n\n")

            # [Functions] — per-axis ramping functions
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

            # [AuxVariables]
            f.write("[AuxVariables]\n")
            for axis in sorted(coupled_axes):
                f.write(f"    [disp_{axis}_residual]\n")
                f.write(f"        order = {self.element_order}\n")
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

            # [Front face flat]
            f.write("[Constraints]\n")
            f.write("    [zface_flat]\n")
            f.write("        type = EqualValueBoundaryConstraint\n")
            f.write("        secondary = 'front'\n")
            f.write("        variable = disp_z\n")
            f.write("        penalty = 1e6\n")
            f.write("        enable = true\n")
            f.write("    []\n")
            f.write("[]\n\n")

            # [Controls] — dynamically determined
            f.write("[Controls]\n")
            f.write("    [switch_loading]\n")
            f.write("        type = TimePeriod\n")
            f.write(f"        start_time = {sim['initialize_time']:.12g}\n")
            f.write(f"        end_time = {(sim['total_time']+1):.12g}\n")

            enable_objs = [f"BCs::{bname}_boundary" for _, bname, _ in coupled_bcs]
            disable_objs = [f"AuxKernels::disp_{axis}_residual" for axis, _, _ in coupled_bcs]

            f.write(f"        enable_objects = '{' '.join(enable_objs)}'\n")
            f.write(f"        disable_objects = '{' '.join(disable_objs)}'\n")
            f.write("        execute_on = 'TIMESTEP_BEGIN'\n")
            f.write("    []\n")
            f.write("[]\n")

        return fixnode_x, fixnode_y, fixnode_z, yroll_x, yroll_y, yroll_z

    def write_postprocess_file(self, ncell=None):
        """
        Generate grain_average_postprocessor.i.
        """

        out = self.save_simulation_folder / "grain_average_postprocessor.i"
   
        if self.eeres_file is None:
            if ncell is None or not isinstance(ncell, int):
                raise ValueError("ncell must be provided as an integer when eeres_file is None")

            ee_file = self.save_simulation_folder / "zero_initial_strain.ee"
            strain_data = np.zeros((ncell, 9), dtype=float)
            np.savetxt(ee_file, strain_data, fmt="%.12g")

            self.eeres_file = ee_file
        else:
            # read self.eeres_file and count number of lines using pandas
            df = pd.read_csv(self.eeres_file, sep=r"[,\s]+", engine="python", header=None)
            self.ncell_ff = df.shape[0]
            shutil.copy(self.eeres_file, self.save_simulation_folder / Path(self.eeres_file).name)

        # --- Generate grain_average_postprocessor.i ---
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
                # cauchy_stress_*
                f.write(f"    # --- cauchy_stress_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [cauchy_stress_{comp}_{i}]\n")
                    f.write("        type = ElementAverageValue\n")
                    f.write(f"        variable = cauchy_stress_{comp}\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")
            
            components = ["11","12","13","21","22","23","31","32","33"]
            for comp in components:
                # nye_tensor_*
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
                # geometric centroid
                f.write(f"    # --- centroid_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [centroid_{comp}_{i}]\n")
                    f.write("        type = FunctionElementAverage\n")
                    f.write(f"        function = coord_{comp}\n")
                    f.write("        use_displaced_mesh = True\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")
                # orientation
                f.write(f"    # --- ori_rodrigues_{comp} ---\n")
                for i in range(1, ncell + 1):
                    f.write(f"    [ori_rodrigues_{comp}_{i}]\n")
                    f.write("        type = ElementAverageValue\n")
                    f.write(f"        variable = ori_rodrigues_{comp}\n")
                    f.write(f"        block = {i}\n")
                    f.write("    []\n")
                f.write("\n")

            # volume
            f.write("    # --- volume ---\n")
            for i in range(1, ncell + 1):
                f.write(f"    [volume_{i}]\n")
                f.write("        type = ElementAverageValue\n")
                f.write(f"        variable = volume\n")
                f.write("        use_displaced_mesh = True\n")
                f.write(f"        block = {i}\n")
                f.write("    []\n")
            f.write("\n")
            

            f.write("[]\n")
    
    def write_orientation_file(self):

        df = pd.read_csv(self.ori_file, sep=r"[,\s]+", engine="python", header=None)
        # df = df.apply(pd.to_numeric, errors="coerce")

        # if df has 9 columnes
        if df.shape[1] == 9:

            mrps = load_orientations(df, field=None)  # existing helper untouched

            np.savetxt(
                self.save_simulation_folder / "mrps_orientation.csv",
                mrps.numpy(),
                delimiter=",",
                comments="",
                fmt="%.12g",
            )
        # do nothing if 3 columns, otherwise raise error
        elif df.shape[1] == 3:
            import torch
            shutil.copy(
                self.ori_file,
                self.save_simulation_folder / "mrps_orientation.csv"
            )
            mrps = torch.tensor(df.values, dtype=torch.float32)
        else:
            raise ValueError("Orientation file must have either 3 (MRPs) or 9 (rotation matrix) columns.")

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
        if self.use_ff_initial_field:
            base_files = ["initial_conditions_ff.i",
                "neml2_cpfe.i",
                "run_cpfe.i",
                "grid_file.i",
                "transfer.i"]
            initial_conditions_file = "initial_conditions_ff.i"
        else:
            base_files = ["initial_conditions.i",
                        "neml2_cpfe.i",
                        "run_cpfe.i",
                        "grid_file.i",
                        "transfer.i"]
            initial_conditions_file = "initial_conditions.i"

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
        self.write_postprocess_file(ncell=ncells)

        if self.use_ff_initial_field:
            ncell_args = [f"ncell={ncells}"]
        else:
            ncell_args = [f"ncell={ncells:.12g}",f"ncell_ff={self.ncell_ff:.12g}"]

        # transfer grid info
        grid_info = self.params["grid_properties"]
        ncell_x, ncell_y, ncell_z = grid_info["number_of_elements"]
        grid_bbox = grid_info["bounding_box"]

        # Build command for subprocess
        vol_correction_cond = "true" if self.element_order == "FIRST" else "false"
        

        log_path = self.save_simulation_folder / "cpfe_run.log"
        argv = [
            "nohup",
            "mpiexec",
            "-n", str(ncore),
            str(self.moose_run_file),
            "-i",
            "run_cpfe.i",
            "boundary_conditions.i",
            initial_conditions_file,
            "grain_average_postprocessor.i",
            "transfer.i",
            "orientation_file=mrps_orientation.csv",
            f"sync_times={self.params['simulation_parameters']['sync_times']}",
            f"scheduler_name={self.params['simulation_parameters']['scheduler_name']}",
            f"device_neml2={self.params['simulation_parameters']['device']}",
            f"device_neml2_batch={self.params['simulation_parameters']['device_batch']}",
            f"nbatchdevice1={self.params['simulation_parameters']['hybrid_batch_sizes'][0]}",
            f"nbatchdevice2={self.params['simulation_parameters']['hybrid_batch_sizes'][1]}",
            f"mesh_file={self.mesh_file.name}",
            f"residual_strain_file={self.eeres_file.name}",
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
            f"burger_scale={self.params['material']['burger_scale']:.12g}",
            f"fixnode_x={fixnode_x:.12g}",
            f"fixnode_y={fixnode_y:.12g}",
            f"fixnode_z={fixnode_z:.12g}",
            f"yroll_x={yroll_x:.12g}",
            f"yroll_y={yroll_y:.12g}",
            f"yroll_z={yroll_z:.12g}",
            f"vol_lock_correction_cond={vol_correction_cond}",
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
                text=True,
                start_new_session=True,
                close_fds=True,
            )

        print(f"CPFE simulation started, PID={proc.pid}")