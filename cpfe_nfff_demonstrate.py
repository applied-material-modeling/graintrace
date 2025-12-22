from synthetic_hedm_generator import SyntheticHEDMGenerator
from construct_nf_mesh import NearFieldMeshBuilder
import os
import matplotlib.pyplot as plt
import sys

fsize = 14
plt.rcParams.update({
    "font.size": fsize,           # Global font size
    "axes.labelsize": fsize,      # Axis label size
    "axes.titlesize": fsize,      # Title size
    "xtick.labelsize": fsize,     # X tick label size
    "ytick.labelsize": fsize,     # Y tick label size
    "legend.fontsize": fsize,     # Legend font size
})

# INPUT -------------------------------------------------------
output_dir = "cpfe_ff_nf_demonstrate"

ff_bounding_box = [-500, 500, -500, 500, -1000, 500]
ff_strain_stdev = 0.005

nf_bounding_box = [-200, 200, -200, 200, -200, 200]

crystal_morpho_args = {"type": "diameq", 
                       "distribution": "lognormal",
                         "params": (500.0, 5.0)}

## GENERATE SYNTHETIC FF + NF STRUCTURE ------------------------
synth_hedm_gen = SyntheticHEDMGenerator(
    output_dir=output_dir,
    ff_bounding_box=ff_bounding_box,
    ff_strain_stdev=ff_strain_stdev, 
    ff_grain_characteristics=crystal_morpho_args,
    nf_bounding_box=nf_bounding_box,
    nf_dz = 50,
    nf_spacing = 50,
    random_seed=42,
)

synth_hedm_gen.run(ff_iterations=10)


## NEAR FIELD MESHING -------------------------------------------

## make sure to communicate the correct spacing and information
nf_folder = os.path.join(output_dir, "NF")          # "cpfe_ff_nf_demonstrate/NF"
save_dir = os.path.join(output_dir, "nf_pipeline")  # dump ALL artifacts here

builder = NearFieldMeshBuilder(
    input_folder=nf_folder,
    save_dir=save_dir,
    angle_convention="bunge",
    angle_type="radians",
    symmetry="432",
    prefix="reconstructed",
    write_intermediate=True,
    write_vtk=True,
)

# --- Stage 1: reconstruct (writes merged_segmented_fixed_grid.npy into save_dir) ---
merged_grid_path = builder.reconstruct(
    dz=50.0,      # must match your SyntheticHEDMGenerator nf_dz
    nx=300,
    ny=300,
    segmentation={
        # keep defaults unless you want to override:
        # "misorientation_tol": 5.0/180*np.pi,
        # "connectivity": 26,
        # "batch_norm": 200_000,
        # "grain_threshold": 1_000,
        # "stop_count": 500,
        # "grain_threshold_final": 10_000,
    },
)
print(f"[OK] Reconstruction complete: {merged_grid_path}")

# --- Stage 2: mesh (requires Sculpt installed + correct paths) ---
# IMPORTANT: You must set these paths for your machine. If you don’t have Sculpt,
# comment this section out and you can still validate reconstruction outputs.
sculpt_config = {
    "mpirun": "/Applications/Coreform-Cubit-2024.8.app/Contents/mpi/bin/mpirun",
    "psculpt": "/Applications/Coreform-Cubit-2024.8.app/Contents/lib/../MacOS/psculpt",
    "nprocs": 12,
    "environment": {
        "OPAL_LIBDIR": "/Applications/Coreform-Cubit-2024.8.app/Contents/lib/../mpi/lib",
        "OPAL_PREFIX": "/Applications/Coreform-Cubit-2024.8.app/Contents/lib/../mpi",
    },
    "epu": "/Applications/Coreform-Cubit-2024.8.app/Contents/lib/../MacOS/epu",
}

try:
    mesh_path = builder.mesh(
        sculpt_config=sculpt_config,
        # sculpt_options=None uses builder defaults
        merged_grid=merged_grid_path,  # explicit restart-safe input
    )
    print(f"[OK] Meshing complete: {mesh_path}")
    print(f"[OK] Mapped orientations: {builder.mapped_orientations_path}")
except FileNotFoundError as e:
    raise
except Exception as e:
    # Hard failure is correct; this makes it obvious if Sculpt/MPI isn't configured.
    print("[FAIL] Meshing did not run. This is expected if Sculpt/MPI paths are incorrect.")
    raise