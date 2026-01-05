from synthetic_hedm_generator import SyntheticHEDMGenerator
from construct_nf_mesh import NearFieldMeshBuilder
import os
import matplotlib.pyplot as plt
import sys
import numpy as np

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
                         "params": (200.0, 5.0)}

nf_dz = 50.0  # Near field dz (thickness) per layer
nf_mesh_nx = 30
nf_mesh_ny = 30

nf_segmentation_input = {
        "misorientation_tol": 1.0/180*np.pi,
        "connectivity": 26,
        "batch_norm": 200_000,
        "grain_threshold": 10,
        "stop_count": 500,
        "grain_threshold_final": 10,
    }

sculpt_options = (
    "--adapt",
    # "-A",
    # "7",
    #"-df",
    #"1",
    "-S",
    "2",
    "-CS",
    "4",
    "--void_mat",
    "0",
    )

## GENERATE SYNTHETIC FF + NF STRUCTURE ------------------------
synth_hedm_gen = SyntheticHEDMGenerator(
    output_dir=output_dir,
    ff_bounding_box=ff_bounding_box,
    ff_strain_stdev=ff_strain_stdev, 
    ff_grain_characteristics=crystal_morpho_args,
    nf_bounding_box=nf_bounding_box,
    nf_dz = nf_dz,
    nf_spacing = 50, #lattice spacing
    random_seed=42,
)

synth_hedm_gen.run(ff_iterations=10)


## NEAR FIELD MESHING -------------------------------------------

nf_folder = os.path.join(output_dir, "NF")          
save_dir = os.path.join(output_dir, "nf_reconstruction") 

builder = NearFieldMeshBuilder(
    input_folder=nf_folder,
    save_dir=save_dir,
    angle_convention="bunge",
    angle_type="degrees",
    symmetry="432",
    prefix="reconstructed",
    write_intermediate=True,
    write_vtk=True,
)

merged_grid_path = builder.reconstruct(
    dz=nf_dz,      # must match your SyntheticHEDMGenerator nf_dz
    nx=nf_mesh_nx,
    ny=nf_mesh_ny,
    segmentation=nf_segmentation_input,

)
print(f"\nReconstruction complete: {merged_grid_path}\n")

sculpt_config = {
    "mpirun": "/opt/Coreform-Cubit-2025.12/bin/mpi/bin/mpirun",
    "psculpt": "/opt/Coreform-Cubit-2025.12/bin/psculpt",
    "epu": "/opt/Coreform-Cubit-2025.12/bin/epu",
    "nprocs": 4,
    "environment": {
        "OPAL_LIBDIR": "/opt/Coreform-Cubit-2025.12/bin/mpi/lib",
        "OPAL_PREFIX": "/opt/Coreform-Cubit-2025.12/bin/mpi",
    },
}

mesh_path = builder.mesh(
    sculpt_config=sculpt_config,
    sculpt_options=sculpt_options,
    merged_grid=merged_grid_path,  # explicit restart-safe input
)
print(f"Meshing complete: {mesh_path}")
print(f"Mapped orientations: {builder.mapped_orientations_path}")