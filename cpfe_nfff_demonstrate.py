from synthetic_hedm_generator import SyntheticHEDMGenerator
from run_cpfe_simulation import CPFESimulation
from construct_voronoi_mesh import VoronoiMeshBuilder
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

ff_bounding_box = [-500, 500, -500, 500, 0, 1500]
ff_strain_stdev = 100.0 #microstrain

nf_bounding_box = [-200, 200, -200, 200, 0, 600]

crystal_morpho_args = {"type": "diameq", 
                       "distribution": "lognormal",
                         "params": (200.0, 5.0)}

nf_dz = 20.0  # Near field dz (thickness) per layer
nf_mesh_nx = 20
nf_mesh_ny = 20

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

initialize_data = True

ncore = 8

nf_folder = os.path.join(output_dir, "NF")          
nf_save_dir = os.path.join(output_dir, "nf_reconstruction") 
output_ff = os.path.join(output_dir, "ff_reconstruction")

if initialize_data:

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

    nf_bounding_box = synth_hedm_gen.nf_bounding_box

    ## NEAR FIELD MESHING -------------------------------------------

    builder_nf = NearFieldMeshBuilder(
        input_folder=nf_folder,
        save_dir=nf_save_dir,
        angle_convention="bunge",
        angle_type="degrees",
        symmetry="432",
        prefix="reconstructed",
        write_intermediate=True,
        write_vtk=True,
    )

    merged_grid_path = builder_nf.reconstruct(
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
        "nprocs": int(ncore),
        "environment": {
            "OPAL_LIBDIR": "/opt/Coreform-Cubit-2025.12/bin/mpi/lib",
            "OPAL_PREFIX": "/opt/Coreform-Cubit-2025.12/bin/mpi",
        },
    }

    mesh_path = builder_nf.mesh(
        sculpt_config=sculpt_config,
        sculpt_options=sculpt_options,
        merged_grid=merged_grid_path,  # explicit restart-safe input
    )
    print(f"Meshing complete: {mesh_path}")
    print(f"Mapped orientations: {builder_nf.mapped_orientations_path}.csv")


    ## RECONSTRUCTED FROM FF DATA -------------------------------------------
    # this to see if we could improve from the geometric centroid vs voronoi centroid
    if not os.path.exists(output_ff):
        os.makedirs(output_ff)

    elastic_strain_identifier = ["eKen11","eKen12","eKen13",
                                    "eKen21","eKen22","eKen23",
                                    "eKen31","eKen32","eKen33"]

    builder_ff = VoronoiMeshBuilder(
      input_csv=output_dir+"/FF/ff.csv",
      output_dir=output_ff,
      bounding_box=ff_bounding_box,
      dim=3,
      weighted=False,
      auto_fix_bbox=False,       
      auto_rotate=False,
      angle_identifier=["Eul0", "Eul1", "Eul2"],
      orientation_descriptor="euler-bunge",
      orientation_active_convention=True,
      elastic_strain_identifier=elastic_strain_identifier,
      strain_unit="microstrain",
    )
    
    builder_ff.build_voronoi(generate_mesh=False,
                    option="centroid",
                    CVT_iter=100,
                    morphoalgo = "subplex",
                    )

## USE SOLUTION AUX TO INITIALIZE EE FOR NF MESH, RUN SIMULATION
# Have a multi app with one way variable transfer for grid REI
output_sim = os.path.join(output_dir, "simulation")
if not os.path.exists(output_sim):
    os.makedirs(output_sim)

sim = CPFESimulation(
    mesh_file=output_dir+"/nf_reconstruction/mesh.e",
    save_simulation_folder=output_sim,
    element_order="FIRST",
    eeres_file=output_ff+"/reconstruction_cpfe_ee.csv",
    ori_file=output_dir+"/nf_reconstruction/orientations.csv",
    dim=3,
    moose_run_file="/home/tranh/projects/puma/puma-opt"
)

# sim.set_parameters("material", **optimized_material)
# nf_bounding_box = np.array([-200, 200, -200, 200, -200, 200])

print(nf_bounding_box.tolist())

sim.set_parameters(
    "boundary",
    bounding_box=nf_bounding_box.tolist(),
    bc={
        "x": {"negative": "stress_free", "positive": "stress_free"},
        "y": {"negative": "stress_free", "positive": "stress_free"},
        "z": {"negative": 0, "positive": 30},
    },
)

sim.run(ncore=int(ncore))

# Post processing
