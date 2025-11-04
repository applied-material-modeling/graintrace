from construct_voronoi_mesh import VoronoiMeshBuilder
from run_cpfe_simulation import CPFESimulation
from tess_to_gnn import NeperTessToGraphNN
from material_calibration import MaterialCalibration
from experiment_rotation_helper import update_experiments, collect_experiment_files
from taylor import TaylorModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

fsize = 14
plt.rcParams.update({
    "font.size": fsize,           # Global font size
    "axes.labelsize": fsize,      # Axis label size
    "axes.titlesize": fsize,      # Title size
    "xtick.labelsize": fsize,     # X tick label size
    "ytick.labelsize": fsize,     # Y tick label size
    "legend.fontsize": fsize,     # Legend font size
})

## INPUT
##------------------------------------
outputdir = "experiment_try1"

generate_sudo = False

input_file = "experiment_2022_raw/0.csv"

exp_data_dir = "experiment_2022_raw"

bounding_box=[-477.0, 528, -487, 532, -1025, 625]
rotate_angles = (0,0,-3.6/180*np.pi)
rotate_unit = "rad"

# row-major ordered, user responsibility, no way to define it
elastic_strain_identifier = ["eKen11","eKen12","eKen13",
                                    "eKen21","eKen22","eKen23",
                                    "eKen31","eKen32","eKen33"]

# this comment will be replaced via a user interface
run_cpfe = False
update_experiments_data = False
run_calibration = True
build_graph = False
build_voronoi = False

if generate_sudo:

    np.random.seed(42)

    bounding_box=[0, 1, 0, 1, 0, 1]
    n_centroids = 10

    # generate n random points (x,y,z) within bounding box

    points = np.random.rand(n_centroids, 3)
    points[:, 0] = points[:, 0] * (bounding_box[1] - bounding_box[0]) + bounding_box[0]
    points[:, 1] = points[:, 1] * (bounding_box[3] - bounding_box[2]) + bounding_box[2]
    points[:, 2] = points[:, 2] * (bounding_box[5] - bounding_box[4]) + bounding_box[4]

    # generate random weights between 0 and 1
    weights = np.random.rand(n_centroids)**3

    # generate random euler angles of value 30degrees in 'x' and 0 everywhere
    euler_angles = np.full((n_centroids, 3), 30)
    euler_angles[:, 0] = 0
    euler_angles[:, 2] = 0

    # generate symmetric tensors in microstrain
    exx = np.random.uniform(-500, 500, n_centroids)
    eyy = np.random.uniform(-500, 500, n_centroids)
    ezz = np.random.uniform(-500, 500, n_centroids)
    exy = np.random.uniform(-500, 500, n_centroids)
    eyz = np.random.uniform(-500, 500, n_centroids)
    exz = np.random.uniform(-500, 500, n_centroids)

    # flatten into 9-component form (symmetric tensor)
    ee = np.column_stack([
        exx, exy, exz,    # row 1
        exy, eyy, eyz,    # row 2
        exz, eyz, ezz     # row 3
    ])

    # create pandas with X, Y, Z, Weight columns

    df = pd.DataFrame(points, columns=["X", "Y", "Z"])
    df["GrainRadius"] = weights
    df["Eul0"] = euler_angles[:, 0]
    df["Eul1"] = euler_angles[:, 1]
    df["Eul2"] = euler_angles[:, 2]
    df["eKen11"] = ee[:, 0]
    df["eKen12"] = ee[:, 1]
    df["eKen13"] = ee[:, 2]
    df["eKen21"] = ee[:, 3]
    df["eKen22"] = ee[:, 4]
    df["eKen23"] = ee[:, 5]
    df["eKen31"] = ee[:, 6]
    df["eKen32"] = ee[:, 7]
    df["eKen33"] = ee[:, 8]

    # save to csv as test.csv
    df.to_csv("test.csv", index=False)
    input_file = "test.csv"

# --- Base test setup ---
if build_voronoi:

  builder = VoronoiMeshBuilder(
      input_csv=input_file,
      output_dir=outputdir,
      bounding_box=bounding_box,
      # parameter below have default values
      dim=3,
      weighted=False,
      auto_fix_bbox=True, 
        # enable correction
      bbox_fix_mode="remove_points", 
        # 'extend_bounding_box' or 'remove_points'
      bbox_tolerance=0.0,       
        # bounding box tolerance % factor
      auto_rotate=False,
        # if True, PCA method applied
        # rotate_angles and rotate_convention are ignored
      rotate_angles= rotate_angles,
      rotate_convention="xyz",
      unit=rotate_unit,
      angle_identifier=["Eul0", "Eul1", "Eul2"],
      orientation_descriptor="euler-bunge",
      orientation_active_convention=True,
      elastic_strain_identifier=elastic_strain_identifier,
      # row-major ordered, user responsibility, no way to define it
      strain_unit="microstrain",
  )

  builder.build_voronoi(generate_mesh=True,
                        mesh_quality_min = 0.7,
                        ## mesh generation parameters
                        relative_el_size=5.0,
                          # 1 roughly 100 elementsper cell
                        option="centroid",
                          # voronoi, centroid, centroidsize
                        CVT_iter=100, # optimization parameter
                        morphoalgo = "subplex",
                        # subplex, lloyd, praxis
                        )

# convert .tess to graph data structure
if build_graph:
  parser = NeperTessToGraphNN(
      tess_path=outputdir+"/voronoi.tess",
      device="cpu",
      dtype=torch.float64
      )

  graph = parser.build_cell_graph()
  os.makedirs(outputdir + "/figures/gnn", exist_ok=True)
  parser.visualize_graph_2D(graph,outpath=outputdir+"/figures/gnn/graph_2D.png")
  parser.visualize_graph_3D(graph,outpath=outputdir+"/figures/gnn/graph_3D.png")

if update_experiments_data:

  print("\n=== Updating Experimental Data with Rotation and Strain ===\n")

  files, stress_levels = collect_experiment_files(exp_data_dir)
  print(f"Found {len(files)} CSV files in '{exp_data_dir}':")
  for f, s in zip(files, stress_levels):
      print(f"  {os.path.basename(f):>12}  (stress = {s:.1f})")
  
  update_experiments(
      input_files=files,
      output_root = os.path.join(outputdir, "rotated_experiments"),
      bounding_box=bounding_box,
      auto_fix_bbox=True,
      bbox_fix_mode="remove_points",
      rotate_angles=rotate_angles,
      unit=rotate_unit,
      angle_identifier=["Eul0", "Eul1", "Eul2"],
      orientation_descriptor="euler-bunge",
      orientation_active_convention=True,
      elastic_strain_identifier=[
          "eKen11","eKen12","eKen13",
          "eKen21","eKen22","eKen23",
          "eKen31","eKen32","eKen33",
      ],
  )

  print(f"\nProcessed all experiments. Results in: {outputdir}")

# Material calibration
if run_calibration:
  print("\n=== Starting Material Calibration ===\n")

  os.makedirs(outputdir + "/figures/material_calibration", exist_ok=True)

  calib = MaterialCalibration(
      model_class=TaylorModel,
      model_args=dict(
          neml2_path="cpfe_base/neml2_cpfe_calibration.i",
          neml2_model_name="model_with_stress",
      ),
      data_args=dict(
          data_dir=outputdir + "/rotated_experiments",
          strain_stress_file=exp_data_dir+"/strain-stress.csv",
          npoints=50,
          full_field_strain_units="microstrain",
          straintype = "eKen",
          # max_stress = 300,
      ),
      save_dir=outputdir + "/figures/material_calibration",
      apply_elastic_correction=False,
      strain_window=(0.0, 0.0015),
  )

  calib.plot_texture(direction=[1, 1, 1])
  calib.plot_stress_strain()

  calib.calibrate(maxiter=50)
  calib.load(outputdir + "/figures/material_calibration" + "/calibrated_material.json")

  calib.plot_stress_strain(include_model=True)
  calib.plot_strain_histogram(include_initial_strain=True)

  # Convert results to parameter dictionary
  translation = {
      "elastic_tensor_E": "elastic_E",
      "elastic_tensor_G": "elastic_G",
      "elastic_tensor_nu": "elastic_nu",
      "slip_strength_constant_strength": "slip_constant_strength",
      "voce_hardening_initial_slope": "voce_hardening_initial_slope",
      "voce_hardening_saturated_hardening": "voce_hardening_saturation",
  }
  optimized_material = {
      translation[k]: float(v) for k, v in zip(calib.model.opt_vars, calib.opt_params)
  }

  print("\nCalibrated material parameters:")
  for k, v in optimized_material.items():
      print(f"  {k} = {v:.6g}")

if run_cpfe:
  # run CPFE simulation
  sim = CPFESimulation(
        mesh_file=outputdir+"/voronoi.msh",
        save_simulation_folder=outputdir+"/simulation_cpfe",
        eeres_file=outputdir+"/voronoi.ee",
        ori_file=outputdir+"/voronoi.ori",
        dim=3,
        moose_run_file="/home/tranh/projects/puma/puma-opt"
    )

  sim.set_parameters("material", **optimized_material)

  sim.set_parameters(
      "boundary",
      bounding_box=builder.bounding_box,
      bc={
          "x": {"negative": "stress_free", "positive": "stress_free"},
          "y": {"negative": "stress_free", "positive": "stress_free"},
          "z": {"negative": 0, "positive": 30},
      },
  )

  sim.run(ncore=20)



