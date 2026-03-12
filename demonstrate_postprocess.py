import numpy as np
from simulation_postprocessing import SimulationResults, FieldFileNaming
from experiment_postprocessing import (
    ExperimentResults,
    FieldFileNaming as ExpFieldFileNaming,
)
import plot_postprocessing as postprocess

# Input simulation
block_csv = "mwe_data/out.csv"
field_dir = "mwe_data/grid_out"

field_naming = FieldFileNaming(
    prefix="out_element_centroid",  # must match your field filenames
    index_width=4,
    sep="_",
    suffix=".csv",
)

output_folder = "postprocess_test1"

test_time_sim = 1.0
test_sim = True

# Input Experiment
grain_folder = "experiment_workflow_aps_28Feb"  # "mwe_data/synthetic_load_exp"

exp_field_naming = ExpFieldFileNaming(
    prefix="stitched_output",
    index_width=1,
    sep="",
    suffix=".csv",
)

test_tim_exp = 0
test_exp = True

# Main ---------------------------------------------------

field_naming = FieldFileNaming(
    prefix="out_element_centroid",  # must match your field filenames
    index_width=4,
    sep="_",
    suffix=".csv",
)

res = SimulationResults(
    block_csv=block_csv,
    field_dir=field_dir,
    field_naming=field_naming,
)

print(f"Loaded block data with {res.n_steps} steps")

exp_res = ExperimentResults(exp_dir=grain_folder, exp_naming=exp_field_naming)

print(f"Loaded experiment data with {exp_res.n_steps} steps")


if test_exp:
    time_index = postprocess.plot_block_properties_distribution(
        exp_res,
        time=test_tim_exp,
        tensor_prefix="eKen",
        order=2,
        output_folder=output_folder,
        bins=10,
    )
    print(f"Tensor distribution plotted at time_index={time_index}")

    postprocess.plot_pole_figure(
        exp_res,
        tensor_prefix="Eul",
        time=test_tim_exp,
        direction=[1, 1, 1],
        crystal_symmetry="432",
        device="cpu",
        output_folder=output_folder,
        construct_odf=False,
        orientation_type="bunge",
        orientation_units="radians",
    )
    print(f"Pole figure 111 plotted")

    postprocess.plot_pole_figure(
        exp_res,
        tensor_prefix="Eul",
        time=test_tim_exp,
        direction=[0, 0, 1],
        crystal_symmetry="432",
        device="cpu",
        output_folder=output_folder,
        construct_odf=False,
        orientation_type="bunge",
        orientation_units="radians",
    )
    print(f"Pole figure plotted")

dfas
if test_sim:
    # 1) scalar distribution
    time_index = postprocess.plot_block_properties_distribution(
        res,
        time=test_time_sim,
        tensor_prefix="volume",
        order=0,
        output_folder=output_folder,
        bins=10,
    )
    print(f"Scalar distribution plotted at time_index={time_index}")

    # 2) vector distribution
    time_index = postprocess.plot_block_properties_distribution(
        res,
        time=test_time_sim,
        tensor_prefix="centroid",
        order=1,
        output_folder=output_folder,
        bins=10,
    )
    print(f"Vector distribution plotted at time_index={time_index}")

    # 3) tensor distribution
    time_index = postprocess.plot_block_properties_distribution(
        res,
        time=test_time_sim,
        tensor_prefix="ee",
        order=2,
        output_folder=output_folder,
        bins=10,
    )
    print(f"Tensor distribution plotted at time_index={time_index}")

    time_index = postprocess.plot_block_properties_distribution(
        res,
        time=test_time_sim,
        tensor_prefix="strain",
        order=2,
        output_folder=output_folder,
        bins=10,
    )
    print(f"Tensor distribution plotted at time_index={time_index}")

    # 4) macroscopic stress-strain
    postprocess.plot_macroscopic_stress_strain(
        res,
        stress_tensor_prefix="cauchy_stress",
        strain_tensor_prefix="strain",
        volume_prefix="volume",
        output_folder=output_folder,
    )
    print(f"Macroscopic stress-strain plotted")

    postprocess.plot_macroscopic_stress_strain(
        res,
        stress_tensor_prefix="cauchy_stress",
        strain_tensor_prefix="ee",
        volume_prefix="volume",
        output_folder=output_folder,
    )
    print(f"Macroscopic stress-ee plotted")

    # 5) properties over time
    postprocess.plot_block_properties_over_time(
        res,
        tensor_prefix="strain",
        order=2,
        output_folder=output_folder,
    )
    print(f"Strain of blocks 1, 2, 3 over time plotted")

    postprocess.plot_block_properties_over_time(
        res,
        tensor_prefix="centroid",
        order=1,
        grain_ids=[2],
        output_folder=output_folder,
    )
    print(f"Centroid of all blocks over time plotted")

    # 6) Texture
    # postprocess.plot_pole_figure(
    #     res,
    #     tensor_prefix="ori_rodrigues",
    #     time = test_time_sim,
    #     direction = [0, 0, 1],
    #     crystal_symmetry = "432",
    #     device = "cpu",
    #     output_folder=output_folder,
    #     construct_odf=False,
    # )
    # print(f"Pole figure plotted")
