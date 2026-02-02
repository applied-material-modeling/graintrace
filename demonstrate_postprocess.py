import numpy as np
from simulation_postprocessing import SimulationResults, FieldFileNaming
from experiment_postprocessing import ExperimentResults, FieldFileNaming as ExpFieldFileNaming
import plot_postprocessing as postprocess

# Input simulation
block_csv = "cpfe_ff_nf_demonstrate_v2/simulation/simulation_out/out.csv"
field_dir = "cpfe_ff_nf_demonstrate_v2/simulation/simulation_out/grid_out"

field_naming = FieldFileNaming(
    prefix="out_element_centroid",   # must match your field filenames
    index_width = 4,
    sep="_",
    suffix=".csv",
)

output_folder = "postprocess_test1"

test_time_sim = 1.0
test_sim = True

# Input Experiment
grain_folder = "testing_during_code_not_upload_to_github/synthetic_load_exp"

exp_field_naming = ExpFieldFileNaming(
    prefix="expsyn",
    index_width=2,
    sep="_",
    suffix=".csv",
)

test_tim_exp = 36
test_exp = True

# Main ---------------------------------------------------

field_naming = FieldFileNaming(
    prefix="out_element_centroid",   # must match your field filenames
    index_width = 4,
    sep="_",
    suffix=".csv",
)

res = SimulationResults(
    block_csv=block_csv,
    field_dir=field_dir,
    field_naming=field_naming,
)

print(f"Loaded block data with {res.n_steps} steps")

exp_res = ExperimentResults(
    exp_dir=grain_folder,
    exp_naming=exp_field_naming
)

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

    # postprocess.plot_pole_figure(
    #     exp_res,
    #     tensor_prefix="Eul",
    #     time = 1.0,
    #     direction = [1, 1, 1],
    #     crystal_symmetry = "432",
    #     device = "cpu",
    #     output_folder=output_folder,
    #     construct_odf=False,
    # )
    # print(f"Pole figure plotted")


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
    postprocess.plot_pole_figure(
        res,
        tensor_prefix="ori_rodrigues",
        time = test_time_sim,
        direction = [0, 0, 1],
        crystal_symmetry = "432",
        device = "cpu",
        output_folder=output_folder,
        construct_odf=False,
    )
    print(f"Pole figure plotted")

