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

import numpy as np
from graintrace.simulation_postprocessing import SimulationResults, FieldFileNaming
from graintrace.experiment_postprocessing import (
    ExperimentResults,
    FieldFileNaming as ExpFieldFileNaming,
)
from graintrace import plot_postprocessing as postprocess

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

# Input Experiment (self-contained: synthetic per-load-step FF CSVs expsyn_<t>time.csv)
grain_folder = "mwe_data/synthetic_load_exp"

exp_field_naming = ExpFieldFileNaming(
    prefix="expsyn",     # files: expsyn_<t>time.csv -> id captured between sep and suffix
    index_width=3,
    sep="_",
    suffix="time.csv",
)

test_tim_exp = 100       # available times are even values 100..160
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
