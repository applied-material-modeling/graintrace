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

from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder
from graintrace.grain_graph_matching import GraphGrainMatcher


ff_bounding_box_1 = [
    -200,
    200,
    -173.205,
    173.205,
    0,
    650,
]
ff_bounding_box_2 = [-200, 200, -173.205, 173.205, 0, 680]

grain_file1 = "mwe_data/synthetic_load_exp/expsyn_146time.csv"
grain_file2 = "mwe_data/synthetic_load_exp/expsyn_160time.csv"

elastic_strain_identifier = [
    "eKen11",
    "eKen12",
    "eKen13",
    "eKen21",
    "eKen22",
    "eKen23",
    "eKen31",
    "eKen32",
    "eKen33",
]

builder_ff1 = VoronoiMeshBuilder(
    input_csv=grain_file1,
    output_dir="test2",
    bounding_box=ff_bounding_box_1,
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

grapha = builder_ff1.build_graph(CVT_iter=10)

builder_ff2 = VoronoiMeshBuilder(
    input_csv=grain_file2,
    output_dir="test3",
    bounding_box=ff_bounding_box_2,
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

graphb = builder_ff2.build_graph(CVT_iter=10)

grain_track = GraphGrainMatcher(
    graph_a=grapha,
    graph_b=graphb,
    output_dir="grain_tracking_output",
)

grain_track.match_grains(
    message_passing_iter=3,
    neighbor_selection_param={"lambda": 0.00125, "iterations": 100, "tolerance": 1e-6},
)
