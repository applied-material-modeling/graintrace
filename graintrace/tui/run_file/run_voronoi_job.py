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

import sys
import os
import json
from pathlib import Path

# Ensure project root in path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder


def main():
    params = json.loads(sys.stdin.read())

    builder = VoronoiMeshBuilder(
        input_csv=params["input_csv"],
        output_dir=params["output_dir"],
        bounding_box=params["bounding_box"],
        dim=params["dim"],
        weighted=params["weighted"],
        auto_fix_bbox=params["auto_fix_bbox"],
        bbox_fix_mode=params["bbox_fix_mode"],
        bbox_tolerance=params["bbox_tolerance"],
        auto_rotate=params["auto_rotate"],
        rotate_angles=params["rotate_angles"],
        rotate_convention=params["rotate_convention"],
        unit=params["unit"],
        angle_identifier=params["angle_identifier"],
        orientation_descriptor=params["orientation_descriptor"],
        orientation_active_convention=params["orientation_active_convention"],
        elastic_strain_identifier=params["elastic_strain_identifier"],
        strain_unit=params["strain_unit"],
    )

    builder.build_voronoi(
        generate_mesh=params["generate_mesh"],
        mesh_quality_min=params["mesh_quality_min"],
        relative_el_size=params["relative_el_size"],
        option=params["option"],
        CVT_iter=params["CVT_iter"],
        morphoalgo=params["morphoalgo"],
    )

    print("Voronoi successfully built.")


if __name__ == "__main__":
    main()