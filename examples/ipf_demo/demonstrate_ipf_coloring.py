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

"""Demonstrate IPF coloring of a CPFE mesh with graintrace.ipf_postprocess.

Input a mesh + per-block Euler-Bunge orientations, write the IPF color legend, then
append per-block RGB fields to a copy of the Exodus mesh. Uses the physically
correct sample_symmetry="1". Open the resulting mesh_rgb.e in ParaView and color by
rgb_x/rgb_y/rgb_z (scalar mapping off).

Self-contained: the mesh and orientations live in this folder; paths are resolved
relative to this file, so it runs from any working directory (moose-src env):

    python examples/ipf_demo/demonstrate_ipf_coloring.py
"""

from __future__ import annotations

from pathlib import Path

from graintrace.ipf_postprocess import IPFProcessor

HERE = Path(__file__).resolve().parent

# Inputs and outputs all live in this folder.
MESH_FILE = str(HERE / "mesh.e")
ORI_CSV = str(HERE / "orientations.csv")
ANGLE_CONVENTION = "bunge"  # orientations.csv holds Euler-Bunge angles
ANGLE_TYPE = "radians"
OUTPUT_DIR = str(HERE / "output")
DIRECTION = [0.0, 0.0, 1.0]  # IPF coloring direction
CRYSTAL_SYMMETRY = "432"  # Cubic crystal symmetry
SAMPLE_SYMMETRY = "1"

ipf = IPFProcessor(
    crystal_symmetry=CRYSTAL_SYMMETRY,
    sample_symmetry=SAMPLE_SYMMETRY,
    save_dir=OUTPUT_DIR,
)

# 1. Color legend for the fundamental triangle.
ipf.ipf_color_chart(savefig_name="ipf_color_chart.png")

# 2. Append per-block IPF RGB to a copy of the mesh.
out = ipf.add_block_rgb_to_exodus(
    mesh_file=MESH_FILE,
    orientations_csv=ORI_CSV,
    output_file="mesh_rgb.e",
    direction=DIRECTION,
    angle_convention=ANGLE_CONVENTION,
    angle_type=ANGLE_TYPE,
)
