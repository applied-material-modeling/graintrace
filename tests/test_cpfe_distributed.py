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

"""Unit tests for the CPFESimulation distributed-mesh (pre-split) option.

These test only the pure command-building logic (`_split_command`) and the default
parameter, so they run without the MOOSE/PUMA/NEML2 stack.
"""

import pytest

from graintrace.run_cpfe_simulation import CPFESimulation


def test_distributed_mesh_defaults_off():
    """Distributed mesh is opt-in: the default is a replicated mesh."""
    assert (
        CPFESimulation.DEFAULT_PARAMS["simulation_parameters"]["distributed_mesh"]
        is False
    )


def test_split_command_builds_expected_list():
    """The pre-split command reuses the -i deck/args tail and appends the split flags."""
    deck_and_args = ["-i", "run_cpfe.i", "transfer.i", "mesh_file=mesh.e"]
    cmd = CPFESimulation._split_command("puma-opt", deck_and_args, 4)
    assert cmd[0] == "puma-opt"
    # the -i deck/args tail is preserved verbatim
    assert cmd[1 : 1 + len(deck_and_args)] == deck_and_args
    assert cmd[cmd.index("--split-mesh") + 1] == "4"
    assert cmd[cmd.index("--split-file") + 1] == "mesh_split.cpr"


def test_split_command_custom_split_file():
    """The split file name is configurable."""
    cmd = CPFESimulation._split_command("p", ["-i", "d.i"], 2, split_file="x.cpr")
    assert cmd[-2:] == ["--split-file", "x.cpr"]


@pytest.mark.parametrize("ncore", [1, 0, -1])
def test_split_command_requires_at_least_two_ranks(ncore):
    """Distributed mesh is pointless on a single rank -> hard error."""
    with pytest.raises(ValueError, match="ncore >= 2"):
        CPFESimulation._split_command("puma-opt", ["-i", "d.i"], ncore)
