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

"""Unit tests for the CPFESimulation distributed-mesh and solver-route options.

These test the pure command-building logic (`_split_command`), the default parameters,
the cpfe_base Executioner decks, and the solver-route deck selection in `run()`, so they
run without the MOOSE/PUMA/NEML2 stack (the one deck-selection test that reaches `run()`'s
Popen path skips if torch is unavailable).
"""

from pathlib import Path

import pytest

import graintrace.run_cpfe_simulation as rcs
from graintrace.run_cpfe_simulation import CPFESimulation

CPFE_BASE = Path(rcs.__file__).parent / "cpfe_base"


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


def test_solver_route_defaults_to_timestep_optimized():
    """The default solver route is the direct-LU, time-step-optimized executioner."""
    assert (
        CPFESimulation.DEFAULT_PARAMS["simulation_parameters"]["solver_route"]
        == "timestep_optimized"
    )


def test_solver_route_decks_exist():
    """Every route in the map resolves to an Executioner deck shipped in cpfe_base."""
    assert set(CPFESimulation.SOLVER_ROUTE_DECKS) == {
        "timestep_optimized",
        "hpc_memory",
    }
    for deck in CPFESimulation.SOLVER_ROUTE_DECKS.values():
        assert (CPFE_BASE / deck).exists(), deck


def test_timestep_optimized_deck_uses_direct_lu():
    """The default route factorizes directly (superlu_dist) and reuses the preconditioner."""
    text = (CPFE_BASE / "executioner_timestep_optimized.i").read_text(encoding="utf-8")
    assert "superlu_dist" in text
    assert "reuse_preconditioner" in text
    assert "gamg" not in text


def test_hpc_memory_deck_uses_gamg():
    """The HPC route uses an iterative fgmres solve preconditioned by GAMG multigrid."""
    text = (CPFE_BASE / "executioner_hpc_memory.i").read_text(encoding="utf-8")
    assert "gamg" in text
    assert "fgmres" in text


def test_run_cpfe_deck_has_no_executioner_block():
    """The [Executioner] block was extracted into the selectable route decks."""
    text = (CPFE_BASE / "run_cpfe.i").read_text(encoding="utf-8")
    # no active block declaration (a reference in a comment is fine)
    assert not any(line.strip() == "[Executioner]" for line in text.splitlines())


@pytest.mark.parametrize(
    "deck",
    [
        "executioner_timestep_optimized.i",
        "executioner_hpc_memory.i",
    ],
)
def test_executioner_decks_disable_residual_and_jacobian_together(deck):
    """Both route decks must keep residual_and_jacobian_together=false.

    The flat loading-face BC is an EqualValueBoundaryConstraint (a NodalConstraint), and newer
    MOOSE hard-errors on residual_and_jacobian_together=true when nodal constraints are present
    (MOOSE issue 33531). Guard against a regression back to true in either deck.
    """
    text = (CPFE_BASE / deck).read_text(encoding="utf-8")
    assert "residual_and_jacobian_together = false" in text
    assert "residual_and_jacobian_together = true" not in text


def _make_min_sim(tmp_path, **sim_params):
    """Build a CPFESimulation over dummy files (no MOOSE run performed)."""
    mesh = tmp_path / "mesh.e"
    mesh.write_text("", encoding="utf-8")
    moose = tmp_path / "puma-opt"
    moose.write_text("", encoding="utf-8")
    ori = tmp_path / "ori.csv"
    # 3-column MRP orientations (two grains) -> copied as-is by write_orientation_file
    ori.write_text("0 0 0\n0.1 0.2 0.3\n", encoding="utf-8")

    sim = CPFESimulation(
        mesh_file=mesh,
        save_simulation_folder=tmp_path / "sim_out",
        moose_run_file=moose,
        element_order="FIRST",
        ori_file=ori,
        use_ff_initial_field=False,
        dim=3,
    )
    if sim_params:
        sim.set_parameters("simulation_parameters", **sim_params)
    return sim


def test_run_rejects_invalid_solver_route(tmp_path):
    """An unknown solver_route is caught early in run() with a clear error."""
    sim = _make_min_sim(tmp_path, solver_route="nonsense")
    with pytest.raises(ValueError, match="Invalid solver_route"):
        sim.run(ncore=1)


@pytest.mark.parametrize(
    "route,deck",
    [
        ("timestep_optimized", "executioner_timestep_optimized.i"),
        ("hpc_memory", "executioner_hpc_memory.i"),
    ],
)
def test_run_selects_executioner_deck(tmp_path, monkeypatch, route, deck):
    """run() copies the route's Executioner deck and lists it in the puma-opt -i argv."""
    pytest.importorskip(
        "torch"
    )  # write_orientation_file (3-col path) builds a torch tensor
    sim = _make_min_sim(tmp_path, solver_route=route, recompile=False)

    # Pre-place a stub so the AOTI compile is skipped (recompile=False).
    aoti = sim.save_simulation_folder / "aoti"
    aoti.mkdir(parents=True, exist_ok=True)
    (aoti / "model_aoti.i").write_text("", encoding="utf-8")

    captured = {}

    class _FakeProc:
        pid = 4321

    def _fake_popen(argv, *args, **kwargs):  # pylint: disable=unused-argument
        captured["argv"] = argv
        return _FakeProc()

    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen)

    sim.run(ncore=1)

    argv = captured["argv"]
    assert deck in argv
    # only the selected deck is merged, never the other route's deck
    other = {"executioner_timestep_optimized.i", "executioner_hpc_memory.i"} - {deck}
    assert not (other & set(argv))
    # the deck was copied into the run folder for the merge
    assert (sim.save_simulation_folder / deck).exists()
