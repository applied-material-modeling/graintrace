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


"""Unit tests for CPFESimulation's loaded-face flat constraint and the dt growth_factor.

Both are pure input-deck generation, so they run without the MOOSE/PUMA/NEML2 stack (the
argv test that reaches ``run()``'s Popen path skips if torch is unavailable).
"""

import re
from pathlib import Path

import pytest

import graintrace.run_cpfe_simulation as rcs
from graintrace.run_cpfe_simulation import CPFESimulation

CPFE_BASE = Path(rcs.__file__).parent / "cpfe_base"
DECKS = ["executioner_timestep_optimized.i", "executioner_hpc_memory.i"]
FREE = "stress_free"


def _make_sim(tmp_path, bc=None, **sim_params):
    """Build a CPFESimulation over dummy files (no MOOSE run performed)."""
    mesh = tmp_path / "mesh.e"
    mesh.write_text("", encoding="utf-8")
    moose = tmp_path / "puma-opt"
    moose.write_text("", encoding="utf-8")
    ori = tmp_path / "ori.csv"
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
    if bc is not None:
        sim.set_parameters("boundary", bc=bc)
    if sim_params:
        sim.set_parameters("simulation_parameters", **sim_params)
    return sim


def _constraints(sim):
    """Write boundary_conditions.i; return {block_name: {key: value}} of [Constraints]."""
    sim.write_bc_file()
    text = (sim.save_simulation_folder / "boundary_conditions.i").read_text(
        encoding="utf-8"
    )
    lines = text.split("[Constraints]\n", 1)[1].splitlines()
    body = "\n".join(lines[: lines.index("[]")])
    blocks = {}
    for name, inner in re.findall(r"\[(\w+)\]\n(.*?)\n\s*\[\]", body, flags=re.S):
        blocks[name] = dict(
            (k.strip(), v.strip())
            for k, v in (ln.split("=", 1) for ln in inner.splitlines())
        )
    return blocks


def _bc(x=(FREE, FREE), y=(FREE, FREE), z=(FREE, FREE)):
    return {
        a: {"negative": v[0], "positive": v[1]}
        for a, v in (("x", x), ("y", y), ("z", z))
    }


def test_flat_constraint_default_z_load_is_legacy(tmp_path):
    """The default z tension keeps one flat constraint on the front face, disp_z."""
    blocks = _constraints(_make_sim(tmp_path))
    assert set(blocks) == {"front_flat"}
    c = blocks["front_flat"]
    assert c["type"] == "EqualValueBoundaryConstraint"
    assert c["secondary"] == "'front'"
    assert c["variable"] == "disp_z"


def test_flat_constraint_follows_x_loading_axis(tmp_path):
    """An x-load constrains the loaded right face on disp_x, not the front on disp_z."""
    blocks = _constraints(_make_sim(tmp_path, bc=_bc(x=(0, 0.002))))
    assert set(blocks) == {"right_flat"}
    assert blocks["right_flat"]["secondary"] == "'right'"
    assert blocks["right_flat"]["variable"] == "disp_x"


def test_flat_constraint_negative_face(tmp_path):
    """A load on a negative face (bottom, y) constrains that face on disp_y."""
    blocks = _constraints(_make_sim(tmp_path, bc=_bc(y=(-0.001, 0))))
    assert set(blocks) == {"bottom_flat"}
    assert blocks["bottom_flat"]["variable"] == "disp_y"


def test_flat_constraint_one_per_loaded_face(tmp_path):
    """Two loaded faces give two constraints, each on its own component."""
    blocks = _constraints(_make_sim(tmp_path, bc=_bc(x=(0, 0.001), z=(0, 0.002))))
    assert set(blocks) == {"right_flat", "front_flat"}
    assert blocks["right_flat"]["variable"] == "disp_x"
    assert blocks["front_flat"]["variable"] == "disp_z"


def test_no_flat_constraint_without_a_load(tmp_path):
    """Fixed (zero) or traction-free faces get no constraint."""
    blocks = _constraints(_make_sim(tmp_path, bc=_bc(z=(0, 0))))
    assert not blocks


def test_growth_factor_defaults_to_legacy_two():
    """Backward compatible: the default dt growth factor is the old hard-coded 2."""
    assert (
        CPFESimulation.DEFAULT_PARAMS["simulation_parameters"]["growth_factor"] == 2.0
    )


@pytest.mark.parametrize("deck", DECKS)
def test_decks_take_growth_factor_from_the_runner(deck):
    """Both Executioner decks read growth_factor from the runner, never a literal."""
    text = (CPFE_BASE / deck).read_text(encoding="utf-8")
    assert re.search(r"growth_factor\s*=\s*\$\{growth_factor\}", text)
    assert not re.search(r"growth_factor\s*=\s*[0-9]", text)


@pytest.mark.parametrize(
    "value,expected", [(2, 2.0), (1.5, 1.5), ("1.25", 1.25), (1, 1.0)]
)
def test_resolve_growth_factor_accepts_numbers_ge_one(tmp_path, value, expected):
    """Numbers >= 1 (incl. numeric strings) are accepted as floats."""
    sim = _make_sim(tmp_path, growth_factor=value)
    assert sim._resolve_growth_factor() == expected


@pytest.mark.parametrize(
    "value", [0.5, 0, -2, float("nan"), float("inf"), "fast", None]
)
def test_resolve_growth_factor_rejects_invalid(tmp_path, value):
    """Values below 1, non-finite or non-numeric are rejected with a clear error."""
    sim = _make_sim(tmp_path, growth_factor=value)
    with pytest.raises(ValueError, match="Invalid growth_factor"):
        sim._resolve_growth_factor()


def test_run_rejects_invalid_growth_factor_early(tmp_path):
    """run() fails before writing/launching anything on a bad growth_factor."""
    sim = _make_sim(tmp_path, growth_factor=0.5)
    with pytest.raises(ValueError, match="Invalid growth_factor"):
        sim.run(ncore=1)


@pytest.mark.parametrize(
    "value,arg", [(None, "growth_factor=2"), (1.5, "growth_factor=1.5")]
)
def test_run_passes_growth_factor_to_puma(tmp_path, monkeypatch, value, arg):
    """run() forwards growth_factor to puma-opt (default 2 when not set)."""
    pytest.importorskip("torch")  # write_orientation_file builds a torch tensor
    kw = {"recompile": False}
    if value is not None:
        kw["growth_factor"] = value
    sim = _make_sim(tmp_path, **kw)
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
    assert arg in captured["argv"]


def _mcp_preview(tmp_path, **kw):
    """Preview (confirm=False) the MCP run_cpfe tool; skips without the mcp extra."""
    pytest.importorskip("mcp.server.fastmcp")
    from graintrace.mcp.tools import (  # pylint: disable=import-outside-toplevel
        simulation,
    )

    out = simulation.run_cpfe(
        mesh_file="m.e",
        ori_file="o.csv",
        moose_run_file="puma-opt",
        save_simulation_folder=str(tmp_path / "sim"),
        bounding_box=[0, 1, 0, 1, 0, 1],
        total_strain=0.002,
        **kw,
    )
    assert out["status"] == "preview"
    return out["resolved_params"]["parameters"]["simulation_parameters"]


def test_mcp_run_cpfe_forwards_growth_factor(tmp_path):
    """run_cpfe defaults growth_factor to 2.0 and forwards an explicit value."""
    assert _mcp_preview(tmp_path)["growth_factor"] == 2.0
    assert _mcp_preview(tmp_path, growth_factor=1.5)["growth_factor"] == 1.5


def test_mcp_run_cpfe_explicit_parameters_win(tmp_path):
    """An explicit simulation_parameters growth_factor overrides the dedicated arg."""
    sp = _mcp_preview(
        tmp_path,
        growth_factor=1.5,
        parameters={"simulation_parameters": {"growth_factor": 1.2}},
    )
    assert sp["growth_factor"] == 1.2
