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

"""Tests for user-selectable slip systems (``graintrace.slip_systems``)."""
from __future__ import annotations

from pathlib import Path

import pytest

import graintrace
from graintrace.slip_systems import SLIP_SYSTEM_PRESETS, SlipSystems

CPFE_BASE = Path(graintrace.__file__).parent / "cpfe_base"
DECKS = ["neml2_cpfe.i", "neml2_cpfe_calibration.i"]
#: preset -> number of slip systems NEML2's CubicCrystal generates
NSLIP = {"fcc": 12, "bcc": 12, "bcc_112": 12, "bcc_110_112": 24, "bcc_110_112_123": 48}


class TestResolve:
    def test_default_is_fcc(self):
        ss = SlipSystems.resolve(None)
        assert ss.name == "fcc"
        assert ss.families == (((1, 1, 0), (1, 1, 1)),)

    @pytest.mark.parametrize("name", sorted(SLIP_SYSTEM_PRESETS))
    def test_presets(self, name):
        ss = SlipSystems.resolve(name.upper())
        assert ss.name == name
        assert len(ss.families) == len(SLIP_SYSTEM_PRESETS[name])

    def test_explicit_pairs_and_dicts_agree(self):
        a = SlipSystems.resolve([((1, 1, 1), (1, 1, 0)), ([1, 1, 1], [1, 2, 3])])
        b = SlipSystems.resolve(
            [
                {"direction": [1, 1, 1], "plane": [1, 1, 0]},
                {"direction": (1, 1, 1), "plane": (1, 2, 3)},
            ]
        )
        assert a.families == b.families
        assert a.name == "custom"
        assert a.to_dict()["families"][1] == {
            "direction": [1, 1, 1],
            "plane": [1, 2, 3],
        }

    def test_passthrough(self):
        ss = SlipSystems.resolve("bcc")
        assert SlipSystems.resolve(ss) is ss

    @pytest.mark.parametrize(
        "bad, msg",
        [
            ("hcp_basal", "unknown slip-system preset"),
            ([], "at least one"),
            ([((1, 1), (1, 1, 1))], "three integers"),
            ([((1, 1, 0.5), (1, 1, 1))], "three integers"),
            ([((0, 0, 0), (1, 1, 1))], "must not be"),
            ([((1, 0, 0), (1, 1, 1))], "no cubic-equivalent"),
            ([(1, 1, 1)], "direction, plane"),
        ],
    )
    def test_rejects(self, bad, msg):
        with pytest.raises(ValueError, match=msg):
            SlipSystems.resolve(bad)

    def test_non_cubic_not_implemented(self):
        with pytest.raises(NotImplementedError, match="cubic"):
            SlipSystems((((1, 1, 0), (1, 1, 1)),), crystal="hexagonal")


class TestApplyToDeck:
    @pytest.mark.parametrize("deck", DECKS)
    def test_default_round_trips_shipped_deck(self, deck):
        text = (CPFE_BASE / deck).read_text(encoding="utf-8")
        assert SlipSystems.resolve(None).apply_to_deck(text) == text

    @pytest.mark.parametrize("deck", DECKS)
    def test_rewrites_both_lines(self, deck):
        text = (CPFE_BASE / deck).read_text(encoding="utf-8")
        out = SlipSystems.resolve("bcc_110_112").apply_to_deck(text)
        assert (
            "torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.int64))'  # @SLIP directions"
            in out
        )
        assert (
            "torch.tensor([[1, 1, 0], [1, 1, 2]], dtype=torch.int64))'  # @SLIP planes"
            in out
        )
        assert out.count("@SLIP") == text.count("@SLIP")

    def test_missing_marker_raises(self):
        with pytest.raises(RuntimeError, match="@SLIP directions"):
            SlipSystems.resolve("bcc").apply_to_deck("[Tensors]\n[]\n")


@pytest.mark.parametrize("deck", DECKS)
@pytest.mark.parametrize("name", sorted(NSLIP))
def test_neml2_generates_expected_slip_count(tmp_path, deck, name):
    """The rewritten deck loads in NEML2 with the expected number of systems."""
    neml2 = pytest.importorskip("neml2")
    text = SlipSystems.resolve(name).apply_to_deck(
        (CPFE_BASE / deck).read_text(encoding="utf-8")
    )
    path = tmp_path / deck
    path.write_text(text, encoding="utf-8")
    assert neml2.load_input(str(path)).get_data("crystal_geometry").nslip == NSLIP[name]


def test_taylor_model_uses_requested_slip(tmp_path):
    """TaylorModel(slip_systems=...) loads a rewritten copy; stress responds to slip."""
    pytest.importorskip("neml2")
    pytest.importorskip("pyzag")
    import torch

    from graintrace.taylor import TaylorModel

    torch.manual_seed(42)
    deck = str(CPFE_BASE / "neml2_cpfe_calibration.i")
    ori = torch.zeros(4, 3, dtype=torch.float64)
    ori[:, 0] = torch.linspace(0.0, 0.3, 4)
    strain = torch.linspace(0.0, 0.01, 21, dtype=torch.float64)
    out = {}
    for spec in (None, "bcc_110_112"):
        tm = TaylorModel(neml2_path=deck, slip_systems=spec, npoints=20)
        with torch.no_grad():
            out[spec] = tm.tmodel.forward(ori, strain, 1e-4)[-1, 2].item()
        if spec is not None:
            assert tm.slip_systems.name == "bcc_110_112"
            assert tm.neml2_path == deck  # the packaged deck is not modified
    assert abs(out[None] - out["bcc_110_112"]) > 1.0


def test_cpfe_bakes_slip_systems(tmp_path):
    """CPFESimulation(slip_systems=...) writes the families into the baked model."""
    from graintrace.run_cpfe_simulation import CPFESimulation

    for f in ("mesh.e", "puma-opt"):
        (tmp_path / f).write_text("", encoding="utf-8")
    sim = CPFESimulation(
        mesh_file=tmp_path / "mesh.e",
        save_simulation_folder=tmp_path / "sim",
        moose_run_file=tmp_path / "puma-opt",
        slip_systems=[{"direction": [1, 1, 1], "plane": [1, 2, 3]}],
    )
    dst = tmp_path / "baked.i"
    sim._bake_neml2_model(
        CPFE_BASE / "neml2_cpfe.i", dst
    )  # pylint: disable=protected-access
    text = dst.read_text(encoding="utf-8")
    assert "torch.tensor([[1, 1, 1]], dtype=torch.int64))'  # @SLIP directions" in text
    assert "torch.tensor([[1, 2, 3]], dtype=torch.int64))'  # @SLIP planes" in text
    assert "@BAKE E nu G" in text

    with pytest.raises(ValueError, match="unknown slip-system preset"):
        CPFESimulation(
            mesh_file=tmp_path / "mesh.e",
            save_simulation_folder=tmp_path / "sim2",
            moose_run_file=tmp_path / "puma-opt",
            slip_systems="hcp",
        )
