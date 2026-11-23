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

"""Ground truth for graintrace's orientation convention (intentionally NOT invariant).

These pin the convention against independent references, so they FAIL on the
pre-0.3 passive-``g`` code:

* MIDAS ``Euler2OrientMat`` (``R = Rz(phi1) Rx(Phi) Rz(phi2)``) == ``euler_to_matrix``;
* neml2 ``ResolvedShear`` fed ``euler_to_mrp`` gives the Schmid factors of the
  crystal-frame loading axis ``R^T z`` (BCC {110}<111>, 12 systems);
* IPF corners [001]/[101]/[111] through the Euler and the MRP paths, and agreement
  with ``neml2.texture.polefigure.inverse_pole_figure_points``;
* the calibration loader and the CPFE 9-column ``.ori`` both reach ``mrp(R)``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("neml2")
torch = pytest.importorskip("torch")

from graintrace import orientation_helper as oh  # noqa: E402

MWE = Path(__file__).resolve().parents[1] / "mwe_data"


def _midas_euler2orientmat(eul_deg):
    """MIDAS ``Euler2OrientMat`` (psi, phi, theta) = (phi1, Phi, phi2), written out."""
    psi, phi, theta = np.deg2rad(np.asarray(eul_deg, dtype=float)).T
    cps, cph, cth = np.cos(psi), np.cos(phi), np.cos(theta)
    sps, sph, sth = np.sin(psi), np.sin(phi), np.sin(theta)
    m = np.empty(psi.shape + (3, 3))
    m[..., 0, 0] = cth * cps - sth * cph * sps
    m[..., 0, 1] = -cth * cph * sps - sth * cps
    m[..., 0, 2] = sph * sps
    m[..., 1, 0] = cth * sps + sth * cph * cps
    m[..., 1, 1] = cth * cph * cps - sth * sps
    m[..., 1, 2] = -sph * cps
    m[..., 2, 0] = sth * sph
    m[..., 2, 1] = cth * sph
    m[..., 2, 2] = cph
    return m


def _random_euler(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(0, 360, n),
            np.degrees(np.arccos(rng.uniform(-1, 1, n))),
            rng.uniform(0, 360, n),
        ],
        axis=1,
    )


class TestMatrixConvention:
    """``euler_to_matrix`` is the active MIDAS / Rz Rx Rz matrix."""

    def test_matches_midas_euler2orientmat(self):
        e = _random_euler()
        got = oh.euler_to_matrix(e, "bunge", "degrees").numpy()
        assert np.allclose(got, _midas_euler2orientmat(e), atol=1e-12)

    def test_is_rz_rx_rz(self):
        from scipy.spatial.transform import Rotation

        e = _random_euler()
        ref = Rotation.from_euler("ZXZ", e, degrees=True).as_matrix()  # intrinsic
        assert np.allclose(oh.euler_to_matrix(e).numpy(), ref, atol=1e-12)

    def test_maps_crystal_to_sample(self):
        """phi1 alone rotates the crystal x axis by +phi1 about sample z."""
        r = oh.euler_to_matrix([[30.0, 0.0, 0.0]]).numpy()[0]
        assert np.allclose(r @ [1, 0, 0], [np.cos(np.pi / 6), np.sin(np.pi / 6), 0])

    def test_rotation_convention_constant(self):
        assert oh.ROTATION_CONVENTION == "active"

    def test_legacy_mrp_to_mrp_is_the_inverse(self):
        e = _random_euler()
        p = oh.euler_to_mrp(e)
        legacy = oh.matrix_to_mrp(oh.euler_to_matrix(e).transpose(-2, -1))
        assert torch.allclose(oh.legacy_mrp_to_mrp(legacy), p, atol=1e-12)


_BCC_HIT = """
[Tensors]
  [a]
    type = Python
    expr = 'Scalar(1.0)'
  []
  [sdirs]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 1], dtype=torch.int64))'
  []
  [splanes]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 0], dtype=torch.int64))'
  []
[]
[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 'a'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]
[Models]
  [rot]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [rss]
    type = ResolvedShear
  []
  [model]
    type = ComposedModel
    models = 'rot rss'
  []
[]
"""


def _bcc_systems():
    """The 12 BCC {110}<111> slip systems (unit d, n), one sign per system."""
    out = []
    dirs = [
        np.array(d, float) for d in itertools.product([1, -1], repeat=3) if d[0] > 0
    ]
    planes = {p for p in itertools.permutations([1, 1, 0])} | {
        p for p in itertools.permutations([1, -1, 0])
    }
    for d in dirs:
        du = d / np.linalg.norm(d)
        for n in planes:
            n = np.array(n, float) / np.linalg.norm(n)
            if abs(du @ n) > 1e-9:
                continue
            if any(np.allclose(n, -m) and np.allclose(du, e) for e, m in out):
                continue
            out.append((du, n))
    return out


class TestNeml2ResolvedShear:
    """neml2 fed ``euler_to_mrp`` resolves uniaxial z-stress with Schmid factors of ``R^T z``."""

    def test_bcc_schmid_factors(self):
        import neml2

        model = neml2.load_string(_BCC_HIT).get_model("model")
        systems = _bcc_systems()
        assert len(systems) == 12
        e = np.vstack([_random_euler(20, 3), [[37.0, 54.7356, 45.0], [0, 0, 0]]])
        sig = 100.0
        stress = torch.zeros(len(e), 6, dtype=torch.float64)
        stress[:, 2] = sig  # Mandel zz
        out = model.call_by_name({"orientation": oh.euler_to_mrp(e), "stress": stress})
        tau = np.abs(out["resolved_shears"].data.numpy())
        assert tau.shape == (len(e), 12)

        # independent reference: the MIDAS matrix, not graintrace's own
        rmat = _midas_euler2orientmat(e)
        z = np.array([0.0, 0.0, 1.0])
        for k in range(len(e)):
            axis = rmat[k].T @ z  # loading axis in the crystal frame
            want = sorted(abs(sig * (axis @ d) * (axis @ n)) for d, n in systems)
            assert np.allclose(np.sort(tau[k]), want, atol=1e-9)
        # the passive reading (crystal axis g^T z = R z) is distinguishably wrong
        axis = rmat[0] @ z
        wrong = sorted(abs(sig * (axis @ d) * (axis @ n)) for d, n in systems)
        assert not np.allclose(np.sort(tau[0]), wrong, atol=1e-3)


class TestIPFConvention:
    """Euler and MRP paths put a known crystal direction at its IPF corner."""

    @staticmethod
    def _ipf():
        from graintrace.ipf_postprocess import IPFProcessor

        return IPFProcessor(crystal_symmetry="432", sample_symmetry="1", save_dir=".")

    # crystal direction along sample z: R^T z = (sinP sin p2, sinP cos p2, cosP),
    # independent of phi1 (non-zero phi1 so a transposed matrix would move it)
    CORNERS = {
        "001": ([37.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        "101": ([37.0, 45.0, 90.0], [1.0, 0.0, 1.0]),
        "111": ([37.0, np.degrees(np.arccos(1 / np.sqrt(3))), 45.0], [1.0, 1.0, 1.0]),
    }

    @pytest.mark.parametrize("corner", ["001", "101", "111"])
    def test_corner_via_euler_and_mrp(self, corner):
        eul, want = self.CORNERS[corner]
        want = np.asarray(want) / np.linalg.norm(want)
        ipf = self._ipf()
        r_eul = oh.euler_to_matrix([eul])
        r_mrp = oh.mrp_to_matrix(oh.euler_to_mrp([eul]))
        for r in (r_eul, r_mrp):
            got = ipf.get_reduced_ipf_directions(r, [0.0, 0.0, 1.0]).numpy()[0]
            assert np.allclose(got, want, atol=1e-7)
        pt = ipf.get_ipf_points(r_eul, [0.0, 0.0, 1.0]).numpy()[0]
        assert np.allclose(pt, want[:2] / (1.0 + want[2]), atol=1e-7)

    def test_agrees_with_neml2_inverse_pole_figure_points(self):
        from neml2.texture.polefigure import inverse_pole_figure_points

        ipf = self._ipf()
        e = _random_euler(40, 5)
        p = oh.euler_to_mrp(e)
        ours = ipf.get_ipf_points(oh.mrp_to_matrix(p), [0.0, 0.0, 1.0]).numpy()
        compared = 0
        for k in range(len(e)):
            pts = inverse_pole_figure_points(
                p[k : k + 1],
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                crystal_symmetry="432",
            )
            pts = np.asarray(pts).reshape(-1, 2)
            if len(pts) == 0:
                continue
            compared += 1
            assert np.min(np.linalg.norm(pts - ours[k], axis=1)) < 1e-6
        assert compared >= len(e) // 2


class TestPipelineHandoffs:
    """Calibration loading and the CPFE 9-column .ori both reach mrp(R)."""

    def test_calibration_texture_from_euler(self):
        pytest.importorskip("pyzag")
        from graintrace.taylor import _texture_mrp

        f = MWE / "ff_calibration" / "0.csv"
        if not f.exists():
            pytest.skip("mwe_data/ff_calibration missing")
        df = pd.read_csv(f)
        got = _texture_mrp(df, ["Eul0", "Eul1", "Eul2"], "bunge", "auto")
        want = oh.euler_to_mrp(
            torch.as_tensor(df[["Eul0", "Eul1", "Eul2"]].to_numpy()), "bunge", "radians"
        )
        assert torch.allclose(got, want, atol=1e-12)
        # this dataset's O11..O33 are MIDAS O = R, so the matrix fallback agrees
        o_only = df.drop(columns=["Eul0", "Eul1", "Eul2"])
        via_o = _texture_mrp(o_only, ["Eul0", "Eul1", "Eul2"], "bunge", "auto")
        assert torch.allclose(
            oh.mrp_to_matrix(via_o), oh.mrp_to_matrix(want), atol=1e-5
        )
        with pytest.raises(ValueError, match="Euler"):
            _texture_mrp(df[["X", "Y", "Z"]], ["Eul0", "Eul1", "Eul2"], "bunge", "auto")

    def test_cpfe_nine_column_ori_is_active_r(self, tmp_path):
        from graintrace.run_cpfe_simulation import CPFESimulation

        e = _random_euler(4, 8)
        r = oh.euler_to_matrix(e).numpy()
        ori = tmp_path / "ori.dat"
        np.savetxt(ori, r.reshape(-1, 9))
        for name in ("mesh.e", "puma-opt"):
            (tmp_path / name).write_text("", encoding="utf-8")
        sim = CPFESimulation(
            mesh_file=tmp_path / "mesh.e",
            save_simulation_folder=tmp_path / "sim",
            moose_run_file=tmp_path / "puma-opt",
            element_order="FIRST",
            ori_file=ori,
            use_ff_initial_field=False,
            dim=3,
        )
        sim.save_simulation_folder.mkdir(parents=True, exist_ok=True)
        assert sim.write_orientation_file() == 4
        mrp = np.loadtxt(tmp_path / "sim" / "mrps_orientation.csv", delimiter=",")
        assert np.allclose(oh.mrp_to_matrix(torch.as_tensor(mrp)).numpy(), r, atol=1e-5)
        assert np.allclose(mrp, oh.euler_to_mrp(e).numpy(), atol=1e-5)
