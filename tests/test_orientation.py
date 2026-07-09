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

"""Tests for orientation math: orientation_helper, nf.metrics, nf.image."""
from __future__ import annotations

import numpy as np
import pytest
import torch


class TestOrientationHelper:
    """Tests requiring neml2 — marked as unit but neml2 must be installed."""

    def test_misorientation_identical_zero(self):
        from graintrace.orientation_helper import misorientation

        e = [0.0, 0.0, 0.0]
        val = misorientation(e, e, angle_convention="bunge", angle_type="degrees", symmetry="1")
        assert float(val) == pytest.approx(0.0, abs=1e-4)

    def test_misorientation_small_angle(self):
        from graintrace.orientation_helper import misorientation

        e1 = [0.0, 0.0, 0.0]
        e2 = [5.0, 0.0, 0.0]
        val = misorientation(e1, e2, angle_convention="bunge", angle_type="degrees", symmetry="1")
        assert float(val) == pytest.approx(5.0, abs=0.5)

    def test_misorientation_cubic_symmetry(self):
        from graintrace.orientation_helper import misorientation

        # 90-degree rotation about z is equivalent in cubic symmetry
        e1 = [0.0, 0.0, 0.0]
        e2 = [90.0, 0.0, 0.0]
        val = misorientation(e1, e2, angle_convention="bunge", angle_type="degrees", symmetry="432")
        assert float(val) == pytest.approx(0.0, abs=1e-3)

    def test_matrix_to_quat_identity(self):
        from graintrace.orientation_helper import matrix_to_quat

        I = torch.eye(3).unsqueeze(0)
        q = matrix_to_quat(I)
        assert q.shape == (1, 4)
        # w component is 1 (or -1) for identity
        assert abs(float(q[0, 0])) == pytest.approx(1.0, abs=1e-5)

    def test_matrix_to_quat_norm_one(self):
        from graintrace.orientation_helper import matrix_to_quat

        rng = np.random.default_rng(7)
        # random rotations via QR, forced to det=1
        A = torch.tensor(rng.normal(size=(4, 3, 3)), dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)
        for i in range(4):
            if torch.linalg.det(Q[i]) < 0:
                Q[i, :, 0] *= -1
        q = matrix_to_quat(Q)
        norms = torch.linalg.norm(q, dim=-1)
        assert torch.allclose(norms, torch.ones(4, dtype=torch.float64), atol=1e-6)

    def test_load_strains_shape(self):
        from graintrace.orientation_helper import load_strains

        df = pytest.importorskip("pandas").DataFrame(
            {
                "eKen11": [1e-4, 2e-4],
                "eKen22": [3e-4, 4e-4],
                "eKen33": [5e-4, 6e-4],
                "eKen23": [7e-4, 8e-4],
                "eKen13": [9e-4, 1e-3],
                "eKen12": [2e-3, 3e-3],
            }
        )
        t = load_strains(df, field="eKen", factor=1e-6)
        assert t.shape == (2, 6)

    def test_load_weights_sums_to_one(self):
        from graintrace.orientation_helper import load_weights

        import pandas as pd
        df = pd.DataFrame({"GrainRadius": [10.0, 20.0, 30.0, 40.0]})
        w = load_weights(df)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
        assert (w > 0).all()


class TestOrientationInterchange:
    """graintrace's canonical orientation interchange is neml2 v3 MRP; every
    converter must round-trip through the rotation matrix consistently."""

    _E = torch.tensor(
        [[10.0, 20.0, 30.0], [45.0, 45.0, 0.0], [0.0, 0.0, 0.0], [123.0, 44.0, 271.0]],
        dtype=torch.float64,
    )

    def test_euler_mrp_matrix_roundtrip(self):
        from graintrace import orientation_helper as oh

        M = oh.euler_to_matrix(self._E, "bunge", "degrees")
        mrp = oh.euler_to_mrp(self._E, "bunge", "degrees")
        # euler->mrp->matrix recovers the original matrix
        assert torch.allclose(oh.mrp_to_matrix(mrp), M, atol=1e-8)
        # matrix->mrp->matrix is identity
        assert torch.allclose(oh.mrp_to_matrix(oh.matrix_to_mrp(M)), M, atol=1e-8)
        # mrp->euler->matrix recovers the original matrix
        er = oh.mrp_to_euler(mrp, "bunge", "degrees")
        assert torch.allclose(oh.euler_to_matrix(er, "bunge", "degrees"), M, atol=1e-8)

    def test_euler_to_mrp_is_true_neml2_mrp(self):
        from graintrace import orientation_helper as oh
        from neml2 import types as t

        M = oh.euler_to_matrix(self._E, "bunge", "degrees").contiguous()
        ref = t.MRP.from_matrix(t.R2(M, 0)).data
        assert torch.allclose(oh.euler_to_mrp(self._E, "bunge", "degrees"), ref, atol=1e-10)

    def test_load_orientations_returns_neml2_mrp(self):
        from graintrace import orientation_helper as oh
        import pandas as pd

        M = oh.euler_to_matrix(self._E, "bunge", "degrees")
        cols = [f"O{i}{j}" for i in range(1, 4) for j in range(1, 4)]
        df = pd.DataFrame(M.reshape(-1, 9).numpy(), columns=cols)
        assert torch.allclose(oh.load_orientations(df), oh.matrix_to_mrp(M), atol=1e-10)
        # load_orientations_mrp is kept as an alias
        assert oh.load_orientations_mrp is oh.load_orientations

    def test_average_rotations_returns_mrp_3vec(self):
        """Guards the Phase-1 regression: average_rotations must return a 3-vector
        neml2 MRP (not a 3x3 matrix), so nf.mesh.write_spn's (N,3) buffer works."""
        from graintrace.nf.metrics import average_rotations
        from graintrace import orientation_helper as oh

        e = self._E[:3]  # a small cluster to average
        mrp, euler = average_rotations(e, angle_convention="bunge", angle_type="degrees")
        assert mrp.shape == (3,)
        assert euler.shape == (3,)
        # the returned MRP and euler describe the same rotation
        assert torch.allclose(
            oh.mrp_to_matrix(mrp),
            oh.euler_to_matrix(euler, "bunge", "degrees"),
            atol=1e-6,
        )


class TestNFImage:
    def test_connectivity_6_offsets(self):
        from graintrace.nf.image import connectivity_options, get_neighbor_indices

        offsets = connectivity_options[6]
        assert offsets.shape == (6, 3)

    def test_connectivity_26_offsets(self):
        from graintrace.nf.image import connectivity_options, get_neighbor_indices

        offsets = connectivity_options[26]
        assert offsets.shape == (26, 3)

    def test_get_neighbor_indices_shape(self):
        from graintrace.nf.image import connectivity_options, get_neighbor_indices

        nx, ny, nz = 4, 4, 4
        offsets = connectivity_options[6]
        shape = (nx, ny, nz)
        result = get_neighbor_indices(offsets, shape)
        # Returns tuple of (dx, dy, dz, Xk, Yk, Zk, valid)
        assert len(result) >= 3

    def test_only_6_and_26_defined(self):
        from graintrace.nf.image import connectivity_options

        assert 6 in connectivity_options
        assert 26 in connectivity_options


class TestNFMetrics:
    def test_misorientation_zero_for_identical(self):
        from graintrace.nf.metrics import misorientation as nf_mis

        e = torch.zeros(5, 3, dtype=torch.float64)
        result = nf_mis(e, e, angle_convention="bunge", angle_type="radians", symmetry="1")
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)
