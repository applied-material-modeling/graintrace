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
        # w component should be 1 (or -1) for identity
        assert abs(float(q[0, 0])) == pytest.approx(1.0, abs=1e-5)

    def test_matrix_to_quat_norm_one(self):
        from graintrace.orientation_helper import matrix_to_quat

        rng = np.random.default_rng(7)
        # Generate random rotation via QR decomposition
        A = torch.tensor(rng.normal(size=(4, 3, 3)), dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)
        # fix det=1
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
