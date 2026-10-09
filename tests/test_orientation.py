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

# orientation_helper's conversions delegate to neml2 (types/ops). The working
# neml2 is the repo-pinned build supplied by PUMA into the PUMA env; on a plain
# checkout without it, skip this module rather than erroring.
pytest.importorskip("neml2")


class TestOrientationHelper:
    """Tests requiring neml2 (marked as unit, but neml2 must be installed)."""

    def test_misorientation_identical_zero(self):
        from graintrace.orientation_helper import misorientation

        e = [0.0, 0.0, 0.0]
        val = misorientation(
            e, e, angle_convention="bunge", angle_type="degrees", symmetry="1"
        )
        assert float(val) == pytest.approx(0.0, abs=1e-4)

    def test_misorientation_small_angle(self):
        from graintrace.orientation_helper import misorientation

        e1 = [0.0, 0.0, 0.0]
        e2 = [5.0, 0.0, 0.0]
        val = misorientation(
            e1, e2, angle_convention="bunge", angle_type="degrees", symmetry="1"
        )
        assert float(val) == pytest.approx(5.0, abs=0.5)

    def test_misorientation_cubic_symmetry(self):
        from graintrace.orientation_helper import misorientation

        # 90-degree rotation about z is equivalent in cubic symmetry
        e1 = [0.0, 0.0, 0.0]
        e2 = [90.0, 0.0, 0.0]
        val = misorientation(
            e1, e2, angle_convention="bunge", angle_type="degrees", symmetry="432"
        )
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


class TestPerturbOrientation:
    """perturb_orientation applies a proper random misorientation of a given
    angular std (degrees), measurable back via misorientation_matrix."""

    _E = torch.tensor(
        [[10.0, 20.0, 30.0], [45.0, 45.0, 0.0], [0.0, 0.0, 0.0], [123.0, 44.0, 271.0]],
        dtype=torch.float64,
    )

    def test_zero_sigma_is_identity(self):
        from graintrace.orientation_helper import perturb_orientation

        out = perturb_orientation(self._E, 0.0, "bunge", "degrees")
        assert torch.allclose(out, self._E, atol=1e-12)

    def test_applied_misorientation_scales_with_sigma(self):
        from graintrace.orientation_helper import (
            perturb_orientation,
            euler_to_matrix,
            misorientation_matrix,
        )

        rng = np.random.default_rng(42)
        sigma = 2.0
        n = 4000
        euler = torch.tensor(
            rng.uniform([0.0, 0.0, 0.0], [360.0, 180.0, 360.0], size=(n, 3)),
            dtype=torch.float64,
        )
        out = perturb_orientation(euler, sigma, "bunge", "degrees", rng)

        R0 = euler_to_matrix(euler, "bunge", "degrees")
        R1 = euler_to_matrix(out, "bunge", "degrees")
        mis = misorientation_matrix(R1, R0, symmetry="1", angle_type="degrees")

        # symmetry="1": measured angle == |applied angle| ~ half-normal(sigma),
        # so E[mis] = sigma*sqrt(2/pi) ~ 0.8*sigma. Loose band around that.
        mean_mis = float(mis.mean())
        assert 0.5 * sigma < mean_mis < 1.1 * sigma
        # nothing should blow up far beyond a few sigma
        assert float(mis.max()) < 8.0 * sigma

    def test_reproducible_with_seed(self):
        from graintrace.orientation_helper import perturb_orientation

        a = perturb_orientation(
            self._E, 1.5, "bunge", "degrees", np.random.default_rng(0)
        )
        b = perturb_orientation(
            self._E, 1.5, "bunge", "degrees", np.random.default_rng(0)
        )
        assert torch.allclose(a, b, atol=1e-12)


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
        assert torch.allclose(
            oh.euler_to_mrp(self._E, "bunge", "degrees"), ref, atol=1e-10
        )

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
        mrp, euler = average_rotations(
            e, angle_convention="bunge", angle_type="degrees"
        )
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
        result = nf_mis(
            e, e, angle_convention="bunge", angle_type="radians", symmetry="1"
        )
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)


def _variant_mixed_grain(n=200, seed=0, symmetry="432"):
    """A tight orientation cluster with half of it in a different symmetry variant.

    Returns ``(euler_clean, euler_mixed)`` in bunge/degrees. Both describe the
    same physical grain: a left-multiplied crystal symmetry operator is a
    zero-misorientation equivalent, so the symmetry-aware mean of the mixed set
    must match the mean of the clean set.
    """
    from graintrace import orientation_helper as oh

    rng = np.random.default_rng(seed)
    base = oh.mrp_to_matrix(torch.tensor([[0.1, -0.05, 0.2]], dtype=torch.float64))
    pert = oh.mrp_to_matrix(
        torch.tensor(rng.normal(0, 0.006, size=(n, 3)), dtype=torch.float64)
    )
    rmat = torch.matmul(base, pert)  # ~1.4 deg spread
    var = rmat.clone()
    ops = oh.symmetry_operators(symmetry)
    var[: n // 2] = torch.matmul(ops[1].unsqueeze(0), var[: n // 2])
    return (
        oh.matrix_to_euler(rmat, "bunge", "degrees"),
        oh.matrix_to_euler(var, "bunge", "degrees"),
    )


def _miso_mrp(a, b, symmetry="432"):
    """Misorientation (degrees) between two MRP 3-vectors."""
    from graintrace import orientation_helper as oh

    return float(
        oh.misorientation_matrix(
            oh.mrp_to_matrix(torch.as_tensor(a).reshape(1, 3)),
            oh.mrp_to_matrix(torch.as_tensor(b).reshape(1, 3)),
            symmetry=symmetry,
        ).item()
    )


class TestAverageRotationsSymmetry:
    """Regression for issue #14: ``average_rotations`` had no crystal-symmetry
    reduction, so a grain whose voxels are reported in different symmetry
    variants averaged to an orientation close to no member of the set. Every NF
    orientation export goes through this function."""

    def test_symmetry_aware_mean_recovers_variant_mixed_grain(self):
        from graintrace.nf.metrics import average_rotations

        clean, mixed = _variant_mixed_grain()
        ref, _ = average_rotations(
            clean, angle_convention="bunge", angle_type="degrees", symmetry="432"
        )
        folded, _ = average_rotations(
            mixed, angle_convention="bunge", angle_type="degrees", symmetry="432"
        )
        naive, _ = average_rotations(
            mixed, angle_convention="bunge", angle_type="degrees", symmetry="1"
        )

        # symmetry-aware: folded back onto the coherent mean
        assert _miso_mrp(folded, ref) < 0.5
        # the old (symmetry-blind) behaviour, kept reachable via symmetry="1"
        assert _miso_mrp(naive, ref) > 5.0

    def test_symmetry_mean_matches_fragmentation_implementation(self):
        """The repo already had a correct symmetry-aware mean; the two must agree."""
        from graintrace import orientation_helper as oh
        from graintrace.fragmentation import FragmentationAnalyzer
        from graintrace.nf.metrics import average_rotations

        _, mixed = _variant_mixed_grain()
        got, _ = average_rotations(
            mixed, angle_convention="bunge", angle_type="degrees", symmetry="432"
        )
        want = FragmentationAnalyzer.mean_orientation_mrp(
            oh.matrix_to_mrp(oh.euler_to_matrix(mixed, "bunge", "degrees")).numpy(),
            "432",
        )
        assert _miso_mrp(got, want) < 1e-6

    def test_symmetry_1_is_unchanged_default(self):
        """Default stays the plain quaternion mean, so direct callers are unaffected."""
        from graintrace.nf.metrics import average_rotations

        clean, _ = _variant_mixed_grain()
        a, _ = average_rotations(clean, angle_convention="bunge", angle_type="degrees")
        b, _ = average_rotations(
            clean, angle_convention="bunge", angle_type="degrees", symmetry="1"
        )
        assert torch.allclose(a, b, atol=1e-12)

    def test_write_spn_threads_symmetry_to_grain_average(self, tmp_path):
        """Blast-radius guard: the NF spn/orientation writer must apply symmetry."""
        from graintrace import orientation_helper as oh
        from graintrace.nf import mesh

        clean, mixed = _variant_mixed_grain(n=64)
        # one grain (phase 1) laid out on a 4x4x4 voxel grid
        grid = torch.zeros(4, 4, 4, 7, dtype=torch.float64)
        grid[..., 0] = 1.0
        grid[..., 1:4] = mixed.reshape(4, 4, 4, 3)

        ori = tmp_path / "orientations.dat"
        mesh.write_spn(
            grid.clone(),
            str(tmp_path / "grid.spn"),
            str(ori),
            angle_convention="bunge",
            angle_type="degrees",
            symmetry="432",
        )
        got = np.loadtxt(ori, delimiter=",").reshape(3)

        ref, _ = average_rotations_reference(clean)
        assert _miso_mrp(got, ref) < 0.5


def average_rotations_reference(euler_clean):
    """Symmetry-aware mean of the uncontaminated grain, for comparison."""
    from graintrace.nf.metrics import average_rotations

    return average_rotations(
        euler_clean, angle_convention="bunge", angle_type="degrees", symmetry="432"
    )


def _line_grid(ids, eulers):
    """(n, 1, 1, 7) grid along x with voxel centres at 0, 1, ... and Euler per voxel."""
    n = len(ids)
    grid = np.zeros((n, 1, 1, 7), dtype=np.float64)
    for i, (gid, eul) in enumerate(zip(ids, eulers)):
        grid[i, 0, 0, 0] = gid
        grid[i, 0, 0, 1:4] = eul
        grid[i, 0, 0, 4:7] = [float(i), 0.0, 0.0]
    return grid


def _shift_mesh_x(mesh_path, delta):
    """Offset every mesh node in +x, standing in for SCULPT dilation/smoothing."""
    import scipy.io as sio

    with sio.netcdf_file(mesh_path, "a") as f:
        f.variables["coordx"][:] = f.variables["coordx"][:] + delta


class TestMapOrientationsGrainConstrained:
    """map_orientations must look up a block against its own grain's voxels only."""

    def _run(self, tmp_path, ids, eulers, shift=0.6):
        from graintrace.nf.mesh import map_orientations, write_voxel_exodus

        grid = _line_grid(ids, eulers)
        mesh_file = str(tmp_path / "mesh.e")
        write_voxel_exodus(
            torch.tensor(grid),
            mesh_file,
            str(tmp_path / "ori_ref"),
            angle_type="degrees",
            symmetry="432",
        )
        reference = np.loadtxt(str(tmp_path / "ori_ref.csv"), delimiter=",").reshape(
            -1, 3
        )

        _shift_mesh_x(mesh_file, shift)
        map_orientations(
            mesh_file,
            grid,
            str(tmp_path / "ori_mapped"),
            angle_type="degrees",
            symmetry="432",
        )
        mapped = np.loadtxt(str(tmp_path / "ori_mapped.csv"), delimiter=",").reshape(
            -1, 3
        )
        return reference, mapped

    def test_void_voxel_does_not_supply_the_identity(self, tmp_path):
        # grain 1 | void | void | grain 2. A +0.6 shift puts grain 1's element
        # centroid nearer the void voxel at x = 1, whose Euler is (0, 0, 0).
        reference, mapped = self._run(
            tmp_path,
            [1, 0, 0, 2],
            [[30.0, 40.0, 50.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [70.0, 10.0, 20.0]],
        )
        assert not np.allclose(mapped[0], 0.0)
        assert np.allclose(mapped, reference, atol=1e-10)

    def test_block_does_not_take_a_neighbour_grains_orientation(self, tmp_path):
        # grain 1 | grain 2 adjacent. A +0.6 shift puts grain 1's element centroid
        # nearer grain 2's voxel, so an unconstrained lookup copies grain 2.
        reference, mapped = self._run(
            tmp_path,
            [1, 2, 0, 0],
            [[30.0, 40.0, 50.0], [70.0, 10.0, 20.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        assert not np.allclose(mapped[0], mapped[1])
        assert np.allclose(mapped, reference, atol=1e-10)

    def test_aligned_mesh_is_unchanged(self, tmp_path):
        # With no offset every element centroid already sits on its own voxel, so
        # the grain-constrained lookup must reproduce the previous behaviour.
        reference, mapped = self._run(
            tmp_path,
            [1, 0, 2, 0],
            [[30.0, 40.0, 50.0], [0.0, 0.0, 0.0], [70.0, 10.0, 20.0], [0.0, 0.0, 0.0]],
            shift=0.0,
        )
        assert np.allclose(mapped, reference, atol=1e-10)

    def test_unassociable_blocks_raise(self, tmp_path):
        import scipy.io as sio

        from graintrace.nf.mesh import map_orientations, write_voxel_exodus

        grid = _line_grid(
            [1, 2, 0, 0],
            [
                [30.0, 40.0, 50.0],
                [70.0, 10.0, 20.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        mesh_file = str(tmp_path / "mesh.e")
        write_voxel_exodus(
            torch.tensor(grid),
            mesh_file,
            str(tmp_path / "ori_ref"),
            angle_type="degrees",
            symmetry="432",
        )
        # Renumber the blocks to ids that name no grain in the grid, and give the
        # grid a third grain so the positional fallback cannot apply either.
        with sio.netcdf_file(mesh_file, "a") as f:
            f.variables["eb_prop1"][:] = np.array([7, 8], dtype=np.int32)
        grid[3, 0, 0, 0] = 3
        grid[3, 0, 0, 1:4] = [10.0, 10.0, 10.0]

        with pytest.raises(ValueError, match="Cannot associate"):
            map_orientations(
                mesh_file,
                grid,
                str(tmp_path / "ori_mapped"),
                angle_type="degrees",
                symmetry="432",
            )
