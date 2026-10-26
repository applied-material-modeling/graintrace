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

"""Tests for :mod:`graintrace.frame_transform`."""

import importlib.util
import unittest

import numpy as np

from graintrace.frame_transform import FrameTransform

# torch is a graintrace runtime dependency, so CI's ``pip install ".[dev]"`` has
# it; neml2 comes from the PUMA build and is not installed there. A module-scope
# importorskip on either one skipped this whole file in CI, including the
# rotation parsing, tensor, algebra, axis-sign and serialisation tests, which are
# pure numpy. Gate only the tests that need the orientation stack, the way
# tests/test_voronoi_rotation_frame.py already does.
#
# The probe is find_spec, not a soft import: orientation_helper imports neml2
# lazily inside its functions, so ``from graintrace.orientation_helper import
# ...`` succeeds without neml2 present and the gate would let the tests run and
# fail at call time rather than skip. tests/test_stitching.py probes the same way.
HAS_ORIENTATION_STACK = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("neml2") is not None
)

if HAS_ORIENTATION_STACK:  # pragma: no cover - dependency availability
    import torch
    from graintrace.orientation_helper import (
        euler_to_matrix,
        matrix_to_euler,
        matrix_to_mrp,
        misorientation_matrix,
        mrp_to_matrix,
    )
else:  # pragma: no cover - the compiled stack is optional
    torch = None
    euler_to_matrix = matrix_to_euler = matrix_to_mrp = None
    misorientation_matrix = mrp_to_matrix = None

requires_orientation_stack = unittest.skipUnless(
    HAS_ORIENTATION_STACK, "requires torch and neml2"
)

# A generic frame change: stage rotation, remount flip, and a small two-axis tilt.
ROTATION_SEQUENCE = [("z", 172.3), ("x", 180.0), ("x", 1.4), ("y", -0.9)]

EULER = np.array(
    [
        [31.0, 47.0, 112.0],
        [10.0, 20.0, 30.0],
        [0.0, 0.0, 0.0],
        [355.0, 1.0, 95.0],
    ]
)


def _generic_transform():
    """A transform with a non-trivial rotation, axis sign and translation."""
    return FrameTransform(
        rotation=ROTATION_SEQUENCE,
        axis_sign=(1, 1, -1),
        translation=(12.0, -3.5, 40.0),
    )


class TestRotationParsing(unittest.TestCase):
    """``rotation`` accepts every documented form and refuses improper input."""

    def test_identity_default(self):
        """No rotation gives the identity in all three maps."""
        T = FrameTransform()
        self.assertTrue(np.allclose(T.Q, np.eye(3)))
        x = np.array([[1.0, 2.0, 3.0]])
        self.assertTrue(np.allclose(T.positions(x), x))

    def test_axis_angle_sequence_is_extrinsic_by_default(self):
        """An (axis, angle) list composes as fixed-axis (extrinsic) rotations."""
        # pylint: disable-next=import-outside-toplevel  # scipy only needed here
        from scipy.spatial.transform import Rotation

        T = FrameTransform(rotation=[("z", 30.0), ("x", 20.0)])
        expected = Rotation.from_euler("zx", [30.0, 20.0], degrees=True).as_matrix()
        self.assertTrue(np.allclose(T.Q, expected))

    def test_intrinsic_sequence_differs_from_extrinsic(self):
        """``sequence='intrinsic'`` is a different, also-valid composition."""
        a = FrameTransform(rotation=[("z", 30.0), ("x", 20.0)], sequence="extrinsic")
        b = FrameTransform(rotation=[("z", 30.0), ("x", 20.0)], sequence="intrinsic")
        self.assertFalse(np.allclose(a.Q, b.Q))

    def test_matrix_quaternion_and_rotvec_agree(self):
        """The matrix, quaternion and rotation-vector forms give the same Q."""
        T = FrameTransform(rotation=[("z", 37.0), ("y", 11.0)])
        Q = T.Q
        from_matrix = FrameTransform(rotation=Q)
        quat = matrix_to_quat_scalar_first(Q)
        from_quat = FrameTransform(rotation=quat)
        rotvec = rotation_vector(Q)
        from_vec = FrameTransform(rotation=rotvec, angle_unit="radians")
        self.assertTrue(np.allclose(from_matrix.Q, Q))
        self.assertTrue(np.allclose(from_quat.Q, Q))
        self.assertTrue(np.allclose(from_vec.Q, Q))

    def test_radians_angle_unit(self):
        """``angle_unit='radians'`` is honoured by the (axis, angle) form."""
        a = FrameTransform(rotation=[("z", 90.0)], angle_unit="degrees")
        b = FrameTransform(rotation=[("z", np.pi / 2)], angle_unit="radians")
        self.assertTrue(np.allclose(a.Q, b.Q))

    def test_improper_rotation_raises(self):
        """A reflection is refused: it would change crystal handedness."""
        mirror = np.diag([1.0, 1.0, -1.0])
        with self.assertRaises(ValueError) as ctx:
            FrameTransform(rotation=mirror)
        self.assertIn("det=+1", str(ctx.exception))

    def test_non_orthogonal_matrix_raises(self):
        """A scaled matrix is refused rather than silently orthogonalised."""
        with self.assertRaises(ValueError) as ctx:
            FrameTransform(rotation=np.diag([2.0, 0.5, 1.0]))
        self.assertIn("orthogonal", str(ctx.exception))

    def test_axis_sign_must_be_unit(self):
        """``axis_sign`` carries no scale, so only +-1 is accepted."""
        with self.assertRaises(ValueError):
            FrameTransform(axis_sign=(1, 1, 2))
        with self.assertRaises(ValueError):
            FrameTransform(axis_sign=(1, 1, 0))

    def test_bad_angle_unit_and_sequence_raise(self):
        """Unknown ``angle_unit``/``sequence`` strings are rejected."""
        with self.assertRaises(ValueError):
            FrameTransform(rotation=[("z", 10.0)], angle_unit="grad")
        with self.assertRaises(ValueError):
            FrameTransform(rotation=[("z", 10.0)], sequence="mixed")

    def test_unknown_axis_raises(self):
        """An axis name outside x/y/z is rejected."""
        with self.assertRaises(ValueError):
            FrameTransform(rotation=[("w", 10.0)])

    def test_non_unit_quaternion_raises(self):
        """A non-unit quaternion is refused, not silently renormalised."""
        with self.assertRaises(ValueError) as ctx:
            FrameTransform(rotation=[2.0, 0.0, 0.0, 0.0])
        self.assertIn("unit length", str(ctx.exception))
        with self.assertRaises(ValueError):
            FrameTransform(rotation=[0.0, 0.0, 0.0, 0.0])

    def test_near_unit_quaternion_is_accepted(self):
        """Round-off within 1e-6 of unit length is fine."""
        h = np.sqrt(0.5)
        T = FrameTransform(rotation=[h * (1 + 5e-7), 0.0, 0.0, h])
        self.assertTrue(
            np.allclose(T.positions([[1.0, 0.0, 0.0]]), [[0.0, 1.0, 0.0]], atol=1e-5)
        )


@requires_orientation_stack
class TestSampleSideCorrectness(unittest.TestCase):
    """The orientation update must be on the sample side: ``R' c = Q R c``."""

    def test_sample_side_rule_degrees(self):
        """For a crystal direction c, the transformed orientation satisfies R' c = Q R c."""
        T = FrameTransform(rotation=ROTATION_SEQUENCE)
        g = euler_to_matrix(torch.tensor(EULER), "bunge", "degrees")
        gp = T.orientation_matrices(g)

        R = g.transpose(-2, -1).numpy()  # active: crystal -> sample
        Rp = gp.transpose(-2, -1).numpy()
        for c in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([1, 1, 1.0])):
            expected = np.einsum("ij,nj->ni", T.Q, R @ c)
            self.assertTrue(np.allclose(Rp @ c, expected))

    def test_sample_side_rule_radians(self):
        """The same rule holds through the radians Euler path."""
        T = FrameTransform(rotation=ROTATION_SEQUENCE)
        eul_rad = np.deg2rad(EULER)
        out = T.euler(eul_rad, "bunge", "radians")
        direct = T.euler(EULER, "bunge", "degrees")
        self.assertTrue(np.allclose(np.rad2deg(out) % 360.0, direct % 360.0))

    def test_crystal_side_update_is_wrong(self):
        """``Q g`` does not reproduce a physical sample rotation (the bug guard)."""
        T = FrameTransform(rotation=ROTATION_SEQUENCE)
        g = euler_to_matrix(torch.tensor(EULER), "bunge", "degrees").numpy()
        wrong = np.einsum("ij,njk->nik", T.Q, g)  # crystal side
        right = T.orientation_matrices(torch.tensor(g)).numpy()
        self.assertFalse(np.allclose(wrong, right))

    def test_cubic_rotation_moves_orientations_in_the_fundamental_zone(self):
        """A 90 degree axis permutation is a real sample rotation, not a no-op.

        Under cubic symmetry the *crystal*-side update by a cubic Q is invisible,
        which is exactly how a wrong-side implementation hides. The sample-side
        update must still move the orientations relative to a fixed reference.
        """
        T = FrameTransform(rotation=[("z", 90.0)])
        g = euler_to_matrix(torch.tensor(EULER), "bunge", "degrees")
        gp = T.orientation_matrices(g)

        ref = euler_to_matrix(torch.tensor(np.zeros_like(EULER)), "bunge", "degrees")
        before = misorientation_matrix(g, ref, symmetry="432")
        after = misorientation_matrix(gp, ref, symmetry="432")
        # Not every orientation has to move, but the set as a whole must.
        self.assertFalse(np.allclose(before.numpy(), after.numpy()))


@requires_orientation_stack
class TestPhysicalConsistency(unittest.TestCase):
    """Positions and orientations must move together."""

    def test_offset_stays_along_the_same_crystal_axis(self):
        """Two grains offset along grain 1's [100] are still offset along it after.

        No ``axis_sign`` here on purpose: a sign entry corrects a *position index*
        convention that the orientations never had, so positions and orientations
        only correspond physically when ``s == (1, 1, 1)``.
        """
        T = FrameTransform(rotation=ROTATION_SEQUENCE, translation=(12.0, -3.5, 40.0))
        e1 = np.array([[31.0, 47.0, 112.0]])
        g1 = euler_to_matrix(torch.tensor(e1), "bunge", "degrees").numpy()[0]
        R1 = g1.T  # crystal -> sample
        direction = R1 @ np.array([1.0, 0.0, 0.0])

        p1 = np.array([10.0, -4.0, 25.0])
        p2 = p1 + 73.0 * direction

        pos = T.positions(np.vstack([p1, p2]))
        offset = pos[1] - pos[0]

        g1p = T.orientation_matrices(torch.tensor(g1)).numpy()
        direction_p = g1p.T @ np.array([1.0, 0.0, 0.0])

        self.assertTrue(
            np.allclose(offset / np.linalg.norm(offset), direction_p, atol=1e-10)
        )

    def test_pairwise_distances_are_invariant(self):
        """A frame change is rigid: pairwise distances do not move."""
        T = _generic_transform()
        rng = np.random.default_rng(42)
        x = rng.normal(0.0, 100.0, size=(20, 3))
        xp = T.positions(x)
        d0 = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        d1 = np.linalg.norm(xp[:, None, :] - xp[None, :, :], axis=-1)
        self.assertTrue(np.allclose(d0, d1))

    def test_pairwise_misorientation_is_invariant(self):
        """A common sample rotation leaves relative orientations untouched."""
        T = _generic_transform()
        g = euler_to_matrix(torch.tensor(EULER), "bunge", "degrees")
        gp = T.orientation_matrices(g)
        before = misorientation_matrix(g[:-1], g[1:], symmetry="432")
        after = misorientation_matrix(gp[:-1], gp[1:], symmetry="432")
        self.assertTrue(np.allclose(before.numpy(), after.numpy()))


class TestTensors(unittest.TestCase):
    """``eps' = Q eps Q^T``, componentwise and in its invariants."""

    @staticmethod
    def _sample_tensors():
        rng = np.random.default_rng(7)
        a = rng.normal(size=(5, 3, 3))
        return a + a.transpose(0, 2, 1)

    def test_componentwise(self):
        """The adapter reproduces the explicit Q eps Q^T product."""
        T = _generic_transform()
        eps = self._sample_tensors()
        got = T.tensors(eps)
        want = np.array([T.Q @ e @ T.Q.T for e in eps])
        self.assertTrue(np.allclose(got, want))

    def test_invariants_preserved(self):
        """Trace and eigenvalues survive the rotation."""
        T = _generic_transform()
        eps = self._sample_tensors()
        got = T.tensors(eps)
        self.assertTrue(
            np.allclose(
                np.trace(eps, axis1=1, axis2=2), np.trace(got, axis1=1, axis2=2)
            )
        )
        self.assertTrue(
            np.allclose(
                np.sort(np.linalg.eigvalsh(eps), axis=-1),
                np.sort(np.linalg.eigvalsh(got), axis=-1),
            )
        )

    def test_flat_nine_column_layout(self):
        """(N, 9) row-major input returns (N, 9) and matches the (N, 3, 3) path."""
        T = _generic_transform()
        eps = self._sample_tensors()
        flat = T.tensors(eps.reshape(-1, 9))
        self.assertEqual(flat.shape, (5, 9))
        self.assertTrue(np.allclose(flat.reshape(-1, 3, 3), T.tensors(eps)))

    def test_axis_sign_does_not_touch_tensors(self):
        """``s`` is a position convention only."""
        eps = self._sample_tensors()
        a = FrameTransform(rotation=ROTATION_SEQUENCE, axis_sign=(1, 1, 1))
        b = FrameTransform(rotation=ROTATION_SEQUENCE, axis_sign=(1, 1, -1))
        self.assertTrue(np.allclose(a.tensors(eps), b.tensors(eps)))

    def test_bad_shape_raises(self):
        """A tensor array that is neither (..., 3, 3) nor (..., 9) is refused."""
        with self.assertRaises(ValueError):
            FrameTransform().tensors(np.zeros((4, 6)))


class TestAxisSign(unittest.TestCase):
    """``axis_sign`` acts on positions only."""

    def test_sign_flips_positions(self):
        """A z reversal negates the z coordinate."""
        T = FrameTransform(axis_sign=(1, 1, -1))
        self.assertTrue(np.allclose(T.positions([[1.0, 2.0, 3.0]]), [[1.0, 2.0, -3.0]]))

    @requires_orientation_stack
    def test_sign_leaves_orientations_alone(self):
        """A z reversal is a position-only map: orientations come back unchanged."""
        T = FrameTransform(axis_sign=(1, 1, -1))
        g = euler_to_matrix(torch.tensor(EULER), "bunge", "degrees")
        self.assertTrue(np.allclose(T.orientation_matrices(g).numpy(), g.numpy()))

    def test_axis_sign_property_recovers_s(self):
        """``axis_sign`` reads back the constructor's sign vector."""
        T = FrameTransform(rotation=ROTATION_SEQUENCE, axis_sign=(-1, 1, -1))
        self.assertTrue(np.allclose(T.axis_sign, [-1, 1, -1]))

    def test_axis_sign_property_is_none_when_not_recoverable(self):
        """After composing with a general rotation the sign is no longer separable.

        The rotation has to be one the sign matrix does not commute with: a z
        reversal does commute with a z rotation, and that composition keeps a
        clean sign vector.
        """
        T = FrameTransform(axis_sign=(1, 1, -1)) @ FrameTransform(
            rotation=[("x", 37.0)]
        )
        self.assertIsNone(T.axis_sign)


class TestAlgebra(unittest.TestCase):
    """``inverse`` and ``@`` behave as the documented group operations."""

    def test_inverse_round_trip_with_sign_and_translation(self):
        """``T.inverse() @ T`` is the identity on positions, orientations and tensors."""
        T = _generic_transform()
        ident = T.inverse() @ T

        rng = np.random.default_rng(3)
        x = rng.normal(0.0, 50.0, size=(8, 3))
        self.assertTrue(np.allclose(ident.positions(x), x))
        self.assertTrue(np.allclose(ident.Q, np.eye(3)))
        self.assertTrue(np.allclose(ident.A, np.eye(3)))
        self.assertTrue(np.allclose(ident.t, np.zeros(3)))

    def test_inverse_undoes_positions_and_tensors(self):
        """Applying T then its inverse recovers positions and tensors."""
        T = _generic_transform()
        Tin = T.inverse()
        rng = np.random.default_rng(11)
        x = rng.normal(0.0, 50.0, size=(6, 3))
        self.assertTrue(np.allclose(Tin.positions(T.positions(x)), x))

        eps = np.array([np.diag([1.0, 2.0, 3.0])])
        self.assertTrue(np.allclose(Tin.tensors(T.tensors(eps)), eps))

    @requires_orientation_stack
    def test_inverse_undoes_the_euler_map(self):
        """Applying T then its inverse recovers the original orientations."""
        T = _generic_transform()
        Tin = T.inverse()
        eul = T.euler(EULER, "bunge", "degrees")
        back = Tin.euler(eul, "bunge", "degrees")
        g0 = euler_to_matrix(torch.tensor(EULER), "bunge", "degrees")
        g1 = euler_to_matrix(torch.tensor(back), "bunge", "degrees")
        self.assertTrue(np.allclose(g0.numpy(), g1.numpy(), atol=1e-10))

    def test_composition_is_associative(self):
        """``(A @ B) @ C == A @ (B @ C)``."""
        A = FrameTransform(rotation=[("z", 31.0)], translation=(1.0, 2.0, 3.0))
        B = FrameTransform(rotation=[("x", 17.0)], axis_sign=(1, -1, 1))
        C = FrameTransform(rotation=[("y", -5.0)], translation=(0.0, -4.0, 2.0))
        left = (A @ B) @ C
        right = A @ (B @ C)
        self.assertTrue(np.allclose(left.A, right.A))
        self.assertTrue(np.allclose(left.Q, right.Q))
        self.assertTrue(np.allclose(left.t, right.t))

    @staticmethod
    def _composition_pair():
        """A composed transform and its two factors."""
        A = FrameTransform(rotation=[("z", 31.0)], translation=(1.0, 2.0, 3.0))
        B = FrameTransform(rotation=[("x", 17.0)], axis_sign=(1, -1, 1))
        return A, B, A @ B

    def test_composition_matches_sequential_positions(self):
        """``(A @ B)(x) == A(B(x))`` for positions."""
        A, B, AB = self._composition_pair()
        rng = np.random.default_rng(5)
        x = rng.normal(size=(4, 3))
        self.assertTrue(np.allclose(AB.positions(x), A.positions(B.positions(x))))

    @requires_orientation_stack
    def test_composition_matches_sequential_orientations(self):
        """``(A @ B)(g) == A(B(g))`` for orientations."""
        A, B, AB = self._composition_pair()
        g = euler_to_matrix(torch.tensor(EULER), "bunge", "degrees")
        self.assertTrue(
            np.allclose(
                AB.orientation_matrices(g).numpy(),
                A.orientation_matrices(B.orientation_matrices(g)).numpy(),
            )
        )

    def test_matmul_with_non_transform_is_not_implemented(self):
        """Composing with an unrelated object defers rather than guessing."""
        with self.assertRaises(TypeError):
            _ = FrameTransform() @ 3


@requires_orientation_stack
class TestMRPPath(unittest.TestCase):
    """The neml2 MRP path, including the load-bearing ``compose`` argument order."""

    def test_mrp_matches_the_matrix_path(self):
        """``mrp`` and ``orientation_matrices`` agree, which pins compose's order.

        ``compose(a, b)`` yields the orientation whose matrix is
        ``matrix(a) @ matrix(b)``; the sample-side update is therefore
        ``compose(orientation, Q^T)``. The swapped order is the crystal-side
        update and fails this test.
        """
        T = FrameTransform(rotation=ROTATION_SEQUENCE)
        g = euler_to_matrix(torch.tensor(EULER), "bunge", "degrees")
        p = matrix_to_mrp(g)

        via_mrp = mrp_to_matrix(T.mrp(p)).numpy()
        via_matrix = T.orientation_matrices(g).numpy()
        self.assertTrue(np.allclose(via_mrp, via_matrix, atol=1e-10))

    def test_mrp_satisfies_the_physical_rule(self):
        """The MRP path obeys ``R' c = Q R c`` too."""
        T = FrameTransform(rotation=ROTATION_SEQUENCE)
        g = euler_to_matrix(torch.tensor(EULER), "bunge", "degrees")
        gp = mrp_to_matrix(T.mrp(matrix_to_mrp(g))).numpy()
        c = np.array([1.0, 0.0, 0.0])
        expected = np.einsum("ij,nj->ni", T.Q, g.transpose(-2, -1).numpy() @ c)
        self.assertTrue(np.allclose(gp.transpose(0, 2, 1) @ c, expected, atol=1e-10))

    def test_mrp_output_is_the_short_rotation_set(self):
        """Random inputs come back with ``|p'| <= 1`` and still equal ``Q R``."""
        # pylint: disable-next=import-outside-toplevel  # scipy only needed here
        from scipy.spatial.transform import Rotation

        g = torch.tensor(
            np.transpose(Rotation.random(500, random_state=4).as_matrix(), (0, 2, 1))
        )
        p = matrix_to_mrp(g)
        # Feed the long (shadow) set for half the inputs: it is the same rotation.
        p[::2] = -p[::2] / (p[::2] ** 2).sum(dim=-1, keepdim=True)
        for seed in range(5):
            Q = Rotation.random(random_state=100 + seed).as_matrix()
            T = FrameTransform(rotation=Q)
            out = T.mrp(p)
            self.assertTrue(torch.all(torch.isfinite(out)))
            self.assertLessEqual(float(out.norm(dim=-1).max()), 1.0 + 1e-12)
            want = T.orientation_matrices(g).numpy()
            self.assertTrue(np.allclose(mrp_to_matrix(out).numpy(), want, atol=1e-9))

    def test_mrp_near_360_degree_composition_stays_finite(self):
        """Rz(a) composed with a frame Rz(a), a -> 180, used to blow up or NaN."""
        # pylint: disable-next=import-outside-toplevel  # scipy only needed here
        from scipy.spatial.transform import Rotation

        for a in (179.0, 179.999, 179.9999999, 180.0):
            R = Rotation.from_euler("z", a, degrees=True).as_matrix()
            g = torch.tensor(R.T[None])
            T = FrameTransform(rotation=[("z", a)])
            out = T.mrp(matrix_to_mrp(g))
            self.assertTrue(torch.all(torch.isfinite(out)), a)
            self.assertLessEqual(float(out.norm(dim=-1).max()), 1.0 + 1e-12)
            want = T.orientation_matrices(g).numpy()
            self.assertTrue(np.allclose(mrp_to_matrix(out).numpy(), want, atol=1e-7), a)

    def test_identity_frame_leaves_mrp_unchanged(self):
        """Q = I is a no-op on short-set inputs (the zero MRP has no shadow)."""
        p = torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]], dtype=torch.float64)
        self.assertTrue(torch.allclose(FrameTransform().mrp(p), p, atol=1e-14))


@requires_orientation_stack
class TestEulerRoundTrips(unittest.TestCase):
    """Notation -> orientation -> transform -> notation, including gimbal lock."""

    def test_identity_transform_round_trips_euler(self):
        """With Q = I the Euler angles come back unchanged (modulo 360)."""
        T = FrameTransform()
        out = T.euler(EULER, "bunge", "degrees")
        g0 = euler_to_matrix(torch.tensor(EULER), "bunge", "degrees").numpy()
        g1 = euler_to_matrix(torch.tensor(out), "bunge", "degrees").numpy()
        self.assertTrue(np.allclose(g0, g1, atol=1e-10))

    def test_gimbal_lock_cases(self):
        """Phi = 0 and Phi = 180 survive the transform round trip."""
        T = FrameTransform(rotation=ROTATION_SEQUENCE)
        locked = np.array([[25.0, 0.0, 40.0], [25.0, 180.0, 40.0], [0.0, 0.0, 0.0]])
        out = T.euler(locked, "bunge", "degrees")
        self.assertTrue(np.all(np.isfinite(out)))
        want = T.orientation_matrices(
            euler_to_matrix(torch.tensor(locked), "bunge", "degrees")
        ).numpy()
        got = euler_to_matrix(torch.tensor(out), "bunge", "degrees").numpy()
        self.assertTrue(np.allclose(got, want, atol=1e-9))

    def test_other_conventions(self):
        """Kocks and Roe go through the same orientation, not angle arithmetic."""
        T = FrameTransform(rotation=ROTATION_SEQUENCE)
        for convention in ("bunge", "kocks", "roe"):
            out = T.euler(EULER, convention, "degrees")
            want = T.orientation_matrices(
                euler_to_matrix(torch.tensor(EULER), convention, "degrees")
            ).numpy()
            got = euler_to_matrix(torch.tensor(out), convention, "degrees").numpy()
            self.assertTrue(np.allclose(got, want, atol=1e-9), convention)


class TestSerialisation(unittest.TestCase):
    """``to_json`` / ``from_json`` round-trip the full transform."""

    def test_json_round_trip(
        self,
    ):
        """A transform read back from JSON maps identically."""
        # pylint: disable-next=import-outside-toplevel  # test-only helper
        import tempfile

        T = _generic_transform()
        with tempfile.TemporaryDirectory() as tmp:
            path = T.to_json(f"{tmp}/t.json")
            back = FrameTransform.from_json(path)
        self.assertTrue(np.allclose(back.A, T.A))
        self.assertTrue(np.allclose(back.Q, T.Q))
        self.assertTrue(np.allclose(back.t, T.t))
        self.assertTrue(np.allclose(back.axis_sign, T.axis_sign))

    def test_from_dict_rejects_a_foreign_record(self):
        """A record without the graintrace schema key is refused."""
        with self.assertRaises(ValueError):
            FrameTransform.from_dict({"rotation": np.eye(3).tolist()})

    def test_composed_transform_round_trips(self):
        """A composed transform has no sign vector and must still survive JSON."""
        T = _generic_transform() @ FrameTransform(rotation=[("x", 37.0)])
        self.assertIsNone(T.axis_sign)
        back = FrameTransform.from_dict(T.to_dict())
        self.assertTrue(np.allclose(back.A, T.A))
        self.assertTrue(np.allclose(back.Q, T.Q))
        self.assertTrue(np.allclose(back.t, T.t))

    def test_from_dict_rejects_a_non_orthogonal_position_matrix(self):
        """A scaled ``position_matrix`` would make positions non-rigid."""
        record = _generic_transform().to_dict()
        record["axis_sign"] = None  # isolate the orthogonality check
        record["position_matrix"] = (
            np.asarray(record["position_matrix"]) * 2.0
        ).tolist()
        with self.assertRaises(ValueError) as ctx:
            FrameTransform.from_dict(record)
        self.assertIn("orthogonal", str(ctx.exception))

    def test_from_dict_rejects_a_position_matrix_that_contradicts_axis_sign(self):
        """``A`` must be ``Q diag(s)`` whenever the record declares an ``s``."""
        record = _generic_transform().to_dict()
        self.assertIsNotNone(record["axis_sign"])
        # Orthogonal, but a different rotation than Q diag(s).
        record["position_matrix"] = FrameTransform(rotation=[("y", 23.0)]).Q.tolist()
        with self.assertRaises(ValueError) as ctx:
            FrameTransform.from_dict(record)
        self.assertIn("axis_sign", str(ctx.exception))

    def test_from_dict_rejects_a_misshaped_position_matrix(self):
        """A record whose ``position_matrix`` is not (3, 3) is refused."""
        record = _generic_transform().to_dict()
        record["position_matrix"] = [[1.0, 0.0], [0.0, 1.0]]
        with self.assertRaises(ValueError):
            FrameTransform.from_dict(record)

    def test_repr_is_informative(self):
        """``repr`` names the determinant, sign and translation."""
        text = repr(_generic_transform())
        self.assertIn("FrameTransform", text)
        self.assertIn("axis_sign", text)


class TestDatasetAdapters(unittest.TestCase):
    """The FF / point-cloud / voxel adapters, and their transform records."""

    @staticmethod
    def _ff_frame():
        # pylint: disable-next=import-outside-toplevel  # pandas at call time
        import pandas as pd

        rng = np.random.default_rng(1)
        n = 6
        eps = rng.normal(0.0, 1e-3, size=(n, 3, 3))
        eps = eps + eps.transpose(0, 2, 1)
        data = {
            "X": rng.normal(0.0, 100.0, n),
            "Y": rng.normal(0.0, 100.0, n),
            "Z": rng.normal(0.0, 100.0, n),
            "GrainRadius": rng.uniform(10.0, 50.0, n),
            "Confidence": rng.uniform(0.7, 1.0, n),
            "Eul0": rng.uniform(0.0, 360.0, n),
            "Eul1": rng.uniform(0.0, 180.0, n),
            "Eul2": rng.uniform(0.0, 360.0, n),
        }
        for k, (i, j) in enumerate([(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]):
            data[f"eKen{i}{j}"] = eps.reshape(n, 9)[:, k]
        return pd.DataFrame(data)

    @requires_orientation_stack
    def test_apply_ff_csv_round_trip(self):
        """``apply_ff_csv`` then the inverse recovers the input table."""
        # pylint: disable-next=import-outside-toplevel  # test-only helpers
        import tempfile

        # pylint: disable-next=import-outside-toplevel  # pandas at call time
        import pandas as pd

        df = self._ff_frame()
        T = _generic_transform()
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/src.csv"
            df.to_csv(src, index=False)
            fwd = T.apply_ff_csv(src, f"{tmp}/fwd.csv", angle_type="degrees")
            back = T.inverse().apply_ff_csv(
                fwd, f"{tmp}/back.csv", angle_type="degrees"
            )
            out = pd.read_csv(back)

            self.assertTrue(np.allclose(out[["X", "Y", "Z"]], df[["X", "Y", "Z"]]))
            self.assertTrue(np.allclose(out["GrainRadius"], df["GrainRadius"]))
            self.assertTrue(np.allclose(out["Confidence"], df["Confidence"]))
            eken = [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
            self.assertTrue(np.allclose(out[eken], df[eken], atol=1e-12))

            # Euler angles come back as the same orientation (not necessarily the
            # same branch of the angle triple).
            g0 = euler_to_matrix(
                torch.tensor(df[["Eul0", "Eul1", "Eul2"]].to_numpy()),
                "bunge",
                "degrees",
            ).numpy()
            g1 = euler_to_matrix(
                torch.tensor(out[["Eul0", "Eul1", "Eul2"]].to_numpy()),
                "bunge",
                "degrees",
            ).numpy()
            self.assertTrue(np.allclose(g0, g1, atol=1e-9))

    @requires_orientation_stack
    def test_apply_ff_csv_writes_a_transform_record(self):
        """Every adapter writes ``<out>.transform.json`` beside its output."""
        # pylint: disable-next=import-outside-toplevel  # test-only helper
        import tempfile

        df = self._ff_frame()
        T = _generic_transform()
        with tempfile.TemporaryDirectory() as tmp:
            dst = T.apply_ff_csv(df, f"{tmp}/out.csv", angle_type="degrees")
            record = FrameTransform.from_json(f"{dst}.transform.json")
        self.assertTrue(np.allclose(record.Q, T.Q))
        self.assertTrue(np.allclose(record.t, T.t))

    @requires_orientation_stack
    def test_apply_ff_csv_transforms_orientation_matrix_columns(self):
        """``O11``..``O33`` are orientation matrices, so they take ``g Q^T``."""
        # pylint: disable-next=import-outside-toplevel  # test-only helper
        import tempfile

        # pylint: disable-next=import-outside-toplevel  # pandas at call time
        import pandas as pd

        df = self._ff_frame()
        g = euler_to_matrix(
            torch.tensor(df[["Eul0", "Eul1", "Eul2"]].to_numpy()), "bunge", "degrees"
        ).numpy()
        ocols = [f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
        df[ocols] = g.reshape(-1, 9)

        T = _generic_transform()
        with tempfile.TemporaryDirectory() as tmp:
            dst = T.apply_ff_csv(df, f"{tmp}/out.csv", angle_type="degrees")
            out = pd.read_csv(dst)
        want = T.orientation_matrices(torch.tensor(g)).numpy().reshape(-1, 9)
        self.assertTrue(np.allclose(out[ocols].to_numpy(), want))

    @requires_orientation_stack
    def test_apply_ff_csv_auto_angle_type(self):
        """``angle_type='auto'`` detects degrees from values beyond 2*pi."""
        # pylint: disable-next=import-outside-toplevel  # test-only helper
        import tempfile

        df = self._ff_frame()
        T = FrameTransform(rotation=[("z", 10.0)])
        with tempfile.TemporaryDirectory() as tmp:
            auto = T.apply_ff_csv(df, f"{tmp}/a.csv", angle_type="auto")
            pinned = T.apply_ff_csv(df, f"{tmp}/b.csv", angle_type="degrees")
            # pylint: disable-next=import-outside-toplevel  # pandas at call time
            import pandas as pd

            self.assertTrue(
                np.allclose(
                    pd.read_csv(auto)[["Eul0", "Eul1", "Eul2"]],
                    pd.read_csv(pinned)[["Eul0", "Eul1", "Eul2"]],
                )
            )

    @requires_orientation_stack
    def test_apply_pointcloud(self):
        """The NF / EBSD point-cloud adapter moves coordinates and orientations."""
        # pylint: disable-next=import-outside-toplevel  # test-only helpers
        import tempfile

        # pylint: disable-next=import-outside-toplevel  # pandas at call time
        import pandas as pd

        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=10),
                "y": rng.normal(size=10),
                "z": rng.normal(size=10),
                "Eul1": rng.uniform(0.0, 2 * np.pi, 10),
                "Eul2": rng.uniform(0.0, np.pi, 10),
                "Eul3": rng.uniform(0.0, 2 * np.pi, 10),
            }
        )
        T = _generic_transform()
        with tempfile.TemporaryDirectory() as tmp:
            dst = T.apply_pointcloud(df, f"{tmp}/pc.csv", angle_type="radians")
            out = pd.read_csv(dst)
        self.assertTrue(
            np.allclose(
                out[["x", "y", "z"]], T.positions(df[["x", "y", "z"]].to_numpy())
            )
        )
        self.assertTrue(np.all(np.isfinite(out[["Eul1", "Eul2", "Eul3"]].to_numpy())))

    @requires_orientation_stack
    def test_apply_voxel_grid(self):
        """The voxel-grid adapter moves the coordinate and Euler slots in place."""
        # pylint: disable-next=import-outside-toplevel  # test-only helper
        import tempfile

        rng = np.random.default_rng(4)
        grid = np.zeros((3, 4, 5, 7))
        grid[..., 0] = rng.integers(0, 3, size=(3, 4, 5))
        grid[..., 1:4] = rng.uniform(0.0, 1.0, size=(3, 4, 5, 3))
        grid[..., 4:7] = rng.normal(0.0, 10.0, size=(3, 4, 5, 3))

        T = _generic_transform()
        with tempfile.TemporaryDirectory() as tmp:
            np.save(f"{tmp}/in.npy", grid)
            out_path = T.apply_voxel_grid(
                f"{tmp}/in.npy", f"{tmp}/out.npy", angle_type="radians"
            )
            out = np.load(out_path)

        self.assertEqual(out.shape, grid.shape)
        self.assertTrue(np.allclose(out[..., 0], grid[..., 0]))  # labels untouched
        self.assertTrue(
            np.allclose(
                out[..., 4:7].reshape(-1, 3),
                T.positions(grid[..., 4:7].reshape(-1, 3)),
            )
        )

    def test_apply_voxel_grid_rejects_a_short_last_axis(self):
        """A grid without the seven expected slots is refused."""
        # pylint: disable-next=import-outside-toplevel  # test-only helper
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            np.save(f"{tmp}/in.npy", np.zeros((2, 2, 2, 4)))
            with self.assertRaises(ValueError):
                FrameTransform().apply_voxel_grid(f"{tmp}/in.npy", f"{tmp}/o.npy")


def matrix_to_quat_scalar_first(Q):
    """Scalar-first ``(w, x, y, z)`` quaternion of a (3, 3) rotation matrix."""
    # pylint: disable-next=import-outside-toplevel  # scipy only needed here
    from scipy.spatial.transform import Rotation

    return np.roll(Rotation.from_matrix(Q).as_quat(), 1)


def rotation_vector(Q):
    """Rotation vector (axis * angle, radians) of a (3, 3) rotation matrix."""
    # pylint: disable-next=import-outside-toplevel  # scipy only needed here
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(Q).as_rotvec()


class TestLazyExport(unittest.TestCase):
    """``FrameTransform`` is reachable from the package root."""

    def test_top_level_import(self):
        """``graintrace.FrameTransform`` resolves through the lazy registry."""
        # pylint: disable-next=import-outside-toplevel  # exercising the lazy path
        import graintrace

        self.assertIs(graintrace.FrameTransform, FrameTransform)


if __name__ == "__main__":
    unittest.main()
