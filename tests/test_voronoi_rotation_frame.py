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

"""The VoronoiMeshBuilder sample rotation must agree with FrameTransform.

These pin the defect behind issue #57: a sample-frame rotation ``Q`` rotates
orientations on the **sample** side, ``g' = g Q^T``. The builder used to select
the crystal-side update ``Q g`` whenever ``orientation_active_convention=True``,
and the ``auto_rotate`` branch stored the transpose of the rotation it had
applied to the coordinates. Neither failed loudly: for a cubic ``Q`` the
crystal-side update is invisible modulo crystal symmetry.

The operative check is a crystal direction, not an orientation: with
``R = g^T`` the crystal-to-sample map, a frame change must send ``R c`` to
``Q R c`` for every crystal direction ``c``.
"""

import numpy as np
import pytest

from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder
from graintrace.frame_transform import FrameTransform

ROTATE_ANGLES = (11.0, -23.0, 47.0)
EULER_COLS = ["Eul0", "Eul1", "Eul2"]
STRAIN_COLS = [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]


def _rotations(n, seed=7):
    """``n`` proper rotation matrices, used as orientation matrices ``g``."""
    # pylint: disable-next=import-outside-toplevel  # scipy is a heavy dep
    from scipy.spatial.transform import Rotation

    return Rotation.random(n, random_state=seed).as_matrix()


def _grain_table(n=6, seed=3):
    """A minimal FF grain table with coordinates, Euler angles and strains."""
    # pylint: disable-next=import-outside-toplevel  # pandas import kept local
    import pandas as pd

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(scale=40.0, size=(n, 3)), columns=["X", "Y", "Z"])
    frame["GrainRadius"] = rng.uniform(5.0, 20.0, size=n)
    frame[EULER_COLS] = rng.uniform(0.0, 180.0, size=(n, 3))
    frame[STRAIN_COLS] = rng.normal(scale=1e-3, size=(n, 9))
    return frame


def _builder(tmp_path, data, **kwargs):
    """A builder that never touches NEPER: ``env`` short-circuits the probe."""
    options = {
        "input_csv": str(tmp_path / "grains.csv"),
        "output_dir": str(tmp_path / "out"),
        "bounding_box": [-100.0, 100.0, -100.0, 100.0, -100.0, 100.0],
        "angle_identifier": EULER_COLS,
        "elastic_strain_identifier": STRAIN_COLS,
        "rotate_angles": ROTATE_ANGLES,
        "unit": "deg",
        "env": {},
    }
    options.update(kwargs)
    builder = VoronoiMeshBuilder(**options)
    builder.data = data
    return builder


def _write_tess(path, ori_mats):
    """A ``.tess`` stub holding only the ``*ori`` section the builder rewrites."""
    lines = [" **cell\n", "  *ori\n", "   rotmat\n"]
    lines += [
        "   " + " ".join(f"{v:.12f}" for v in m.flatten()) + "\n" for m in ori_mats
    ]
    lines += [" **vertex\n", "   0\n"]
    path.write_text("".join(lines), encoding="utf-8")


class TestRotateRecordsWhatItApplied:
    """``rotate`` must record the rotation the coordinates actually received."""

    def test_manual_rotation_matches_frame_transform(self, tmp_path):
        data = _grain_table()
        original = data[["X", "Y", "Z"]].to_numpy(dtype=float).copy()
        builder = _builder(tmp_path, data)
        builder.rotate()

        frame = builder.frame_transform
        assert frame is not None
        np.testing.assert_allclose(
            builder.data[["X", "Y", "Z"]].to_numpy(dtype=float),
            frame.positions(original),
            atol=1e-12,
        )
        np.testing.assert_allclose(builder.rotate_matrix, frame.Q, atol=1e-12)

    def test_auto_rotate_rotation_is_the_one_applied_to_positions(self, tmp_path):
        """The PCA branch applies ``E^T`` plus a shift; it used to store ``E``."""
        pytest.importorskip("sklearn")

        data = _grain_table(n=40, seed=11)
        original = data[["X", "Y", "Z"]].to_numpy(dtype=float).copy()
        builder = _builder(tmp_path, data, auto_rotate=True)
        builder.rotate()

        frame = builder.frame_transform
        assert frame is not None
        # The recorded transform reproduces the moved coordinates exactly,
        # translation included -- that is what "records what it applied" means.
        np.testing.assert_allclose(
            builder.data[["X", "Y", "Z"]].to_numpy(dtype=float),
            frame.positions(original),
            atol=1e-9,
        )
        np.testing.assert_allclose(builder.rotate_matrix, frame.Q, atol=1e-12)

    @pytest.mark.parametrize("seed", range(48))
    def test_auto_rotate_is_always_proper(self, tmp_path, seed):
        """Sign alignment must run before the handedness fix (seeds 13/23/30 failed)."""
        pytest.importorskip("sklearn")
        # pylint: disable-next=import-outside-toplevel  # pandas/scipy kept local
        import pandas as pd

        # pylint: disable-next=import-outside-toplevel
        from scipy.spatial.transform import Rotation

        rng = np.random.default_rng(seed)
        x = rng.normal(size=(300, 3)) * [50.0, 30.0, 10.0]
        x = x @ Rotation.random(random_state=seed).as_matrix().T
        data = pd.DataFrame(x, columns=["X", "Y", "Z"])
        builder = _builder(tmp_path, data, auto_rotate=True)
        builder.rotate()

        frame = builder.frame_transform
        assert np.isclose(np.linalg.det(frame.Q), 1.0)
        # Each principal axis keeps a non-negative component along its global axis
        # (up to the last axis, which may be flipped to stay right-handed).
        assert np.all(np.diag(frame.Q)[:2] >= 0.0)
        np.testing.assert_allclose(
            builder.data[["X", "Y", "Z"]].to_numpy(dtype=float),
            frame.positions(x),
            atol=1e-9,
        )


class TestOrientationsAreUpdatedOnTheSampleSide:
    """``g' = g Q^T``, pinned through a crystal direction."""

    def test_tess_orientations_follow_the_sample_side_rule(self, tmp_path):
        data = _grain_table()
        builder = _builder(tmp_path, data)
        builder.rotate()

        g0 = _rotations(len(data))
        tess = tmp_path / "reconstruction.tess"
        _write_tess(tess, g0)
        builder.apply_rotation_to_properties(tess_file=str(tess))

        g1 = builder.data[builder.ori_rotmat_id].to_numpy(dtype=float).reshape(-1, 3, 3)
        q = builder.rotate_matrix

        np.testing.assert_allclose(g1, g0 @ q.T, atol=1e-7)

        # The physical statement: every crystal direction lands where the frame
        # change sends it. R = g^T maps crystal to sample.
        crystal_dir = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        moved = np.einsum("nij,j->ni", np.transpose(g1, (0, 2, 1)), crystal_dir)
        expected = np.einsum(
            "ij,njk,k->ni", q, np.transpose(g0, (0, 2, 1)), crystal_dir
        )
        np.testing.assert_allclose(moved, expected, atol=1e-7)

    def test_crystal_side_update_is_not_what_is_applied(self, tmp_path):
        """Guards the regression: ``Q g`` must not reappear."""
        data = _grain_table()
        builder = _builder(tmp_path, data)
        builder.rotate()

        g0 = _rotations(len(data))
        tess = tmp_path / "reconstruction.tess"
        _write_tess(tess, g0)
        builder.apply_rotation_to_properties(tess_file=str(tess))

        g1 = builder.data[builder.ori_rotmat_id].to_numpy(dtype=float).reshape(-1, 3, 3)
        assert not np.allclose(g1, builder.rotate_matrix @ g0, atol=1e-6)

    def test_strains_are_rotated_as_second_order_tensors(self, tmp_path):
        data = _grain_table()
        strain0 = data[STRAIN_COLS].to_numpy(dtype=float).reshape(-1, 3, 3).copy()
        builder = _builder(tmp_path, data)
        builder.rotate()

        g0 = _rotations(len(data))
        tess = tmp_path / "reconstruction.tess"
        _write_tess(tess, g0)
        builder.apply_rotation_to_properties(tess_file=str(tess))

        q = builder.rotate_matrix
        strain1 = builder.data[STRAIN_COLS].to_numpy(dtype=float).reshape(-1, 3, 3)
        np.testing.assert_allclose(
            strain1, np.einsum("ij,njk,lk->nil", q, strain0, q), atol=1e-12
        )


class TestOrientationActiveConventionIsDeprecated:
    """The flag no longer selects a branch; it warns and is ignored."""

    def test_true_warns(self, tmp_path):
        with pytest.deprecated_call():
            _builder(tmp_path, _grain_table(), orientation_active_convention=True)

    def test_false_does_not_warn(self, tmp_path, recwarn):
        _builder(tmp_path, _grain_table(), orientation_active_convention=False)
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_both_settings_now_give_the_sample_side_result(self, tmp_path):
        g0 = _rotations(6)
        results = []
        for flag in (False, True):
            data = _grain_table()
            with pytest.warns(DeprecationWarning) if flag else _no_warning_ctx():
                builder = _builder(tmp_path, data, orientation_active_convention=flag)
            builder.rotate()
            tess = tmp_path / f"reconstruction_{flag}.tess"
            _write_tess(tess, g0)
            builder.apply_rotation_to_properties(tess_file=str(tess))
            results.append(builder.data[builder.ori_rotmat_id].to_numpy(dtype=float))
        np.testing.assert_allclose(results[0], results[1], atol=1e-12)


class TestOrientationsDatRewrite:
    """``orientations.dat`` must carry the same sample-side update."""

    def test_rewritten_euler_match_frame_transform(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("neml2")

        data = _grain_table()
        euler0 = data[EULER_COLS].to_numpy(dtype=float).copy()
        builder = _builder(tmp_path, data)
        builder.rotate()

        path = tmp_path / "orientations.dat"
        # pylint: disable-next=protected-access  # the rewrite is internal
        builder._rewrite_orientations_dat_rotated(str(path))

        expected = FrameTransform(rotation=builder.rotate_matrix).euler(
            euler0, "bunge", "degrees"
        )
        np.testing.assert_allclose(
            builder.data[EULER_COLS].to_numpy(dtype=float), expected, atol=1e-9
        )
        np.testing.assert_allclose(np.loadtxt(path), expected, atol=1e-6)


def _no_warning_ctx():
    """A no-op context manager, so the loop above can stay one shape."""
    # pylint: disable-next=import-outside-toplevel  # stdlib, used once
    from contextlib import nullcontext

    return nullcontext()
