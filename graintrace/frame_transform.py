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

"""Technique-agnostic specimen-frame change for FF / NF / EBSD datasets.

:class:`FrameTransform` takes **one** dataset and a **user-stated** transform and
rewrites positions, crystal orientations and sample-frame tensors consistently
into a target specimen frame (for example Z = loading axis).

The transform is

.. math::

    x' = Q\\,(s \\odot x) + t, \\qquad
    g' = g\\,Q^{T}, \\qquad
    \\varepsilon' = Q\\,\\varepsilon\\,Q^{T}

where ``Q`` is a proper rotation (main stage rotation, remount flip and specimen
tilt composed into one), ``s`` is a per-axis sign convention that acts on
**positions only**, and ``t`` is the origin offset. ``g`` is the orientation
matrix in :mod:`graintrace.orientation_helper`'s convention, i.e.
``g = euler_to_matrix(...) = R_active^T`` where ``R_active`` maps crystal to
sample.

Scale is deliberately **not** part of a frame change: a per-axis scale difference
between two datasets is either a wrong reconstruction input (layer spacing, pixel
size) or real deformation, and neither is a coordinate-frame property.

graintrace does not estimate transforms. If the frames are known, each dataset is
converted independently and the datasets then agree by construction; registration
(estimating an unknown transform) stays a one-off diagnostic outside the library.

.. important::

   The orientation update is on the **sample side**. For a cubic ``Q`` (say a 90
   degree axis permutation) the crystal-side update ``Q g`` is invisible modulo
   crystal symmetry, so a wrong-side implementation can pass orientation-only
   checks while doing nothing physical. The pinning check is
   ``R' c = Q R c`` for a crystal direction ``c``; see
   ``tests/test_frame_transform.py``.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np

# Entries of ``axis_sign`` must be exactly +1 or -1: a frame change carries no scale.
_ALLOWED_SIGNS = (-1.0, 1.0)

# Tolerance on det(Q) == +1 and on the orthogonality of a user-supplied matrix.
_ROTATION_ATOL = 1e-8

_AXIS_NAMES = {"x": "x", "y": "y", "z": "z", "X": "x", "Y": "y", "Z": "z"}


def _is_axis_angle_sequence(rotation) -> bool:
    """True when ``rotation`` is the ordered ``(axis, angle)`` list form.

    The list form holds axis *names*, so it must be recognised before anything
    tries to coerce ``rotation`` to a float array.

    Args:
        rotation: any accepted ``rotation`` value.

    Returns:
        bool: whether the value is a non-empty sequence of 2-element pairs.
    """
    if not isinstance(rotation, (list, tuple)) or len(rotation) == 0:
        return False
    first = rotation[0]
    return isinstance(first, (list, tuple)) and len(first) == 2


def _as_scipy_rotation(rotation, angle_unit: str, sequence: str):
    """Build a ``scipy`` ``Rotation`` from any of the accepted ``rotation`` forms.

    Args:
        rotation: ``None`` (identity), a (3, 3) matrix, a scalar-first quaternion
            ``(w, x, y, z)``, a rotation vector ``(rx, ry, rz)`` whose magnitude is
            the angle in ``angle_unit``, or an ordered sequence of
            ``(axis, angle)`` pairs such as
            ``[("z", 172.3), ("x", 180.0), ("y", -0.9)]``.
        angle_unit: ``"degrees"`` or ``"radians"``; applies to the rotation-vector
            and ``(axis, angle)`` forms.
        sequence: ``"extrinsic"`` (fixed axes, the default) or ``"intrinsic"``
            (axes carried along by the preceding rotations). Only read by the
            ``(axis, angle)`` form.

    Returns:
        scipy.spatial.transform.Rotation: the composed rotation.

    Raises:
        ValueError: if ``angle_unit``, ``sequence``, an axis name, or the overall
            shape of ``rotation`` is not recognised.
    """
    # pylint: disable-next=import-outside-toplevel  # scipy is imported lazily
    from scipy.spatial.transform import Rotation

    if angle_unit not in ("degrees", "radians"):
        raise ValueError(
            f"angle_unit must be 'degrees' or 'radians', got {angle_unit!r}"
        )
    if sequence not in ("extrinsic", "intrinsic"):
        raise ValueError(
            f"sequence must be 'extrinsic' or 'intrinsic', got {sequence!r}"
        )
    degrees = angle_unit == "degrees"

    if rotation is None:
        return Rotation.identity()

    # Ordered list of (axis, angle) pairs -- the documented human-readable form.
    if _is_axis_angle_sequence(rotation):
        axes, angles = [], []
        for item in rotation:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(
                    "a rotation sequence must be a list of (axis, angle) pairs; "
                    f"got the entry {item!r}"
                )
            axis, angle = item
            if axis not in _AXIS_NAMES:
                raise ValueError(f"rotation axis must be one of x/y/z, got {axis!r}")
            axes.append(_AXIS_NAMES[axis])
            angles.append(float(angle))
        # Composed one axis at a time rather than through a single from_euler
        # call: a frame change routinely needs four or more terms (stage
        # rotation, flip, then both tilt components) and from_euler caps at 3.
        # Extrinsic rotations are about the fixed axes, so each new term
        # pre-multiplies; intrinsic rotations carry the axes along, so each new
        # term post-multiplies.
        total = Rotation.identity()
        for axis, angle in zip(axes, angles):
            step = Rotation.from_euler(axis, angle, degrees=degrees)
            total = total * step if sequence == "intrinsic" else step * total
        return total

    arr = np.asarray(rotation, dtype=float)
    if arr.shape == (3, 3):
        return Rotation.from_matrix(arr)
    if arr.shape == (4,):
        # graintrace quaternions are scalar-first (w, x, y, z); scipy is scalar-last.
        return Rotation.from_quat(np.roll(arr, -1))
    if arr.shape == (3,):
        vec = arr if not degrees else np.deg2rad(arr)
        return Rotation.from_rotvec(vec)
    raise ValueError(
        "rotation must be None, a (3, 3) matrix, a scalar-first quaternion (4,), "
        "a rotation vector (3,), or a list of (axis, angle) pairs; got shape "
        f"{arr.shape}"
    )


def _rotation_matrix(rotation, angle_unit: str, sequence: str) -> np.ndarray:
    """Return the proper (3, 3) rotation matrix for any accepted ``rotation`` form.

    Args:
        rotation: see :func:`_as_scipy_rotation`.
        angle_unit: ``"degrees"`` or ``"radians"``.
        sequence: ``"extrinsic"`` or ``"intrinsic"``.

    Returns:
        numpy.ndarray: (3, 3) float64 rotation matrix.

    Raises:
        ValueError: if a supplied (3, 3) matrix is not a proper rotation. ``scipy``
            silently orthogonalises a near-rotation, so the matrix form is checked
            here before it is handed over.
    """
    arr = None
    if rotation is not None and not _is_axis_angle_sequence(rotation):
        arr = np.asarray(rotation, dtype=float)
    if arr is not None and arr.shape == (3, 3):
        # scipy's from_matrix orthogonalises; refuse an improper or non-orthogonal
        # input here rather than silently repairing it.
        if not np.allclose(arr.T @ arr, np.eye(3), atol=1e-6):
            raise ValueError(
                "rotation matrix is not orthogonal (R^T R != I); a frame change "
                "carries no scale or shear"
            )
        if not np.isclose(np.linalg.det(arr), 1.0, atol=1e-6):
            raise ValueError(
                "rotation must be proper (det=+1); use axis_sign for axis reversal"
            )
    return np.asarray(
        _as_scipy_rotation(rotation, angle_unit, sequence).as_matrix(), dtype=float
    )


def _check_proper(Q: np.ndarray) -> np.ndarray:
    """Validate that ``Q`` is a proper rotation and return it as float64.

    Args:
        Q: candidate (3, 3) matrix.

    Returns:
        numpy.ndarray: ``Q`` as a contiguous float64 array.

    Raises:
        ValueError: if ``Q`` is not (3, 3) or ``det(Q) != +1``. Mirroring an
            orientation changes crystal handedness, so an improper matrix is
            never silently accepted.
    """
    Q = np.ascontiguousarray(Q, dtype=float)
    if Q.shape != (3, 3):
        raise ValueError(f"rotation must be a (3, 3) matrix, got shape {Q.shape}")
    if not np.isclose(np.linalg.det(Q), 1.0, atol=_ROTATION_ATOL):
        raise ValueError(
            "rotation must be proper (det=+1); use axis_sign for axis reversal"
        )
    return Q


class FrameTransform:
    """Sample-frame change ``x' = Q (s * x) + t``, ``g' = g Q^T``, ``eps' = Q eps Q^T``.

    The transform is stated by the user, never estimated. Positions, crystal
    orientations and sample-frame tensors are all moved by the same rotation, so a
    dataset converted with it is consistent by construction.

    Internally the position map and the orientation map are stored separately: the
    position matrix ``A = Q diag(s)`` may be improper (an axis reversal), while the
    orientation rotation ``Q`` is always proper. Keeping them apart is what makes
    :meth:`inverse` and :meth:`__matmul__` exact when ``s != (1, 1, 1)``, since
    ``x = s * (Q^T (x' - t))`` puts the sign *after* the rotation.

    Args:
        rotation: the composed frame rotation ``Q``. Accepts ``None`` (identity), a
            (3, 3) matrix, a scalar-first quaternion ``(w, x, y, z)``, a rotation
            vector, or an ordered list of ``(axis, angle)`` pairs such as
            ``[("z", 172.3), ("x", 180.0), ("x", 1.4), ("y", -0.9)]`` (main stage
            rotation, remount flip, then the two tilt components). Tilt is part of
            the rotation, not a separate step.
        axis_sign: per-axis sign convention ``s``, entries exactly ``+1`` or ``-1``.
            It corrects an index convention (for example NF layers numbered
            top-down against a target Z that points up) and is applied to
            **positions only**, never to orientations or tensors.
        translation: origin offset ``t``, applied after the rotation.
        angle_unit: ``"degrees"`` (default) or ``"radians"``, for the angle-bearing
            ``rotation`` forms.
        sequence: ``"extrinsic"`` (default, fixed axes) or ``"intrinsic"``, for the
            ``(axis, angle)`` list form.

    Raises:
        ValueError: if ``rotation`` is not a proper rotation, or ``axis_sign`` has
            an entry other than ``+1``/``-1`` (a frame change carries no scale).

    Example:
        >>> import numpy as np
        >>> from graintrace.frame_transform import FrameTransform
        >>> T = FrameTransform(rotation=[("z", 90.0)])
        >>> np.round(T.positions([[1.0, 0.0, 0.0]]), 12)
        array([[0., 1., 0.]])
    """

    def __init__(
        self,
        rotation=None,
        axis_sign: Sequence[float] = (1, 1, 1),
        translation: Sequence[float] = (0.0, 0.0, 0.0),
        angle_unit: str = "degrees",
        sequence: str = "extrinsic",
    ):
        self.Q = _check_proper(_rotation_matrix(rotation, angle_unit, sequence))

        s = np.asarray(axis_sign, dtype=float).reshape(-1)
        if s.shape != (3,):
            raise ValueError(f"axis_sign must have 3 entries, got {s.shape[0]}")
        if not np.all(np.isin(s, _ALLOWED_SIGNS)):
            raise ValueError(
                "axis_sign entries must be +1 or -1 (no scaling in a frame change); "
                f"got {axis_sign!r}"
            )

        t = np.asarray(translation, dtype=float).reshape(-1)
        if t.shape != (3,):
            raise ValueError(f"translation must have 3 entries, got {t.shape[0]}")

        self.t = t
        #: Position matrix; ``A = Q diag(s)`` and may be improper when ``s`` reverses an axis.
        self.A = self.Q @ np.diag(s)

    # ------------------------------------------------------------------
    # construction from stored parts
    # ------------------------------------------------------------------
    @classmethod
    def _from_parts(
        cls, A: np.ndarray, Q: np.ndarray, t: np.ndarray
    ) -> "FrameTransform":
        """Build a transform directly from its three stored parts.

        Used by :meth:`inverse` and :meth:`__matmul__`, whose results need not
        factor as ``Q diag(s)`` with ``s`` a sign vector.

        Args:
            A: (3, 3) position matrix (orthogonal, possibly improper).
            Q: (3, 3) proper orientation rotation.
            t: (3,) translation.

        Returns:
            FrameTransform: the transform with exactly these parts.
        """
        obj = cls.__new__(cls)
        obj.Q = _check_proper(Q)
        obj.A = np.ascontiguousarray(A, dtype=float)
        obj.t = np.ascontiguousarray(t, dtype=float).reshape(3)
        return obj

    @property
    def axis_sign(self) -> Optional[np.ndarray]:
        """The per-axis sign vector ``s``, or ``None`` if the transform has none.

        ``s`` is recoverable as ``Q^T A`` whenever that is a diagonal matrix of
        ``+-1``. After a composition with a rotation it generally is not, and the
        position map is then only expressible as the matrix :attr:`A`.
        """
        d = self.Q.T @ self.A
        off = d - np.diag(np.diag(d))
        if not np.allclose(off, 0.0, atol=_ROTATION_ATOL):
            return None
        diag = np.diag(d)
        if not np.allclose(np.abs(diag), 1.0, atol=_ROTATION_ATOL):
            return None
        return np.sign(diag)

    def __repr__(self) -> str:
        s = self.axis_sign
        sign_txt = "None" if s is None else f"({s[0]:+.0f}, {s[1]:+.0f}, {s[2]:+.0f})"
        return (
            f"FrameTransform(Q=<det {np.linalg.det(self.Q):+.3f}>, "
            f"axis_sign={sign_txt}, translation={tuple(self.t)})"
        )

    # ------------------------------------------------------------------
    # core maps
    # ------------------------------------------------------------------
    def positions(self, x) -> np.ndarray:
        """Transform positions: ``x' = A x + t`` with ``A = Q diag(s)``.

        Args:
            x: (..., 3) positions.

        Returns:
            numpy.ndarray: (..., 3) float64 positions in the target frame.
        """
        arr = np.asarray(x, dtype=float)
        if arr.shape[-1] != 3:
            raise ValueError(f"positions must be (..., 3), got shape {arr.shape}")
        return arr @ self.A.T + self.t

    def orientation_matrices(self, g):
        """Transform orientation matrices on the **sample** side: ``g' = g Q^T``.

        Args:
            g: (..., 3, 3) orientation matrices in
                :func:`~graintrace.orientation_helper.euler_to_matrix`'s convention
                (``g = R_active^T``, where ``R_active`` maps crystal to sample). A
                ``torch.Tensor`` is returned as a tensor with its dtype and device
                preserved; anything else is returned as a float64 tensor.

        Returns:
            torch.Tensor: (..., 3, 3) transformed orientation matrices.
        """
        # pylint: disable-next=import-outside-toplevel  # torch is a heavy dep
        import torch

        if isinstance(g, torch.Tensor):
            gt = g
        else:
            # np.array, not asarray: a read-only input would otherwise make a
            # non-writable tensor and torch warns about it.
            gt = torch.as_tensor(np.array(g, dtype=float), dtype=torch.float64)
        Qt = torch.as_tensor(self.Q.T, dtype=gt.dtype, device=gt.device)
        return gt @ Qt

    def mrp(self, p):
        """Transform neml2 v3 MRP orientations, staying in neml2 throughout.

        ``R' = Q R`` is applied by composing the frame rotation onto the
        orientation inside neml2, so the orientation never leaves the CPFE
        interchange format for a hand-built matrix step.

        The ``compose`` argument order is load bearing and is pinned by
        ``tests/test_frame_transform.py``: ``compose(a, b)`` produces the
        orientation whose matrix is ``matrix(a) @ matrix(b)``, so the sample-side
        update ``g' = g Q^T`` is ``compose(orientation, Q^T)``. Swapping the
        arguments gives the crystal-side update, which is the error this class
        exists to prevent.

        The result is the **short-rotation** MRP set (``|p'| <= 1``, rotation
        angle ``<= 180`` degrees). An MRP ``p`` and its shadow ``-p / |p|^2``
        (neml2 ``shadow``) are the same rotation; the inputs are brought onto
        the short set and the frame rotation's set is chosen so that the
        ``compose`` denominator stays ``>= 1``, which keeps the result finite
        even when the composed rotation is close to 360 degrees.

        Args:
            p: (..., 3) neml2 v3 MRP orientations (either set).

        Returns:
            torch.Tensor: (..., 3) transformed MRP orientations, ``|p'| <= 1``.
        """
        # pylint: disable-next=import-outside-toplevel  # neml2/torch are heavy deps
        import torch

        # pylint: disable-next=import-outside-toplevel  # neml2 is an optional dep
        from neml2.types import MRP, compose

        # pylint: disable-next=import-outside-toplevel  # avoid a module-level cycle
        from .orientation_helper import matrix_to_mrp

        pt = _short_mrp(
            torch.as_tensor(np.asarray(p, dtype=float), dtype=torch.float64)
        )
        q_frame = _short_mrp(
            matrix_to_mrp(torch.as_tensor(self.Q.T, dtype=torch.float64))
        ).expand_as(pt)
        # compose's denominator is 1 + |a|^2 |b|^2 - 2 a.b; with a.b <= 0 it is >= 1.
        # Swapping q for its shadow flips the sign of a.b and leaves the rotation.
        flip = (pt * q_frame).sum(dim=-1, keepdim=True) > 0.0
        q_frame = torch.where(flip, _shadow(q_frame), q_frame)
        composed = compose(MRP(pt.contiguous(), 0), MRP(q_frame.contiguous(), 0))
        return _short_mrp(composed.data)

    def euler(
        self,
        eul,
        convention: str = "bunge",
        angle_type: str = "degrees",
    ) -> np.ndarray:
        """Transform Euler angles: notation -> orientation -> transform -> notation.

        No arithmetic is done on the angles themselves; adding angles is wrong for
        any rotation that is not about the first Bunge axis.

        Args:
            eul: (..., 3) Euler angles.
            convention: ``'bunge'``, ``'kocks'`` or ``'roe'``.
            angle_type: ``'degrees'`` or ``'radians'``; applies to input and output.

        Returns:
            numpy.ndarray: (..., 3) Euler angles in the same convention and unit.
        """
        # pylint: disable-next=import-outside-toplevel  # torch is a heavy dep
        import torch

        # pylint: disable-next=import-outside-toplevel  # avoid a module-level cycle
        from .orientation_helper import euler_to_matrix, matrix_to_euler

        # np.array (not asarray): a read-only input would make a non-writable tensor.
        e = torch.as_tensor(np.array(eul, dtype=float), dtype=torch.float64)
        g = euler_to_matrix(e, convention, angle_type)
        return matrix_to_euler(
            self.orientation_matrices(g), convention, angle_type
        ).numpy()

    def tensors(self, eps) -> np.ndarray:
        """Transform sample-frame rank-2 tensors: ``eps' = Q eps Q^T``.

        Args:
            eps: (..., 3, 3) tensors, or (..., 9) row-major flattened tensors
                (the ``eKen``/``eFab``/stress column layout).

        Returns:
            numpy.ndarray: the transformed tensors, in the same shape as the input.
        """
        arr = np.asarray(eps, dtype=float)
        flat9 = arr.shape[-1] == 9 and arr.shape[-2:] != (3, 3)
        if flat9:
            mats = arr.reshape(arr.shape[:-1] + (3, 3))
        elif arr.shape[-2:] == (3, 3):
            mats = arr
        else:
            raise ValueError(
                f"tensors must be (..., 3, 3) or (..., 9) row-major, got {arr.shape}"
            )
        out = np.einsum("ij,...jk,lk->...il", self.Q, mats, self.Q)
        return out.reshape(arr.shape) if flat9 else out

    # ------------------------------------------------------------------
    # algebra
    # ------------------------------------------------------------------
    def inverse(self) -> "FrameTransform":
        """Return the inverse frame change.

        With ``x' = A x + t`` the inverse is ``x = A^T (x' - t)``, so the inverted
        parts are ``(A^T, Q^T, -A^T t)``. The sign vector lands *after* the
        rotation on the way back, which is why ``A`` and ``Q`` are stored
        separately.

        Returns:
            FrameTransform: ``T.inverse() @ T`` is the identity.
        """
        A_inv = self.A.T
        return FrameTransform._from_parts(A_inv, self.Q.T, -(A_inv @ self.t))

    def __matmul__(self, other: "FrameTransform") -> "FrameTransform":
        """Compose two frame changes: ``(self @ other)(x) == self(other(x))``.

        The position matrices and the orientation rotations compose separately,
        and the translation picks up ``other``'s offset through ``self``'s
        position matrix.

        Args:
            other: the transform applied first.

        Returns:
            FrameTransform: the composed transform.
        """
        if not isinstance(other, FrameTransform):
            return NotImplemented
        return FrameTransform._from_parts(
            self.A @ other.A,
            self.Q @ other.Q,
            self.A @ other.t + self.t,
        )

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-serialisable record of the transform.

        Returns:
            dict: the three stored parts plus the recovered ``axis_sign`` when the
            transform still has one, under a ``"graintrace_frame_transform"``
            schema key.
        """
        s = self.axis_sign
        return {
            "schema": "graintrace_frame_transform",
            "version": 1,
            "rotation": self.Q.tolist(),
            "position_matrix": self.A.tolist(),
            "translation": self.t.tolist(),
            "axis_sign": None if s is None else [int(v) for v in s],
            "convention": {
                "position": "x' = A x + t, A = Q diag(s)",
                "orientation": "g' = g Q^T (sample side; g = euler_to_matrix(...))",
                "tensor": "eps' = Q eps Q^T",
            },
        }

    @classmethod
    def from_dict(cls, record: dict) -> "FrameTransform":
        """Rebuild a transform from :meth:`to_dict`'s record.

        Args:
            record: a mapping as produced by :meth:`to_dict`.

        Returns:
            FrameTransform: the stored transform.

        Raises:
            ValueError: if the record is not a graintrace frame-transform record,
                if ``position_matrix`` is not orthogonal, or if it disagrees with
                a non-null ``axis_sign``.
        """
        if record.get("schema") != "graintrace_frame_transform":
            raise ValueError(
                "not a graintrace frame-transform record (missing or wrong 'schema')"
            )
        Q = np.asarray(record["rotation"], dtype=float)
        A = np.asarray(record.get("position_matrix", Q), dtype=float)
        t = np.asarray(record.get("translation", (0.0, 0.0, 0.0)), dtype=float)

        # Unlike inverse()/__matmul__, which build from parts this class computed
        # itself, a record is outside data: a non-orthogonal A would make
        # positions() a non-rigid map while orientation_matrices() stayed rigid,
        # and nothing downstream would notice. A is orthogonal for every
        # transform this class can produce, composed ones included, so it is the
        # invariant to check here.
        if A.shape != (3, 3):
            raise ValueError(f"position_matrix must be (3, 3), got {A.shape}")
        if not np.allclose(A @ A.T, np.eye(3), atol=_ROTATION_ATOL):
            raise ValueError(
                "position_matrix is not orthogonal; a frame change cannot scale "
                "or skew positions while leaving orientations rigid"
            )

        # When the record claims a sign vector, A must be exactly Q diag(s).
        # Composed transforms store axis_sign=None and are left unconstrained.
        stored_sign = record.get("axis_sign")
        if stored_sign is not None:
            want = Q @ np.diag(np.asarray(stored_sign, dtype=float))
            if not np.allclose(A, want, atol=_ROTATION_ATOL):
                raise ValueError(
                    "position_matrix disagrees with axis_sign: expected "
                    "A = Q diag(axis_sign)"
                )

        return cls._from_parts(A, Q, t)

    def to_json(self, path: str) -> str:
        """Write the transform record to ``path`` as JSON.

        Args:
            path: destination file path.

        Returns:
            str: ``path``.
        """
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "FrameTransform":
        """Read a transform back from a :meth:`to_json` file.

        Args:
            path: the JSON file written by :meth:`to_json`.

        Returns:
            FrameTransform: the stored transform.
        """
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # ------------------------------------------------------------------
    # dataset adapters
    # ------------------------------------------------------------------
    def _write_record(self, dst: str) -> str:
        """Write ``<dst>.transform.json`` beside an adapter's output.

        Args:
            dst: the output file the record accompanies.

        Returns:
            str: the record path.
        """
        return self.to_json(f"{dst}.transform.json")

    @staticmethod
    def _resolve_angle_type(values: np.ndarray, angle_type: str) -> str:
        """Resolve ``angle_type="auto"`` from the magnitude of the Euler values.

        Args:
            values: the Euler angle array to inspect.
            angle_type: ``'degrees'``, ``'radians'`` or ``'auto'``.

        Returns:
            str: ``'degrees'`` or ``'radians'``.

        Raises:
            ValueError: if ``angle_type`` is not one of the three accepted strings.
        """
        if angle_type in ("degrees", "radians"):
            return angle_type
        if angle_type != "auto":
            raise ValueError(
                f"angle_type must be 'degrees', 'radians' or 'auto', got {angle_type!r}"
            )
        # Same rule the rest of graintrace uses: anything beyond 2*pi must be degrees.
        return "degrees" if np.nanmax(np.abs(values)) > 2.0 * np.pi else "radians"

    def apply_ff_csv(
        self,
        src: str,
        dst: str,
        coord_cols: Sequence[str] = ("X", "Y", "Z"),
        euler_cols: Sequence[str] = ("Eul0", "Eul1", "Eul2"),
        convention: str = "bunge",
        angle_type: str = "auto",
        tensor_prefixes: Sequence[str] = ("eKen", "eFab"),
        extra_matrix_prefixes: Sequence[str] = ("O",),
    ) -> str:
        """Move a far-field grain table into the target frame.

        Positions, Euler angles, every present 9-component sample-frame tensor and
        every present 9-component orientation matrix are transformed; grain radius,
        volume, confidence and ids are left untouched. A
        ``<dst>.transform.json`` record is written beside the output.

        Args:
            src: input CSV path, or a ``pandas.DataFrame``.
            dst: output CSV path.
            coord_cols: the three position columns.
            euler_cols: the three Euler columns, or ``None`` to skip orientations.
            convention: Euler convention, ``'bunge'``/``'kocks'``/``'roe'``.
            angle_type: ``'degrees'``, ``'radians'`` or ``'auto'`` (detect from the
                values: anything beyond ``2*pi`` is degrees).
            tensor_prefixes: prefixes whose ``<prefix>11``..``<prefix>33`` columns
                are sample-frame tensors transformed as ``Q eps Q^T``. Missing
                prefixes are skipped.
            extra_matrix_prefixes: prefixes whose ``<prefix>11``..``<prefix>33``
                columns hold an orientation **matrix** ``g``, transformed as
                ``g Q^T``. Missing prefixes are skipped.

        Returns:
            str: ``dst``.
        """
        # pylint: disable-next=import-outside-toplevel  # pandas at call time
        import pandas as pd

        df = pd.read_csv(src).copy() if isinstance(src, str) else src.copy()

        coord_cols = list(coord_cols)
        df[coord_cols] = self.positions(df[coord_cols].to_numpy(dtype=float))

        if euler_cols is not None:
            euler_cols = list(euler_cols)
            eul = df[euler_cols].to_numpy(dtype=float)
            df[euler_cols] = self.euler(
                eul, convention, self._resolve_angle_type(eul, angle_type)
            )

        for prefix in tensor_prefixes:
            cols = _component_columns(df, prefix)
            if cols is not None:
                df[cols] = self.tensors(df[cols].to_numpy(dtype=float))

        for prefix in extra_matrix_prefixes:
            cols = _component_columns(df, prefix)
            if cols is not None:
                g = df[cols].to_numpy(dtype=float).reshape(-1, 3, 3)
                df[cols] = self.orientation_matrices(g).numpy().reshape(-1, 9)

        _ensure_parent(dst)
        df.to_csv(dst, index=False)
        self._write_record(dst)
        return dst

    def apply_pointcloud(
        self,
        df_or_csv,
        dst: str,
        coord_cols: Sequence[str] = ("x", "y", "z"),
        euler_cols: Sequence[str] = ("Eul1", "Eul2", "Eul3"),
        convention: str = "bunge",
        angle_type: str = "radians",
    ) -> str:
        """Move a NF / EBSD point cloud into the target frame.

        A rotation that mixes z into x and y (a flip or a tilt) cannot be written
        back as 2D ``.mic`` layers, so this adapter works on the stacked point
        cloud and outputs 3D coordinates. To keep a layered NF dataset native
        instead, move the *other* dataset in with :meth:`inverse`.

        Args:
            df_or_csv: input CSV path or ``pandas.DataFrame``.
            dst: output CSV path.
            coord_cols: the three position columns.
            euler_cols: the three Euler columns, or ``None`` to skip orientations.
            convention: Euler convention.
            angle_type: ``'degrees'``, ``'radians'`` or ``'auto'``.

        Returns:
            str: ``dst``.
        """
        # pylint: disable-next=import-outside-toplevel  # pandas at call time
        import pandas as pd

        df = (
            pd.read_csv(df_or_csv).copy()
            if isinstance(df_or_csv, str)
            else df_or_csv.copy()
        )

        coord_cols = list(coord_cols)
        df[coord_cols] = self.positions(df[coord_cols].to_numpy(dtype=float))

        if euler_cols is not None:
            euler_cols = list(euler_cols)
            eul = df[euler_cols].to_numpy(dtype=float)
            df[euler_cols] = self.euler(
                eul, convention, self._resolve_angle_type(eul, angle_type)
            )

        _ensure_parent(dst)
        df.to_csv(dst, index=False)
        self._write_record(dst)
        return dst

    def apply_voxel_grid(
        self,
        npy_in: str,
        npy_out: str,
        convention: str = "bunge",
        angle_type: str = "radians",
    ) -> str:
        """Move a ``[phase, Eul1, Eul2, Eul3, X, Y, Z]`` voxel grid into the target frame.

        This is the layout written by the NF reconstruction
        (``merged_segmented_fixed_grid.npy``): the last axis holds the phase/grain
        label, three Euler angles and three coordinates. The grid keeps its array
        shape -- only the stored coordinates and orientations move, so after a
        rotation that mixes axes the grid is no longer axis-aligned and should be
        treated as a point cloud downstream.

        Args:
            npy_in: input ``.npy`` path.
            npy_out: output ``.npy`` path.
            convention: Euler convention of columns 1..3.
            angle_type: ``'degrees'``, ``'radians'`` or ``'auto'``.

        Returns:
            str: ``npy_out``.

        Raises:
            ValueError: if the last axis is shorter than 7 entries.
        """
        grid = np.load(npy_in)
        if grid.shape[-1] < 7:
            raise ValueError(
                "voxel grid must have at least 7 entries on the last axis "
                f"([phase, Eul1..3, X, Y, Z]); got {grid.shape[-1]}"
            )
        out = np.array(grid, dtype=float, copy=True)
        flat = out.reshape(-1, out.shape[-1])

        eul = flat[:, 1:4]
        flat[:, 1:4] = self.euler(
            eul, convention, self._resolve_angle_type(eul, angle_type)
        )
        flat[:, 4:7] = self.positions(flat[:, 4:7])

        _ensure_parent(npy_out)
        np.save(npy_out, out.reshape(grid.shape))
        self._write_record(npy_out)
        return npy_out


def _shadow(p):
    """neml2's shadow MRP set ``-p / |p|^2`` (the same rotation), safe at ``p = 0``.

    Args:
        p: (..., 3) MRP tensor.

    Returns:
        torch.Tensor: the shadow set; rows with ``p = 0`` are returned unchanged
        (the identity has no finite shadow and never needs one here).
    """
    # pylint: disable-next=import-outside-toplevel  # torch is a heavy dep
    import torch

    # pylint: disable-next=import-outside-toplevel  # neml2 is an optional dep
    from neml2.types import MRP, shadow

    nsq = (p * p).sum(dim=-1, keepdim=True)
    zero = nsq == 0.0
    safe = torch.where(zero, torch.ones_like(p), p).contiguous()
    return torch.where(zero, p, shadow(MRP(safe, 0)).data)


def _short_mrp(p):
    """Map each MRP onto the short-rotation set ``|p| <= 1`` via its shadow.

    Args:
        p: (..., 3) MRP tensor.

    Returns:
        torch.Tensor: the same rotations with ``|p| <= 1``.
    """
    # pylint: disable-next=import-outside-toplevel  # torch is a heavy dep
    import torch

    long_set = (p * p).sum(dim=-1, keepdim=True) > 1.0
    return torch.where(long_set, _shadow(p), p)


def _component_columns(df, prefix: str) -> Optional[list]:
    """Return the nine ``<prefix>11``..``<prefix>33`` columns, or ``None`` if absent.

    Args:
        df: the dataframe to inspect.
        prefix: the component-column prefix, e.g. ``"eKen"`` or ``"O"``.

    Returns:
        list | None: the nine column names in row-major order when all are
        present, otherwise ``None``.
    """
    cols = [f"{prefix}{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    return cols if all(c in df.columns for c in cols) else None


def _ensure_parent(path: str) -> None:
    """Create the parent directory of ``path`` if it does not already exist.

    Args:
        path: the file about to be written.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
