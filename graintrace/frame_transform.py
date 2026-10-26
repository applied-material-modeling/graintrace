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

"""Stated specimen-frame change for FF / NF / EBSD datasets.

``x' = Q (s * x) + t``, ``g' = g Q^T`` (sample side, ``g = euler_to_matrix(...)``),
``eps' = Q eps Q^T``. ``Q`` is a proper rotation, ``s`` a per-axis sign applied to
positions only, ``t`` a translation. There is no scale term.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np

_ALLOWED_SIGNS = (-1.0, 1.0)
_ROTATION_ATOL = 1e-8
_QUAT_NORM_ATOL = 1e-6
_AXIS_NAMES = {"x": "x", "y": "y", "z": "z", "X": "x", "Y": "y", "Z": "z"}


def _is_axis_angle_sequence(rotation) -> bool:
    """True when ``rotation`` is a non-empty list of ``(axis, angle)`` pairs."""
    if not isinstance(rotation, (list, tuple)) or len(rotation) == 0:
        return False
    first = rotation[0]
    return isinstance(first, (list, tuple)) and len(first) == 2


def _as_scipy_rotation(rotation, angle_unit: str, sequence: str):
    """Build a scipy ``Rotation`` from any accepted ``rotation`` form.

    Args:
        rotation: see :class:`FrameTransform`.
        angle_unit: ``"degrees"`` or ``"radians"``.
        sequence: ``"extrinsic"`` or ``"intrinsic"`` (``(axis, angle)`` form only).

    Returns:
        scipy.spatial.transform.Rotation: the composed rotation.

    Raises:
        ValueError: on an unknown unit, sequence, axis or shape, or a non-unit quaternion.
    """
    # pylint: disable-next=import-outside-toplevel
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

    if _is_axis_angle_sequence(rotation):
        total = Rotation.identity()
        for item in rotation:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(
                    "a rotation sequence must be a list of (axis, angle) pairs; "
                    f"got the entry {item!r}"
                )
            axis, angle = item
            if axis not in _AXIS_NAMES:
                raise ValueError(f"rotation axis must be one of x/y/z, got {axis!r}")
            # One term at a time: from_euler caps at three axes.
            step = Rotation.from_euler(_AXIS_NAMES[axis], float(angle), degrees=degrees)
            total = total * step if sequence == "intrinsic" else step * total
        return total

    arr = np.asarray(rotation, dtype=float)
    if arr.shape == (3, 3):
        return Rotation.from_matrix(arr)
    if arr.shape == (4,):
        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > _QUAT_NORM_ATOL:
            raise ValueError(
                f"quaternion must be unit length (|q| = 1 within {_QUAT_NORM_ATOL:g}); "
                f"got |q| = {norm:.6g}"
            )
        return Rotation.from_quat(np.roll(arr, -1))  # scalar-first -> scipy order
    if arr.shape == (3,):
        vec = arr if not degrees else np.deg2rad(arr)
        return Rotation.from_rotvec(vec)
    raise ValueError(
        "rotation must be None, a (3, 3) matrix, a scalar-first quaternion (4,), "
        "a rotation vector (3,), or a list of (axis, angle) pairs; got shape "
        f"{arr.shape}"
    )


def _rotation_matrix(rotation, angle_unit: str, sequence: str) -> np.ndarray:
    """Return the (3, 3) rotation matrix for any accepted ``rotation`` form.

    Raises:
        ValueError: if a supplied matrix is not orthogonal or not proper.
    """
    arr = None
    if rotation is not None and not _is_axis_angle_sequence(rotation):
        arr = np.asarray(rotation, dtype=float)
    if arr is not None and arr.shape == (3, 3):
        # scipy's from_matrix would silently orthogonalise.
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
    """Return ``Q`` as float64, raising ``ValueError`` unless it is a proper (3, 3) rotation."""
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

    Positions use ``A = Q diag(s)`` (possibly improper); orientations and tensors
    use the proper ``Q``. Both are stored so :meth:`inverse` and ``@`` stay exact.

    Args:
        rotation: ``None``, a (3, 3) proper matrix, a scalar-first unit quaternion
            ``(w, x, y, z)``, a rotation vector, or ``(axis, angle)`` pairs such as
            ``[("z", 172.3), ("x", 180.0), ("y", -0.9)]``.
        axis_sign: per-axis ``+1``/``-1``, applied to positions only.
        translation: origin offset ``t``, applied after the rotation.
        angle_unit: ``"degrees"`` (default) or ``"radians"``.
        sequence: ``"extrinsic"`` (default, fixed axes) or ``"intrinsic"``.

    Raises:
        ValueError: on an improper rotation or an ``axis_sign`` entry other than +-1.

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
        #: Position matrix ``A = Q diag(s)``.
        self.A = self.Q @ np.diag(s)

    @classmethod
    def _from_parts(
        cls, A: np.ndarray, Q: np.ndarray, t: np.ndarray
    ) -> "FrameTransform":
        """Build a transform from its stored parts ``(A, Q, t)``."""
        obj = cls.__new__(cls)
        obj.Q = _check_proper(Q)
        obj.A = np.ascontiguousarray(A, dtype=float)
        obj.t = np.ascontiguousarray(t, dtype=float).reshape(3)
        return obj

    @property
    def axis_sign(self) -> Optional[np.ndarray]:
        """The sign vector ``s``, or ``None`` when ``Q^T A`` is not a +-1 diagonal (composed transforms)."""
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

    def positions(self, x) -> np.ndarray:
        """Transform positions: ``x' = A x + t``.

        Args:
            x: (..., 3) positions.

        Returns:
            numpy.ndarray: (..., 3) positions in the target frame.
        """
        arr = np.asarray(x, dtype=float)
        if arr.shape[-1] != 3:
            raise ValueError(f"positions must be (..., 3), got shape {arr.shape}")
        return arr @ self.A.T + self.t

    def orientation_matrices(self, g):
        """Transform orientation matrices on the sample side: ``g' = g Q^T``.

        Args:
            g: (..., 3, 3) matrices in :func:`~graintrace.orientation_helper.euler_to_matrix`'s
                convention; a tensor keeps its dtype and device.

        Returns:
            torch.Tensor: (..., 3, 3) transformed matrices.
        """
        # pylint: disable-next=import-outside-toplevel
        import torch

        if isinstance(g, torch.Tensor):
            gt = g
        else:
            # np.array, not asarray: a read-only input makes torch warn.
            gt = torch.as_tensor(np.array(g, dtype=float), dtype=torch.float64)
        Qt = torch.as_tensor(self.Q.T, dtype=gt.dtype, device=gt.device)
        return gt @ Qt

    def mrp(self, p):
        """Transform neml2 v3 MRP orientations in neml2; returns the short set ``|p'| <= 1``.

        Args:
            p: (..., 3) neml2 v3 MRP orientations (either shadow set).

        Returns:
            torch.Tensor: (..., 3) transformed MRP, ``|p'| <= 1``.
        """
        # pylint: disable=import-outside-toplevel
        import torch
        from neml2.types import MRP, compose

        from .orientation_helper import matrix_to_mrp

        pt = _short_mrp(
            torch.as_tensor(np.asarray(p, dtype=float), dtype=torch.float64)
        )
        q_frame = _short_mrp(
            matrix_to_mrp(torch.as_tensor(self.Q.T, dtype=torch.float64))
        ).expand_as(pt)
        # compose(a, b) has matrix(a) @ matrix(b), so g Q^T is compose(p, Q^T).
        # Its denominator 1 + |a|^2|b|^2 - 2 a.b stays >= 1 when a.b <= 0.
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
        """Transform Euler angles via the orientation matrix (no angle arithmetic).

        Args:
            eul: (..., 3) Euler angles.
            convention: ``'bunge'``, ``'kocks'`` or ``'roe'``.
            angle_type: ``'degrees'`` or ``'radians'``, for input and output.

        Returns:
            numpy.ndarray: (..., 3) Euler angles in the same convention and unit.
        """
        # pylint: disable=import-outside-toplevel
        import torch

        from .orientation_helper import euler_to_matrix, matrix_to_euler

        e = torch.as_tensor(np.array(eul, dtype=float), dtype=torch.float64)
        g = euler_to_matrix(e, convention, angle_type)
        return matrix_to_euler(
            self.orientation_matrices(g), convention, angle_type
        ).numpy()

    def tensors(self, eps) -> np.ndarray:
        """Transform sample-frame rank-2 tensors: ``eps' = Q eps Q^T``.

        Args:
            eps: (..., 3, 3) tensors or (..., 9) row-major components.

        Returns:
            numpy.ndarray: transformed tensors, same shape as the input.
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

    def inverse(self) -> "FrameTransform":
        """Return the inverse, with parts ``(A^T, Q^T, -A^T t)``."""
        A_inv = self.A.T
        return FrameTransform._from_parts(A_inv, self.Q.T, -(A_inv @ self.t))

    def __matmul__(self, other: "FrameTransform") -> "FrameTransform":
        """Compose: ``(self @ other)`` applies ``other`` first, then ``self``."""
        if not isinstance(other, FrameTransform):
            return NotImplemented
        return FrameTransform._from_parts(
            self.A @ other.A,
            self.Q @ other.Q,
            self.A @ other.t + self.t,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable record of the transform."""
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
            ValueError: on a foreign record, a non-orthogonal ``position_matrix``, or
                one that disagrees with a non-null ``axis_sign``.
        """
        if record.get("schema") != "graintrace_frame_transform":
            raise ValueError(
                "not a graintrace frame-transform record (missing or wrong 'schema')"
            )
        Q = np.asarray(record["rotation"], dtype=float)
        A = np.asarray(record.get("position_matrix", Q), dtype=float)
        t = np.asarray(record.get("translation", (0.0, 0.0, 0.0)), dtype=float)

        if A.shape != (3, 3):
            raise ValueError(f"position_matrix must be (3, 3), got {A.shape}")
        if not np.allclose(A @ A.T, np.eye(3), atol=_ROTATION_ATOL):
            raise ValueError(
                "position_matrix is not orthogonal; a frame change cannot scale "
                "or skew positions while leaving orientations rigid"
            )

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
        """Write :meth:`to_dict` to ``path`` as JSON and return ``path``."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "FrameTransform":
        """Read a transform written by :meth:`to_json`."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def _write_record(self, dst: str) -> str:
        """Write ``<dst>.transform.json`` beside an adapter's output."""
        return self.to_json(f"{dst}.transform.json")

    @staticmethod
    def _resolve_angle_type(values: np.ndarray, angle_type: str) -> str:
        """Resolve ``angle_type="auto"``: any value beyond ``2*pi`` means degrees."""
        if angle_type in ("degrees", "radians"):
            return angle_type
        if angle_type != "auto":
            raise ValueError(
                f"angle_type must be 'degrees', 'radians' or 'auto', got {angle_type!r}"
            )
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
        """Move an FF grain table into the target frame and write ``<dst>.transform.json``.

        Args:
            src: input CSV path or ``pandas.DataFrame``.
            dst: output CSV path.
            coord_cols: the three position columns.
            euler_cols: the three Euler columns, or ``None`` to skip.
            convention: ``'bunge'``, ``'kocks'`` or ``'roe'``.
            angle_type: ``'degrees'``, ``'radians'`` or ``'auto'``.
            tensor_prefixes: ``<prefix>11..33`` tensor columns (``Q eps Q^T``); missing ones are skipped.
            extra_matrix_prefixes: ``<prefix>11..33`` orientation-matrix columns (``g Q^T``); missing ones are skipped.

        Returns:
            str: ``dst``.
        """
        # pylint: disable-next=import-outside-toplevel
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
        """Move an NF / EBSD point cloud into the target frame and write ``<dst>.transform.json``.

        Args:
            df_or_csv: input CSV path or ``pandas.DataFrame``.
            dst: output CSV path.
            coord_cols: the three position columns.
            euler_cols: the three Euler columns, or ``None`` to skip.
            convention: Euler convention.
            angle_type: ``'degrees'``, ``'radians'`` or ``'auto'``.

        Returns:
            str: ``dst``.
        """
        # pylint: disable-next=import-outside-toplevel
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
        """Move a ``[phase, Eul1..3, X, Y, Z]`` voxel grid (``.npy``) into the target frame.

        The array shape is kept; after a rotation that mixes axes the grid is no
        longer axis aligned.

        Args:
            npy_in: input ``.npy`` path.
            npy_out: output ``.npy`` path.
            convention: Euler convention of columns 1..3.
            angle_type: ``'degrees'``, ``'radians'`` or ``'auto'``.

        Returns:
            str: ``npy_out``.

        Raises:
            ValueError: if the last axis has fewer than 7 entries.
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

    def apply_ee_csv(self, src: str, dst: str) -> str:
        """Move a headerless ``x, y, z`` + 9 row-major strain file (``reconstruction_cpfe_ee.csv``).

        Writes ``dst`` comma-delimited without a header, plus ``<dst>.transform.json``.

        Args:
            src: input file, comma- or whitespace-delimited.
            dst: output CSV path.

        Returns:
            str: ``dst``.

        Raises:
            ValueError: if the file does not have 12 columns.
        """
        # pylint: disable-next=import-outside-toplevel
        import pandas as pd

        data = pd.read_csv(src, sep=r"[,\s]+", engine="python", header=None).to_numpy(
            dtype=float
        )
        if data.ndim != 2 or data.shape[1] != 12:
            raise ValueError(
                "an elastic-strain file must have 12 columns (x, y, z + 9 row-major "
                f"strain components); got shape {data.shape}"
            )
        out = np.empty_like(data)
        out[:, :3] = self.positions(data[:, :3])
        out[:, 3:] = self.tensors(data[:, 3:])

        _ensure_parent(dst)
        pd.DataFrame(out).to_csv(dst, index=False, header=False, float_format="%.12g")
        self._write_record(dst)
        return dst

    def inside_voxel_solid(
        self,
        points,
        solid_mask,
        origin: Sequence[float],
        spacing,
        margin_vox: int = 0,
    ) -> np.ndarray:
        """Flag target-frame points that land on solid voxels of a source-frame grid.

        Points are mapped back with :meth:`inverse` and snapped to the nearest voxel
        centre; points outside the grid are ``False``. Use it for the
        ``in_overlap_a``/``in_overlap_b`` masks of ``detect_splits``.

        Args:
            points: (N, 3) points in the target frame.
            solid_mask: (nx, ny, nz) boolean array, axis ``k`` along coordinate ``k``.
            origin: source-frame centre of voxel ``(0, 0, 0)``.
            spacing: voxel size, scalar or ``(dx, dy, dz)``.
            margin_vox: erode the solid by this many voxels first (default 0).

        Returns:
            numpy.ndarray: (N,) boolean mask.

        Raises:
            ValueError: on a non-3D mask, non-positive spacing, negative margin or non-(N, 3) points.
        """
        mask = np.asarray(solid_mask, dtype=bool)
        if mask.ndim != 3:
            raise ValueError(f"solid_mask must be 3D (nx, ny, nz), got {mask.shape}")
        h = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
        if np.any(h <= 0):
            raise ValueError(f"spacing must be positive, got {spacing!r}")
        if int(margin_vox) < 0:
            raise ValueError(f"margin_vox must be >= 0, got {margin_vox!r}")
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got shape {pts.shape}")

        if margin_vox:
            # pylint: disable-next=import-outside-toplevel
            from scipy.ndimage import binary_erosion

            mask = binary_erosion(mask, iterations=int(margin_vox), border_value=0)

        src = self.inverse().positions(pts)
        idx = np.rint((src - np.asarray(origin, dtype=float)) / h).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < np.asarray(mask.shape)), axis=1)
        out = np.zeros(len(pts), dtype=bool)
        ii = idx[inside]
        out[inside] = mask[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out


def voxel_solid_from_grid(grid, spacing=None):
    """Return ``(solid_mask, origin, spacing)`` of a ``[phase, Eul1..3, X, Y, Z]`` voxel grid.

    Phase ``> 0`` is solid. The result feeds :meth:`FrameTransform.inside_voxel_solid`.

    Args:
        grid: (nx, ny, nz, >=7) array or ``.npy`` path (e.g. ``merged_segmented_fixed_grid.npy``).
        spacing: voxel size; inferred from neighbouring voxel coordinates when ``None``.

    Returns:
        tuple: boolean (nx, ny, nz) mask, (3,) centre of voxel ``(0, 0, 0)``, (3,) spacing.

    Raises:
        ValueError: on a malformed grid, or a single-voxel axis with no ``spacing``.
    """
    arr = np.load(grid) if isinstance(grid, (str, os.PathLike)) else np.asarray(grid)
    if arr.ndim != 4 or arr.shape[-1] < 7:
        raise ValueError(
            "voxel grid must be (nx, ny, nz, >=7) [phase, Eul1..3, X, Y, Z]; "
            f"got {arr.shape}"
        )
    origin = np.asarray(arr[0, 0, 0, 4:7], dtype=float)
    if spacing is None:
        h = np.empty(3)
        for k in range(3):
            if arr.shape[k] < 2:
                raise ValueError(
                    f"cannot infer spacing along axis {k} (one voxel); pass spacing"
                )
            step = [0, 0, 0]
            step[k] = 1
            h[k] = abs(float(arr[tuple(step)][4 + k]) - origin[k])
    else:
        h = np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).copy()
    return arr[..., 0] > 0, origin, h


def _shadow(p):
    """neml2 shadow MRP ``-p / |p|^2`` (same rotation); ``p = 0`` is returned unchanged."""
    # pylint: disable=import-outside-toplevel
    import torch
    from neml2.types import MRP, shadow

    zero = (p * p).sum(dim=-1, keepdim=True) == 0.0
    safe = torch.where(zero, torch.ones_like(p), p).contiguous()
    return torch.where(zero, p, shadow(MRP(safe, 0)).data)


def _short_mrp(p):
    """Map each MRP onto the short-rotation set ``|p| <= 1``."""
    # pylint: disable-next=import-outside-toplevel
    import torch

    long_set = (p * p).sum(dim=-1, keepdim=True) > 1.0
    return torch.where(long_set, _shadow(p), p)


def _component_columns(df, prefix: str) -> Optional[list]:
    """The nine ``<prefix>11..33`` columns in row-major order, or ``None`` if any is absent."""
    cols = [f"{prefix}{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    return cols if all(c in df.columns for c in cols) else None


def _ensure_parent(path: str) -> None:
    """Create the parent directory of ``path`` if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
