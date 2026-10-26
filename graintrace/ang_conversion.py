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

"""Read EDAX/TSL ``.ang`` orientation maps and convert them for the mesh builders.

:class:`AngConverter` writes per-layer ``.mic`` files for
:class:`~graintrace.NearFieldMeshBuilder` or one merged ``x, y, z, Eul0, Eul1, Eul2``
CSV for :class:`~graintrace.VoxelMeshBuilder`. Angles are passed through as TSL
writes them (Bunge), only converted to the requested unit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

#: The eight leading columns of every TSL ``.ang`` layout; extra columns are dropped.
ANG_COLUMNS = ("phi1", "PHI", "phi2", "x", "y", "IQ", "CI", "phase")

ANGLE_TYPES = ("radians", "degrees")

_MIC_HEADER = "%OrientationRowNr\tNrMatches\tRunTime\tX\tY\tTriEdgeSize\tUpDown\tEul1\tEul2\tEul3\tConfidence\tPhaseNr"


def _check_unit(name: str, value: str) -> str:
    """Return ``value`` if it is one of :data:`ANGLE_TYPES`, else raise ``ValueError``."""
    if value not in ANGLE_TYPES:
        raise ValueError(f"{name}={value!r}; expected one of {list(ANGLE_TYPES)}")
    return value


def read_ang(path: PathLike) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Read one ``.ang`` file (8-column headerless, or ``#``-headed with 8+ columns).

    Args:
        path: Path to the ``.ang`` file.

    Returns:
        ``(df, header)``: ``df`` holds the :data:`ANG_COLUMNS`; ``header`` maps the
        upper-cased keys of ``# KEY: value`` lines (e.g. ``GRID``, ``XSTEP``) to values.

    Raises:
        ValueError: if the file has fewer than eight data columns.
    """
    header: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                if line.strip():
                    break
                continue
            key, sep, value = line.lstrip("#").partition(":")
            if sep and key.strip():
                header[key.strip().upper()] = value.strip()

    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None)
    if df.shape[1] < len(ANG_COLUMNS):
        raise ValueError(
            f"{path}: {df.shape[1]} data columns, expected at least {len(ANG_COLUMNS)} ({', '.join(ANG_COLUMNS)})"
        )
    df = df.iloc[:, : len(ANG_COLUMNS)]
    df.columns = list(ANG_COLUMNS)
    return df, header


class AngConverter:
    """Convert one ``.ang`` file, or a folder of per-layer files, for the mesh builders.

    Folder layers are ordered by the integer after ``file_token`` in the filename and
    placed at ``z = (index - lowest index) * zstep``, so a missing layer leaves a gap.
    """

    def __init__(
        self,
        *,
        input_path: PathLike,
        save_dir: PathLike,
        file_token: str = "layer",
        angle_type: str = "radians",
        output_angle_type: str = "degrees",
        confidence_min: Optional[float] = None,
    ) -> None:
        """Set up the converter.

        Args:
            input_path: A ``.ang`` file, or a folder holding one per layer.
            save_dir: Output folder; created if absent.
            file_token: Filename token the layer index follows (e.g. ``layer`` in ``layer03.ang``).
            angle_type: Unit of the input Euler angles. TSL OIM writes radians.
            output_angle_type: Unit of the written angles; pass the same value as the
                downstream builder's ``angle_type``.
            confidence_min: Drop points whose ``CI`` is below this (``0.0`` drops TSL's
                negative-CI unindexed points). ``None`` keeps every point.

        Raises:
            ValueError: if ``angle_type`` or ``output_angle_type`` is not a known unit.
        """
        self.input_path = Path(input_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.file_token = str(file_token)
        self.angle_type = _check_unit("angle_type", angle_type)
        self.output_angle_type = _check_unit("output_angle_type", output_angle_type)
        self.confidence_min = confidence_min

    def layer_offsets(self) -> List[Tuple[int, Path]]:
        """Pair each input file with its layer index minus the lowest one.

        Returns:
            ``(offset, path)`` pairs in layer order; a single-file input is ``[(0, path)]``.

        Raises:
            FileNotFoundError: if the folder holds no ``.ang`` files.
            ValueError: if a filename has no layer index after ``file_token``.
        """
        if self.input_path.is_file():
            return [(0, self.input_path)]

        # pylint: disable=import-outside-toplevel
        from .nf.convert import layer_number

        files = sorted(self.input_path.glob("*.ang"))
        if not files:
            raise FileNotFoundError(f"No .ang files found in {self.input_path}")
        numbers = {f: layer_number(str(f), token=self.file_token) for f in files}
        ordered = sorted(files, key=numbers.__getitem__)
        return [(numbers[f] - numbers[ordered[0]], f) for f in ordered]

    def _read_layer(self, path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Read one layer, apply the CI filter and convert the angles to the output unit."""
        df, header = read_ang(path)
        euler = df[["phi1", "PHI", "phi2"]].to_numpy(dtype=float)
        if self.confidence_min is not None:
            keep = df["CI"].to_numpy() >= self.confidence_min
            df, euler = df[keep], euler[keep]
            if df.empty:
                raise ValueError(
                    f"{path}: no points left after CI >= {self.confidence_min}"
                )
        if self.angle_type != self.output_angle_type:
            euler = (
                np.deg2rad(euler)
                if self.output_angle_type == "radians"
                else np.rad2deg(euler)
            )
        out = pd.DataFrame({"x": df["x"].to_numpy(float), "y": df["y"].to_numpy(float)})
        out[["Eul0", "Eul1", "Eul2"]] = euler
        out["CI"] = df["CI"].to_numpy(float)
        return out, header

    def to_ebsd_csv(
        self, *, zstep: float, output_name: str = "ebsd_merged.csv"
    ) -> Path:
        """Write the merged ``x, y, z, Eul0, Eul1, Eul2`` CSV that VoxelMeshBuilder reads.

        Args:
            zstep: Layer spacing (micrometers).
            output_name: Filename inside ``save_dir``.

        Returns:
            Path to the written CSV.
        """
        frames = []
        for offset, path in self.layer_offsets():
            layer, _ = self._read_layer(path)
            layer.insert(2, "z", offset * float(zstep))
            frames.append(layer[["x", "y", "z", "Eul0", "Eul1", "Eul2"]])
        out_path = self.save_dir / output_name
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
        return out_path

    def to_mic(
        self,
        *,
        zstep: float,
        tri_edge_size: Optional[float] = None,
        prefix: Optional[str] = None,
    ) -> List[Path]:
        """Write one ``.mic`` per layer for NearFieldMeshBuilder.

        ``Confidence`` is the ``.ang`` CI and ``PhaseNr`` is always 1. Output names keep
        any gap in the layer numbering so ``nf_to_pointcloud`` reports it.

        Args:
            zstep: Layer spacing (micrometers), written to ``%GlobalPosition``. The NF
                reader recomputes ``z`` from its own ``dz``; pass the same value there.
            tri_edge_size: ``TriEdgeSize`` value; defaults to the layer's header ``XSTEP``, else 1.0.
            prefix: Output stem; layer offset ``i`` becomes ``<prefix><i+1>.mic``. Defaults to ``file_token``.

        Returns:
            The written ``.mic`` paths, in layer order.
        """
        prefix = self.file_token if prefix is None else str(prefix)
        written: List[Path] = []
        for offset, path in self.layer_offsets():
            layer, header = self._read_layer(path)
            if tri_edge_size is None:
                edge = float((header.get("XSTEP") or "1").split()[0])
            else:
                edge = float(tri_edge_size)
            n = len(layer)
            cols = [
                np.arange(1, n + 1),
                np.ones(n),
                np.zeros(n),
                layer["x"],
                layer["y"],
                np.full(n, edge),
                np.full(n, -1.0),
                layer["Eul0"],
                layer["Eul1"],
                layer["Eul2"],
                layer["CI"],
                np.ones(n),
            ]
            out_path = self.save_dir / f"{prefix}{offset + 1}.mic"
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(
                    f"%TriEdgeSize {edge:.6f}\n%NumPhases 1\n%GlobalPosition {offset * float(zstep):.6f}\n{_MIC_HEADER}\n"
                )
                np.savetxt(fh, np.column_stack(cols), delimiter="\t", fmt="%.8g")
            written.append(out_path)
        return written
