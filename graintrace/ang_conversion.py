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

"""Convert EDAX/TSL ``.ang`` orientation maps into the two schemas graintrace reads.

``.ang`` is what EDAX/TSL OIM writes, so it is the native acquisition format of
one of the three data types graintrace is built around. Nothing in the package
reads it directly: :class:`~graintrace.NearFieldMeshBuilder` reads per-layer
``.mic`` files and :class:`~graintrace.VoxelMeshBuilder` reads a merged
``x, y, z, Eul0, Eul1, Eul2`` CSV. :class:`AngConverter` writes either target.

The orientation convention and the angle unit are **never guessed**. They are
taken from the ``#`` header when it pins them, otherwise from the explicit
``angle_convention`` / ``angle_type`` arguments, and a conversion where neither
source supplies them raises :class:`ValueError`. Silently assuming the wrong
unit turns a microstructure into noise without failing anywhere downstream,
which is the one failure mode this module exists to prevent.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

#: Angle units this module accepts, normalised to these two spellings.
ANGLE_TYPES = ("radians", "degrees")

#: Euler conventions ``graintrace.orientation_helper`` understands.
ANGLE_CONVENTIONS = ("bunge", "kocks", "roe")

#: The minimal ``.ang`` column layout: three Euler angles, two coordinates,
#: image quality, confidence index, phase index. Every TSL variant starts with
#: these eight columns in this order; richer exports append more.
ANG_BASE_COLUMNS = (
    "phi1",
    "PHI",
    "phi2",
    "x",
    "y",
    "IQ",
    "CI",
    "phase",
)

#: The two extra columns of the common 10-column TSL export.
ANG_EXTRA_COLUMNS = ("SEM", "fit")

# Header keys (case- and separator-insensitive) that pin the angle unit, mapped
# to the spelling in ANGLE_TYPES. TSL itself does not always write one, which is
# why the explicit argument exists.
_UNIT_KEYS = ("angleunits", "angularunits", "eulerangleunits", "units")
_UNIT_VALUES = {
    "rad": "radians",
    "radian": "radians",
    "radians": "radians",
    "deg": "degrees",
    "degree": "degrees",
    "degrees": "degrees",
}

# Header keys that pin the Euler convention.
_CONVENTION_KEYS = (
    "eulerconvention",
    "eulerangleconvention",
    "angleconvention",
    "orientationconvention",
    "convention",
)

#: Column name in ``.mic`` output that the NF reader renames on the way in.
_MIC_COLUMNS = (
    "%OrientationRowNr",
    "NrMatches",
    "RunTime",
    "X",
    "Y",
    "TriEdgeSize",
    "UpDown",
    "Eul1",
    "Eul2",
    "Eul3",
    "Confidence",
    "PhaseNr",
)


def _normalise_key(text: str) -> str:
    """Strip separators and case from a header key so lookups are forgiving."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass
class AngHeader:
    """What the ``#`` comment block of a ``.ang`` file pins down.

    Every field is ``None`` when the header does not supply it. A headerless
    (8-column, no ``#`` lines) file yields an instance with all fields unset,
    which is a valid input -- the caller must then supply ``angle_convention``
    and ``angle_type`` explicitly.

    Attributes:
        angle_type: ``"radians"`` or ``"degrees"`` if a unit key was present.
        angle_convention: ``"bunge"``, ``"kocks"`` or ``"roe"`` if pinned.
        column_names: Column names parsed from a ``COLUMN_HEADERS`` key,
            normalised to the graintrace spellings where they are recognised.
        xstep: In-plane step in x (micrometers).
        ystep: In-plane step in y (micrometers).
        grid: Grid type reported by the header, e.g. ``"SqrGrid"``.
        keys: Every ``key: value`` pair found, with the raw key text preserved.
    """

    angle_type: Optional[str] = None
    angle_convention: Optional[str] = None
    column_names: Optional[Tuple[str, ...]] = None
    xstep: Optional[float] = None
    ystep: Optional[float] = None
    grid: Optional[str] = None
    keys: Dict[str, str] = field(default_factory=dict)


# Map the names TSL writes in COLUMN_HEADERS onto the graintrace spellings in
# ANG_BASE_COLUMNS. Anything unrecognised is kept verbatim as an extra column.
_COLUMN_ALIASES = {
    "phi1": "phi1",
    "phi": "PHI",
    "phi2": "phi2",
    "x": "x",
    "y": "y",
    "xposition": "x",
    "yposition": "y",
    "iq": "IQ",
    "imagequality": "IQ",
    "ci": "CI",
    "confidenceindex": "CI",
    "phase": "phase",
    "phaseindex": "phase",
    "sem": "SEM",
    "semsignal": "SEM",
    "fit": "fit",
}


def _parse_column_headers(value: str) -> Tuple[str, ...]:
    """Split a ``COLUMN_HEADERS`` value into normalised column names."""
    parts = [p.strip() for p in re.split(r"[,\t]+", value) if p.strip()]
    return tuple(_COLUMN_ALIASES.get(_normalise_key(p), p) for p in parts)


def read_ang_header(path: PathLike) -> AngHeader:
    """Parse the ``#`` comment block of a ``.ang`` file.

    Only the leading run of ``#`` lines is read; parsing stops at the first data
    line, so this is cheap on large maps.

    Args:
        path: Path to the ``.ang`` file.

    Returns:
        AngHeader: the parsed header. A file with no ``#`` lines gives an
        instance with every field unset, which is not an error.
    """
    header = AngHeader()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break
            body = stripped.lstrip("#").strip()
            if not body:
                continue
            # TSL writes both "KEY: value" and "KEY   value"; accept either.
            if ":" in body:
                raw_key, value = body.split(":", 1)
            else:
                parts = body.split(None, 1)
                if len(parts) != 2:
                    continue
                raw_key, value = parts
            raw_key = raw_key.strip()
            value = value.strip()
            if not raw_key:
                continue
            header.keys[raw_key] = value
            key = _normalise_key(raw_key)

            if key in _UNIT_KEYS:
                unit = _UNIT_VALUES.get(_normalise_key(value))
                if unit is None:
                    raise ValueError(
                        f"{path}: header key {raw_key!r} has unrecognised angle "
                        f"unit {value!r}; expected one of {sorted(_UNIT_VALUES)}."
                    )
                header.angle_type = unit
            elif key in _CONVENTION_KEYS:
                convention = _normalise_key(value)
                if convention not in ANGLE_CONVENTIONS:
                    raise ValueError(
                        f"{path}: header key {raw_key!r} has unrecognised Euler "
                        f"convention {value!r}; expected one of "
                        f"{list(ANGLE_CONVENTIONS)}."
                    )
                header.angle_convention = convention
            elif key == "columnheaders":
                header.column_names = _parse_column_headers(value)
            elif key == "xstep":
                header.xstep = _header_float(path, raw_key, value)
            elif key == "ystep":
                header.ystep = _header_float(path, raw_key, value)
            elif key == "grid":
                header.grid = _header_word(path, raw_key, value)

    return header


def _header_word(path: PathLike, raw_key: str, value: str) -> str:
    """Return the first whitespace-separated token of a header value.

    Raises:
        ValueError: if the key carries no value. TSL does write empty-valued
            keys, so this is a malformed file rather than an internal error,
            and it is reported the way every other failure here is.
    """
    parts = value.split()
    if not parts:
        raise ValueError(
            f"{path}: header key {raw_key!r} has no value. Give it one or "
            f"remove the line."
        )
    return parts[0]


def _header_float(path: PathLike, raw_key: str, value: str) -> float:
    """Return a header value's first token as a float.

    Raises:
        ValueError: if the key carries no value, or one that is not a number.
    """
    word = _header_word(path, raw_key, value)
    try:
        return float(word)
    except ValueError as exc:
        raise ValueError(
            f"{path}: header key {raw_key!r} has value {value!r}, which is not "
            f"a number."
        ) from exc


def _resolve_setting(
    name: str,
    explicit: Optional[str],
    from_header: Optional[str],
    allowed: Sequence[str],
    path: PathLike,
    hint: str,
    normalise_unit: bool = False,
) -> str:
    """Resolve one orientation setting from the explicit argument or the header.

    The explicit argument wins over the header so a user can override a file
    they know is mislabelled. Neither source supplying a value is an error, not
    a cue to fall back to a default.

    Args:
        name: The argument name, used only in the error messages.
        explicit: The caller's value, or ``None``.
        from_header: The value the file header pinned, or ``None``.
        allowed: The accepted values.
        path: The file being read, used only in the error messages.
        hint: Extra guidance appended when neither source set the value.
        normalise_unit: Map unit spellings (``rad``, ``deg``, ...) onto
            :data:`ANGLE_TYPES` first. Set it for angle-unit settings; it is a
            parameter rather than a test on ``name`` so adding an argument
            whose name contains another cannot change the behaviour.

    Raises:
        ValueError: if the value is unrecognised, or if neither source set it.
    """
    if explicit is not None:
        value = _normalise_key(str(explicit))
        if normalise_unit:
            value = _UNIT_VALUES.get(value, value)
        if value not in allowed:
            raise ValueError(
                f"{name}={explicit!r} is not recognised; expected one of {list(allowed)}."
            )
        return value
    if from_header is not None:
        return from_header
    raise ValueError(
        f"{path}: {name} is not pinned by the file header and was not given. "
        f"Pass {name}= explicitly. {hint}"
    )


_UNIT_HINT = (
    "EDAX/TSL OIM writes Euler angles in radians, so angle_type='radians' is "
    "right for an unmodified OIM export, but graintrace will not assume it: a "
    "degrees file read as radians produces a plausible-looking and wrong "
    "microstructure."
)
_CONVENTION_HINT = (
    "EDAX/TSL OIM writes Bunge (phi1, PHI, phi2) angles, so "
    "angle_convention='bunge' is right for an unmodified OIM export."
)


def read_ang(
    path: PathLike,
    *,
    angle_convention: Optional[str] = None,
    angle_type: Optional[str] = None,
) -> Tuple[pd.DataFrame, AngHeader]:
    """Read one ``.ang`` file into a DataFrame, with its resolved conventions.

    Handles both the minimal 8-column headerless layout and the ``#``-commented
    variants (10-column TSL exports, and files whose header declares its own
    ``COLUMN_HEADERS``). Column count drives the naming when the header does not
    declare it: the first eight are :data:`ANG_BASE_COLUMNS`, the next two
    :data:`ANG_EXTRA_COLUMNS`, and anything beyond that is named ``col10``,
    ``col11`` and so on.

    Args:
        path: Path to the ``.ang`` file.
        angle_convention: ``"bunge"``, ``"kocks"`` or ``"roe"``. Overrides the
            header. Required when the header does not pin it.
        angle_type: ``"radians"`` or ``"degrees"``. Overrides the header.
            Required when the header does not pin it.

    Returns:
        tuple: ``(df, header)`` where ``df`` holds at least the columns of
        :data:`ANG_BASE_COLUMNS`, and ``header`` is the parsed
        :class:`AngHeader` with ``angle_type`` and ``angle_convention`` filled
        in with the resolved values.

    Raises:
        ValueError: if the file is empty, has fewer than eight columns, or if
            either orientation setting is pinned by neither the header nor the
            arguments.
    """
    path = Path(path)
    header = read_ang_header(path)

    resolved_unit = _resolve_setting(
        "angle_type",
        angle_type,
        header.angle_type,
        ANGLE_TYPES,
        path,
        _UNIT_HINT,
        normalise_unit=True,
    )
    resolved_convention = _resolve_setting(
        "angle_convention",
        angle_convention,
        header.angle_convention,
        ANGLE_CONVENTIONS,
        path,
        _CONVENTION_HINT,
    )

    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None)
    if df.empty:
        raise ValueError(f"{path}: no data rows found below the header.")

    ncols = df.shape[1]
    if ncols < len(ANG_BASE_COLUMNS):
        raise ValueError(
            f"{path}: found {ncols} data columns, expected at least "
            f"{len(ANG_BASE_COLUMNS)} ({', '.join(ANG_BASE_COLUMNS)})."
        )

    if header.column_names is not None:
        if len(header.column_names) != ncols:
            raise ValueError(
                f"{path}: the header declares {len(header.column_names)} columns "
                f"but the data has {ncols}. Fix the header or strip it."
            )
        names: List[str] = list(header.column_names)
    else:
        names = list(ANG_BASE_COLUMNS)
        names += list(ANG_EXTRA_COLUMNS[: max(0, ncols - len(names))])
        names += [f"col{i}" for i in range(len(names), ncols)]

    df.columns = names

    missing = [c for c in ANG_BASE_COLUMNS[:5] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: the header's COLUMN_HEADERS does not name the required "
            f"columns {missing}; it must cover phi1, PHI, phi2, x and y."
        )

    header.angle_type = resolved_unit
    header.angle_convention = resolved_convention
    return df, header


def _euler_to_bunge_degrees(
    euler: np.ndarray, convention: str, angle_type: str
) -> np.ndarray:
    """Convert an (N, 3) Euler array to Bunge angles in degrees."""
    radians = euler if angle_type == "radians" else np.deg2rad(euler)
    if convention == "bunge":
        bunge = radians
    elif convention == "kocks":
        # (psi, theta, phi) -> (psi + pi/2, theta, pi/2 - phi), per
        # graintrace.orientation_helper._to_bunge.
        bunge = np.column_stack(
            [
                radians[:, 0] + math.pi / 2.0,
                radians[:, 1],
                math.pi / 2.0 - radians[:, 2],
            ]
        )
    else:  # roe
        bunge = np.column_stack(
            [
                radians[:, 0] + math.pi / 2.0,
                radians[:, 1],
                radians[:, 2] - math.pi / 2.0,
            ]
        )
    return np.rad2deg(bunge)


class AngConverter:
    """Convert a folder (or one file) of ``.ang`` layers into a graintrace schema.

    Two targets, one per downstream reader:

    * :meth:`to_mic` writes one ``.mic`` per layer for
      :class:`~graintrace.NearFieldMeshBuilder`.
    * :meth:`to_ebsd_csv` writes a single merged ``x, y, z, Eul0, Eul1, Eul2``
      CSV for :class:`~graintrace.VoxelMeshBuilder`.

    Both take ``z`` from the layer index times the layer step, which is the
    decision every hand-rolled converter has had to rediscover. Layers are
    ordered by the integer that follows ``file_token`` in the filename; a single
    input file is layer 0 and needs no token.

    Output angles are always Bunge. The unit of the written angles is chosen by
    ``output_angle_type`` and must be passed on to the downstream builder's
    ``angle_type``.
    """

    def __init__(
        self,
        *,
        input_path: PathLike,
        save_dir: PathLike,
        file_token: str = "layer",
        angle_convention: Optional[str] = None,
        angle_type: Optional[str] = None,
        output_angle_type: str = "degrees",
        confidence_min: Optional[float] = None,
    ) -> None:
        """Set up a converter over one ``.ang`` file or a folder of them.

        Args:
            input_path: A ``.ang`` file, or a folder holding one per layer.
            save_dir: Output folder; created if it does not exist.
            file_token: Filename token the layer index follows, e.g. ``layer``
                in ``layer03.ang``. Ignored for a single-file input.
            angle_convention: ``"bunge"``, ``"kocks"`` or ``"roe"`` for the
                input files. Overrides the header; required when no header
                pins it.
            angle_type: ``"radians"`` or ``"degrees"`` for the input files.
                Overrides the header; required when no header pins it.
            output_angle_type: Unit of the written Bunge angles, ``"degrees"``
                (default) or ``"radians"``. Pass the same value as the
                downstream builder's ``angle_type``.
            confidence_min: Drop rows whose ``CI`` is below this. ``None``
                (default) keeps every row. TSL marks unindexed points with a
                negative CI, so ``0.0`` is the usual floor.

        Raises:
            ValueError: if ``output_angle_type`` is not a recognised unit.
            FileNotFoundError: if the input holds no ``.ang`` files.
        """
        self.input_path = Path(input_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.file_token = str(file_token)
        self.angle_convention = angle_convention
        self.angle_type = angle_type
        self.output_angle_type = _resolve_setting(
            "output_angle_type",
            output_angle_type,
            None,
            ANGLE_TYPES,
            self.input_path,
            "",
            normalise_unit=True,
        )
        self.confidence_min = None if confidence_min is None else float(confidence_min)

    def layer_files(self) -> List[Path]:
        """List the input ``.ang`` files in layer order.

        A single-file input gives a one-element list. A folder is sorted by the
        integer following ``file_token``.

        Returns:
            list[Path]: the files, ordered by layer index.

        Raises:
            FileNotFoundError: if the folder holds no ``.ang`` files.
        """
        return [path for _, path in self.layer_offsets()]

    def layer_offsets(self) -> List[Tuple[int, Path]]:
        """Pair every input file with its offset from the lowest layer index.

        The offset is the integer in the filename minus the smallest such
        integer in the folder, **not** the file's position in the sorted list.
        A folder holding ``layer1`` and ``layer3`` therefore gives offsets
        ``0`` and ``2``: the missing layer leaves a gap in ``z`` instead of
        silently compressing the stack to a contiguous one. A single-file input
        is offset ``0`` and its filename is never parsed.

        Returns:
            list[tuple[int, Path]]: ``(offset, path)``, ordered by layer index.

        Raises:
            FileNotFoundError: if the folder holds no ``.ang`` files.
            ValueError: if a filename carries no layer index after
                ``file_token``.
        """
        if self.input_path.is_file():
            return [(0, self.input_path)]

        # Lazy: nf.convert pulls the VTK writer stack, which this path does not need
        # until a folder is actually listed.
        # pylint: disable=import-outside-toplevel
        from .nf.convert import layer_number

        files = sorted(self.input_path.glob("*.ang"))
        if not files:
            raise FileNotFoundError(f"No .ang files found in {self.input_path}")

        numbers = {f: layer_number(str(f), token=self.file_token) for f in files}
        ordered = sorted(files, key=numbers.__getitem__)
        first = numbers[ordered[0]]
        return [(numbers[f] - first, f) for f in ordered]

    def _read_layer(self, path: Path) -> pd.DataFrame:
        """Read one layer and reduce it to x, y and Bunge angles in the output unit."""
        df, header = read_ang(
            path,
            angle_convention=self.angle_convention,
            angle_type=self.angle_type,
        )
        if self.confidence_min is not None:
            if "CI" not in df.columns:
                raise ValueError(
                    f"{path}: confidence_min was given but the file has no CI column."
                )
            df = df[df["CI"] >= self.confidence_min]
            if df.empty:
                raise ValueError(
                    f"{path}: no points left after the CI >= "
                    f"{self.confidence_min} filter."
                )

        bunge_deg = _euler_to_bunge_degrees(
            df[["phi1", "PHI", "phi2"]].to_numpy(dtype=float),
            header.angle_convention,
            header.angle_type,
        )
        if self.output_angle_type == "radians":
            bunge = np.deg2rad(bunge_deg)
        else:
            bunge = bunge_deg

        out = pd.DataFrame(
            {
                "x": df["x"].to_numpy(dtype=float),
                "y": df["y"].to_numpy(dtype=float),
                "Eul0": bunge[:, 0],
                "Eul1": bunge[:, 1],
                "Eul2": bunge[:, 2],
            }
        )
        out["CI"] = (
            df["CI"].to_numpy(dtype=float)
            if "CI" in df.columns
            else np.full(len(df), 1.0)
        )
        out["phase"] = (
            df["phase"].to_numpy(dtype=float)
            if "phase" in df.columns
            else np.ones(len(df))
        )
        out.attrs["xstep"] = header.xstep
        return out

    def to_ebsd_csv(
        self,
        *,
        zstep: float,
        output_name: str = "ebsd_merged.csv",
    ) -> Path:
        """Write the merged ``x, y, z, Eul0, Eul1, Eul2`` CSV VoxelMeshBuilder reads.

        Args:
            zstep: Spacing between layers (micrometers). ``z`` is the layer's
                index in the filename minus the lowest such index, times
                ``zstep``, so the first layer sits at ``z = 0`` and a missing
                layer leaves a ``2 * zstep`` gap rather than closing up.
            output_name: Filename written inside ``save_dir``.

        Returns:
            Path: the written CSV.
        """
        zstep = float(zstep)
        frames = []
        for offset, path in self.layer_offsets():
            layer = self._read_layer(path)
            layer.insert(2, "z", offset * zstep)
            frames.append(layer[["x", "y", "z", "Eul0", "Eul1", "Eul2"]])

        merged = pd.concat(frames, ignore_index=True)
        out_path = self.save_dir / output_name
        merged.to_csv(out_path, index=False)
        return out_path

    def to_mic(
        self,
        *,
        zstep: float,
        tri_edge_size: Optional[float] = None,
        prefix: Optional[str] = None,
    ) -> List[Path]:
        """Write one ``.mic`` per layer for NearFieldMeshBuilder.

        Each file carries the three ``%`` metadata lines the reader skips
        (``%TriEdgeSize``, ``%NumPhases``, ``%GlobalPosition``) followed by the
        tab-delimited column row. ``%GlobalPosition`` records the layer's ``z``;
        note that ``NearFieldMeshBuilder`` recomputes ``z`` from its own ``dz``
        argument, so pass the same value there as ``zstep`` here.

        A gap in the input layer numbering is preserved, in the filenames as
        well as in ``%GlobalPosition``: ``layer1.ang`` and ``layer3.ang`` give
        ``<prefix>1.mic`` and ``<prefix>3.mic``. That is deliberate.
        :func:`~graintrace.nf.convert.nf_to_pointcloud` recomputes ``z`` from
        the file order and refuses a non-contiguous layer set, so renumbering
        the outputs contiguously would hide the missing layer from the one
        check that catches it and place the stack at the wrong ``z``.

        Args:
            zstep: Spacing between layers (micrometers), written into
                ``%GlobalPosition`` as the layer's offset times ``zstep``.
            tri_edge_size: Value for the ``TriEdgeSize`` column and the matching
                metadata line. Resolved per layer: defaults to that layer's
                header ``XSTEP``, or ``1.0`` when its header reports none, so
                layers with different steps each get their own.
            prefix: Output filename stem; the layer at offset ``i`` from the
                first becomes ``<prefix><i+1>.mic``. Defaults to
                ``file_token``.

        Returns:
            list[Path]: the written ``.mic`` files, in layer order.
        """
        zstep = float(zstep)
        prefix = self.file_token if prefix is None else str(prefix)

        written: List[Path] = []
        for offset, path in self.layer_offsets():
            layer = self._read_layer(path)
            edge = (
                float(tri_edge_size)
                if tri_edge_size is not None
                else float(layer.attrs.get("xstep") or 1.0)
            )
            count = len(layer)
            frame = pd.DataFrame(
                {
                    "%OrientationRowNr": np.arange(1, count + 1, dtype=float),
                    "NrMatches": np.ones(count),
                    "RunTime": np.zeros(count),
                    "X": layer["x"].to_numpy(),
                    "Y": layer["y"].to_numpy(),
                    "TriEdgeSize": np.full(count, edge),
                    "UpDown": np.full(count, -1.0),
                    "Eul1": layer["Eul0"].to_numpy(),
                    "Eul2": layer["Eul1"].to_numpy(),
                    "Eul3": layer["Eul2"].to_numpy(),
                    "Confidence": layer["CI"].to_numpy(),
                    "PhaseNr": layer["phase"].to_numpy(),
                }
            )
            out_path = self.save_dir / f"{prefix}{offset + 1}.mic"
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(f"%TriEdgeSize {edge:.6f}\n")
                fh.write("%NumPhases 1\n")
                fh.write(f"%GlobalPosition {offset * zstep:.6f}\n")
                frame[list(_MIC_COLUMNS)].to_csv(fh, sep="\t", index=False)
            written.append(out_path)

        return written
