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

"""Per-grain cell geometry from a Neper tessellation, for HEDM stitching.

Given a set of grain centroids (one scan or a stitched accumulator), build a
(Laguerre / power) Voronoi tessellation *at the measured centroids* — no CVT
relaxation — and read back each grain's true polyhedral z-extent ``[Zmin, Zmax]``
and volume centroid. This replaces the equivalent-sphere ``z +/- GrainRadius``
approximation used by the region classifier for elongated / anisotropic grains.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def default_neper_env() -> dict:
    """Environment that lets a ``~/.local`` Neper install find its GSL/OpenBLAS libs.

    Mirrors the env built by ``VoronoiMeshBuilder.check_dependencies`` (see
    ``construct_voronoi_mesh.py``) but without the install side effects.
    """
    prefix = os.path.join(os.path.expanduser("~"), ".local")
    env = os.environ.copy()
    env["PATH"] = f"{prefix}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")
    return env


def _read_tess_sections(tess_path: str) -> dict:
    """Group a ``.tess`` file's non-empty lines by top-level (``**``) section name."""
    sections: dict = {}
    current = None
    with open(tess_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("**"):
                current = s.strip("*").strip().lower()
                sections[current] = []
                continue
            if current is not None:
                sections[current].append(s)
    return sections


def _parse_tess_cells(
    tess_path: str,
) -> Tuple[np.ndarray, List[List[int]], List[List[int]]]:
    """Slim parser for the geometry needed per cell.

    Returns ``(vertices[Nv, 3], face_vertices, cell_faces)`` with all ids 0-based.
    ``face_vertices[f]`` are the vertex ids of face f; ``cell_faces[c]`` are the
    (absolute) face ids of cell c. Blocks follow the Neper ``.tess`` layout:
    ``**vertex`` (count, then ``id x y z dom``), ``**face`` (count, then 4 lines
    per face; line 1 = ``id nverts v1..vn``), ``**polyhedron`` (count, then
    ``id nfaces f1..fn`` per cell).
    """
    sec = _read_tess_sections(tess_path)

    vlines = sec.get("vertex", [])
    nv = int(vlines[0].split()[0])
    verts = np.empty((nv, 3), dtype=float)
    for k in range(nv):
        p = vlines[1 + k].split()
        verts[k] = (float(p[1]), float(p[2]), float(p[3]))

    flines = sec.get("face", [])
    nf = int(flines[0].split()[0])
    face_vertices: List[List[int]] = []
    i = 1
    for _ in range(nf):
        hdr = flines[i].split()
        nverts = int(hdr[1])
        face_vertices.append([int(x) - 1 for x in hdr[2 : 2 + nverts]])
        i += 4  # header + edges + face-eqn + trailing line

    plines = sec.get("polyhedron", [])
    nc = int(plines[0].split()[0])
    cell_faces: List[List[int]] = []
    for k in range(nc):
        p = plines[1 + k].split()
        nfaces = int(p[1])
        cell_faces.append([abs(int(x)) - 1 for x in p[2 : 2 + nfaces]])

    return verts, face_vertices, cell_faces


def compute_cell_geometry(
    df: pd.DataFrame,
    bbox: Sequence[float],
    weighted: bool = True,
    coord_cols: Sequence[str] = ("X", "Y", "Z"),
    radius_col: str = "GrainRadius",
    work_dir: Optional[str] = None,
    env: Optional[dict] = None,
    neper_bin: str = "neper",
    keep_files: bool = False,
) -> pd.DataFrame:
    """Tessellate ``df``'s grains inside ``bbox`` and return per-grain cell geometry.

    The tessellation is a (Laguerre when ``weighted``) Voronoi diagram of the exact
    measured centroids — ``-morphooptistop iter=0`` disables CVT relaxation so cells
    stay anchored at the grains. Cell *i* corresponds to input row *i* (the same
    seed-order assumption ``VoronoiMeshBuilder`` relies on for orientation mapping).

    Args:
        df: grain table with ``coord_cols`` (and ``radius_col`` when ``weighted``).
        bbox: domain ``[xlo, xhi, ylo, yhi, zlo, zhi]``. z should be the scan FOV
            window so that a cell touching the z-limit means the grain is truncated.
        weighted: Laguerre weight = effective grain volume ``(4/3)*pi*r**3``.
        work_dir: scratch dir for Neper I/O (a temp dir is used and removed if None).
        env: subprocess env (defaults to :func:`default_neper_env`).

    Returns:
        DataFrame indexed like ``df`` with columns ``Zmin, Zmax, Xc, Yc, Zc, Vol``.
    """
    n = len(df)
    cols = ["Zmin", "Zmax", "Xc", "Yc", "Zc", "Vol"]
    if n == 0:
        return pd.DataFrame(columns=cols)

    if env is None:
        env = default_neper_env()

    created_tmp = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="scan_tess_")
    os.makedirs(work_dir, exist_ok=True)

    try:
        coords = df[list(coord_cols)].to_numpy(dtype=float)
        # Run Neper from inside work_dir with dotless relative filenames: Neper
        # truncates the -o path at the first '.', so an absolute path through a
        # dotted directory (e.g. ~/.claude_tmp) would be mangled.
        np.savetxt(os.path.join(work_dir, "points.dat"), coords, fmt="%.12g")

        morpho_ini = "coo:file(points.dat)"
        if weighted:
            if radius_col not in df.columns:
                raise ValueError(
                    f"weighted tessellation requires a '{radius_col}' column."
                )
            r = df[radius_col].to_numpy(dtype=float)
            w = (4.0 / 3.0) * np.pi * (r**3)
            total = w.sum()
            if total > 0:
                w = w / total
            np.savetxt(os.path.join(work_dir, "weights.dat"), w, fmt="%.12g")
            morpho_ini += ",weight:file(weights.dat)"

        xlo, xhi, ylo, yhi, zlo, zhi = (float(v) for v in bbox)
        sx, sy, sz = xhi - xlo, yhi - ylo, zhi - zlo
        if sx <= 0 or sy <= 0 or sz <= 0:
            raise ValueError(f"Invalid tessellation bbox (non-positive extent): {bbox}")
        domain = f"cube({sx},{sy},{sz}):translate({xlo},{ylo},{zlo})"

        cmd = [
            neper_bin,
            "-T",
            "-n",
            str(n),
            "-id",
            "1",
            "-dim",
            "3",
            "-domain",
            domain,
            "-morpho",
            "voronoi",
            "-morphooptiini",
            morpho_ini,
            "-morphooptistop",
            "iter=0",
            "-o",
            "scan_tess",
            "-format",
            "tess",
            "-statcell",
            "x,y,z,vol",
        ]
        subprocess.run(
            cmd,
            check=True,
            env=env,
            cwd=work_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        out = os.path.join(work_dir, "scan_tess")
        verts, face_vertices, cell_faces = _parse_tess_cells(out + ".tess")
        if len(cell_faces) != n:
            raise RuntimeError(
                f"Neper produced {len(cell_faces)} cells for {n} grains "
                "(seed/cell ordering broke)."
            )

        zmin = np.empty(n, dtype=float)
        zmax = np.empty(n, dtype=float)
        for c, fids in enumerate(cell_faces):
            vids = set()
            for f in fids:
                vids.update(face_vertices[f])
            zc = verts[list(vids), 2]
            zmin[c] = zc.min()
            zmax[c] = zc.max()

        stat = np.loadtxt(out + ".stcell").reshape(n, 4)

        return pd.DataFrame(
            {
                "Zmin": zmin,
                "Zmax": zmax,
                "Xc": stat[:, 0],
                "Yc": stat[:, 1],
                "Zc": stat[:, 2],
                "Vol": stat[:, 3],
            },
            index=df.index,
        )
    finally:
        if created_tmp and not keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)
