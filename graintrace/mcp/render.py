# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Off-screen 3D rendering to PNG via PyVista (VTK/EGL): no ParaView, no display.

Used for the spatial views the demo/MCP need: grains/reconstruction, CPFE field on
the probe grid, and REI rare-cluster regions. Curves/distributions stay in matplotlib
(graintrace already ships those). Rendering is forced off-screen; on this headless box
VTK falls back to EGL/OSMesa (verified). Exodus (.e) is readable but the policy is to
point users at ParaView for interactive mesh inspection rather than auto-render it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# Must be set before importing pyvista so it starts off-screen.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")


def _plotter():
    """Import pyvista off-screen (after the env var is set) and return the module."""
    # pyvista must be imported only after PYVISTA_OFF_SCREEN is set (see above).
    import pyvista as pv  # pylint: disable=import-outside-toplevel

    pv.OFF_SCREEN = True
    return pv


def list_fields(path: str) -> dict:
    """Return the point/cell array names available in a VTK/mesh file."""
    pv = _plotter()
    mesh = pv.read(path)
    return {
        "point_arrays": list(getattr(mesh, "point_data", {}).keys()),
        "cell_arrays": list(getattr(mesh, "cell_data", {}).keys()),
    }


def _pick_scalar(mesh, field: Optional[str]):
    """Resolve a usable scalar name; if `field` names a multi-component array,
    a magnitude is added. Returns the scalar name to color by (or None)."""
    if field is None:
        return None
    for store in (mesh.point_data, mesh.cell_data):
        if field in store:
            arr = store[field]
            if arr.ndim > 1 and arr.shape[1] > 1:
                mag = f"{field}_magnitude"
                store[mag] = np.linalg.norm(arr, axis=1)
                return mag
            return field
    # component columns like nye_tensor_11..33 -> build a norm
    comp = [
        c
        for c in list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        if c.startswith(field)
    ]
    if comp:
        store = mesh.point_data if comp[0] in mesh.point_data else mesh.cell_data
        stacked = np.stack([store[c] for c in comp], axis=1)
        name = f"{field}_norm"
        store[name] = np.linalg.norm(stacked, axis=1)
        return name
    return None


def render_vtk(
    path: str,
    out_png: str,
    field: Optional[str] = None,
    title: Optional[str] = None,
    threshold_rare: bool = False,
    window_size: tuple = (1100, 900),
    cmap: str = "viridis",
) -> str:
    """Render a VTK/VTU/mesh file to a PNG off-screen.

    field: array to color by (multi-component -> magnitude; a base like
      'nye_tensor' -> norm over its *_11..33 columns). None -> solid surface.
    threshold_rare: if a 'rare_cluster_id' array is present, show the full field
      faintly + the rare blocks (id>=2) opaque and highlighted.
    Returns the PNG path.
    """
    pv = _plotter()
    mesh = pv.read(path)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    p = pv.Plotter(off_screen=True, window_size=list(window_size))
    scalar = _pick_scalar(mesh, field)

    if threshold_rare and (
        "rare_cluster_id" in mesh.point_data or "rare_cluster_id" in mesh.cell_data
    ):
        p.add_mesh(mesh, color="lightgray", opacity=0.08)
        try:
            rare = mesh.threshold(1.5, scalars="rare_cluster_id")
            p.add_mesh(
                rare,
                scalars="rare_cluster_id",
                cmap="turbo",
                show_scalar_bar=True,
                opacity=1.0,
            )
        # Best-effort: fall back to plain coloring if thresholding fails.
        except Exception:  # pylint: disable=broad-exception-caught
            p.add_mesh(mesh, scalars="rare_cluster_id", cmap="turbo")
    elif scalar is not None:
        p.add_mesh(mesh, scalars=scalar, cmap=cmap, show_scalar_bar=True)
    else:
        # solid: try RGB arrays (IPF) if present, else a neutral color
        rgb_name = next(
            (
                n
                for n in ("RGB", "rgb", "ipf_rgb", "colors")
                if n in mesh.point_data or n in mesh.cell_data
            ),
            None,
        )
        if rgb_name is not None:
            p.add_mesh(mesh, scalars=rgb_name, rgb=True)
        else:
            p.add_mesh(mesh, color="lightsteelblue", show_edges=False)

    if title:
        p.add_text(title, font_size=11)
    p.add_axes()
    p.camera_position = "iso"
    p.screenshot(out_png)
    p.close()
    return out_png


def render_grains_ipf(
    vtk_or_mesh_with_rgb: str, out_png: str, title: Optional[str] = None
) -> str:
    """Render a mesh that already carries per-block RGB (from IPFProcessor)."""
    return render_vtk(vtk_or_mesh_with_rgb, out_png, field=None, title=title)
