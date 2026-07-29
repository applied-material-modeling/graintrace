# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Tool: render a VTK/mesh field to a PNG (pyvista off-screen) for chat display."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from graintrace.mcp.app import mcp, workdir


def _img_and_info(png_path: str, info: dict):
    """Return the rendered PNG inline (so it displays in chat / the model can see
    and analyze it) PLUS a JSON text with the path + metadata. Falls back to just
    the dict if inline images aren't supported by the client."""
    import json
    try:
        from mcp.server.fastmcp import Image
        return [Image(path=png_path), json.dumps(info)]
    except Exception:
        return info


@mcp.tool()
def visualize(
    path: str,
    field: Optional[str] = None,
    out_png: Optional[str] = None,
    rare_only: bool = False,
):
    """Render a VTK/VTU/mesh file to a PNG off-screen (pyvista/EGL -- no ParaView,
    no display) so grains / reconstructions / fields / REI hotspots can be shown.

    Parameters
    ----------
    path : a .vtk/.vtu/.msh file (e.g. a reconstruction VTK, a grid_out field, or
        the REI rare-cluster VTK). For an Exodus (.e) mesh, prefer opening it in
        ParaView -- this renders a static snapshot only.
    field : array to color by. A multi-component array uses its magnitude; a base
        name like 'nye_tensor' uses the norm over its *_11..33 columns. None ->
        solid surface (uses per-block RGB/IPF colors if present).
    out_png : output PNG path (defaults under the MCP workdir /plots).
    rare_only : if the file has a 'rare_cluster_id' array, dim the background and
        highlight only the rare blocks.

    Returns the PNG path (under the MCP workdir) and the arrays available to color
    by. Open the PNG (or use list_outputs) to view it.
    """
    from graintrace.mcp import render

    if out_png is None:
        out_png = str(workdir() / "plots" / (Path(path).stem + ".png"))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    try:
        available = render.list_fields(path)
    except Exception as exc:
        return {"error": f"could not read {path}: {exc}"}

    try:
        render.render_vtk(path, out_png, field=field, threshold_rare=rare_only,
                          title=field or Path(path).name)
    except Exception as exc:
        return {"error": f"render failed: {exc}", "available_fields": available}

    return _img_and_info(out_png, {
        "png_path": out_png,
        "size_bytes": os.path.getsize(out_png) if os.path.exists(out_png) else 0,
        "available_fields": available,
        "note": "PNG shown inline + written under the MCP workdir.",
    })


_COORD_SETS = [("X", "Y", "Z"), ("x", "y", "z")]


@mcp.tool()
def plot_centroids(
    csv_path: Optional[str] = None,
    csv_paths: Optional[list] = None,
    color_by: Optional[str] = None,
    out_png: Optional[str] = None,
    title: Optional[str] = None,
    point_size: float = 6.0,
):
    """Scatter grain CENTROIDS from a grain CSV (xy / xz / yz projections) to a PNG
    -- for pre/post-stitch centroid figures and quick microstructure QC. Uses
    matplotlib (no VTK); this is the CSV counterpart to `visualize` (which needs a
    mesh/VTK).

    Parameters
    ----------
    csv_path : one grain CSV with X,Y,Z (or x,y,z). e.g. a stitched output.
    csv_paths : OR a list of per-scan CSVs to overlay, colored by scan index
        (the pre-stitch view). Mutually exclusive with csv_path.
    color_by : column to color points by (e.g. 'ScanID', 'GrainRadius', 'Eul0').
        Ignored when csv_paths is given (colored by scan instead).
    out_png : output PNG path (defaults under the MCP workdir /plots).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if bool(csv_path) == bool(csv_paths):
        return {"error": "provide exactly one of csv_path or csv_paths"}

    # assemble a dataframe + a color series
    frames, labels = [], []
    srcs = [csv_path] if csv_path else list(csv_paths)
    for i, s in enumerate(srcs):
        try:
            d = pd.read_csv(s)
        except Exception as exc:
            return {"error": f"could not read {s}: {exc}"}
        d = d.copy()
        d["_scan"] = i
        frames.append(d)
        labels.append(os.path.basename(str(s)))
    df = pd.concat(frames, ignore_index=True)

    coord = next((c for c in _COORD_SETS if set(c) <= set(df.columns)), None)
    if coord is None:
        return {"error": f"no coordinate columns (X,Y,Z or x,y,z) in {labels}"}
    cx, cy, cz = coord

    if csv_paths:
        cvals, clabel = df["_scan"], "scan index"
    elif color_by and color_by in df.columns:
        cvals, clabel = df[color_by], color_by
    else:
        cvals, clabel = None, None

    if out_png is None:
        stem = "centroids_" + (Path(srcs[0]).stem if csv_path else "prestitch")
        out_png = str(workdir() / "plots" / f"{stem}.png")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    pairs = [(cx, cy), (cx, cz), (cy, cz)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    sc = None
    for ax, (a, b) in zip(axes, pairs):
        sc = ax.scatter(df[a], df[b], c=cvals, s=point_size, cmap="tab10"
                        if csv_paths else "viridis", alpha=0.8)
        ax.set_xlabel(a); ax.set_ylabel(b); ax.set_aspect("equal", "box")
        ax.set_title(f"{a}-{b}")
    if cvals is not None and sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, label=clabel)
    fig.suptitle(title or (f"Centroids ({len(df)} grains)"
                           + (f" by {clabel}" if clabel else "")))
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return _img_and_info(out_png, {
        "png_path": out_png, "n_points": int(len(df)),
        "coords": coord, "colored_by": clabel, "sources": labels,
        "note": "PNG shown inline + written under the MCP workdir."})
