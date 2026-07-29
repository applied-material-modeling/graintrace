# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Tool: stitch overlapping FF-HEDM z-scan layers into one grain set."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from graintrace.mcp.app import mcp, workdir
from graintrace.mcp.confirm import gate

# Operative typed defaults (recipe stitching.md holds the prose guidance).
_DEFAULTS: Dict[str, Any] = {
    "position_tolerance": 50.0,
    "orientation_tolerance": 5.0,
    "radius_tolerance": -1.0,
    "weights": {"pos": 0.1, "ori": 1.0, "rad": 0.0},
    "min_neighbors": 5,
    "orientation_convention": "bunge",
    "orientation_units": "degrees",
    "symmetry": "432",
    "refine_extents": False,
}


@mcp.tool()
def stitch_scans(
    scan_files: List[str],
    zlo: Optional[float] = None,
    zhi: Optional[float] = None,
    overlap_fraction: Optional[float] = None,
    output_csv: Optional[str] = None,
    sample_json: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> dict:
    """Merge overlapping FF-HEDM z-scan layers into one consistent grain set
    (wraps `RegionBaseStitching`).

    Read `get_recommended_parameters('stitching')` first. Call once with
    confirm=false to preview the resolved parameters, then again with
    confirm=true after the user approves.

    Parameters
    ----------
    scan_files : list of per-layer CSV paths, with Z already shifted per layer.
    zlo, zhi : z-range of the stitched volume (micrometers).
    overlap_fraction : scan overlap as a fraction (e.g. 0.2 for 20%).
    output_csv : output path (defaults under the MCP workdir).
    params : overrides for any of: position_tolerance, orientation_tolerance,
        radius_tolerance, weights, min_neighbors, orientation_convention,
        orientation_units ('degrees'|'radians'), symmetry, refine_extents,
        tess_weighted, update_centroid, xy_bounding_box. NOTE: if
        orientation_units='radians', pass orientation_tolerance in radians.

    Needs NEPER only when refine_extents=true; otherwise pure Python.
    """
    from graintrace.hedm_stitching_techniques.region_base_stitching import (
        RegionBaseStitching,
    )
    from graintrace.mcp import sample_meta

    smeta = sample_meta.resolve_sample(sample_json)
    if zlo is None:
        zlo = smeta.get("zlo")
    if zhi is None:
        zhi = smeta.get("zhi")
    if overlap_fraction is None:
        overlap_fraction = smeta.get("overlap_fraction")

    p = {**_DEFAULTS, **(params or {})}
    if "orientation_units" in smeta:
        p["orientation_units"] = smeta["orientation_units"]
    if "symmetry" in smeta:
        p["symmetry"] = smeta["symmetry"]
    if "orientation_convention" in smeta:
        p["orientation_convention"] = smeta["orientation_convention"]

    # Scan geometry is NOT in the CSVs -- require it (directly or via sample_json).
    missing = []
    if zlo is None or zhi is None:
        missing.append("z-range (zlo, zhi) of the stitched volume, in um")
    if overlap_fraction is None:
        missing.append("overlap_fraction between adjacent scans (e.g. 0.25 for 25%)")

    if output_csv is None:
        output_csv = str(workdir() / "stitching" / "stitched_output.csv")

    resolved = {
        "scan_files": scan_files,
        "output_csv": output_csv,
        "zlo": zlo,
        "zhi": zhi,
        "overlap_fraction": overlap_fraction,
        **p,
    }
    needs = ["neper"] if p.get("refine_extents") else []

    def _run():
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        stitcher = RegionBaseStitching(
            scan_files=scan_files,
            output_csv=output_csv,
            position_tolerance=p["position_tolerance"],
            orientation_tolerance=p["orientation_tolerance"],
            radius_tolerance=p["radius_tolerance"],
            weights=p["weights"],
            min_neighbors=p["min_neighbors"],
            orientation_convention=p["orientation_convention"],
            orientation_units=p["orientation_units"],
            symmetry=p["symmetry"],
            output_column=p.get("output_column"),
            refine_extents=p["refine_extents"],
            tess_weighted=p.get("tess_weighted", True),
            update_centroid=p.get("update_centroid", False),
            xy_bounding_box=p.get("xy_bounding_box"),
        )
        result = stitcher.run(zlo=zlo, zhi=zhi, overlap_fraction=overlap_fraction)

        # RegionBaseStitching does NOT carry the residual elastic-strain tensor
        # through, so downstream ff_reconstruct (which reads eKen*/eFab*) would
        # fail. Re-attach per-grain strain from the original scans by nearest
        # centroid, exactly like the demo driver does.
        import pandas as pd
        reattached = None
        try:
            df = pd.read_csv(output_csv)
            scan0 = pd.read_csv(scan_files[0], nrows=1)
            for pref in ("eKen", "eFab"):
                scols = [f"{pref}{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
                if set(scols).issubset(scan0.columns) and not set(scols).issubset(df.columns):
                    from scipy.spatial import cKDTree
                    coord = next((c for c in (["X", "Y", "Z"], ["x", "y", "z"])
                                  if set(c) <= set(df.columns)), None)
                    if coord is None:
                        break
                    allscan = pd.concat([pd.read_csv(s) for s in scan_files],
                                        ignore_index=True)
                    tree = cKDTree(allscan[coord].to_numpy())
                    _, idx = tree.query(df[coord].to_numpy())
                    df[scols] = allscan[scols].to_numpy()[idx]
                    df.to_csv(output_csv, index=False)
                    reattached = pref
                    break
        except Exception as exc:
            reattached = f"failed: {exc}"

        n = None
        try:
            n = len(getattr(result, "grains", result))
        except Exception:
            try:
                n = len(pd.read_csv(output_csv))
            except Exception:
                pass
        return {"output_csv": output_csv, "n_stitched_grains": n,
                "residual_strain_reattached": reattached}

    return gate(
        tool="stitch_scans",
        confirm=confirm,
        resolved_params=resolved,
        needs=needs,
        will_write=[output_csv],
        run=_run,
        background=False,
        notes="Pure-Python unless refine_extents=true (then NEPER is used).",
        missing_required=missing,
    )
