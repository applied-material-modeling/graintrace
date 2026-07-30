# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Inspect a raw HEDM CSV and resolve the experiment metadata a CSV can't carry.

A raw FF-HEDM grain CSV holds only per-grain centroids, radius, Euler angles, and
(optionally) elastic strain. It does NOT carry sample dimensions, loading
conditions, scan geometry, or units, and graintrace silently defaults those
(no unit auto-detection exists; CPFE even defaults to a unit-cube domain). This
module lets the MCP tools (1) inspect what IS present and infer suggestions, and
(2) load a ``sample.json`` that supplies the rest, so a tool can tell the caller
exactly what to confirm before running.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_COORD_SETS = [("X", "Y", "Z"), ("x", "y", "z")]
_EULER_SETS = [
    ("Eul0", "Eul1", "Eul2"),
    ("Eul1", "Eul2", "Eul3"),
    ("phi1", "Phi", "phi2"),
]
_STRAIN_PREFIXES = ("eKen", "eFab")
_TWO_PI = 2.0 * math.pi


def _first_present(cols, sets):
    cset = set(cols)
    for s in sets:
        if set(s) <= cset:
            return list(s)
    return None


def inspect_csv(path: str, pad_frac: float = 0.02) -> Dict[str, Any]:
    """Report columns, a suggested bounding box (coord range + pad), an Euler
    unit *guess* (|angle|>2pi => degrees), and strain-column presence."""
    df = pd.read_csv(path)
    cols = list(df.columns)
    info: Dict[str, Any] = {"path": str(path), "n_rows": int(len(df)), "columns": cols}

    coord = _first_present(cols, _COORD_SETS)
    info["coord_columns"] = coord
    if coord:
        mins = [float(df[c].min()) for c in coord]
        maxs = [float(df[c].max()) for c in coord]
        pads = [(hi - lo) * pad_frac or 1.0 for lo, hi in zip(mins, maxs)]
        bbox = []
        for lo, hi, p in zip(mins, maxs, pads):
            bbox.extend([round(lo - p, 4), round(hi + p, 4)])
        info["coord_ranges"] = {c: [lo, hi] for c, lo, hi in zip(coord, mins, maxs)}
        info["suggested_bounding_box"] = bbox  # [xlo,xhi,ylo,yhi,zlo,zhi]

    euler = _first_present(cols, _EULER_SETS)
    info["euler_columns"] = euler
    if euler:
        maxabs = float(df[euler].abs().to_numpy().max())
        info["euler_abs_max"] = round(maxabs, 4)
        info["orientation_units_guess"] = "degrees" if maxabs > _TWO_PI else "radians"
        info["orientation_units_confidence"] = (
            "high (values exceed 2pi -> degrees)"
            if maxabs > _TWO_PI
            else "LOW (all |angles| <= 2pi; could be radians OR small-angle degrees; CONFIRM)"
        )

    strain_prefix = next(
        (p for p in _STRAIN_PREFIXES if f"{p}11" in cols or f"{p}_11" in cols), None
    )
    info["strain_columns_present"] = strain_prefix is not None
    info["strain_prefix"] = strain_prefix
    info["has_grain_radius"] = "GrainRadius" in cols
    return info


# ---- sample.json ------------------------------------------------------------


def load_sample_json(path: str) -> Dict[str, Any]:
    """Parse a sample.json file into a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_sample(sample_json: Optional[str]) -> Dict[str, Any]:
    """Flatten a sample.json into the fields the tools need. Missing -> absent key."""
    if not sample_json:
        return {}
    s = load_sample_json(sample_json)
    out: Dict[str, Any] = {}
    samp = s.get("sample", {})
    units = samp.get("units", {})
    if "bounding_box_um" in samp:
        out["bounding_box"] = samp["bounding_box_um"]
    if "orientation" in units:
        out["orientation_units"] = units["orientation"]  # 'degrees'/'radians'
        out["unit"] = "deg" if str(units["orientation"]).startswith("deg") else "rad"
    if "orientation_convention" in units:
        out["orientation_convention"] = units["orientation_convention"]
    if "strain" in units:
        out["strain_unit"] = units["strain"]
    if "symmetry" in units:
        out["symmetry"] = units["symmetry"]
    sg = s.get("scan_geometry", {})
    for k in ("zlo", "zhi", "overlap_fraction", "n_scans"):
        if k in sg:
            out[k] = sg[k]
    ld = s.get("loading", {})
    for k in ("total_strain", "loaded_axis", "bc", "mode"):
        if k in ld:
            out[k] = ld[k]
    esc = s.get("elastic_strain_columns", {})
    if "prefix" in esc:
        out["elastic_prefix"] = esc["prefix"]
    return out


# ---- per-tool required non-inferrable inputs --------------------------------

# Human-readable checklist of what a CSV cannot provide, per tool.
REQUIRED: Dict[str, List[str]] = {
    "stitch_scans": ["zlo", "zhi", "overlap_fraction", "orientation_units"],
    "ff_reconstruct": ["bounding_box", "unit (orientation deg/rad)"],
    "run_cpfe": [
        "bounding_box (sample dimensions)",
        "loading (total_strain or explicit bc)",
    ],
}


def required_inputs_for(tool: str) -> List[str]:
    """Return the non-inferrable inputs a given tool requires."""
    return REQUIRED.get(tool, [])


def checklist(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """What the user must confirm for a blind file, with inferred suggestions."""
    out: Dict[str, Any] = {
        "must_confirm": {
            "sample_dimensions": "bounding box [xlo,xhi,ylo,yhi,zlo,zhi] in um "
            "(NOT in the CSV; CPFE otherwise defaults to a unit cube)",
            "loading_conditions": "total applied strain (or explicit BCs) + loaded axis "
            "(NOT in the CSV)",
            "scan_geometry": "z-range (zlo,zhi) and scan overlap fraction for stitching",
            "units": "orientation deg/rad and strain unit (microstrain vs strain); "
            "graintrace does NOT auto-detect these",
        },
        "how_to_supply": "Pass a sample.json (see the 'experiment_metadata' recipe) or "
        "provide these values explicitly after confirming with the user.",
    }
    if csv_path:
        try:
            out["inspection"] = inspect_csv(csv_path)
        # Best-effort: report any inspection failure instead of raising.
        except Exception as exc:  # pylint: disable=broad-exception-caught
            out["inspection_error"] = str(exc)
    return out
