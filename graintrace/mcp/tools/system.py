# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Introspection tools: dependency status, recipes, jobs, and outputs.

These are read-only and ungated (they run no graintrace computation), so they do
not take a ``confirm`` flag.
"""

from __future__ import annotations

from typing import List, Optional

from graintrace.mcp import deps, jobs, recipes, sample_meta
from graintrace.mcp.app import mcp, workdir


# ---- external stack status ---------------------------------------------------


@mcp.tool()
def dependency_status() -> dict:
    """Report which external tools (NEPER, MOOSE/PUMA, CUBIT/SCULPT, NEML2,
    pyzag, GMSH) are built and available on this machine.

    Call this early: it tells you which workflow segments can actually run here.
    A segment whose tools are missing will refuse to run with a build hint.
    """
    return deps.summary()


# ---- recommendation recipes --------------------------------------------------


@mcp.tool()
def list_recommended_recipes() -> List[str]:
    """List available recommendation recipes (vetted parameter presets per
    setup). Read one with `get_recommended_parameters` before proposing
    parameters for a run."""
    return recipes.list_recipes()


@mcp.tool()
def get_recommended_parameters(name: str) -> dict:
    """Return a recommendation recipe: recommended defaults plus prose guidance
    on how to choose parameters for a given setup.

    `name` is one of `list_recommended_recipes()` (e.g. 'ff_reconstruction',
    'stitching', 'cpfe_simulation'). Use its `defaults` to prefill a tool's
    `params`, and relay its guidance to the user.
    """
    rec = recipes.get_recipe(name)
    if rec is None:
        return {
            "error": f"no recipe named '{name}'",
            "available": recipes.list_recipes(),
        }
    return rec


@mcp.tool()
def inspect_experiment(path: str) -> dict:
    """Inspect a raw HEDM CSV before running anything, and list the experiment
    metadata that a CSV CANNOT provide (sample dimensions, loading conditions,
    scan geometry, units) so you can ask the user for it.

    ALWAYS call this first when handed a raw grain CSV. It returns the columns
    present, a suggested bounding box from the coordinate ranges, an Euler
    unit *guess* (note: graintrace does NOT auto-detect units, confirm it),
    whether residual-strain columns exist, and a must-confirm checklist. Then
    either collect a sample.json or confirm these values with the user before
    calling ff_reconstruct / stitch_scans / run_cpfe with confirm=true.
    """
    return sample_meta.checklist(csv_path=path)


@mcp.resource("recipe://{name}")
def recipe_resource(name: str) -> str:
    """Serve a recommendation recipe as a readable Markdown resource."""
    rec = recipes.get_recipe(name)
    if rec is None:
        return (
            f"# Unknown recipe '{name}'\nAvailable: {', '.join(recipes.list_recipes())}"
        )
    return rec["markdown"]


# ---- background jobs ---------------------------------------------------------


@mcp.tool()
def job_status(job_id: str) -> dict:
    """Get the status/result of a background job started by a heavy tool
    (CPFE, reconstruction, meshing, calibration)."""
    job = jobs.get(job_id)
    if job is None:
        return {
            "error": f"no job '{job_id}'",
            "known_jobs": [j["job_id"] for j in jobs.all_jobs()],
        }
    snap = job.snapshot()
    snap["recent_log"] = jobs.tail(job_id, 40)
    return snap


@mcp.tool()
def list_jobs() -> List[dict]:
    """List all background jobs from this server session, newest first."""
    return jobs.all_jobs()


@mcp.tool()
def job_log(job_id: str, lines: int = 100) -> str:
    """Return the last `lines` of a background job's captured stdout/stderr."""
    return jobs.tail(job_id, lines)


# ---- outputs -----------------------------------------------------------------


@mcp.tool()
def list_outputs(subdir: Optional[str] = None) -> dict:
    """List files produced under the MCP workdir (where all tool outputs land).

    Pass `subdir` to scope to one run's output folder. Useful for finding the
    meshes, CSVs, plots, and VTKs a step wrote so you can report them back.
    """
    root = workdir()
    base = (root / subdir).resolve() if subdir else root
    if not str(base).startswith(str(root)):
        return {"error": "subdir escapes the workdir"}
    if not base.exists():
        return {
            "workdir": str(root),
            "listing_of": str(base),
            "files": [],
            "note": "does not exist",
        }
    files = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            files.append({"path": str(p), "size_bytes": p.stat().st_size})
    return {
        "workdir": str(root),
        "listing_of": str(base),
        "n_files": len(files),
        "files": files[:500],
    }
