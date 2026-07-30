# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Tool: run a MOOSE/PUMA crystal-plasticity FE simulation (CPFESimulation)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from graintrace.mcp import deps
from graintrace.mcp.app import mcp, workdir
from graintrace.mcp.confirm import gate


_AXIS_IDX = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}


@mcp.tool()
def run_cpfe(
    mesh_file: str,
    ori_file: str,
    moose_run_file: Optional[str] = None,
    save_simulation_folder: Optional[str] = None,
    eeres_file: Optional[str] = None,
    sample_json: Optional[str] = None,
    bounding_box: Optional[List[float]] = None,
    total_strain: Optional[float] = None,
    loaded_axis: str = "z",
    grid_elements: Optional[List[int]] = None,
    parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    init_params: Optional[Dict[str, Any]] = None,
    ncore: int = 4,
    confirm: bool = False,
) -> dict:
    """Run a crystal-plasticity FE simulation with NEML2 v3 + MOOSE/PUMA
    (wraps `CPFESimulation`). Read `get_recommended_parameters('cpfe_simulation')`
    first. Heaviest step: GPU-bound, minutes to hours; always a background job.

    LOADING CONDITIONS ARE NOT IN THE MESH. Supply `bounding_box` (sample
    dimensions) + `total_strain` (+ `loaded_axis`), or a `sample_json`, and this
    tool builds the `boundary` (bc: uniaxial displacement = total_strain*axis_extent,
    other faces stress-free) and `grid_properties` (probe grid inset by 1e-4)
    sections for you. Without them it returns 'needs_input'; ask the user rather
    than run on the silent unit-cube default.

    Parameters
    ----------
    mesh_file : .msh (FF) or .e (NF) mesh.
    ori_file : per-grain/element orientations in NEML2 v3 MRP.
    moose_run_file : path to your built `puma-opt` binary.
    bounding_box : [xlo,xhi,ylo,yhi,zlo,zhi] um (sample dimensions).
    total_strain : applied macroscopic strain along `loaded_axis` (e.g. 0.002).
    loaded_axis : 'x'|'y'|'z' (default z).
    grid_elements : probe grid [nx,ny,nz] (default [20,20,20]).
    sample_json : experiment metadata supplying bounding_box + loading.
    parameters : advanced; explicit {section: {kwargs}} for set_parameters
        (material / simulation_parameters / boundary / grid_properties). Anything
        you provide here overrides the auto-built sections.
    init_params : CPFESimulation overrides (element_order, dim, use_ff_initial_field).
    ncore : MPI ranks (== number of GPUs for a device list).

    Needs `puma-opt` (MOOSE/PUMA) and a working NEML2 v3 build.
    """
    # Lazy: heavy graintrace submodule + mcp helpers (pandas), imported on run.
    # pylint: disable=import-outside-toplevel
    from graintrace.run_cpfe_simulation import CPFESimulation
    from graintrace.mcp import sample_meta, tool_paths

    if moose_run_file is None:
        moose_run_file = tool_paths.puma_opt()  # from tools.json if configured
    if save_simulation_folder is None:
        save_simulation_folder = str(workdir() / "simulation")
    init = {
        "element_order": "SECOND",
        "dim": 3,
        "use_ff_initial_field": True,
        **(init_params or {}),
    }
    smeta = sample_meta.resolve_sample(sample_json)
    if bounding_box is None:
        bounding_box = smeta.get("bounding_box")
    if total_strain is None:
        total_strain = smeta.get("total_strain")
    loaded_axis = (smeta.get("loaded_axis") or loaded_axis or "z").lower()
    grid_elements = grid_elements or [20, 20, 20]

    # Copy so we never mutate the caller's dict.
    sections = {k: dict(v) for k, v in (parameters or {}).items()}

    # Auto-build boundary + grid_properties from sample dimensions + loading,
    # unless the caller supplied them explicitly.
    missing = []
    if "boundary" not in sections or "grid_properties" not in sections:
        if bounding_box is None:
            missing.append(
                "bounding_box (sample dimensions [xlo,xhi,ylo,yhi,zlo,zhi] um)"
            )
        if "boundary" not in sections and total_strain is None:
            missing.append(
                "loading: total_strain (+ loaded_axis) or an explicit boundary bc"
            )
    if not missing:
        lo_i, hi_i = _AXIS_IDX.get(loaded_axis, (4, 5))
        if "boundary" not in sections:
            displace = float(total_strain) * (bounding_box[hi_i] - bounding_box[lo_i])
            bc = {
                a: {"negative": "stress_free", "positive": "stress_free"}
                for a in ("x", "y", "z")
            }
            bc[loaded_axis] = {"negative": 0, "positive": displace}
            sections["boundary"] = {"bounding_box": bounding_box, "bc": bc}
        if "grid_properties" not in sections:
            grid_bb = list(bounding_box)
            for i in (0, 2, 4):
                grid_bb[i] += 1e-4
            for i in (1, 3, 5):
                grid_bb[i] -= 1e-4
            sections["grid_properties"] = {
                "number_of_elements": grid_elements,
                "bounding_box": grid_bb,
            }

    # GPU policy: default the device to the GPU when available.
    if deps.gpu_available():
        sp = sections.setdefault("simulation_parameters", {})
        sp.setdefault("device", deps.default_cpfe_device())

    suggestions = {}
    if missing:
        suggestions = {
            "loaded_axis": loaded_axis,
            "hint": "Pass bounding_box + total_strain (or a sample_json).",
        }

    resolved = {
        "mesh_file": mesh_file,
        "ori_file": ori_file,
        "moose_run_file": moose_run_file,
        "save_simulation_folder": save_simulation_folder,
        "eeres_file": eeres_file,
        "init_params": init,
        "bounding_box": bounding_box,
        "total_strain": total_strain,
        "loaded_axis": loaded_axis,
        "parameters": sections,
        "ncore": ncore,
    }

    def _run():
        sim = CPFESimulation(
            mesh_file=mesh_file,
            save_simulation_folder=save_simulation_folder,
            moose_run_file=moose_run_file,
            eeres_file=eeres_file,
            ori_file=ori_file,
            **init,
        )
        for section, kwargs in sections.items():
            sim.set_parameters(section, **kwargs)
        sim.run(ncore=ncore)
        return {"save_simulation_folder": save_simulation_folder}

    return gate(
        tool="run_cpfe",
        confirm=confirm,
        resolved_params=resolved,
        needs=["puma-opt", "neml2-aoti"],
        will_write=[save_simulation_folder],
        run=_run,
        background=True,
        notes="GPU-bound; can take a long time. Poll job_status for progress.",
        missing_required=missing,
        suggestions=suggestions,
    )
