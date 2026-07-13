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


@mcp.tool()
def run_cpfe(
    mesh_file: str,
    ori_file: str,
    moose_run_file: str,
    save_simulation_folder: Optional[str] = None,
    eeres_file: Optional[str] = None,
    parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    init_params: Optional[Dict[str, Any]] = None,
    ncore: int = 8,
    confirm: bool = False,
) -> dict:
    """Run a crystal-plasticity FE simulation with NEML2 v3 + MOOSE/PUMA
    (wraps `CPFESimulation`). Read `get_recommended_parameters('cpfe_simulation')`
    first. Heaviest step -- GPU-bound, minutes to hours; always a background job.

    Parameters
    ----------
    mesh_file : .msh (FF) or .e (NF) mesh.
    ori_file : per-grain/element orientations in NEML2 v3 MRP. Convert FF
        orientations.dat with orientation_helper.euler_to_mrp first.
    moose_run_file : path to your built `puma-opt` binary.
    save_simulation_folder : output dir (defaults under the MCP workdir).
    eeres_file : per-grain initial elastic strain CSV, or None for zero strain.
    init_params : overrides for CPFESimulation(...) -- element_order
        ('FIRST'|'SECOND'), dim, use_ff_initial_field.
    parameters : dict of {section: {kwargs}} passed to set_parameters. Sections:
        'material', 'simulation_parameters', 'boundary', 'grid_properties'. See
        the recipe for the keys of each. `boundary` needs bounding_box + bc;
        `grid_properties` needs number_of_elements + bounding_box (inset 0.0001).
    ncore : MPI ranks (== number of GPUs for a device list).

    Needs `puma-opt` (MOOSE/PUMA) and a working NEML2 v3 build.
    """
    from graintrace.run_cpfe_simulation import CPFESimulation

    if save_simulation_folder is None:
        save_simulation_folder = str(workdir() / "simulation")
    init = {
        "element_order": "SECOND", "dim": 3, "use_ff_initial_field": True,
        **(init_params or {}),
    }
    # Copy so we never mutate the caller's dict. GPU policy: if a GPU is
    # available and no device was specified, default to it (cuda:0).
    sections = {k: dict(v) for k, v in (parameters or {}).items()}
    if deps.gpu_available():
        sp = sections.setdefault("simulation_parameters", {})
        sp.setdefault("device", deps.default_cpfe_device())
    resolved = {
        "mesh_file": mesh_file, "ori_file": ori_file,
        "moose_run_file": moose_run_file,
        "save_simulation_folder": save_simulation_folder,
        "eeres_file": eeres_file, "init_params": init,
        "parameters": sections, "ncore": ncore,
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
        tool="run_cpfe", confirm=confirm, resolved_params=resolved,
        needs=["puma-opt", "neml2-aoti"], will_write=[save_simulation_folder],
        run=_run, background=True,
        notes="GPU-bound; can take a long time. Poll job_status for progress.",
    )
