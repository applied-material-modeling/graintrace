# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Shared FastMCP application instance and small cross-tool helpers.

Kept separate from ``server.py`` so tool modules can ``from graintrace.mcp.app
import mcp`` and register with ``@mcp.tool()`` without importing the runner
(avoids a circular import: ``server`` imports the tool modules to trigger
registration, the tool modules import only this file).
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# The one shared server instance. Instructions are surfaced to the client model
# so it knows the confirm-before-run contract without being told each time.
mcp = FastMCP(
    "graintrace",
    instructions=(
        "graintrace drives the HEDM -> CPFE -> rare-event workflow "
        "(grain stitching, FF/NF/EBSD microstructure reconstruction, crystal-"
        "plasticity material calibration, MOOSE/PUMA CPFE simulation, and "
        "rare-event identification).\n\n"
        "Contract you MUST follow:\n"
        "1. Tools that run computation take a `params` dict and a `confirm` "
        "flag. ALWAYS call first with confirm=false to get a preview, show the "
        "resolved parameters and required external tools to the user, and only "
        "call again with confirm=true after they explicitly approve.\n"
        "2. Before proposing parameters, read the matching recipe via "
        "`get_recommended_parameters` (or the `recipe://` resources) -- they "
        "hold vetted defaults per setup.\n"
        "3. Heavy runs (CPFE, meshing, reconstruction) return a job id; poll "
        "`job_status` for progress instead of blocking.\n"
        "4. If a required external tool (NEPER, MOOSE/PUMA, CUBIT/SCULPT, "
        "NEML2) is not built, the tool says so plainly -- relay that to the "
        "user; do not retry.\n"
        "5. GPU policy: if a GPU is available (check `dependency_status` -> "
        "`gpu`), ALWAYS use it. Set the CPFE device to 'cuda:0' (or a "
        "space-separated list like 'cuda:0 cuda:1' for multi-GPU) and the "
        "calibration device to 'cuda'. Only fall back to 'cpu' when no GPU is "
        "present. Tools that pick the device themselves already default to GPU "
        "when one is available."
    ),
)


def workdir() -> Path:
    """Root directory for MCP-created outputs.

    Defaults to ``$GRAINTRACE_MCP_WORKDIR`` or ``./graintrace_mcp_out`` under the
    current working directory. Created on demand. All tool outputs land here so a
    chat session's artifacts are easy to find and serve back as resources.
    """
    root = Path(os.environ.get("GRAINTRACE_MCP_WORKDIR", "graintrace_mcp_out"))
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root
