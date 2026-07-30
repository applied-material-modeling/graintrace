# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""graintrace MCP server.

Exposes the graintrace HEDM -> CPFE -> REI workflow as Model Context Protocol
(MCP) tools so that any MCP-capable chat client (Claude Desktop, Claude Code, or
any other MCP client) can drive graintrace. The server itself contains no LLM;
the connecting client supplies the reasoning model and the user's key. See
``graintrace/mcp/README.md`` for setup.

Run with::

    graintrace-mcp                 # stdio transport (Claude Desktop / Code)
    graintrace-mcp --http          # streamable-http transport on :8000

Design invariants:
  * No graintrace computation runs until the caller passes ``confirm=True``.
    Every executing tool first returns a *preview* (resolved parameters, the
    external tools it needs, and the files it will write).
  * Missing external tools (NEPER / MOOSE-PUMA / CUBIT-SCULPT / NEML2) are
    reported as a clear "not built yet" message when that code path is reached,
    never as a raw traceback.
"""

from __future__ import annotations

from graintrace.mcp.app import mcp

__all__ = ["mcp"]
