# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Compatibility layer over the ``mcp`` Python SDK, versions 1.x and 2.x.

``mcp`` 2.0 renamed the high-level server ``mcp.server.fastmcp.FastMCP`` to
``mcp.server.mcpserver.MCPServer`` and moved the HTTP host/port from
``server.settings`` to ``run()`` keyword arguments. graintrace imports the server
class and ``Image`` from here, so it runs on either major version.
"""

from __future__ import annotations

try:  # mcp >= 2
    from mcp.server.mcpserver import Image, MCPServer

    MCP_SDK_MAJOR = 2
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer
    from mcp.server.fastmcp import Image

    MCP_SDK_MAJOR = 1

__all__ = ["Image", "MCPServer", "MCP_SDK_MAJOR", "run_streamable_http"]


def run_streamable_http(server, host: str, port: int) -> None:
    """Serve ``server`` over streamable-http on ``host:port`` (mcp 1.x or 2.x).

    Args:
        server: the ``MCPServer`` / ``FastMCP`` instance.
        host: bind host.
        port: bind port.
    """
    if MCP_SDK_MAJOR >= 2:
        server.run(transport="streamable-http", host=host, port=port)
    else:
        server.settings.host = host
        server.settings.port = port
        server.run(transport="streamable-http")
