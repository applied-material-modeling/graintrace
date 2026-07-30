# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Runner for the graintrace MCP server.

Importing the tool modules below registers every tool/resource on the shared
FastMCP instance. ``main()`` is the ``graintrace-mcp`` console entry point.

    graintrace-mcp                 # stdio (Claude Desktop / Claude Code / mcpo)
    graintrace-mcp --http          # streamable-http on 127.0.0.1:8000
    graintrace-mcp --http --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse

from graintrace.mcp.app import mcp

# Register tools by import side effect. Order is cosmetic.
# pylint: disable=unused-import
from graintrace.mcp.tools import system  # noqa: F401,E402
from graintrace.mcp.tools import stitching  # noqa: F401,E402
from graintrace.mcp.tools import reconstruction  # noqa: F401,E402
from graintrace.mcp.tools import simulation  # noqa: F401,E402
from graintrace.mcp.tools import calibration  # noqa: F401,E402
from graintrace.mcp.tools import synthetic  # noqa: F401,E402
from graintrace.mcp.tools import analysis  # noqa: F401,E402
from graintrace.mcp.tools import tracking  # noqa: F401,E402
from graintrace.mcp.tools import viz  # noqa: F401,E402
from graintrace.mcp.tools import codebase  # noqa: F401,E402

# pylint: enable=unused-import


def main() -> None:
    """Parse CLI args and run the MCP server over stdio or streamable-http."""
    parser = argparse.ArgumentParser(prog="graintrace-mcp", description=__doc__)
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over streamable-http instead of stdio.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port.")
    args = parser.parse_args()

    # Make the neml2 AOTI runtime + neml2-compile/puma-opt subprocesses find the
    # env's newer libstdc++ (CXXABI). CPFE subprocesses inherit os.environ.
    from graintrace.mcp import deps  # pylint: disable=import-outside-toplevel

    deps.ensure_runtime_ld_library_path()

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
