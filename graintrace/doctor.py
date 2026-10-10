# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Report which parts of the graintrace stack are installed on this machine.

``pip install graintrace`` ships the Python code only. NEPER, CUBIT/SCULPT,
MOOSE/PUMA and NEML2 are built separately, and which of them you have decides
which workflows can run at all. This module turns the probes that the MCP
server already uses into a plain command, so that checking an install does not
require an MCP client::

    python -m graintrace.doctor            # human-readable table
    python -m graintrace.doctor --json     # machine-readable, for scripts
    python -m graintrace.doctor --require neper,cubit   # exit 1 if either is missing

Every missing entry prints the remedy for that specific dependency. A missing
tool is not an error by itself -- the command exits 0 unless ``--require`` names
something that is absent -- because almost every external tool is optional for
some subset of the workflows.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Sequence


def _load_deps() -> ModuleType:
    """Import ``graintrace.mcp.deps`` without importing the MCP server package.

    ``graintrace/mcp/__init__.py`` imports the FastMCP application, which needs
    the optional ``mcp`` extra, while the probes themselves need none of it.
    This command has to work on a bare ``pip install graintrace``, so when the
    real package has not already been imported a stub standing in for
    ``graintrace.mcp`` is registered first. It carries the right ``__path__``,
    which is all the import machinery needs to find ``deps`` and the
    ``tool_paths`` module that one of the probes pulls in.
    """
    if "graintrace.mcp" not in sys.modules:
        stub = ModuleType("graintrace.mcp")
        stub.__path__ = [str(Path(__file__).resolve().parent / "mcp")]
        sys.modules["graintrace.mcp"] = stub
    return importlib.import_module("graintrace.mcp.deps")


def report() -> List[dict]:
    """Run every dependency probe and return one dict per dependency.

    Each dict has ``name``, ``ok``, ``detail`` (where it was found, or why it is
    missing) and ``build_hint`` (how to get it).
    """
    deps = _load_deps()
    return [
        {
            "name": s.name,
            "ok": s.ok,
            "detail": s.detail,
            "build_hint": s.build_hint,
        }
        for s in deps.check_all()
    ]


def format_report(rows: Sequence[dict]) -> str:
    """Render :func:`report` output as an aligned table plus remedies."""
    width = max((len(r["name"]) for r in rows), default=0)
    lines = ["graintrace dependency report", ""]
    for r in rows:
        mark = "ok     " if r["ok"] else "MISSING"
        lines.append(f"  {mark}  {r['name']:<{width}}  {r['detail']}")
    missing = [r for r in rows if not r["ok"]]
    if missing:
        lines += ["", "How to get the missing pieces:"]
        for r in missing:
            lines.append(f"  {r['name']}: {r['build_hint']}")
    else:
        lines += ["", "Everything graintrace can drive is present on this machine."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m graintrace.doctor``. Returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m graintrace.doctor",
        description="Report which external graintrace dependencies are installed.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of a table"
    )
    parser.add_argument(
        "--require",
        default="",
        metavar="NAME[,NAME...]",
        help="exit 1 unless every named dependency is present",
    )
    args = parser.parse_args(argv)

    rows = report()
    print(json.dumps(rows, indent=2) if args.json else format_report(rows))

    required = [n.strip() for n in args.require.split(",") if n.strip()]
    if not required:
        return 0
    by_name = {r["name"]: r for r in rows}
    absent = [n for n in required if not by_name.get(n, {"ok": False})["ok"]]
    if absent:
        print(f"\nrequired but not available: {', '.join(absent)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
